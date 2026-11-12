// Ultralytics YOLO datasets: detect, segment, pose and OBB (rotated boxes). Label rows written from
// Seamtag's shapes and read back into them, data.yaml, the rule that finds an image's label file,
// and a guess at which task a dataset's rows are for. The rules are Ultralytics 8.4.211's
// (ultralytics/data/utils.py: verify_image_label, img2label_paths and check_det_dataset), written
// again from what they do: Ultralytics is AGPL-3.0, so none of its code is copied here. Each
// image's own label file stays the working copy; these files are made from it and read into it.
import yaml from 'js-yaml'
import type { DatasetImage, Issue, KeypointSchema, MaskBits, PoseInstance, Severity } from './dataset'
import { pyFixed } from './formats'
import { newShape, type Shape } from './labelfile'
import { circlePolygon, CLOSED_TYPES } from './outline'
import { flipIndex, keypointCount, pixelBox, poseInstances, PY_SPACE, pyFloat, pyInt, pyStr, shapeBox, shapesFromPose, short, ULTRALYTICS_YAML, yamlString } from './pose'
import { convexHull, findContours, minAreaRect } from './polygonize'

/** The tasks Ultralytics trains on labelled boxes, outlines, keypoints and rotated boxes */
export const YOLO_TASKS = ['detect', 'segment', 'pose', 'obb'] as const
/** detect (boxes), segment (outlines), pose (keypoints) or obb (rotated boxes) */
export type YoloTask = (typeof YOLO_TASKS)[number]
/** Each task as Ultralytics names it, and what it's trained on, for the window */
export const TASK_NAMES: Record<YoloTask, string> = { detect: 'Detect (boxes)', segment: 'Segment (outlines)', pose: 'Pose (keypoints)', obb: 'OBB (rotated boxes)' }
const TASK_WORD: Record<YoloTask, string> = { detect: 'detect', segment: 'segment', pose: 'pose', obb: 'OBB' }

type Box = [number, number, number, number]
type Point = [number, number]

const issue = (severity: Severity, code: string, image: string | null, message: string, shape?: number): Issue =>
  (shape === undefined ? { severity, code, image, message } : { severity, code, image, shape, message })

// Files and folders

/** Ultralytics' IMG_FORMATS: what follows the last dot of the files it takes for images, in any case */
export const IMG_FORMATS = new Set(['avif', 'bmp', 'dng', 'heic', 'heif', 'jp2', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'])

/** Is this an image Ultralytics reads? Decided as its get_img_files decides, by what follows the last dot */
export const isYoloImage = (file: string) => IMG_FORMATS.has(file.slice(file.lastIndexOf('.') + 1).toLowerCase())

/**
 * Ultralytics' img2label_paths: the last /images/ in an image's path becomes /labels/, and what
 * follows its last dot becomes txt, so ds/images/train/a/b.jpg has ds/labels/train/a/b.txt. An
 * image with no images folder in its path has its labels beside it. Ultralytics only sees whole
 * paths; a path relative to a dataset's folder counts from that folder, so images/a.jpg has
 * labels/a.txt.
 */
export function labelPathFor(image: string, sep = '/'): string {
  const from = `${sep}images${sep}`
  const whole = sep + image
  const at = whole.lastIndexOf(from)
  const swapped = at < 0 ? whole : `${whole.slice(0, at)}${sep}labels${sep}${whole.slice(at + from.length)}`
  const dot = swapped.lastIndexOf('.')
  return `${dot < 0 ? swapped : swapped.slice(0, dot)}.txt`.slice(sep.length)
}

// Rows as verify_image_label reads them: f.read().strip().splitlines(), blank lines left out, and
// each line split at whitespace, all as Python does it. A byte order mark isn't whitespace there.
const SPLIT = new RegExp(`[${PY_SPACE}]+`)
const LINES = new RegExp('\\r\\n|[\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]')
type Row = { line: number; tokens: string[] }
const rowsOf = (text: string): Row[] =>
  text.split(LINES).map((l, i) => ({ line: i + 1, tokens: l.split(SPLIT).filter(Boolean) })).filter((r) => r.tokens.length)

const BOM = '\ufeff'
// Characters that don't show, such as a byte order mark, as Python's repr() writes them, so a
// message never quotes a value whose fault can't be seen; a long value cut short
const shown = (token: string) => short(token).replace(/\p{C}/gu, (c) => {
  const code = c.codePointAt(0)!
  const [prefix, digits] = code <= 0xff ? ['x', 2] : code <= 0xffff ? ['u', 4] : ['U', 8]
  return `\\${prefix}${code.toString(16).padStart(digits, '0')}`
})

/** A label file's bytes as Python reads them with encoding="utf-8": null when they aren't UTF-8. A byte order mark stays. */
export function labelText(bytes: Uint8Array): string | null {
  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes)
  } catch {
    return null
  }
}

// Ultralytics' verdict on a label file

/** What verify_image_label makes of a label file */
export type Verdict = {
  /** ok; missing (no file) and empty (no rows) are backgrounds; corrupt images are left out of training */
  status: 'ok' | 'missing' | 'empty' | 'corrupt'
  /** Why it's corrupt, as a clause that follows "as" */
  reason: string | null
  /** The line it's about, from 1 */
  line: number | null
  /** The lines of rows that repeat an earlier one, which Ultralytics drops */
  repeated: number[]
  /** The rows were read as outlines (some had more than 6 values) */
  outline: boolean
  /** Each row as Ultralytics keeps it, in float32: class and box (centre, size), in file order */
  boxes: number[][]
  /** The outlines' points, for outline rows */
  outlines: number[][][]
  /** Pose: each row's keypoints, x, y and visibility */
  keypoints: number[][][] | null
}

const f32 = Math.fround
// Ultralytics compares float32 values with Python floats, which NumPy 2 rounds to float32 first
const TOP = f32(1.01), BOTTOM = f32(-0.01)
const nanMax = (values: number[]) => values.reduce((m, v) => (Number.isNaN(m) || Number.isNaN(v) ? NaN : Math.max(m, v)), -Infinity)
const nanMin = (values: number[]) => values.reduce((m, v) => (Number.isNaN(m) || Number.isNaN(v) ? NaN : Math.min(m, v)), Infinity)
const bits = (values: number[]) => Array.from(new Uint32Array(new Float32Array(values).buffer)).join(',')

// Ultralytics' segments2boxes and xyxy2xywh, in float32 as NumPy does them
function outlineBox(points: number[][]): number[] {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1])
  const x0 = nanMin(xs), y0 = nanMin(ys), x1 = nanMax(xs), y1 = nanMax(ys)
  return [f32(f32(x0 + x1) / 2), f32(f32(y0 + y1) / 2), f32(x1 - x0), f32(y1 - y0)]
}

/**
 * Ultralytics 8.4.211's verify_image_label on a label file's text (null for no file), for a task,
 * a class count and data.yaml's kpt_shape. The image itself isn't checked. Its rules, in order:
 * a row of 5 + K × ndim values is cut to its box for a task other than pose; a file with a row of
 * more than 6 values is outlines, where a row of 5 values is refused ("labels mix segment and
 * detection rows") and each outline becomes its box; then every row must have 5 values (pose:
 * 5 + K × ndim), every coordinate at most 1.01 (pose: the keypoints' x and y only; outlines:
 * their boxes), nothing below 0 unless the lowest is at least -0.01 and no size is below 0,
 * classes whole numbers below the class count. Repeated rows are dropped, outlines only when their
 * points repeat too.
 */
export function verifyRows(text: string | null, task: YoloTask, nc: number, kptShape: [number, number] | null = null): Verdict {
  const base: Verdict = { status: 'ok', reason: null, line: null, repeated: [], outline: false, boxes: [], outlines: [], keypoints: null }
  if (text === null) return { ...base, status: 'missing' }
  const corrupt = (reason: string, line: number | null): Verdict => ({ ...base, status: 'corrupt', reason, line })
  // strip() leaves a byte order mark on the first value, and every path ends in float() of it
  if (text.startsWith(BOM)) return corrupt('it starts with a byte order mark, which Ultralytics reads as part of the first number', 1)
  const pose = task === 'pose'
  const [nkpt, ndim] = kptShape ?? [0, 0]
  let rows = rowsOf(text)
  // "pose labels for a box task: keep the box, drop the keypoints"
  if (nkpt && !pose) rows = rows.map((r) => (r.tokens.length === 5 + nkpt * ndim ? { ...r, tokens: r.tokens.slice(0, 5) } : r))
  if (!rows.length) return { ...base, status: 'empty' }
  let lb: number[][]
  const outlines: number[][][] = []
  const outline = !pose && rows.some((r) => r.tokens.length > 6)
  if (outline) {
    const box = rows.find((r) => r.tokens.length === 5)
    if (box) return corrupt('it mixes box rows of 5 values with outline rows of 7 or more', box.line)
    for (const r of rows) if (pyFloat(r.tokens[0]) === null) return corrupt(`line ${r.line}'s class, ${shown(r.tokens[0])}, isn't a number`, r.line)
    for (const r of rows) {
      const values = r.tokens.slice(1).map(pyFloat)
      const bad = values.findIndex((v) => v === null)
      if (bad >= 0) return corrupt(`line ${r.line} has ${shown(r.tokens[bad + 1])}, which isn't a number`, r.line)
      if (values.length % 2) return corrupt(`line ${r.line} has an odd number of coordinates`, r.line)
      if (!values.length) return corrupt(`line ${r.line} has a class but no points`, r.line)
      const points: number[][] = []
      for (let k = 0; k < values.length; k += 2) points.push([f32(values[k]!), f32(values[k + 1]!)])
      outlines.push(points)
    }
    lb = rows.map((r, i) => [f32(pyFloat(r.tokens[0])!), ...outlineBox(outlines[i])])
  } else {
    // NumPy makes one array of all rows, so they must be the same length and all numbers
    const n = rows[0].tokens.length
    const odd = rows.find((r) => r.tokens.length !== n)
    if (odd) return corrupt(`line ${odd.line} has ${odd.tokens.length} values where line ${rows[0].line} has ${n}`, odd.line)
    lb = []
    for (const r of rows) {
      const values = r.tokens.map(pyFloat)
      const bad = values.findIndex((v) => v === null)
      if (bad >= 0) return corrupt(`line ${r.line} has ${shown(r.tokens[bad])}, which isn't a number`, r.line)
      lb.push(values.map((v) => f32(v!)))
    }
  }
  const want = pose ? 5 + nkpt * ndim : 5
  // "labels require {5 + nkpt * ndim} columns each", "labels require 5 columns"
  if (lb[0].length !== want) {
    return corrupt(pose ? `its rows have ${lb[0].length} values where kpt_shape [${nkpt}, ${ndim}] needs ${want}` : `its rows have ${lb[0].length} values where a box has 5`, rows[0].line)
  }
  const pointsOf = (r: number[]) => {
    if (!pose) return r.slice(1)
    const out: number[] = []
    for (let k = 0; k < nkpt; k++) out.push(r[5 + k * ndim], r[6 + k * ndim])
    return out
  }
  // "non-normalized or out of bounds coordinates", with 1% to spare
  if (!(nanMax(lb.flatMap(pointsOf)) <= TOP)) {
    const i = lb.findIndex((r) => !(nanMax(pointsOf(r)) <= TOP))
    return corrupt(`line ${rows[i].line} has ${outline ? 'an outline whose box reaches' : 'a coordinate'} past 1.01, or one that isn't a number`, rows[i].line)
  }
  // "negative label values": a little below 0 passes when no size is negative
  const lowest = nanMin(lb.flat())
  if (!(lowest >= 0 || (lowest >= BOTTOM && nanMin(lb.flatMap((r) => [r[3], r[4]])) >= 0))) {
    const i = lb.findIndex((r) => !(nanMin(r) >= 0))
    return corrupt(`line ${rows[i].line} has a value below 0`, rows[i].line)
  }
  // "non-integer class labels"
  const fraction = lb.findIndex((r) => r[0] % 1 !== 0)
  if (fraction >= 0) return corrupt(`line ${rows[fraction].line}'s class, ${shown(rows[fraction].tokens[0])}, isn't a whole number`, rows[fraction].line)
  // "Label class 3 exceeds dataset class count 3"
  const past = lb.findIndex((r) => !(r[0] < nc))
  if (past >= 0) return corrupt(`line ${rows[past].line}'s class, ${shown(rows[past].tokens[0])}, is past the ${nc} classes (0 to ${nc - 1})`, rows[past].line)
  // np.unique over the rows; "distinct polygons can share a class and box", so outlines are compared again with their points, bit for bit
  const repeats = (key: (r: number[], i: number) => string) => {
    const seen = new Set<string>()
    return lb.map((r, i) => { const k = key(r, i); const again = seen.has(k); seen.add(k); return again })
  }
  let again = repeats((r) => r.join(','))
  if (again.some(Boolean) && outlines.length) again = repeats((r, i) => `${bits([r[0]])}|${bits(outlines[i].flat())}`)
  const keep = lb.map((_, i) => !again[i])
  const keypoints = pose
    ? lb.filter((_, i) => keep[i]).map((r) => Array.from({ length: nkpt }, (_, k) => {
      const [x, y] = [r[5 + k * ndim], r[6 + k * ndim]]
      // Without visibility, a keypoint below 0 on either axis is missing
      return ndim === 3 ? [x, y, r[7 + k * ndim]] : [x, y, x < 0 || y < 0 ? 0 : 1]
    }))
    : null
  return {
    ...base,
    repeated: rows.filter((_, i) => again[i]).map((r) => r.line),
    outline,
    boxes: lb.filter((_, i) => keep[i]).map((r) => r.slice(0, 5)),
    outlines: outlines.filter((_, i) => keep[i]),
    keypoints
  }
}

// Writing rows

/** An image's label file for a task: its text, how many rows it has, and what happened to shapes on the way */
export type Rows = { text: string; rows: number; issues: Issue[] }

const KIND: Record<string, string> = { line: 'a line', linestrip: 'a line strip', point: 'a point', points: 'a set of points' }

// A quadrilateral's sides, each from one corner to the next
function sides(points: number[][]) {
  return [0, 1, 2, 3].map((i) => [points[(i + 1) % 4][0] - points[i][0], points[(i + 1) % 4][1] - points[i][1]])
}

/** Are a quadrilateral's opposite sides equal and parallel, within a share of its longest side? */
function isParallelogram(points: number[][], tolerance = 0.01): boolean {
  if (points.length !== 4) return false
  const [a, b, c, d] = sides(points)
  const long = Math.max(...[a, b, c, d].map(([x, y]) => Math.hypot(x, y)))
  return long > 0 && Math.hypot(a[0], a[1]) > 0 && Math.hypot(b[0], b[1]) > 0 &&
    Math.hypot(a[0] + c[0], a[1] + c[1]) <= tolerance * long && Math.hypot(b[0] + d[0], b[1] + d[1]) <= tolerance * long
}

/** Is this quadrilateral a rectangle: opposite sides equal and parallel, and square corners, within a share of its longest side? */
export function isRectangle(points: number[][], tolerance = 0.01): boolean {
  if (!isParallelogram(points, tolerance)) return false
  const [a, b] = sides(points)
  return Math.abs(a[0] * b[0] + a[1] * b[1]) <= tolerance * Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1])
}

const cornersOf = ([x0, y0, x1, y1]: Box): Point[] => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

function signedArea(ring: number[][]) {
  let sum = 0
  for (let i = 0; i < ring.length; i++) {
    const [ax, ay] = ring[i], [bx, by] = ring[(i + 1) % ring.length]
    sum += ax * by - bx * ay
  }
  return sum / 2
}

/** Points that lie on a straight line between their neighbours add nothing to an outline */
function dropStraight(ring: number[][]): number[][] {
  const out: number[][] = []
  const straight = (a: number[], b: number[], c: number[]) =>
    (b[0] - a[0]) * (c[1] - b[1]) === (b[1] - a[1]) * (c[0] - b[0]) && (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) > 0
  for (const p of ring) {
    while (out.length >= 2 && straight(out[out.length - 2], out[out.length - 1], p)) out.pop()
    out.push(p)
  }
  // The ring closes back to its start
  while (out.length >= 3 && straight(out[out.length - 2], out[out.length - 1], out[0])) out.pop()
  while (out.length >= 3 && straight(out[out.length - 1], out[0], out[1])) out.shift()
  return out
}

/**
 * A mask's outer outlines, one for each separate part, and how many holes it has. scikit-image's
 * find_contours (polygonize.ts) puts pixel centres at whole numbers; in the image, pixel (x, y)
 * spans x to x + 1, so its centre is at x + 0.5 and the outlines run along the pixels' edges,
 * with the corners cut. Points on straight runs between two others are left out, as OpenCV's
 * CHAIN_APPROX_SIMPLE leaves them out.
 */
export function maskOutlines(m: MaskBits): { outer: number[][][]; holes: number } {
  const W = m.width + 2, H = m.height + 2
  // A row and column of background around it, so parts at the edge close too
  const padded = new Uint8Array(W * H)
  for (let y = 0; y < m.height; y++) for (let x = 0; x < m.width; x++) padded[(y + 1) * W + x + 1] = m.data[y * m.width + x] ? 1 : 0
  const outer: number[][][] = []
  let holes = 0
  for (const contour of findContours({ data: padded, width: W, height: H })) {
    let ring = contour.map(([r, c]) => [m.x + c - 0.5, m.y + r - 0.5])
    const [first, last] = [ring[0], ring[ring.length - 1]]
    if (first[0] === last[0] && first[1] === last[1]) ring = ring.slice(0, -1)
    // labelme's compute_polygons_from_mask keeps contours of positive area: the others go round holes
    if (signedArea(ring) > 0) outer.push(dropStraight(ring))
    else holes++
  }
  return { outer, holes }
}

/** The corners of a mask's pixels that a hull around them can touch: each row's outer two pixels' */
function maskCorners(m: MaskBits): Point[] {
  const out: Point[] = []
  for (let y = 0; y < m.height; y++) {
    let lo = -1, hi = -1
    for (let x = 0; x < m.width; x++) if (m.data[y * m.width + x]) { if (lo < 0) lo = x; hi = x }
    if (lo >= 0) out.push([m.x + lo, m.y + y], [m.x + lo, m.y + y + 1], [m.x + hi + 1, m.y + y], [m.x + hi + 1, m.y + y + 1])
  }
  return out
}

/** Consecutive repeats out, the closing point too */
function distinct(points: string[][]): string[][] {
  const out = points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1])
  while (out.length > 1 && out[0][0] === out[out.length - 1][0] && out[0][1] === out[out.length - 1][1]) out.pop()
  return out
}

// What every writer of rows does: shares of the image to 6 places, kept within 0 to 1, a row
// written once, and an issue for each change on the way
function rowWriter(image: DatasetImage) {
  const issues: Issue[] = []
  const say = (severity: Severity, code: string, message: string, shape?: number) => issues.push(issue(severity, code, image.name, `${image.name}: ${message}`, shape))
  const lines: string[] = []
  const written = new Set<string>()
  const { width: W, height: H } = image
  return {
    say,
    done: (): Rows => ({ text: lines.map((l) => `${l}\n`).join(''), rows: lines.length, issues }),
    /** False, with an error, when the image's labels or size can't be read */
    ready() {
      if (image.broken) say('error', 'yolo-broken', `its label file can't be read (${image.broken}), so it's left out`)
      else if (!(W > 0 && H > 0)) say('error', 'yolo-no-size', "the image's size can't be read, so its labels can't be written as shares of it")
      else return true
      return false
    },
    point: ([px, py]: number[]) => [pyFixed(Math.min(Math.max(px, 0), W) / W), pyFixed(Math.min(Math.max(py, 0), H) / H)],
    box([x0, y0, x1, y1]: Box) {
      const [a, c] = [x0, x1].map((v) => Math.min(Math.max(v, 0), W)), [b, d] = [y0, y1].map((v) => Math.min(Math.max(v, 0), H))
      return `${pyFixed((a + c) / 2 / W)} ${pyFixed((b + d) / 2 / H)} ${pyFixed((c - a) / W)} ${pyFixed((d - b) / H)}`
    },
    /** Why a box can't be written: it has no width or height, or no part of it is in the image */
    misplaced: ([x0, y0, x1, y1]: Box): 'flat' | 'outside' | null =>
      !(x1 - x0 > 0 && y1 - y0 > 0) ? 'flat' : !(Math.min(x1, W) - Math.max(x0, 0) > 0 && Math.min(y1, H) - Math.max(y0, 0) > 0) ? 'outside' : null,
    // Cut at the image's edge: shapes a pixel or less past it were drawn to the edge
    cut(box: Box, label: string, shape?: number) {
      const over = Math.max(-box[0], -box[1], box[2] - W, box[3] - H)
      if (over > 1e-9) say(over > 1 ? 'warning' : 'info', 'yolo-clamped', `${label} reaches past the image's edge, so it's cut there`, shape)
    },
    add(row: string, label: string, shape?: number) {
      if (written.has(row)) say('info', 'yolo-repeated', `${label} gives the same row as a shape before it, so it's written once, as Ultralytics would keep it`, shape)
      else {
        written.add(row)
        lines.push(row)
      }
    }
  }
}

/**
 * YOLO pose rows for an image's posed objects (pose.ts's poseInstances): each object's box and K
 * keypoints of x, y and visibility (2 seen, 1 hidden), where K is the most any class of the
 * schema has, so every row is as long and visibility is always there, as kpt_shape [K, 3] says.
 * Missing keypoints, those outside the image and the padding are 0 0 0. `classes` gives the
 * class numbers: the schema's classes in order.
 */
export function poseRows(image: DatasetImage, instances: PoseInstance[], classes: string[], schema: KeypointSchema): Rows {
  const w = rowWriter(image)
  if (!w.ready()) return w.done()
  const k = keypointCount(schema)
  for (const p of instances) {
    const cls = classes.indexOf(p.label)
    if (cls < 0) {
      w.say('warning', 'yolo-unknown-label', `${p.label} isn't in the class list, so it's left out`, p.shapes[0])
      continue
    }
    const where = w.misplaced(p.box)
    // Ultralytics trains without an object whose box has no width or height, keypoints and all:
    // Instances.remove_zero_area_boxes after Mosaic, RandomPerspective.box_candidates without it
    // (ultralytics/data/augment.py), though verify_image_label takes the row
    if (where === 'flat') {
      w.say('warning', 'yolo-no-area', `${p.label}'s box has no width or height, so it's left out with its keypoints, as Ultralytics would drop it in training`, p.shapes[0])
      continue
    }
    if (where === 'outside') {
      w.say('warning', 'yolo-outside', `${p.label} lies outside the image, so it's left out`, p.shapes[0])
      continue
    }
    w.cut(p.box, p.label, p.shapes[0])
    const names = schema.classes[p.label] ?? []
    const values: string[] = []
    for (let kk = 0; kk < k; kk++) {
      const kp = p.keypoints[kk] ?? null
      if (kp && (kp[0] < 0 || kp[1] < 0 || kp[0] > image.width || kp[1] > image.height)) {
        const shape = p.shapes.find((i) => image.shapes[i]?.shape_type === 'point' && image.shapes[i].label === names[kk])
        w.say('warning', 'yolo-keypoint-outside', `${names[kk]} of ${p.label} is outside the image, so it's written as missing`, shape)
        values.push('0 0 0')
      } else values.push(kp ? `${w.point(kp).join(' ')} ${kp[2]}` : '0 0 0')
    }
    w.add(`${cls} ${w.box(p.box)} ${values.join(' ')}`, p.label, p.shapes[0])
  }
  return w.done()
}

// numpy's argmin over the squared distance of every pair of points, as Ultralytics' min_index
// takes it: the first smallest, by the first outline's point, then the second's
function closest(a: number[][], b: number[][]): [number, number] {
  let best = Infinity, ia = 0, ib = 0
  for (let i = 0; i < a.length; i++) {
    const [ax, ay] = a[i]
    for (let j = 0; j < b.length; j++) {
      const dx = ax - b[j][0], dy = ay - b[j][1], d = dx * dx + dy * dy
      if (d < best) {
        best = d
        ia = i
        ib = j
      }
    }
  }
  return [ia, ib]
}

/**
 * Outlines of one object as a single outline, joined as Ultralytics 8.4.211's merge_multi_segment
 * joins a COCO object's polygons in convert_coco (ultralytics/data/converter.py), written again
 * from what it does. Each outline is tied to the next at their two closest points. The path goes
 * round the first outline from its tie and back to it, along each middle outline from its tie
 * with the one before to its tie with the one after (turned the other way round when that tie
 * comes first in its points), round the last outline, then back along the middle outlines' other
 * sides, so it crosses every tie there and back. One outline comes back as it is.
 */
export function joinOutlines(rings: number[][][]): number[][] {
  if (rings.length < 2) return rings[0] ?? []
  const n = rings.length
  const ties: number[][] = rings.map(() => [])
  for (let i = 1; i < n; i++) {
    const [a, b] = closest(rings[i - 1], rings[i])
    ties[i - 1].push(a)
    ties[i].push(b)
  }
  // Each outline from its first tie round to it again, and where its second tie is on the way
  const walks = rings.map((ring, i) => {
    let [from, to = from] = ties[i]
    let points = ring
    if (ties[i].length === 2 && from > to) {
      points = [...ring].reverse()
      ;[from, to] = [ring.length - 1 - from, ring.length - 1 - to]
    }
    return { round: [...points.slice(from), ...points.slice(0, from), points[from]], to: to - from }
  })
  const path = [...walks[0].round]
  for (let i = 1; i < n - 1; i++) path.push(...walks[i].round.slice(0, walks[i].to + 1))
  path.push(...walks[n - 1].round)
  for (let i = n - 2; i >= 1; i--) path.push(...walks[i].round.slice(walks[i].to))
  return path
}

/**
 * An image's label file for a task: one row an object, as shares of the image written with
 * Python's '%.6f' and kept within 0 to 1, so Ultralytics 8.3 to 8.4 accept every row. Shapes
 * sharing a label and a group_id are one object, as COCO's export and labelme 7.9.1's
 * labelme2coco take them; any other shape is one on its own, a mask in separate parts too.
 * - detect: every object's box (a mask's around its pixels; several shapes' around them all)
 * - segment: outlines; a box as its 4 corners, so box and outline rows never share a file;
 *   a circle as labelme2coco's polygon; a mask as its outline, holes filled; an object of
 *   several parts as one outline, joined as Ultralytics' convert_coco joins them (joinOutlines)
 * - obb: 4 corners; a rotated rectangle's as stored, a box's from the top left round, any other
 *   shape's or object of several shapes' smallest rotated rectangle
 * - pose: the posed objects poseInstances finds with the schema, as poseRows writes them
 * Lines, line strips, points (outside pose) and sets of points are left out, and so is a row
 * that repeats another, which Ultralytics would drop. An image with no rows gets an empty file,
 * which Ultralytics reads as a background. `classes` gives the class numbers: for pose, the
 * schema's classes in order.
 */
export function labelRows(image: DatasetImage, task: YoloTask, classes: string[], schema?: KeypointSchema): Rows {
  if (task === 'pose') {
    if (!schema) throw new Error('A YOLO pose export needs a keypoint file')
    const { instances, issues } = poseInstances(image, schema)
    const keypointNames = new Set(Object.values(schema.classes).flat())
    const used = new Set(instances.flatMap((p) => p.shapes))
    const told = new Set(issues.map((f) => f.shape))
    image.shapes.forEach((s, i) => {
      if (used.has(i) || told.has(i)) return
      if (CLOSED_TYPES.has(s.shape_type) && Object.hasOwn(schema.classes, s.label)) return
      if (s.shape_type === 'point' && keypointNames.has(s.label)) return
      issues.push(issue('warning', 'yolo-not-pose', image.name, `${image.name}: ${s.label} isn't a class or keypoint in the keypoint file, so it's left out`, i))
    })
    const rows = poseRows(image, instances, classes, schema)
    return { ...rows, issues: [...issues, ...rows.issues] }
  }
  const w = rowWriter(image)
  if (!w.ready()) return w.done()
  const { say } = w
  const { width: W, height: H } = image
  type Part = { index: number; shape: Shape; mask: MaskBits | null; box: Box }
  // A shape that can be part of an object, or null, saying why it can't
  const partOf = (i: number): Part | null => {
    const s = image.shapes[i], type = s.shape_type
    const leave = (severity: Severity, code: string, message: string) => { say(severity, code, message, i); return null }
    if (!CLOSED_TYPES.has(type)) return leave('warning', 'yolo-open-shape', `${s.label} is ${KIND[type] ?? `a ${type}`}, which YOLO ${TASK_WORD[task]} files can't hold, so it's left out`)
    if (!classes.includes(s.label)) return leave('warning', 'yolo-unknown-label', `${s.label} isn't in the class list, so it's left out`)
    const mask = type === 'mask' ? image.masks[i] ?? null : null
    if (type === 'mask' && !mask) return leave('error', 'yolo-mask-unreadable', `${s.label}'s mask can't be decoded, so it's left out`)
    const box = mask ? pixelBox(mask) : shapeBox(s)
    if (!box) return leave('warning', 'yolo-no-area', `${s.label} has no pixels, so it's left out`)
    const where = w.misplaced(box)
    if (where === 'flat') return leave('warning', 'yolo-no-area', `${s.label} has no area, so it's left out`)
    if (where === 'outside') return leave('warning', 'yolo-outside', `${s.label} lies outside the image, so it's left out`)
    w.cut(box, s.label, i)
    return { index: i, shape: s, mask, box }
  }
  // Objects in the order of their first shapes, each shape of an object read in turn
  const objects = new Map<string | number, number[]>()
  image.shapes.forEach((s, i) => {
    const key = s.group_id === null ? i : `${s.label}\u0000${s.group_id}`
    objects.set(key, [...(objects.get(key) ?? []), i])
  })
  for (const members of objects.values()) {
    const parts = members.map(partOf).filter((p): p is Part => p !== null)
    if (!parts.length) continue
    const first = parts[0], { label, group_id: group } = first.shape, cls = classes.indexOf(label), at = first.index
    const shapes = `${label} is ${parts.length} shapes of group ${group}, one object,`
    if (task === 'detect') {
      if (parts.length > 1) say('info', 'yolo-joined', `${shapes} so it's one box around them all`, at)
      const box = parts.map((p) => p.box).reduce((a, b): Box => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])])
      w.add(`${cls} ${w.box(box)}`, label, at)
      continue
    }
    if (task === 'obb') {
      const s = first.shape, type = s.shape_type
      let corners: number[][] | null = null
      if (parts.length === 1) {
        corners = type === 'rectangle' ? cornersOf(first.box) : type === 'oriented_rectangle' ? s.points : null
        if (type === 'polygon' && isRectangle(s.points)) corners = s.points
      }
      if (!corners) {
        // Every square around a circle is as small as any other: the upright one. With other
        // shapes, a circle counts as its polygon.
        const around = parts.flatMap((p): Point[] => p.mask ? maskCorners(p.mask) : p.shape.shape_type === 'circle'
          ? circlePolygon(p.shape.points[0], p.shape.points[1]) as Point[] : p.shape.shape_type === 'rectangle' ? cornersOf(p.box) : p.shape.points as Point[])
        const hull = parts.length === 1 && type === 'circle' ? cornersOf(first.box) : convexHull(around)
        if (!hull) {
          say('warning', 'yolo-no-area', `${label} has no area, so it's left out`, at)
          continue
        }
        corners = parts.length === 1 && type === 'circle' ? hull : minAreaRect(hull)
        say('warning', 'yolo-obb-fitted', parts.length > 1 ? `${shapes} so the smallest rotated box around them all is written` : `${label} is a ${type}, so the smallest rotated box around it is written`, at)
      }
      const points = distinct(corners.map(w.point))
      if (points.length < 3) say('warning', 'yolo-few-points', `${label} has fewer than 3 points apart, so it's left out`, at)
      else w.add(`${cls} ${points.map((p) => p.join(' ')).join(' ')}`, label, at)
      continue
    }
    // Segment: every part's outlines, a mask's one for each separate part of it
    const rings: { ring: number[][]; index: number }[] = []
    for (const p of parts) {
      const s = p.shape
      if (p.mask) {
        const { outer, holes } = maskOutlines(p.mask)
        if (holes) say('warning', 'yolo-mask-holes', `${s.label}'s mask has ${holes === 1 ? 'a hole' : `${holes} holes`}, which a YOLO outline can't hold, so ${holes === 1 ? "it's" : "they're"} filled`, p.index)
        for (const ring of outer) rings.push({ ring, index: p.index })
      } else if (s.shape_type === 'circle') rings.push({ ring: circlePolygon(s.points[0], s.points[1])!, index: p.index })
      else rings.push({ ring: s.shape_type === 'rectangle' ? cornersOf(p.box) : s.points, index: p.index })
    }
    const kept = rings.filter(({ ring, index }) => {
      if (distinct(ring.map(w.point)).length >= 3) return true
      say('warning', 'yolo-few-points', rings.length > 1
        ? `${label} has a part with fewer than 3 points apart, so that part is left out`
        : `${label} has fewer than 3 points apart, so it's left out`, index)
      return false
    })
    if (!kept.length) continue
    if (kept.length > 1) {
      say('info', 'yolo-joined', parts.length > 1
        ? `${shapes} so its ${kept.length} separate parts are joined into one outline, as Ultralytics' convert_coco joins an object's parts`
        : `${label}'s mask is in ${kept.length} separate parts, one object, so they're joined into one outline, as Ultralytics' convert_coco joins an object's parts`, at)
    }
    // Parts are joined where they lie in the image, as COCO's export keeps them
    const inside = kept.map(({ ring }) => ring.map(([x, y]) => [Math.min(Math.max(x, 0), W), Math.min(Math.max(y, 0), H)]))
    const points = distinct(joinOutlines(inside).map(w.point))
    w.add(`${cls} ${points.map((p) => p.join(' ')).join(' ')}`, label, at)
  }
  return w.done()
}

// data.yaml

/** The keys Seamtag writes in data.yaml. Folders are relative to the YAML's own folder. */
export type DataYamlSpec = {
  names: string[]
  train: string
  val: string
  test?: string | null
  /** Pose: K keypoints a row, each x, y and visibility */
  kptShape?: [number, number] | null
  flipIdx?: number[] | null
  /** Pose: each class's keypoints' names, by class number */
  kptNames?: string[][] | null
}

/**
 * data.yaml as Ultralytics reads it: train, val and test folders relative to the file, with no
 * path key, so the root is the file's own folder wherever it's run from; names as a map from
 * class number to name, and nc; for pose, kpt_shape, flip_idx and kpt_names as
 * cfg/datasets/coco8-pose.yaml writes them. Names are quoted where a YAML reader would take
 * them for something else (yes, 1, a: b).
 */
export function writeDataYaml(d: DataYamlSpec): string {
  const lines = ['# Folders are relative to this file: with no path key, Ultralytics takes its folder as the root',
    `train: ${yamlString(d.train)}`, `val: ${yamlString(d.val)}`]
  if (d.test) lines.push(`test: ${yamlString(d.test)}`)
  if (d.kptShape) {
    lines.push('', `kpt_shape: [${d.kptShape[0]}, ${d.kptShape[1]}]`)
    if (d.flipIdx?.length) lines.push(`flip_idx: [${d.flipIdx.join(', ')}]`)
  }
  lines.push('', `nc: ${d.names.length}`, 'names:', ...d.names.map((n, i) => `  ${i}: ${yamlString(n)}`))
  if (d.kptNames?.some((names) => names?.length)) {
    lines.push('', 'kpt_names:')
    d.kptNames.forEach((names, i) => {
      if (names?.length) lines.push(`  ${i}:`, ...names.map((n) => `    - ${yamlString(n)}`))
    })
  }
  return lines.join('\n') + '\n'
}

/** data.yaml's pose keys from a keypoint file: its classes as names, kpt_shape [K, 3], kpt_names and flip_idx */
export function poseYamlKeys(schema: KeypointSchema): { names: string[]; kptShape: [number, number]; kptNames: string[][]; flipIdx: number[] | null; issues: Issue[] } {
  const names = Object.keys(schema.classes)
  const { flipIdx, issues } = flipIndex(schema)
  return { names, kptShape: [keypointCount(schema), 3], kptNames: names.map((n) => schema.classes[n]), flipIdx, issues }
}

/** A split in data.yaml: a folder, an image, a .txt list of images, or a list of those */
export type SplitSpec = string | string[] | null

/** data.yaml as Ultralytics' check_det_dataset takes it */
export type DataYaml = {
  /** The root folder, when given; relative to the working folder in Ultralytics */
  path: string | null
  train: SplitSpec
  val: SplitSpec
  test: SplitSpec
  /** Class names, by class number */
  names: string[]
  kptShape: [number, number] | null
  kptNames: Record<number, string[]> | null
  flipIdx: number[] | null
}

/**
 * The most classes Seamtag names itself, class_0 on, when a dataset gives only their number or
 * none at all: ImageNet-21k has 21,843. Each is a string kept while the dataset is open.
 */
export const MOST_CLASSES = 100_000

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const typeName = (v: unknown) => (Array.isArray(v) ? 'a list' : isObject(v) ? 'a map' : typeof v === 'number' ? 'a number' : typeof v === 'boolean' ? 'true or false' : 'text')
const loadYaml = (text: string) => yaml.load(text.startsWith(BOM) ? text.slice(1) : text, { schema: ULTRALYTICS_YAML })

/**
 * Is this YAML a dataset's? check_det_dataset needs train and val, each a folder, a file or a
 * list, so a map holding one of them (or validation, read as val) is taken for one. X-AnyLabeling's
 * keypoint file, a model's settings and a training run's args.yaml (val: true) aren't.
 */
export function isDatasetYaml(text: string): boolean {
  try {
    const raw = loadYaml(text)
    return isObject(raw) && ['train', 'val', 'validation'].some((k) => typeof raw[k] === 'string' || Array.isArray(raw[k]))
  } catch {
    return false
  }
}

/**
 * data.yaml, read as Ultralytics 8.4.211 reads it (its own YAML reader, then check_det_dataset's
 * checks): train and val are needed (validation is taken for val), names as a list or a map from
 * class number to name, or nc alone for class_0, class_1 and so on, and both must agree. What
 * Ultralytics refuses comes back as errors.
 */
export function readDataYaml(text: string, file = 'data.yaml'): { data: DataYaml; issues: Issue[] } {
  const issues: Issue[] = []
  const data: DataYaml = { path: null, train: null, val: null, test: null, names: [], kptShape: null, kptNames: null, flipIdx: null }
  const say = (severity: Severity, code: string, message: string) => issues.push(issue(severity, code, null, `${file} ${message}`))
  let raw: unknown
  try {
    raw = loadYaml(text)
  } catch (e) {
    say('error', 'yolo-yaml', `isn't valid YAML (${(e as Error).message.split('\n')[0]}), so it can't be read`)
    return { data, issues }
  }
  if (raw === undefined || raw === null) raw = {}
  if (!isObject(raw)) {
    say('error', 'yolo-yaml', "isn't a map of keys and values, so it can't be read")
    return { data, issues }
  }
  const d = { ...raw }
  // Ultralytics' DATASET_KEY_TYPES
  const kinds: Record<string, string[]> = { path: ['text'], train: ['text', 'a list'], val: ['text', 'a list'], test: ['text', 'a list'], names: ['a list', 'a map'], kpt_shape: ['a list'], flip_idx: ['a list'] }
  for (const [key, allowed] of Object.entries(kinds)) {
    if (d[key] === null || d[key] === undefined || allowed.includes(typeName(d[key]))) continue
    say('error', 'yolo-yaml', `has ${typeName(d[key])} for ${key}, where Ultralytics needs ${allowed.join(' or ')}`)
    delete d[key]
  }
  if (!('val' in d) && 'validation' in d) {
    say('info', 'yolo-yaml', 'has validation where Ultralytics expects val, which it reads the same')
    d.val = d.validation
  }
  for (const key of ['train', 'val']) if (!(key in d)) say('error', 'yolo-yaml', `has no ${key}, which Ultralytics needs`)
  const split = (key: string): SplitSpec => {
    const v = d[key]
    if (typeof v === 'string') return v || null
    if (!Array.isArray(v)) return null
    if (v.some((x) => typeof x !== 'string')) {
      say('error', 'yolo-yaml', `has a ${key} list with something other than folder names in it`)
      return null
    }
    return v.length ? (v as string[]) : null
  }
  ;[data.train, data.val, data.test] = [split('train'), split('val'), split('test')]
  if (typeof d.path === 'string') data.path = d.path

  let nc: number | null = null
  if (d.names === null || d.names === undefined) {
    if (!('nc' in d)) say('error', 'yolo-yaml', 'has neither names nor nc, so the classes are unknown')
  }
  if ('nc' in d) {
    // nc may be written 10, 10.0 or '10', but not 1.9
    const v = d.nc
    const n = typeof v === 'number' ? v : typeof v === 'boolean' ? Number(v) : typeof v === 'string' ? pyFloat(v) : null
    if (n === null || !Number.isInteger(n)) say('error', 'yolo-yaml', `has nc: ${shown(pyStr(v))}, which isn't a whole number`)
    else nc = n
  }
  if (d.names !== null && d.names !== undefined) {
    // check_class_names: numbers as keys, names as text, numbered from 0 with none missing
    const entries: [number, string][] = []
    const pairs: [unknown, unknown][] = Array.isArray(d.names) ? d.names.map((v, i) => [i, v]) : Object.entries(d.names as Record<string, unknown>)
    let ok = true
    for (const [k, v] of pairs) {
      const id = typeof k === 'number' ? k : pyInt(String(k))
      if (id === null) { say('error', 'yolo-yaml', `has ${shown(String(k))} as a class number in names`); ok = false; continue }
      if (isObject(v) || Array.isArray(v)) { say('error', 'yolo-yaml', `has ${typeName(v)} as the name of class ${id}`); ok = false; continue }
      entries.push([id, pyStr(v)])
    }
    if (nc !== null && pairs.length !== nc) { say('error', 'yolo-yaml', `has ${pairs.length} names but nc: ${nc}, which must agree`); ok = false }
    entries.sort((a, b) => a[0] - b[0])
    if (ok && !entries.length) say('error', 'yolo-yaml', 'has no class names, and Ultralytics needs at least one')
    else if (ok && entries[entries.length - 1][0] >= entries.length) {
      say('error', 'yolo-yaml', `numbers its ${entries.length} classes ${entries[0][0]} to ${entries[entries.length - 1][0]}, where they must run from 0 to ${entries.length - 1}`)
    } else if (ok) data.names = entries.map(([, n]) => n)
    for (const [k, v] of pairs) {
      if (v !== null && typeof v !== 'boolean') continue
      const how = v === null ? 'a name left empty or written null, None or ~' : 'true or false without quotes'
      say('warning', 'yolo-yaml-name', `names class ${shown(String(k))} ${pyStr(v)}, as Ultralytics reads ${how}: put the name in quotes to keep it`)
    }
  } else if (nc !== null) {
    // A name is made for each class, so nc alone may ask for no more than MOST_CLASSES
    if (nc < 1) say('error', 'yolo-yaml', `has nc: ${nc}, and Ultralytics needs at least one class`)
    else if (nc > MOST_CLASSES) say('error', 'yolo-yaml', `has nc: ${nc} but no names, and Seamtag names at most ${MOST_CLASSES.toLocaleString('en-GB')} classes itself, so the classes are unknown`)
    else data.names = Array.from({ length: nc }, (_, i) => `class_${i}`)
  }

  if (d.kpt_shape !== undefined && d.kpt_shape !== null) {
    const v = d.kpt_shape as unknown[]
    if (v.length === 2 && v.every((n) => typeof n === 'number' && Number.isInteger(n)) && (v[0] as number) > 0 && [2, 3].includes(v[1] as number)) {
      data.kptShape = [v[0] as number, v[1] as number]
    } else say('error', 'yolo-yaml', 'has a kpt_shape that isn\'t [keypoints, 2 or 3], such as [17, 3]')
  }
  if (Array.isArray(d.flip_idx) && d.flip_idx.length) {
    const v = d.flip_idx as unknown[]
    const k = data.kptShape?.[0]
    if (!v.every((n) => typeof n === 'number' && Number.isInteger(n) && n >= 0 && (k === undefined || n < k))) say('error', 'yolo-yaml', 'has a flip_idx with something other than keypoint numbers in it')
    else if (k !== undefined && v.length !== k) say('error', 'yolo-yaml', `has a flip_idx of ${v.length} keypoints where kpt_shape has ${k}, which stops Ultralytics when it mirrors images`)
    else data.flipIdx = v as number[]
  }
  if (d.kpt_names !== undefined && d.kpt_names !== null) {
    const v = d.kpt_names
    const out: Record<number, string[]> = {}
    const pairs: [unknown, unknown][] = Array.isArray(v) ? v.map((x, i) => [i, x]) : isObject(v) ? Object.entries(v) : []
    for (const [k, list] of pairs) {
      const id = typeof k === 'number' ? k : pyInt(String(k))
      if (id !== null && Array.isArray(list)) out[id] = list.map(pyStr)
    }
    if (Object.keys(out).length) data.kptNames = out
    else say('warning', 'yolo-yaml', "has kpt_names that aren't lists of names by class number, so they're left out")
  }
  return { data, issues }
}

// Which task

/** What a dataset's label rows look like, gathered file by file, for guessing the task */
export type RowStats = {
  files: number
  /** Files with no rows */
  empty: number
  rows: number
  /** How many rows have each number of values (class and numbers) */
  lengths: Record<number, number>
  /** Files with box rows (5 values) beside outline rows (7 or more), which Ultralytics skips */
  mixed: number
  /** Rows of 9 values: upright rectangles, turned ones, and other quadrilaterals */
  quads: { upright: number; rotated: number; other: number }
  /** Rows, by length, whose every third value after the box is 0, 1 or 2, as keypoints' visibility */
  visible: Record<number, number>
}

/** Counts with nothing counted yet */
export const newRowStats = (): RowStats => ({ files: 0, empty: 0, rows: 0, lengths: {}, mixed: 0, quads: { upright: 0, rotated: 0, other: 0 }, visible: {} })

/**
 * A quadrilateral's kind: a rectangle upright or turned, or another shape. With the image's size,
 * corners are tested in pixels; without, shares of the image skew a turned rectangle into a
 * parallelogram, so that's what's tested.
 */
export function quadKind(points: number[][], size?: [number, number] | null): 'upright' | 'rotated' | 'other' {
  const p = size ? points.map(([px, py]) => [px * size[0], py * size[1]]) : points
  if (!(size ? isRectangle(p) : isParallelogram(p))) return 'other'
  const turned = [0, 1, 2, 3].some((i) => {
    const dx = Math.abs(p[(i + 1) % 4][0] - p[i][0]), dy = Math.abs(p[(i + 1) % 4][1] - p[i][1])
    return Math.min(dx, dy) > 1e-3 * Math.hypot(dx, dy)
  })
  return turned ? 'rotated' : 'upright'
}

/** Adds a label file's rows to the counts; `size` is its image's, when known */
export function countRows(stats: RowStats, text: string, size?: [number, number] | null) {
  const rows = rowsOf(text)
  stats.files++
  if (!rows.length) {
    stats.empty++
    return
  }
  let boxes = false, outlines = false
  for (const { tokens } of rows) {
    const n = tokens.length
    stats.rows++
    stats.lengths[n] = (stats.lengths[n] ?? 0) + 1
    if (n === 5) boxes = true
    if (n > 6) outlines = true
    const values = tokens.map(pyFloat)
    if (values.some((v) => v === null || !Number.isFinite(v))) continue
    const v = values as number[]
    if (n === 9) stats.quads[quadKind([[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]]], size)]++
    if (n > 5 && (n - 5) % 3 === 0) {
      let fits = true
      for (let i = 7; i < n; i += 3) if (![0, 1, 2].includes(v[i])) fits = false
      if (fits) stats.visible[n] = (stats.visible[n] ?? 0) + 1
    }
  }
  if (boxes && outlines) stats.mixed++
}

/** The task a dataset's rows look made for, other tasks they also fit, and why */
export type TaskGuess = {
  task: YoloTask
  /** The keypoints a row holds and the values each has, when the rows are or could be pose rows */
  kptShape: [number, number] | null
  /** Other tasks the rows fit, for the window to offer */
  others: YoloTask[]
  /** The rows fit more than one task, so the window should ask */
  ask: boolean
  /** Why, as a sentence */
  reason: string
  stats: RowStats
}

/**
 * Which task a dataset is for, decided once for the whole dataset. data.yaml's kpt_shape means
 * pose. Otherwise the rows decide: all 5 values, boxes; all 9, rotated boxes when every one is a
 * rectangle and some are turned, else 4-point outlines, and the window asks either way; all one
 * length with every third value after the box 0, 1 or 2, keypoints with visibility, as
 * Ultralytics' _infer_ndjson_kpt_shape guesses (ultralytics/data/converter.py); odd lengths of 7
 * or more, outlines, which could also be keypoints without visibility.
 */
export function guessTask(stats: RowStats, kptShape: [number, number] | null = null): TaskGuess {
  const guess = (task: YoloTask, reason: string, others: YoloTask[] = [], kpt: [number, number] | null = null, ask = others.length > 0): TaskGuess =>
    ({ task, kptShape: kpt, others, ask, reason, stats })
  const count = (n: number) => stats.lengths[n] ?? 0
  const lengths = Object.keys(stats.lengths).map(Number).filter((n) => count(n)).sort((a, b) => a - b)
  if (kptShape) {
    const want = 5 + kptShape[0] * kptShape[1]
    const off = stats.rows - count(want)
    return guess('pose', `data.yaml gives kpt_shape [${kptShape.join(', ')}]${off ? `, though ${off} of the ${stats.rows} rows don't have the ${want} values it needs` : ''}`, [], kptShape)
  }
  if (!lengths.length) {
    return guess('detect', stats.files ? 'The label files have no rows, so every image is a background' : 'No label files were found, so every image is a background', [], null, false)
  }
  const values = lengths.join(', ')
  if (lengths.length === 1) {
    const n = lengths[0]
    if (n === 5) return guess('detect', 'Every row is a box: a class and 4 numbers')
    if (n === 9) {
      const { upright, rotated, other } = stats.quads
      if (!other && rotated) return guess('obb', `Every row is the 4 corners of a rectangle, and ${rotated} are turned, as rotated boxes are; they could also be 4-point outlines`, ['segment'])
      return guess('segment', `Every row has 4 points${other ? `, and ${other} aren't rectangles` : upright ? ', all of them upright rectangles' : ''}, so they look like outlines; they could also be rotated boxes`, ['obb'])
    }
    const k3 = (n - 5) / 3, k2 = (n - 5) / 2
    if (n > 5 && Number.isInteger(k3) && (stats.visible[n] ?? 0) === stats.rows) {
      return guess('pose', `Every row is a box and ${k3} keypoints whose visibility is 0, 1 or 2`, n >= 7 && n % 2 === 1 ? ['segment'] : [], [k3, 3])
    }
    if (n >= 7 && n % 2 === 1) return guess('segment', `Every row is an outline of ${(n - 1) / 2} points; they could also be boxes with ${k2} keypoints of x and y`, ['pose'], [k2, 2])
    if (n > 5 && Number.isInteger(k3)) return guess('pose', `Every row has ${n} values, as a box and ${k3} keypoints have, though not every visibility is 0, 1 or 2`, [], [k3, 3], true)
    return guess('detect', `Every row has ${n} values, which no task reads`, [], null, true)
  }
  if (lengths.some((n) => n >= 7 && n % 2 === 1)) {
    const boxes = count(5)
    const note = boxes ? `, and ${boxes} box rows, which segment training can't take${stats.mixed ? `; ${stats.mixed} files mix both, which Ultralytics skips` : ''}` : ''
    return guess('segment', `Rows have ${values} values: outlines${note}`, boxes ? ['detect'] : [])
  }
  return guess('detect', `Rows have ${values} values, which only boxes of 5 fit`, [], null, true)
}

// Reading rows

/** How to read a label file: its task, its dataset's classes and keypoints, and its image's size */
export type ReadOptions = {
  task: YoloTask
  /** Class names, by class number */
  names: string[]
  /** The image's size, which the rows are shares of */
  width: number
  height: number
  kptShape?: [number, number] | null
  /** Pose: each class's keypoints' names */
  kptNames?: Record<number, string[]> | null
  /** The image, for the issues */
  image?: string | null
  /** The label file's name in messages */
  file?: string
  /** The group_id the first posed object gets; each next one takes the next number */
  firstGroup?: number
}

// Pixels from shares, to a millionth of a pixel: '%.6f' shares times whole pixels have no more digits
const pixel = (share: number, size: number) => Math.round(share * size * 1e6) / 1e6

/**
 * A label file's rows as Seamtag's shapes, with points in pixels as they are, not rounded or cut:
 * a box as a rectangle, an outline as a polygon, a rotated box as an oriented rectangle when its
 * corners make one (else a polygon), a pose row as a rectangle and its keypoints as points named
 * after them (from kpt_names, else kp0, kp1 ...) sharing a new group_id; keypoints with
 * visibility 0 are left out, 1 is marked difficult. Rows of another task's shape are read as
 * Ultralytics would train them (an outline in a detect dataset is its box) or as they are, with
 * a note. What Ultralytics would refuse comes back as issues, and rows that can't be read are
 * left out with one.
 */
export function readRows(text: string, o: ReadOptions): { shapes: Shape[]; issues: Issue[]; verdict: Verdict } {
  const verdict = verifyRows(text, o.task, o.names.length, o.kptShape ?? null)
  const issues: Issue[] = []
  const file = o.file ?? 'The label file'
  const say = (severity: Severity, code: string, message: string) => issues.push(issue(severity, code, o.image ?? null, `${file}${message}`))
  const shapes: Shape[] = []
  // Python keeps a byte order mark, as Windows Notepad's older UTF-8 saves write one; the rows are read without it
  const bom = text.startsWith(BOM)
  if (bom) {
    const rest = verifyRows(text.slice(1), o.task, o.names.length, o.kptShape ?? null)
    const also = rest.status === 'corrupt' ? `, as it would without the mark, since ${rest.reason}` : ''
    say('warning', 'yolo-corrupt', `: it starts with a byte order mark, which Ultralytics reads as part of the first number, so it would leave this image out of training${also}; it's read here without the mark`)
  } else if (verdict.status === 'corrupt') say('warning', 'yolo-corrupt', `: Ultralytics would leave this image out of training, as ${verdict.reason}`)
  if (verdict.repeated.length) say('info', 'yolo-repeated', `: ${verdict.repeated.length === 1 ? 'a row repeats' : `${verdict.repeated.length} rows repeat`} an earlier one and ${verdict.repeated.length === 1 ? "isn't" : "aren't"} read, as Ultralytics drops them`)
  const W = o.width, H = o.height
  const [nkpt, ndim] = o.kptShape ?? [0, 0]
  if (o.task === 'pose' && !nkpt) {
    say('error', 'yolo-no-kpt-shape', ": pose rows can't be read without data.yaml's kpt_shape")
    return { shapes, issues, verdict }
  }
  const skip = new Set(verdict.repeated)
  let group = o.firstGroup ?? 1
  const box = (v: number[]): number[][] => [[pixel(v[1] - v[3] / 2, W), pixel(v[2] - v[4] / 2, H)], [pixel(v[1] + v[3] / 2, W), pixel(v[2] + v[4] / 2, H)]]
  const pointsOf = (v: number[]) => {
    const out: number[][] = []
    for (let k = 1; k + 1 < v.length; k += 2) out.push([pixel(v[k], W), pixel(v[k + 1], H)])
    return out
  }
  // A value can be a number and still too large for the image: in pixels it's Infinity, which no
  // label file can hold. Ultralytics would refuse the file; the row is left out here.
  const placed = (points: number[][]) => points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
  const tooLarge = (at: string) => say('warning', 'yolo-row', `${at} has a value too large to place on the image, so it's left out`)
  for (const { line, tokens } of rowsOf(bom ? text.slice(1) : text)) {
    if (skip.has(line)) continue
    const at = `: line ${line}`
    const parsed = tokens.map(pyFloat)
    const bad = parsed.findIndex((v) => v === null || !Number.isFinite(v))
    if (bad >= 0) {
      say('warning', 'yolo-row', `${at} has ${shown(tokens[bad])}, which isn't a number, so it's left out`)
      continue
    }
    let v = parsed as number[]
    const c = f32(v[0])
    if (!Number.isInteger(c) || c < 0 || c >= o.names.length) {
      say('warning', 'yolo-row', `${at}'s class, ${shown(tokens[0])}, isn't one of the ${o.names.length} classes (0 to ${o.names.length - 1}), so it's left out`)
      continue
    }
    const label = o.names[c]
    let n = v.length
    if (o.task === 'pose') {
      if (n !== 5 + nkpt * ndim) {
        say('warning', 'yolo-row', `${at} has ${n} values where kpt_shape [${nkpt}, ${ndim}] needs ${5 + nkpt * ndim}, so it's left out`)
        continue
      }
      const keypoints = Array.from({ length: nkpt }, (_, k): [number, number, number] => {
        const [kx, ky] = [v[5 + k * ndim], v[6 + k * ndim]]
        // Without visibility Ultralytics counts a keypoint below 0 as missing; 0 0 is how tools write one too
        const vis = ndim === 3 ? v[7 + k * ndim] : kx < 0 || ky < 0 || (kx === 0 && ky === 0) ? 0 : 2
        return [pixel(kx, W), pixel(ky, H), vis]
      })
      const [[x0, y0], [x1, y1]] = box(v)
      // Keypoints not labelled are left out whatever they hold
      if (!placed([[x0, y0], [x1, y1], ...keypoints.filter((k) => k[2] > 0)])) { tooLarge(at); continue }
      shapes.push(...shapesFromPose(label, [x0, y0, x1, y1], keypoints, o.kptNames?.[c] ?? [], group++))
      continue
    }
    if (nkpt && n === 5 + nkpt * ndim) {
      say('info', 'yolo-keypoints-dropped', `${at} has keypoints, which ${TASK_WORD[o.task]} training doesn't read, so only its box is`)
      v = v.slice(0, 5)
      n = 5
    }
    if (n === 5) {
      const points = box(v)
      if (!placed(points)) { tooLarge(at); continue }
      if (o.task !== 'detect') say('warning', 'yolo-box-row', `${at} is a box, which ${o.task === 'obb' ? 'rotated box' : 'outline'} training can't take; it's read as a box`)
      shapes.push(newShape(label, 'rectangle', points))
    } else if (n >= 7 && n % 2 === 1) {
      const points = pointsOf(v)
      if (!placed(points)) { tooLarge(at); continue }
      if (o.task === 'detect') {
        say('info', 'yolo-outline-as-box', `${at} is an outline, which detection training reads as the box around it, as it's read here`)
        // A loop, not Math.min(...xs): a row can hold more points than a call takes arguments
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
        for (const [x, y] of points) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
        shapes.push(newShape(label, 'rectangle', [[x0, y0], [x1, y1]]))
      } else if (o.task === 'obb' && n === 9 && isRectangle(points)) {
        shapes.push(newShape(label, 'oriented_rectangle', points))
      } else {
        if (o.task === 'obb') say('info', 'yolo-obb-outline', `${at} ${n === 9 ? "isn't a rectangle" : `has ${points.length} points`}, so it's read as an outline; Ultralytics trains the smallest rotated box around it`)
        shapes.push(newShape(label, 'polygon', points))
      }
    } else {
      say('warning', 'yolo-row', `${at} has ${n} values, which make neither a box (5) nor an outline (an odd number from 7), so it's left out`)
    }
  }
  return { shapes, issues, verdict }
}
