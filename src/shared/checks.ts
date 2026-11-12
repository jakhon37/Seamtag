// Checks over a folder's labels, as research weighed them against labelme 7.9.1, Ultralytics
// 8.4.211, pycocotools 2.0.11, Datumaro 1.13.12 and X-AnyLabeling 4.3.2: what can't be read,
// shapes that are broken or doubtful, and labels that drift apart. All of it is plain
// functions over what src/main/dataset.ts read, so the window can run it again and apply the
// fixes that change nothing but the problem.
import ClipperLib from 'clipper-lib'
import { type DatasetImage, type Issue, type MaskBits, type Severity, sizeText } from './dataset'
import { boxOf } from './formats'
import { isAi, type Shape, type ShapeType } from './labelfile'
import { CLOSED_TYPES, outlineOf } from './outline'

/** Every check by its code: how much it matters, and what the window calls a list of them */
export const CHECKS = {
  'label-file-unreadable': { severity: 'error', title: "Label files that can't be read" },
  'image-unreadable': { severity: 'error', title: "Images that can't be read" },
  'size-mismatch': { severity: 'error', title: 'Label files that give another image size' },
  'empty-label': { severity: 'error', title: 'Shapes with no label' },
  'mask-undecodable': { severity: 'error', title: "Masks whose pixels can't be read" },
  'shared-label-file': { severity: 'error', title: 'Images that share one label file' },
  'point-count': { severity: 'warning', title: 'Shapes with the wrong number of points' },
  'zero-area': { severity: 'warning', title: 'Shapes that cover no area' },
  'repeated-points': { severity: 'warning', title: 'Points repeated one after another' },
  'self-intersecting': { severity: 'warning', title: 'Polygons that cross themselves' },
  'outside-image': { severity: 'warning', title: 'Shapes reaching outside the image' },
  'duplicate-shape': { severity: 'warning', title: 'Exact copies of a shape' },
  'near-duplicate': { severity: 'warning', title: 'Shapes almost on top of another of their label' },
  'unknown-label': { severity: 'warning', title: 'Labels not in the label list' },
  'label-variants': { severity: 'warning', title: 'One label written several ways' },
  'ai-pending': { severity: 'warning', title: 'AI proposals not yet accepted' },
  'orphan-label-file': { severity: 'warning', title: 'Label files without their image' },
  'image-path-mismatch': { severity: 'warning', title: 'Label files that name another image' },
  'duplicate-image': { severity: 'warning', title: 'Identical image files' },
  'never-labelled': { severity: 'info', title: 'Images never labelled' },
  'labelled-empty': { severity: 'info', title: 'Images labelled with no shapes' },
  'not-verified': { severity: 'info', title: 'Images not verified' },
  'unused-label': { severity: 'info', title: 'Labels in the list that no shape uses' }
} as const satisfies Record<string, { severity: Severity; title: string }>
export type CheckCode = keyof typeof CHECKS

/** Images whose label file is one and the same file (src/main/dataset.ts' labelFileClashes) */
export type Clash = { file: string; images: string[] }
/**
 * A label file no image reads (src/main/dataset.ts' orphanLabels): its path in the folder, or
 * in full outside it, and the image its imagePath names. `holdsImage` when it holds its own
 * image, so it still opens on its own. `entries` for a CreateML list that images do read, with
 * the images it holds labels of that none of them is.
 */
export type OrphanFile = { file: string; names: string | null; holdsImage?: boolean; entries?: string[] }

export type CheckOptions = {
  /** The label list (config's labels, or the project's classes); null when none is set */
  labels: string[] | null
  clashes?: Clash[]
  /** Groups of identical image files (findDuplicateImages) */
  duplicates?: string[][]
  orphans?: OrphanFile[]
  /** How the whole folder uses its labels (tallyLabels), when the images given are a slice of it */
  tally?: LabelTally
}

/** How a folder's labels are used: shapes by label, AI proposals too, and the labels of shapes that aren't proposals */
export type LabelTally = { uses: Map<string, number>; accepted: Set<string> }

/** The images' labels added to a tally, a new one when none is given: a folder's can be counted a slice at a time */
export function tallyLabels(images: DatasetImage[], tally: LabelTally = { uses: new Map(), accepted: new Set() }): LabelTally {
  // AI proposals too: they're in the files
  for (const image of images) {
    for (const s of image.shapes) {
      tally.uses.set(s.label, (tally.uses.get(s.label) ?? 0) + 1)
      if (!isAi(s)) tally.accepted.add(s.label)
    }
  }
  return tally
}

// Each image's clash, worked out once for each list of them: a folder checked a slice at a time
// gives every slice the folder's list, which one -O labels.json fills with every image
const clashLookups = new WeakMap<Clash[], Map<string, Clash>>()
function clashesByImage(clashes: Clash[]) {
  let byImage = clashLookups.get(clashes)
  if (!byImage) {
    byImage = new Map()
    for (const c of clashes) for (const name of c.images) byImage.set(name, c)
    clashLookups.set(clashes, byImage)
  }
  return byImage
}

// A point further outside the image than this is reported. Ultralytics 8.4.211 drops an image
// whose shares pass 1.01 (data/utils.py verify_image_label); a pixel is what a hand misses by.
const OUTSIDE = 1
// Points closer than this are one point
const SAME = 1e-6
// Exact copies: the same points to a hundredth of a pixel, as Seamtag's own geometry keeps them
const COPY = 0.01
// Near-duplicates: intersection over union at least this
const NEAR = 0.9
// Less than a square pixel marks nothing a model can learn from
const MIN_AREA = 1

// labelme 7.9.1 refuses files whose shapes have another exact count (_label_file.py,
// _validate_shape_semantics), and polygons and line strips need 3 and 2 (_shape/_model.py)
const POINTS: Record<ShapeType, { count: number; exact: boolean; name: string }> = {
  polygon: { count: 3, exact: false, name: 'a polygon' },
  rectangle: { count: 2, exact: true, name: 'a rectangle' },
  oriented_rectangle: { count: 4, exact: true, name: 'a rotated rectangle' },
  circle: { count: 2, exact: true, name: 'a circle' },
  line: { count: 2, exact: true, name: 'a line' },
  point: { count: 1, exact: true, name: 'a point' },
  linestrip: { count: 2, exact: false, name: 'a line strip' },
  points: { count: 1, exact: false, name: 'a points shape' },
  mask: { count: 2, exact: true, name: 'a mask' }
}
const pointsWrong = (s: Shape) => {
  const rule = POINTS[s.shape_type]
  return rule.exact ? s.points.length !== rule.count : s.points.length < rule.count
}

const what = (s: Shape, k: number) => (s.label.trim() ? `Shape ${k + 1} (${s.label})` : `Shape ${k + 1}`)
const fileName = (file: string) => file.split(/[\\/]/).pop()!
const counted = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`
const num = (v: number) => String(Math.round(v * 10) / 10)
const same = (a: number[], b: number[]) => Math.abs(a[0] - b[0]) <= SAME && Math.abs(a[1] - b[1]) <= SAME

/** a, b and c; past six, the first five and how many more. `count` is how many there are when `items` holds only the first few. */
function listed(items: string[], count = items.length) {
  const shown = count > 6 ? [...items.slice(0, 5), `${count - 5} more`] : items
  return shown.length < 2 ? shown.join('') : `${shown.slice(0, -1).join(', ')} and ${shown[shown.length - 1]}`
}

/** The first of a group's names but one, as many as listed shows, without going through a group of thousands for each of them */
function othersIn(group: string[], name: string) {
  const others: string[] = []
  for (const n of group) {
    if (others.length === 6) break
    if (n !== name) others.push(n)
  }
  return others
}

// Geometry. Clipper works in whole numbers: this keeps a hundredth of a pixel, as geometry.ts does.
const SCALE = 100
type Path = { X: number; Y: number }[]
const toPath = (ring: number[][]): Path => ring.map(([x, y]) => ({ X: Math.round(x * SCALE), Y: Math.round(y * SCALE) }))

/** The area clipper gives two rings combined, filled even-odd as the canvas, PIL and labelme fill them */
function clipArea(type: number, subject: number[][], clip?: number[][]) {
  if (subject.length < 3 || (clip && clip.length < 3)) return 0
  const c = new ClipperLib.Clipper()
  c.AddPath(toPath(subject), ClipperLib.PolyType.ptSubject, true)
  if (clip) c.AddPath(toPath(clip), ClipperLib.PolyType.ptClip, true)
  const out: Path[] = []
  c.Execute(type, out, ClipperLib.PolyFillType.pftEvenOdd, ClipperLib.PolyFillType.pftEvenOdd)
  // Outer rings count up and holes down
  return Math.abs(out.reduce((sum, p) => sum + ClipperLib.Clipper.Area(p), 0)) / SCALE ** 2
}

function ringArea(ring: number[][]) {
  let sum = 0
  ring.forEach((p, i) => {
    const q = ring[(i + 1) % ring.length]
    sum += p[0] * q[1] - q[0] * p[1]
  })
  return Math.abs(sum) / 2
}

function bitCount(m: MaskBits) {
  // A loop, not reduce: Statistics and the checks count every mask of the folder each time
  const { data } = m
  let n = 0
  for (let i = 0; i < data.length; i++) if (data[i]) n++
  return n
}

/**
 * A closed shape's area in square pixels: a mask's set pixels, a polygon filled even-odd.
 * Null for lines and points, and for a mask whose pixels didn't decode.
 */
export function shapeArea(s: Shape, bits: MaskBits | null = null): number | null {
  if (!CLOSED_TYPES.has(s.shape_type) || pointsWrong(s)) return null
  if (s.shape_type === 'mask') return bits ? bitCount(bits) : null
  if (s.shape_type === 'rectangle') {
    const [[x0, y0], [x1, y1]] = s.points
    return Math.abs(x1 - x0) * Math.abs(y1 - y0)
  }
  if (s.shape_type === 'circle') {
    const [[cx, cy], [px, py]] = s.points
    return Math.PI * ((px - cx) ** 2 + (py - cy) ** 2)
  }
  const ring = cleaned(s.points, true).points
  if (ring.length < 3) return 0
  // The shoelace sum is the area of an outline that never meets itself. One that does is
  // filled even-odd, as the canvas, PIL and labelme fill it: the sum would set a bow tie's
  // lobes against each other, and count twice a hole whose cut joins it to the outline the
  // same way round, as other tools write them.
  return meeting(ring, meet) ? clipArea(ClipperLib.ClipType.ctUnion, ring) : ringArea(ring)
}

/** The points left after taking out those that repeat the one before, and a polygon's last point when it repeats its first */
function cleaned(points: number[][], closed: boolean) {
  const out: number[][] = []
  for (const p of points) if (!out.length || !same(p, out[out.length - 1])) out.push(p)
  const inRow = points.length - out.length
  while (closed && out.length > 1 && same(out[out.length - 1], out[0])) out.pop()
  return { points: out, inRow, closing: points.length - inRow - out.length }
}

/** Which way c lies from the line a to b: 0 when it's on it, give or take the products' rounding */
function side(a: number[], b: number[], c: number[]) {
  const v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
  const scale = (Math.abs(b[0] - a[0]) + Math.abs(b[1] - a[1])) * (Math.abs(c[0] - a[0]) + Math.abs(c[1] - a[1]))
  return Math.abs(v) <= 1e-12 * scale ? 0 : Math.sign(v)
}

type Meets = (p: number[], q: number[], r: number[], s: number[]) => boolean
/** Edge p q crosses edge r s: each one's ends lie either side of the other */
const crosses: Meets = (p, q, r, s) => side(p, q, r) * side(p, q, s) < 0 && side(r, s, p) * side(r, s, q) < 0
/** c, on the line through a and b, lies between them */
const between = (a: number[], b: number[], c: number[]) =>
  Math.min(a[0], b[0]) <= c[0] && c[0] <= Math.max(a[0], b[0]) && Math.min(a[1], b[1]) <= c[1] && c[1] <= Math.max(a[1], b[1])
/** Edge p q crosses, touches or runs along edge r s */
const meet: Meets = (p, q, r, s) => crosses(p, q, r, s) ||
  (side(p, q, r) === 0 && between(p, q, r)) || (side(p, q, s) === 0 && between(p, q, s)) ||
  (side(r, s, p) === 0 && between(r, s, p)) || (side(r, s, q) === 0 && between(r, s, q))

/**
 * The first two edges of a ring, not neighbours, that `meets` says meet, by their first
 * points' indexes. Edges are swept along x, so long outlines cost little more than short ones.
 */
function meeting(ring: number[][], meets: Meets): [number, number] | null {
  const n = ring.length
  if (n < 4) return null
  const edges = ring.map((p, i) => {
    const q = ring[(i + 1) % n]
    return { i, x0: Math.min(p[0], q[0]), x1: Math.max(p[0], q[0]), y0: Math.min(p[1], q[1]), y1: Math.max(p[1], q[1]) }
  }).sort((a, b) => a.x0 - b.x0)
  let active: typeof edges = []
  for (const e of edges) {
    active = active.filter((a) => a.x1 >= e.x0)
    for (const a of active) {
      const gap = Math.abs(a.i - e.i)
      if (gap === 1 || gap === n - 1 || a.y1 < e.y0 || e.y1 < a.y0) continue
      if (meets(ring[a.i], ring[(a.i + 1) % n], ring[e.i], ring[(e.i + 1) % n])) return [a.i, e.i]
    }
    active.push(e)
  }
  return null
}

/**
 * Where two edges of a ring cross, or null. Only a true crossing counts: edges that touch or
 * run along each other don't, so the zero-width cut that joins a hole to its outline in
 * Seamtag's polygons (geometry.ts withHoles) isn't one.
 */
export function crossing(ring: number[][]): [number, number] | null {
  const pair = meeting(ring, crosses)
  if (!pair) return null
  const n = ring.length
  const [p, q, r, s] = [ring[pair[0]], ring[(pair[0] + 1) % n], ring[pair[1]], ring[(pair[1] + 1) % n]]
  const d = (q[0] - p[0]) * (s[1] - r[1]) - (q[1] - p[1]) * (s[0] - r[0])
  const t = ((r[0] - p[0]) * (s[1] - r[1]) - (r[1] - p[1]) * (s[0] - r[0])) / d
  return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/** A shape's points in a form where copies match: a box's corners sorted, a circle's centre and radius */
function canonical(s: Shape): number[][] {
  if (s.shape_type === 'rectangle' && s.points.length === 2) {
    const [[x0, y0], [x1, y1]] = s.points
    return [[Math.min(x0, x1), Math.min(y0, y1)], [Math.max(x0, x1), Math.max(y0, y1)]]
  }
  if (s.shape_type === 'circle' && s.points.length === 2) {
    const [[cx, cy], [px, py]] = s.points
    return [[cx, cy], [Math.hypot(px - cx, py - cy), 0]]
  }
  return s.points
}

// The masks' pixels are read once before going through them: the folder's reader unpacks them
// when they're read (src/main/dataset.ts)
function sameBits(a: MaskBits, b: MaskBits) {
  if (a.x !== b.x || a.y !== b.y || a.width !== b.width || a.height !== b.height) return false
  const [da, db] = [a.data, b.data]
  for (let i = 0; i < da.length; i++) if (!da[i] !== !db[i]) return false
  return true
}

// The copy to keep: a person's over a model's proposal, then one with a description, flags or a group
const worth = (s: Shape) => (isAi(s) ? 0 : 8) + (s.description.trim() ? 4 : 0) +
  (Object.entries(s.flags).some(([f, v]) => v && f !== 'ai') ? 2 : 0) + (s.group_id !== null ? 1 : 0)

/**
 * For each shape, the shape of its exact copies to keep: same label and type, points within a
 * hundredth of a pixel, and for masks the same pixels. A shape with no copy stands for itself.
 */
export function copiesOf(shapes: Shape[], masks: (MaskBits | null)[] = []): number[] {
  const parent = shapes.map((_, k) => k)
  const top = (k: number): number => (parent[k] === k ? k : (parent[k] = top(parent[k])))
  const groups = new Map<string, number[]>()
  shapes.forEach((s, k) => {
    if (!s.points.length) return
    const key = `${s.label}\0${s.shape_type}\0${s.points.length}`
    const group = groups.get(key)
    if (group) group.push(k)
    else groups.set(key, [k])
  })
  for (const group of groups.values()) {
    if (group.length < 2) continue
    const points = new Map(group.map((k) => [k, canonical(shapes[k])]))
    const x = (k: number) => points.get(k)![0][0]
    group.sort((a, b) => x(a) - x(b) || a - b)
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length && x(group[j]) - x(group[i]) <= COPY; j++) {
        const [a, b] = [group[i], group[j]]
        const pa = points.get(a)!, pb = points.get(b)!
        if (!pa.every((p, n) => Math.abs(p[0] - pb[n][0]) <= COPY && Math.abs(p[1] - pb[n][1]) <= COPY)) continue
        if (shapes[a].shape_type === 'mask') {
          const [ma, mb] = [masks[a], masks[b]]
          if (ma && mb ? !sameBits(ma, mb) : shapes[a].mask !== shapes[b].mask) continue
        }
        parent[top(b)] = top(a)
      }
    }
  }
  const keep = new Map<number, number>()
  shapes.forEach((s, k) => {
    const root = top(k), best = keep.get(root)
    if (best === undefined || worth(s) > worth(shapes[best])) keep.set(root, k)
  })
  return shapes.map((_, k) => keep.get(top(k))!)
}

type Box = [number, number, number, number]
type Region = { k: number; box: Box; ring: number[][] | null; bits: MaskBits | null; area?: number }

function ringBox(ring: number[][]): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const [x, y] of ring) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y)
  }
  return [x0, y0, x1, y1]
}

/** The box around a mask's set pixels, right and bottom exclusive; null when none is set */
function bitsBox(m: MaskBits): Box | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  const { data } = m
  for (let y = 0; y < m.height; y++) {
    for (let x = 0; x < m.width; x++) {
      if (!data[y * m.width + x]) continue
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y)
    }
  }
  return x1 < 0 ? null : [m.x + x0, m.y + y0, m.x + x1 + 1, m.y + y1 + 1]
}

function bitsOverlap(a: MaskBits, b: MaskBits) {
  let n = 0
  const [da, db] = [a.data, b.data]
  for (let y = Math.max(a.y, b.y); y < Math.min(a.y + a.height, b.y + b.height); y++) {
    for (let x = Math.max(a.x, b.x); x < Math.min(a.x + a.width, b.x + b.width); x++) {
      if (da[(y - a.y) * a.width + x - a.x] && db[(y - b.y) * b.width + x - b.x]) n++
    }
  }
  return n
}

const regionArea = (r: Region) => (r.area ??= r.bits ? bitCount(r.bits) : clipArea(ClipperLib.ClipType.ctUnion, r.ring!))

/** Intersection over union, or 0 when the boxes alone show it can't reach NEAR */
function overlap(a: Region, b: Region, rectangles: boolean) {
  const w = Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0])
  const h = Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1])
  if (w <= 0 || h <= 0) return 0
  const [areaA, areaB] = [regionArea(a), regionArea(b)]
  // The overlap is no bigger than the boxes' or the smaller area, and the union no smaller than the larger
  const larger = Math.max(areaA, areaB)
  if (!larger || Math.min(areaA, areaB) / larger < NEAR || (w * h) / larger < NEAR) return 0
  const inter = rectangles ? w * h : a.bits && b.bits ? bitsOverlap(a.bits, b.bits) : clipArea(ClipperLib.ClipType.ctIntersection, a.ring!, b.ring!)
  return inter / (areaA + areaB - inter)
}

/** Pairs of shapes of one label covering nearly the same area: each later shape with the earlier one it overlaps most */
function nearCopies(image: DatasetImage, copyOf: number[]) {
  const byLabel = new Map<string, Region[]>()
  image.shapes.forEach((s, k) => {
    if (!CLOSED_TYPES.has(s.shape_type) || pointsWrong(s)) return
    const bits = s.shape_type === 'mask' ? image.masks[k] ?? null : null
    const ring = s.shape_type === 'mask' ? null : outlineOf(s)
    const box = bits ? bitsBox(bits) : ring ? ringBox(ring) : null
    if (!box) return
    // A box's area is its own: no need to clip it
    const region: Region = { k, box, ring, bits, ...(s.shape_type === 'rectangle' ? { area: (box[2] - box[0]) * (box[3] - box[1]) } : {}) }
    const group = byLabel.get(s.label)
    if (group) group.push(region)
    else byLabel.set(s.label, [region])
  })
  const best = new Map<number, { other: number; iou: number }>()
  for (const regions of byLabel.values()) {
    regions.sort((a, b) => a.box[0] - b.box[0])
    let active: Region[] = []
    for (const r of regions) {
      active = active.filter((a) => a.box[2] > r.box[0])
      for (const a of active) {
        // Masks are compared by their pixels, outlines by clipping: not one with the other
        if (!a.bits !== !r.bits || copyOf[a.k] === copyOf[r.k]) continue
        const rectangles = image.shapes[a.k].shape_type === 'rectangle' && image.shapes[r.k].shape_type === 'rectangle'
        const iou = overlap(a, r, rectangles)
        if (iou < NEAR) continue
        const [later, other] = a.k > r.k ? [a.k, r.k] : [r.k, a.k]
        if (iou > (best.get(later)?.iou ?? 0)) best.set(later, { other, iou })
      }
      active.push(r)
    }
  }
  return [...best].sort((a, b) => a[0] - b[0])
}

/** How far a shape reaches past each edge, the worst first */
function overreach(s: Shape, width: number, height: number) {
  const [x0, y0, x1, y1] = s.shape_type === 'circle' ? boxOf(s) : ringBox(s.points)
  const edges: [string, number][] = [['left', -x0], ['top', -y0], ['right', x1 - width], ['bottom', y1 - height]]
  return edges.sort((a, b) => b[1] - a[1])[0]
}

/** A label as it reads once case, spaces, _ and - stop mattering */
const variantKey = (label: string) => label.toLowerCase().replace(/[\s_-]+/g, ' ').trim()

/**
 * Every problem in a folder's labels, the worst first, then in the folder's order. Each names
 * its image (null for the folder as a whole) and, where it's one shape's, the shape's index.
 */
export function checkDataset(images: DatasetImage[], { labels, clashes = [], duplicates = [], orphans = [], tally }: CheckOptions): Issue[] {
  const issues: Issue[] = []
  const add = (code: CheckCode, image: string | null, message: string, shape?: number) =>
    issues.push({ severity: CHECKS[code].severity, code, image, ...(shape === undefined ? {} : { shape }), message })
  const listedLabels = labels ? new Set(labels) : null
  const clashed = clashesByImage(clashes)
  // Verification is in use once any image is verified; until then, saying so of every image is noise
  const verifying = images.some((i) => i.verified)

  // Labels across the folder
  const { uses, accepted } = tally ?? tallyLabels(images)
  // Variants of one label: the one in the label list wins, then the most used
  const variants = new Map<string, string[]>()
  for (const label of uses.keys()) {
    const key = variantKey(label)
    if (key) variants.set(key, [...(variants.get(key) ?? []), label])
  }
  const preferred = new Map<string, string>()
  for (const group of variants.values()) {
    if (group.length < 2) continue
    const rank = (l: string) => (listedLabels?.has(l) ? 1e12 : 0) + uses.get(l)!
    const winner = [...group].sort((a, b) => rank(b) - rank(a) || a.localeCompare(b))[0]
    for (const l of group) if (l !== winner) preferred.set(l, winner)
  }

  for (const image of images) {
    const { name } = image
    if (image.broken) {
      add('label-file-unreadable', name, `${fileName(image.file ?? name)} can't be read, so ${name}'s labels can't be checked or exported: ${image.broken}`)
      continue
    }
    if (image.unreadable) add('image-unreadable', name, `${name} can't be read, so its labels can't be checked against it or exported: ${image.unreadable}`)
    const sized = !image.unreadable && image.width > 0 && image.height > 0
    const clash = clashed.get(name)
    if (image.file && sized && image.labelSize && (image.labelSize[0] !== image.width || image.labelSize[1] !== image.height)) {
      const [w, h] = image.labelSize
      add('size-mismatch', name, `${fileName(image.file)} gives the image size as ${sizeText(w, h)}, but ${name} is ${sizeText(image.width, image.height)}, ` +
        `so its shapes may be out of place${image.format === 'labelme' ? ", and labelme 7 won't open it" : ''}`)
    }
    if (clash) {
      add('shared-label-file', name, `${fileName(clash.file)} is the label file of ${listed(clash.images)}, so saving one replaces the others' labels`)
    } else if (image.file && image.labelImage !== undefined && image.labelImage !== name) {
      add('image-path-mismatch', name, `${fileName(image.file)} names ${image.labelImage ?? 'an image outside the folder'} as its image, not ${name}, ` +
        'so labelme would show these labels on that image')
    }
    if (!image.file) add('never-labelled', name, `${name} has no label file yet`)
    else if (!image.shapes.length) add('labelled-empty', name, `${name} is labelled with no shapes, so it counts as a background`)
    if (verifying && image.file && !image.verified && image.format !== 'yolo') add('not-verified', name, `${name} isn't verified yet`)
    const ai = image.shapes.flatMap((s, k) => (isAi(s) ? [k] : []))
    if (ai.length) add('ai-pending', name, `${name} has ${counted(ai.length, 'AI proposal')} not yet accepted`, ai[0])

    // YOLO's boxes are shares of the image: with no size to scale them by, their pixels mean nothing
    const geometry = sized || image.format !== 'yolo'
    const unknown = new Map<string, number[]>()
    const odd = new Map<string, number>()
    image.shapes.forEach((s, k) => {
      const shape = what(s, k)
      if (!s.label.trim()) add('empty-label', name, `${shape} has no label, so no export can name its class`, k)
      else if (preferred.has(s.label)) {
        if (!odd.has(s.label)) odd.set(s.label, k)
      } else if (listedLabels && !listedLabels.has(s.label)) unknown.set(s.label, [...(unknown.get(s.label) ?? []), k])
      if (s.shape_type === 'mask' && !image.masks[k]) {
        add('mask-undecodable', name, s.mask ? `${shape} is a mask whose pixels can't be read, so it can't be exported` : `${shape} is a mask, but the file holds no pixels for it`, k)
      }
      if (!geometry) return
      const rule = POINTS[s.shape_type]
      if (pointsWrong(s)) {
        add('point-count', name, `${shape} has ${counted(s.points.length, 'point')}: ${rule.name} needs ${rule.exact ? 'exactly' : 'at least'} ${rule.count}`, k)
        return
      }
      const area = shapeArea(s, image.masks[k] ?? null)
      if (area !== null && area < MIN_AREA) add('zero-area', name, `${shape} ${area > 0 ? 'covers less than a pixel' : `covers no area: ${noArea(s)}`}`, k)
      if (s.shape_type === 'polygon' || s.shape_type === 'linestrip') {
        const { points, inRow, closing } = cleaned(s.points, s.shape_type === 'polygon')
        if (inRow || closing) {
          const parts = [...(inRow ? [`has ${counted(inRow, 'point that repeats', 'points that repeat')} the one before`] : []), ...(closing ? ['ends with its first point again'] : [])]
          const them = inRow + closing === 1 ? 'it' : 'them'
          // Too few points left: taking them out would only swap this warning for another, so there's no fix
          add('repeated-points', name, `${shape} ${parts.join(', and ')}: ${points.length < rule.count
            ? `without ${them}, ${counted(points.length, 'point is', 'points are')} left, and ${rule.name} needs at least ${rule.count}`
            : `taking ${them} out leaves its outline as it is`}`, k)
        }
      }
      if (s.shape_type === 'polygon' || s.shape_type === 'oriented_rectangle') {
        const at = crossing(cleaned(s.points, true).points)
        if (at) add('self-intersecting', name, `${shape} crosses itself near (${Math.round(at[0])}, ${Math.round(at[1])}), so tools may fill it differently`, k)
      }
      if (sized) {
        const [edge, by] = overreach(s, image.width, image.height)
        if (by > OUTSIDE) add('outside-image', name, `${shape} reaches ${num(by)} px past the ${edge} edge of the image`, k)
      }
    })
    for (const [label, at] of unknown) add('unknown-label', name, `Label “${label}” isn't in the label list (${at.length === 1 ? 'shape' : 'shapes'} ${listed(at.map((k) => String(k + 1)))})`, at[0])
    for (const [label, k] of odd) {
      const to = preferred.get(label)!
      add('label-variants', name, listedLabels?.has(to)
        ? `Label “${label}” looks like “${to}” from the label list, written another way`
        : `Label “${label}” looks like “${to}” written another way (${counted(uses.get(to)!, 'shape uses', 'shapes use')} “${to}”)`, k)
    }
    if (!geometry) continue
    const copyOf = copiesOf(image.shapes, image.masks)
    copyOf.forEach((keep, k) => {
      if (keep !== k) add('duplicate-shape', name, `${what(image.shapes[k], k)} is an exact copy of shape ${keep + 1}`, k)
    })
    for (const [k, { other, iou }] of nearCopies(image, copyOf)) {
      add('near-duplicate', name, `${what(image.shapes[k], k)} covers almost the same area as shape ${other + 1} (they overlap ${Math.round(iou * 100)}%)`, k)
    }
  }

  for (const group of duplicates) {
    for (const name of group) {
      add('duplicate-image', name, `${name} has the same bytes as ${listed(othersIn(group, name), group.length - 1)}, ` +
        'so a split can put one copy in training and another in validation')
    }
  }
  for (const o of orphans) {
    if (o.entries) {
      add('orphan-label-file', null, `${o.file} holds labels of ${listed(o.entries)}, but no image in the folder reads them from it, so they're never shown or exported`)
      continue
    }
    const more = [...(o.names ? [`its imagePath names ${o.names}`] : []), ...(o.holdsImage ? ['it holds its own image, so it still opens on its own'] : [])]
    add('orphan-label-file', null, `${o.file} has no image in the folder, so its labels are never shown or exported${more.length ? `; ${more.join(', and ')}` : ''}`)
  }
  if (labels) {
    for (const label of labels) if (!accepted.has(label)) add('unused-label', null, `Label “${label}” is in the label list, but no accepted shape uses it`)
  }

  const order = new Map(images.map((i, k) => [i.name, k]))
  const at = (name: string | null) => (name === null ? -1 : order.get(name) ?? images.length)
  const rank = { error: 0, warning: 1, info: 2 }
  return issues.sort((a, b) => rank[a.severity] - rank[b.severity] || at(a.image) - at(b.image) || (a.shape ?? -1) - (b.shape ?? -1))
}

/** Why a shape with the right number of points covers no area */
function noArea(s: Shape) {
  // A mask is kept as labelme pastes it, clipped to the image
  if (s.shape_type === 'mask') return 'its mask has no pixels set inside the image'
  if (s.shape_type === 'circle') return 'its rim is at its centre'
  if (s.shape_type === 'rectangle') {
    const [[x0, y0], [x1, y1]] = s.points
    return x0 === x1 && y0 === y1 ? 'its corners are one point' : x0 === x1 ? 'it has no width' : 'it has no height'
  }
  return 'its points lie on one line'
}

// Fixes. Each gives an image's new shape list, or null when there's nothing it would change;
// the window applies them to the open image as edits that can be undone.

/**
 * Takes out points that repeat the one before them, and a polygon's last point when it repeats
 * its first. The outline stays as it is. A shape that would be left with fewer points than its
 * type needs stays as it is too: labelme2coco refuses a polygon of 2, so that's no fix.
 */
export function withoutRepeatedPoints(shapes: Shape[], only?: number): Shape[] | null {
  let changed = false
  const next = shapes.map((s, k) => {
    if ((only !== undefined && k !== only) || (s.shape_type !== 'polygon' && s.shape_type !== 'linestrip')) return s
    const { points } = cleaned(s.points, s.shape_type === 'polygon')
    if (points.length === s.points.length || points.length < POINTS[s.shape_type].count) return s
    changed = true
    return { ...s, points: points.map(([x, y]) => [x, y]) }
  })
  return changed ? next : null
}

/** Deletes exact copies of shapes, keeping the copy worth most: a person's over a model's, then one with a description, flags or a group */
export function withoutDuplicates(shapes: Shape[], masks: (MaskBits | null)[] = []): Shape[] | null {
  const copyOf = copiesOf(shapes, masks)
  const next = shapes.filter((_, k) => copyOf[k] === k)
  return next.length === shapes.length ? null : next
}

/**
 * Clips boxes reaching more than a pixel outside the image to its edges, or the one box given
 * however little it reaches out. A box wholly outside is left: clipping would leave nothing.
 */
export function boxesClipped(shapes: Shape[], width: number, height: number, only?: number): Shape[] | null {
  if (!(width > 0 && height > 0)) return null
  let changed = false
  const next = shapes.map((s, k) => {
    if ((only !== undefined && k !== only) || s.shape_type !== 'rectangle' || s.points.length !== 2) return s
    if (only === undefined && overreach(s, width, height)[1] <= OUTSIDE) return s
    const points = s.points.map(([x, y]) => [Math.min(Math.max(x, 0), width), Math.min(Math.max(y, 0), height)])
    const [[x0, y0], [x1, y1]] = points
    if (x0 === x1 || y0 === y1 || points.every((p, n) => p[0] === s.points[n][0] && p[1] === s.points[n][1])) return s
    changed = true
    return { ...s, points }
  })
  return changed ? next : null
}

/** The safe fixes by the code of the issue they fix */
export const FIXES: Partial<Record<CheckCode, (image: DatasetImage, shape?: number) => Shape[] | null>> = {
  'repeated-points': (image, shape) => withoutRepeatedPoints(image.shapes, shape),
  'duplicate-shape': (image) => withoutDuplicates(image.shapes, image.masks),
  'outside-image': (image, shape) => boxesClipped(image.shapes, image.width, image.height, shape)
}

/** The image's shapes with an issue fixed, or null when there's no safe fix for it */
export function fixIssue(image: DatasetImage, issue: Issue): Shape[] | null {
  return FIXES[issue.code as CheckCode]?.(image, issue.shape) ?? null
}

export type RenamePlan = {
  from: string
  to: string
  /** The images whose labels change, with their new shape lists and how many shapes were renamed */
  images: { name: string; shapes: Shape[]; renamed: number }[]
  /** Images whose label file is left alone, and why */
  skipped: { name: string; reason: string }[]
  /** Shapes renamed in all */
  shapes: number
}

/** A rename as a change to an image's shapes: rewriteLabels makes it to what the file holds as it writes */
export const renaming = (from: string, to: string) => (shapes: Shape[]) => shapes.map((s) => (s.label === from ? { ...s, label: to } : s))

/**
 * What renaming a label across the folder would change, for the window to show before it's
 * done: AI proposals are renamed too, and keep waiting to be accepted. Label files that can't
 * be read, or that several images share, are left alone. Each image's shapes are as read:
 * make the rename with `renaming`, so edits made since stay.
 */
export function renameLabel(images: DatasetImage[], from: string, to: string, clashes: Clash[] = []): RenamePlan {
  if (!to.trim()) throw new Error('A label needs a name, so nothing is renamed')
  const clashed = new Map<string, Clash>()
  for (const c of clashes) for (const name of c.images) clashed.set(name, c)
  const plan: RenamePlan = { from, to, images: [], skipped: [], shapes: 0 }
  if (from === to) return plan
  for (const image of images) {
    if (image.broken) {
      plan.skipped.push({ name: image.name, reason: `${fileName(image.file ?? image.name)} can't be read, so it's left unchanged` })
      continue
    }
    const renamed = image.shapes.filter((s) => s.label === from).length
    if (!renamed) continue
    const clash = clashed.get(image.name)
    if (clash) {
      plan.skipped.push({ name: image.name, reason: `${fileName(clash.file)} holds the labels of ${listed(clash.images)}, so it's left unchanged` })
      continue
    }
    plan.images.push({ name: image.name, shapes: renaming(from, to)(image.shapes), renamed })
    plan.shapes += renamed
  }
  return plan
}
