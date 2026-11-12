// Keypoints. X-AnyLabeling 4.3.2's pose file (classes: {person: [nose, left_eye, ...]} and
// has_visible), read and written as a KeypointSchema, with Seamtag's own skeleton and flip pairs
// beside them; the posed objects in an image's shapes; and the shapes a posed object becomes on
// import. A posed object is a box and the points sharing its group_id, each point labelled with
// its keypoint's name, as X-AnyLabeling's pose export takes them
// (anylabeling/views/labeling/label_converter.py, custom_to_yolo).
import yaml from 'js-yaml'
import type { DatasetImage, Issue, KeypointSchema, MaskBits, PoseInstance, Severity } from './dataset'
import { boxOf } from './formats'
import { newShape, type Shape } from './labelfile'
import { CLOSED_TYPES } from './outline'

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const issue = (severity: Severity, code: string, image: string | null, message: string, shape?: number): Issue =>
  (shape === undefined ? { severity, code, image, message } : { severity, code, image, shape, message })

// YAML

// Ultralytics 8.4.211 reads data.yaml with its own YAML class (ultralytics/utils/__init__.py): YAML 1.2's
// words for nothing, true and false plus Python's None, True and False, so yes, no, on and off stay text,
// and a plain value starting with a digit, sign or dot and ending with a digit or dot is a number when
// Python's int() or float() takes it. These types make js-yaml read plain values the same way.
const PY_NULL = new Set(['', '~', 'null', 'Null', 'NULL', 'None'])
const PY_BOOL: Record<string, boolean> = { true: true, True: true, TRUE: true, false: false, False: false, FALSE: false }
const PY_SPECIAL: Record<string, number> = {
  '.inf': Infinity, '.Inf': Infinity, '.INF': Infinity, '+.inf': Infinity, '+.Inf': Infinity, '+.INF': Infinity,
  '-.inf': -Infinity, '-.Inf': -Infinity, '-.INF': -Infinity, '.nan': NaN, '.NaN': NaN, '.NAN': NaN
}

/** Python's whitespace, which str.split() and str.strip() split and strip at */
export const PY_SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
const TRIM = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, 'g')

// Python takes any Unicode decimal digit. They come in runs of ten from 0, so a digit's value is
// how far it is from the start of its run, counted in tens.
const DIGIT = /\p{Nd}/u
function asciiDigits(s: string) {
  return s.replace(/\p{Nd}/gu, (d) => {
    const cp = d.codePointAt(0)!
    let start = cp
    while (start > 0 && DIGIT.test(String.fromCodePoint(start - 1))) start--
    return String((cp - start) % 10)
  })
}

const PY_DIGITS = '\\d(?:_?\\d)*'
const PY_FLOAT = new RegExp(`^[+-]?(?:(?:${PY_DIGITS}(?:\\.(?:${PY_DIGITS})?)?|\\.${PY_DIGITS})(?:[eE][+-]?${PY_DIGITS})?|inf|infinity|nan)$`, 'i')
const PY_INT = new RegExp(`^[+-]?${PY_DIGITS}$`)

/** Python's float() of a string, or null where it raises ValueError */
export function pyFloat(s: string): number | null {
  const t = asciiDigits(s.replace(TRIM, ''))
  if (!PY_FLOAT.test(t)) return null
  const lower = t.toLowerCase().replace(/_/g, '')
  if (/nan$/.test(lower)) return NaN
  if (/inf(inity)?$/.test(lower)) return lower[0] === '-' ? -Infinity : Infinity
  return Number(lower)
}

/** Python's int() of a string, or null where it raises ValueError */
export function pyInt(s: string): number | null {
  const t = asciiDigits(s.replace(TRIM, ''))
  return PY_INT.test(t) ? Number(t.replace(/_/g, '')) : null
}

const pyNumber = (s: string) => {
  if (s in PY_SPECIAL) return PY_SPECIAL[s]
  return /^[0-9+\-.]/.test(s) && /[0-9.]$/.test(s) ? (pyInt(s) ?? pyFloat(s)) : null
}

/** YAML as Ultralytics 8.4.211 reads data.yaml: js-yaml with plain values resolved as its YAML._scalar does */
export const ULTRALYTICS_YAML = yaml.FAILSAFE_SCHEMA.extend({
  implicit: [
    new yaml.Type('tag:yaml.org,2002:null', { kind: 'scalar', resolve: (s: string | null) => s === null || PY_NULL.has(s), construct: () => null }),
    new yaml.Type('tag:yaml.org,2002:bool', { kind: 'scalar', resolve: (s: string | null) => s !== null && s in PY_BOOL, construct: (s: string) => PY_BOOL[s] }),
    new yaml.Type('tag:yaml.org,2002:float', { kind: 'scalar', resolve: (s: string | null) => s !== null && pyNumber(s) !== null, construct: (s: string) => pyNumber(s) })
  ]
})

/** Python's str() of a value YAML gave, as Ultralytics' check_class_names turns names into text */
export function pyStr(v: unknown): string {
  if (v === null || v === undefined) return 'None'
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return Number.isNaN(v) ? 'nan' : v === Infinity ? 'inf' : v === -Infinity ? '-inf' : String(v)
  return String(v)
}

// A name is written plain only when every YAML reader gives it back as it is: Ultralytics' own,
// PyYAML (YAML 1.1, as X-AnyLabeling, CVAT and older Ultralytics read), and js-yaml. So it starts
// with a letter or _, holds letters, digits and _ with spaces, dots, dashes and slashes between
// them, and isn't a word YAML 1.1 or Python reads as true, false or nothing.
const PLAIN = /^[\p{L}_][\p{L}\p{N}_]*(?:[ .\-/]+[\p{L}\p{N}_]+)*$/u
const RESERVED = /^(?:y|n|yes|no|on|off|true|false|null|none)$/i
// C1 controls, line and paragraph separators, byte order marks and non-characters
const QUOTED_ESCAPES = /[\x7f-\x9f\u2028\u2029\ufeff\ufffe\uffff]/g

/**
 * A text value for data.yaml or a pose file: plain when that's safe, otherwise double-quoted as
 * Ultralytics' YAML._repr writes it (JSON's escapes, and \x escapes for C1 controls), also
 * escaping line and paragraph separators and byte order marks, which some readers take as line ends
 */
export function yamlString(s: string): string {
  if (PLAIN.test(s) && !RESERVED.test(s)) return s
  return JSON.stringify(s).replace(QUOTED_ESCAPES, (c) => {
    const code = c.charCodeAt(0)
    return code <= 0xff ? `\\x${code.toString(16).padStart(2, '0')}` : `\\u${code.toString(16).padStart(4, '0')}`
  })
}

// The pose file

/** A keypoint file's contents: the schema, and X-AnyLabeling's has_visible (rows with visibility) */
export type PoseFile = { schema: KeypointSchema; hasVisible: boolean }

function poseFail(message: string): never {
  throw new Error(`The keypoint file ${message}`)
}

// PyYAML, which X-AnyLabeling reads pose files with, takes YAML 1.1's yes, no, on and off for true and false
const YES = /^(?:y|yes|on|true)$/i, NO = /^(?:n|no|off|false)$/i
const asName = (v: unknown) => (typeof v === 'string' ? v : typeof v === 'number' ? String(v) : null)

function readPairs(raw: unknown, key: string, schema: KeypointSchema): Record<string, [string, string][]> | undefined {
  if (raw === undefined || raw === null) return undefined
  if (!isObject(raw)) poseFail(`needs ${key} as a class name and its pairs of keypoints on each line`)
  const out: Record<string, [string, string][]> = {}
  for (const [cls, pairs] of Object.entries(raw as Record<string, unknown>)) {
    const names = Object.hasOwn(schema.classes, cls) ? schema.classes[cls] : null
    if (!names) poseFail(`names ${cls} under ${key}, which isn't one of its classes`)
    if (pairs === null) { out[cls] = []; continue }
    if (!Array.isArray(pairs)) poseFail(`needs ${key} of ${cls} as a list of pairs, such as [left_eye, right_eye]`)
    out[cls] = (pairs as unknown[]).map((pair) => {
      const ends = Array.isArray(pair) ? pair.map(asName) : []
      if (ends.length !== 2 || ends.some((e) => e === null)) poseFail(`needs each pair under ${key} of ${cls} to name two keypoints`)
      for (const e of ends) if (!names.includes(e!)) poseFail(`pairs ${e} under ${key} of ${cls}, which isn't one of its keypoints`)
      if (ends[0] === ends[1]) poseFail(`pairs ${ends[0]} with itself under ${key} of ${cls}`)
      return [ends[0]!, ends[1]!] as [string, string]
    })
    if (key === 'flip') {
      const seen = new Set<string>()
      for (const name of out[cls].flat()) {
        if (seen.has(name)) poseFail(`flips ${name} of ${cls} in two pairs`)
        seen.add(name)
      }
    }
  }
  return out
}

/**
 * X-AnyLabeling's pose file, as its LabelConverter reads one: classes maps each class to its
 * keypoints' names in order, and has_visible (true when left out) says rows hold visibility.
 * Seamtag's optional skeleton and flip keys give pairs of keypoints by name, per class.
 */
export function readPoseYaml(text: string): PoseFile {
  let data: unknown
  try {
    data = yaml.load(text, { schema: yaml.CORE_SCHEMA })
  } catch (e) {
    poseFail(`isn't valid YAML (${(e as Error).message.split('\n')[0]})`)
  }
  if (!isObject(data)) poseFail('needs classes: each class name and its keypoints, as X-AnyLabeling writes them')
  const raw = (data as Record<string, unknown>).classes
  if (!isObject(raw) || !Object.keys(raw).length) poseFail('needs classes: each class name and its keypoints, as X-AnyLabeling writes them')
  const classes: Record<string, string[]> = {}
  for (const [cls, list] of Object.entries(raw as Record<string, unknown>)) {
    if (!Array.isArray(list) || !list.length) poseFail(`needs a list of keypoint names for ${cls}`)
    const names = (list as unknown[]).map(asName)
    names.forEach((n, k) => {
      if (n === null || n === '') poseFail(`has no name for keypoint ${k + 1} of ${cls}`)
      if (names.indexOf(n) !== k) poseFail(`names ${n} twice in ${cls}`)
    })
    classes[cls] = names as string[]
  }
  const schema: KeypointSchema = { classes }
  const skeleton = readPairs((data as Record<string, unknown>).skeleton, 'skeleton', schema)
  const flip = readPairs((data as Record<string, unknown>).flip, 'flip', schema)
  if (skeleton) schema.skeleton = skeleton
  if (flip) schema.flip = flip
  const visible = (data as Record<string, unknown>).has_visible
  let hasVisible = true
  if (typeof visible === 'boolean') hasVisible = visible
  else if (typeof visible === 'string' && (YES.test(visible) || NO.test(visible))) hasVisible = YES.test(visible)
  else if (visible !== undefined && visible !== null) poseFail('needs has_visible to be true or false')
  return { schema, hasVisible }
}

/** A pose file X-AnyLabeling reads: classes and has_visible, then Seamtag's skeleton and flip pairs when given */
export function writePoseYaml(schema: KeypointSchema, hasVisible = true): string {
  const lines = ['# Keypoints of each class, in order, as X-AnyLabeling lists them', 'classes:']
  for (const [cls, names] of Object.entries(schema.classes)) {
    lines.push(`  ${yamlString(cls)}:`, ...names.map((n) => `    - ${yamlString(n)}`))
  }
  lines.push(`has_visible: ${hasVisible}`)
  for (const [key, pairs] of [['skeleton', schema.skeleton], ['flip', schema.flip]] as const) {
    if (!pairs || !Object.keys(pairs).length) continue
    lines.push(key === 'skeleton' ? '# Lines between keypoints, for COCO' : '# Keypoints that swap places in a mirrored image', `${key}:`)
    for (const [cls, list] of Object.entries(pairs)) {
      lines.push(list.length ? `  ${yamlString(cls)}:` : `  ${yamlString(cls)}: []`)
      for (const [a, b] of list) lines.push(`    - [${yamlString(a)}, ${yamlString(b)}]`)
    }
  }
  return lines.join('\n') + '\n'
}

/** The largest number of keypoints a class has: every row of a YOLO pose dataset holds that many */
export const keypointCount = (schema: KeypointSchema) => Math.max(0, ...Object.values(schema.classes).map((n) => n.length))

/**
 * Ultralytics' flip_idx: for each keypoint, the one it becomes in a mirrored image. Only from the
 * schema's flip pairs, when every class with keypoints has them and they swap the same places;
 * otherwise null, and Ultralytics turns off flipping (ultralytics/data/augment.py, v8_transforms).
 */
export function flipIndex(schema: KeypointSchema): { flipIdx: number[] | null; issues: Issue[] } {
  const k = keypointCount(schema)
  const classes = Object.entries(schema.classes).filter(([, names]) => names.length)
  const missing = classes.filter(([cls]) => !(schema.flip && Object.hasOwn(schema.flip, cls)))
  if (!classes.length || missing.length === classes.length) return { flipIdx: null, issues: [] }
  if (missing.length) {
    return {
      flipIdx: null,
      issues: [issue('warning', 'pose-flip', null, `The keypoint file gives flip pairs for some classes but not ${missing.map(([c]) => c).join(', ')}, so data.yaml has no flip_idx and Ultralytics won't mirror images`)]
    }
  }
  let first: number[] | null = null
  for (const [cls, names] of classes) {
    const index = Array.from({ length: k }, (_, i) => i)
    for (const [a, b] of schema.flip![cls]) {
      const ia = names.indexOf(a), ib = names.indexOf(b)
      if (ia < 0 || ib < 0) {
        return { flipIdx: null, issues: [issue('warning', 'pose-flip', null, `The flip pair ${a} and ${b} of ${cls} names a keypoint ${cls} doesn't have, so data.yaml has no flip_idx and Ultralytics won't mirror images`)] }
      }
      index[ia] = ib
      index[ib] = ia
    }
    if (!first) first = index
    else if (index.some((v, i) => v !== first![i])) {
      return {
        flipIdx: null,
        issues: [issue('warning', 'pose-flip', null, "The keypoint file's classes swap different keypoints when mirrored, and data.yaml has one flip_idx for all, so it's left out and Ultralytics won't mirror images")]
      }
    }
  }
  return { flipIdx: first, issues: [] }
}

// Posed objects in an image

/** The box of a mask's pixels, each pixel spanning x to x + 1; null when it has none */
export function pixelBox(m: MaskBits): [number, number, number, number] | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (let y = 0; y < m.height; y++) {
    for (let x = 0; x < m.width; x++) {
      if (!m.data[y * m.width + x]) continue
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      y1 = y
    }
  }
  return x1 < 0 ? null : [m.x + x0, m.y + y0, m.x + x1 + 1, m.y + y1 + 1]
}

/** The box around a closed shape: a mask's around its pixels, which labelme's box gives both corners of */
export function shapeBox(s: Shape, mask?: MaskBits | null): [number, number, number, number] | null {
  if (!s.points.length) return null
  if (s.shape_type !== 'mask') return boxOf(s)
  if (mask) return pixelBox(mask)
  const [x0, y0, x1, y1] = boxOf(s)
  return [x0, y0, x1 + 1, y1 + 1]
}

const pointVisibility = (s: Shape): 1 | 2 => (s.flags.difficult === true || s.flags.occluded === true ? 1 : 2)

/**
 * The posed objects in an image: each group_id holding a box (any closed shape labelled with a
 * class of the schema) and points labelled with its keypoints' names, and boxes of those classes
 * that have no group, with no keypoints. A point marked difficult or occluded is labelled but
 * hidden (COCO's visibility 1), others are seen (2). Problems come back as issues: errors for
 * groups that can't be read one way only, which a pose export stops on.
 */
export function poseInstances(image: DatasetImage, schema: KeypointSchema): { instances: PoseInstance[]; issues: Issue[] } {
  const issues: Issue[] = []
  const at = image.name
  const say = (severity: Severity, code: string, message: string, shape?: number) => issues.push(issue(severity, code, at, `${at}: ${message}`, shape))
  const isClass = (label: string) => Object.hasOwn(schema.classes, label)
  const keypointNames = new Set(Object.values(schema.classes).flat())
  type Group = { first: number; regions: number[]; points: number[] }
  const groups = new Map<number, Group>()
  const instances: (PoseInstance & { first: number })[] = []
  const join = (g: number, i: number) => {
    if (!groups.has(g)) groups.set(g, { first: i, regions: [], points: [] })
    return groups.get(g)!
  }
  image.shapes.forEach((s, i) => {
    if (CLOSED_TYPES.has(s.shape_type) && isClass(s.label)) {
      if (s.group_id !== null) join(s.group_id, i).regions.push(i)
      else {
        const box = shapeBox(s, image.masks[i])
        if (!box) return say('warning', 'pose-empty-box', `${s.label} has no pixels, so it's left out`, i)
        say('info', 'pose-no-keypoints', `${s.label} has no group, so it has no keypoints`, i)
        instances.push({ first: i, label: s.label, group: null, box, keypoints: schema.classes[s.label].map(() => null), shapes: [i] })
      }
    } else if (s.shape_type === 'point') {
      if (s.group_id !== null) join(s.group_id, i).points.push(i)
      else if (keypointNames.has(s.label)) say('warning', 'pose-no-group', `point ${s.label} has no group, so it isn't a keypoint of any object and is left out`, i)
    }
  })
  for (const [g, group] of groups) {
    const named = group.points.filter((i) => keypointNames.has(image.shapes[i].label))
    // Groups with neither a box of a pose class nor a known keypoint are labelme's groups of other shapes
    if (!group.regions.length && !named.length) continue
    if (group.regions.length > 1) {
      say('error', 'pose-two-boxes', `group ${g} has ${group.regions.length} boxes (${group.regions.map((i) => image.shapes[i].label).join(', ')}), so it's left out: give each object its own group`, group.regions[1])
      continue
    }
    let label: string
    let box: [number, number, number, number] | null
    if (group.regions.length) {
      const r = group.regions[0]
      label = image.shapes[r].label
      box = shapeBox(image.shapes[r], image.masks[r])
      if (!box) {
        say('error', 'pose-empty-box', `group ${g}'s ${label} has no pixels, so the group is left out`, r)
        continue
      }
    } else {
      const names = named.map((i) => image.shapes[i].label)
      const fits = Object.keys(schema.classes).filter((c) => names.every((n) => schema.classes[c].includes(n)))
      if (fits.length !== 1) {
        say('error', 'pose-no-class', `group ${g} has keypoints but no box, and ${fits.length ? `they fit ${fits.join(' and ')}` : 'no class has all of them'}, so it's left out: add a box with the class's label`, named[0])
        continue
      }
      label = fits[0]
      const xs = named.map((i) => image.shapes[i].points[0][0]), ys = named.map((i) => image.shapes[i].points[0][1])
      box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
      say('warning', 'pose-no-box', `group ${g} has keypoints of ${label} but no box, so the box around its points is used`, named[0])
    }
    const names = schema.classes[label]
    const keypoints: PoseInstance['keypoints'] = names.map(() => null)
    const used = [...group.regions]
    for (const i of group.points) {
      const p = image.shapes[i]
      const k = names.indexOf(p.label)
      if (k < 0) {
        say('error', 'pose-unknown-keypoint', `point ${p.label} in group ${g} isn't one of ${label}'s keypoints in the keypoint file`, i)
        continue
      }
      if (keypoints[k]) {
        say('error', 'pose-repeated-keypoint', `group ${g} has ${p.label} twice, so the second is left out`, i)
        continue
      }
      keypoints[k] = [p.points[0][0], p.points[0][1], pointVisibility(p)]
      used.push(i)
    }
    instances.push({ first: group.first, label, group: g, box, keypoints, shapes: used.sort((a, b) => a - b) })
  }
  instances.sort((a, b) => a.first - b.first)
  return { instances: instances.map(({ first: _first, ...p }) => p), issues }
}

/**
 * A posed object as Seamtag's shapes, for import: a rectangle with its class's label and a point
 * for each keypoint that's there, named after it, all sharing group_id `group`. Keypoints with
 * visibility 0 aren't labelled and are left out; 1 (labelled but hidden) is marked difficult, as
 * X-AnyLabeling marks it. Unnamed keypoints are called kp0, kp1 and so on.
 */
export function shapesFromPose(label: string, box: [number, number, number, number], keypoints: ([number, number, number] | null)[], names: string[], group: number): Shape[] {
  const shapes = [newShape(label, 'rectangle', [[box[0], box[1]], [box[2], box[3]]], { group_id: group })]
  keypoints.forEach((k, i) => {
    if (!k || !(k[2] > 0)) return
    shapes.push(newShape(names[i] ?? `kp${i}`, 'point', [[k[0], k[1]]], { group_id: group, ...(k[2] === 1 ? { flags: { difficult: true } } : {}) }))
  })
  return shapes
}
