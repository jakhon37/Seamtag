// File > Export dataset…: an open folder's labels written as a COCO or Ultralytics YOLO dataset.
// Each image's own label file stays the working copy; the dataset is made from it, as X-AnyLabeling
// and CVAT make theirs. A check first reads every label file and runs the format's writers over it
// without writing anything, so the window shows what goes in and what the format can't hold. The
// export then builds the dataset in a hidden folder beside the one chosen and moves it into place
// at the end, so a stopped or failed export leaves nothing half written. Nothing is ever written
// inside the image folder or the folder its label files are saved in.
import { constants, createReadStream, promises as fs } from 'fs'
import path from 'path'
import { cocoLabels, type CocoDataset, type CocoEntry, type CocoInfo, type CocoTask, writeCoco } from '../shared/coco'
import { type DatasetImage, type Issue, type KeypointSchema, type PoseInstance, type Severity, SEVERITIES } from '../shared/dataset'
import type { LabelFormat } from '../shared/formats'
import { isAi } from '../shared/labelfile'
import { CLOSED_TYPES } from '../shared/outline'
import { poseInstances, readPoseYaml, writePoseYaml } from '../shared/pose'
import { isYoloImage, labelPathFor, labelRows, poseYamlKeys, type YoloTask, writeDataYaml } from '../shared/ultralytics'
import { labelFileClashes, readDataset } from './dataset'
import { folderOf, imagePath, isEmbedded, readDoc, writeAtomic } from './project'

/** The datasets the window exports */
export const EXPORT_FORMATS = ['coco', 'coco-keypoints', 'yolo-detect', 'yolo-segment', 'yolo-pose', 'yolo-obb'] as const
export type ExportFormat = (typeof EXPORT_FORMATS)[number]
const TASKS: Partial<Record<ExportFormat, YoloTask>> = { 'yolo-detect': 'detect', 'yolo-segment': 'segment', 'yolo-pose': 'pose', 'yolo-obb': 'obb' }
const FORMAT_WORDS: Record<ExportFormat, string> = {
  coco: 'COCO', 'coco-keypoints': 'COCO with keypoints', 'yolo-detect': 'YOLO detect', 'yolo-segment': 'YOLO segment', 'yolo-pose': 'YOLO pose', 'yolo-obb': 'YOLO OBB'
}
/** Does this format need a keypoints file, for each class's keypoint names? */
export const needsKeypoints = (format: ExportFormat) => format === 'coco-keypoints' || format === 'yolo-pose'

/**
 * How images are split: not at all; by shares in percent, after sorting them by path and
 * shuffling them with a seed, so a seed always gives the same split; or by the train, val or
 * test folder each image is in
 */
export type SplitChoice = { mode: 'none' } | { mode: 'ratio'; train: number; val: number; test: number; seed: number } | { mode: 'folders' }

export type ExportOptions = {
  format: ExportFormat
  split: SplitChoice
  /** Copied; hard-linked, or copied where a link can't be made; or, for COCO, left where they are */
  images: 'copy' | 'link' | 'leave'
  /** Never-labelled images go in as backgrounds. Images labelled with no shapes always do. */
  backgrounds: boolean
  /** AI proposals not yet accepted are left out, rather than stopping the export */
  leaveAi: boolean
  /** The keypoints file (X-AnyLabeling's pose YAML), for COCO with keypoints and YOLO pose */
  keypoints: string | null
  /** The window's save format: an image with label files in two formats is read in this one */
  prefer: LabelFormat
  /** The label list, which numbers the categories and classes first */
  labels: string[]
}

export type ExportProgress = { phase: 'read' | 'check' | 'write' | 'finish'; done: number; total: number }
type Context = { signal?: AbortSignal; onProgress?: (p: ExportProgress) => void; version?: string }

/** A split as the export makes it: its name, its folder in the dataset, and what it holds */
export type SplitCount = { name: string; folder: string; images: number; objects: number; backgrounds: number; posed: number }
/** A label's objects in each split, in the splits' order */
export type LabelCount = { label: string; objects: number[] }
/** Problems of one kind, with the first few of them */
export type IssueGroup = { key: string; code: string; severity: Severity; title: string; count: number; issues: Issue[] }
/** A keypoints file as the window shows it */
export type KeypointsSummary = { file: string; classes: { name: string; keypoints: number }[]; flip: boolean; skeleton: boolean }

export type Preflight = {
  /** Asks for more of its issues: moreIssues */
  id: number
  splits: SplitCount[]
  labels: LabelCount[]
  groups: IssueGroup[]
  counts: Record<Severity, number>
  /** An error stops the export */
  blocked: boolean
  keypoints: KeypointsSummary | null
  /** What the export does beyond the files themselves, said before it does it */
  notes: string[]
}

export type ExportResult = {
  stopped: boolean
  out: string
  splits: SplitCount[]
  /** Images copied, linked, copied where a link failed, and left in place */
  images: { copied: number; linked: number; fellBack: number; left: number }
  /** Every file written, images too */
  files: number
  /** The annotation files and data.yaml, as paths in the dataset */
  written: string[]
  /** The warnings and notes the check showed, which the export went ahead with */
  accepted: { severity: Severity; title: string; count: number }[]
  notes: string[]
}

// What each kind of issue is called in the window, sentence case; a severity of its own where a
// code has two
const TITLES: Record<string, string> = {
  'export-no-keypoints': 'No keypoints file',
  'export-keypoints-unreadable': "The keypoints file can't be read",
  'export-nothing': 'Nothing to export',
  'export-no-classes': 'No classes',
  'export-ai-pending': 'AI proposals not yet accepted',
  'export-shared-label-file': 'Images sharing one label file',
  'export-same-path': 'Images that would be written to one path',
  'export-same-label': 'Images whose YOLO label files would be one file',
  'export-in-label-file': 'Images held inside their label files',
  'export-not-yolo-image': "Images Ultralytics doesn't read",
  'export-no-split-folder': 'Images under no train, val or test folder',
  'export-no-train': 'No images for train',
  'export-no-val': 'No images for val',
  'export-never-labelled': 'Never-labelled images left out',
  'export-background': 'Never-labelled images as backgrounds',
  'export-ai-left-out': 'AI proposals left out',
  'export-ai-only': 'Images with only AI proposals, left out',
  'coco-broken-labels': "Label files that can't be read",
  'coco-unreadable-image': "Images that can't be read",
  'coco-no-inside': "Lines and points COCO can't hold",
  'coco-bad-shape': 'Shapes with the wrong number of points',
  'coco-mask-unreadable': "Masks that can't be read",
  'coco-outside': "Shapes past the image's edge",
  'coco-too-few-points': 'Polygons of fewer than 3 points',
  'coco-no-radius': 'Circles with no radius',
  'coco-no-pixels': 'Objects that cover no whole pixel',
  'coco-pose-class': 'Posed objects with no keypoint names',
  'coco-pose-keypoints': 'Posed objects with the wrong keypoint count',
  'coco-unlisted-label': 'Labels not in the label list',
  'coco-skeleton': 'Skeleton lines to unknown keypoints',
  'yolo-broken': "Label files that can't be read",
  'yolo-no-size': "Images whose size can't be read",
  'yolo-clamped:warning': "Shapes cut at the image's edge",
  'yolo-clamped:info': "Shapes cut at the image's edge by a pixel or less",
  'yolo-repeated': 'Shapes that repeat a row',
  'yolo-unknown-label': 'Labels not in the class list',
  'yolo-no-area': 'Shapes with no area',
  'yolo-outside': 'Shapes outside the image',
  'yolo-keypoint-outside': 'Keypoints outside the image',
  'yolo-not-pose': "Shapes that aren't part of a pose",
  'yolo-open-shape': "Lines and points YOLO can't hold",
  'yolo-mask-unreadable': "Masks that can't be read",
  'yolo-obb-fitted': 'Shapes written as the rotated box around them',
  'yolo-mask-holes': 'Mask holes filled',
  'yolo-joined': 'Objects of several parts joined',
  'yolo-few-points': 'Outlines of fewer than 3 points',
  'pose-empty-box:error': 'Pose groups whose box has no pixels',
  'pose-empty-box:warning': 'Boxes with no pixels',
  'pose-no-keypoints': 'Boxes with no keypoints',
  'pose-no-group': 'Keypoints with no group',
  'pose-two-boxes': 'Pose groups with two boxes',
  'pose-no-class': 'Pose groups with no box or class',
  'pose-no-box': 'Pose groups with no box',
  'pose-unknown-keypoint': "Points that aren't the class's keypoints",
  'pose-repeated-keypoint': 'Keypoints given twice',
  'pose-flip': 'No flip pairs for mirroring'
}
const titleOf = (code: string, severity: Severity) => TITLES[`${code}:${severity}`] ?? TITLES[code] ?? code

const issue = (severity: Severity, code: string, image: string | null, message: string, shape?: number): Issue =>
  shape === undefined ? { severity, code, image, message } : { severity, code, image, shape, message }
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString('en-GB')} ${n === 1 ? one : many}`
// A few names in full; the rest, as when every image shares one -O labels.json, as a count
const listed = (names: string[]) => names.length < 2 ? names.join('')
  : names.length > 4 ? `${names.slice(0, 3).join(', ')} and ${plural(names.length - 3, 'other image')}` : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
const byCode = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)
const posixJoin = (...parts: string[]) => parts.filter(Boolean).join('/')
// The main process answers the window between chunks of work, so its saves never wait on an export
const tick = () => new Promise<void>((resolve) => setImmediate(resolve))
const reason = (e: unknown) => (e instanceof Error ? e.message : String(e))

// Windows and macOS find a.PNG by the name a.png
const FOLDS = process.platform === 'win32' || process.platform === 'darwin'
const fold = (p: string) => (FOLDS ? p.toLowerCase() : p)
const same = (a: string, b: string) => fold(a) === fold(b)
const inside = (child: string, parent: string) => {
  const rel = path.relative(fold(parent), fold(child))
  return !!rel && rel !== '..' && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel)
}

/** A path with its links followed as far as it exists, so a link into the image folder counts as the image folder */
async function realPath(p: string): Promise<string> {
  let head = path.resolve(p)
  const tail: string[] = []
  for (;;) {
    try {
      return path.join(await fs.realpath(head), ...tail)
    } catch {
      const up = path.dirname(head)
      if (up === head) return path.resolve(p)
      tail.unshift(path.basename(head))
      head = up
    }
  }
}

/**
 * Why a folder can't take an export, or null when it can. It must be new or empty, so nothing
 * in it is written over, and neither the image folder nor inside it, nor the folder the label
 * files are saved in (-O).
 */
export async function folderProblem(root: string, out: string): Promise<string | null> {
  if (!out || !path.isAbsolute(out)) return 'Choose a folder to export to'
  const folder = folderOf(root)
  const [real, images] = await Promise.all([realPath(out), realPath(folder.dir)])
  if (same(real, images)) return "That's the image folder: choose a new or empty folder outside it"
  if (inside(real, images)) return "That's inside the image folder: choose a new or empty folder outside it"
  if (folder.outputDir) {
    const labels = await realPath(folder.outputDir)
    if (same(real, labels)) return "That's the folder the label files are saved in: choose a new or empty folder outside it"
    if (inside(real, labels)) return "That's inside the folder the label files are saved in: choose a new or empty folder outside it"
  }
  if (folder.outputFile && same(real, await realPath(path.dirname(folder.outputFile)))) {
    return "That's the folder the label file is saved in: choose a new or empty folder"
  }
  const stat = await fs.lstat(out).catch(() => null)
  // The export is moved into place by its name at the end, which a link can't take
  if (stat?.isSymbolicLink()) return "That's a link to a folder: choose the folder itself"
  if (stat && !stat.isDirectory()) return "That's a file: choose a new or empty folder"
  if (stat) {
    const entries = await fs.readdir(out).catch(() => null)
    if (!entries) return "That folder can't be read: choose another"
    if (entries.length) return "That folder isn't empty: choose a new or empty folder, so nothing in it is written over"
  } else if (!(await fs.stat(path.dirname(out)).catch(() => null))?.isDirectory()) {
    return "The folder it would go in doesn't exist"
  }
  return null
}

// Splits

const SPLIT_FOLDERS: Record<string, string> = { train: 'train', val: 'val', valid: 'val', validation: 'val', test: 'test' }
const SPLITS = ['train', 'val', 'test'] as const

/** The split an image's folders give it: its first folder named train, val, valid, validation or test, in any case, which leaves its path */
export function splitFolder(name: string): { split: string | null; rel: string } {
  const parts = name.split('/')
  for (let i = 0; i < parts.length - 1; i++) {
    const split = SPLIT_FOLDERS[parts[i].toLowerCase()]
    if (split) return { split, rel: [...parts.slice(0, i), ...parts.slice(i + 1)].join('/') }
  }
  return { split: null, rel: name }
}

/** A seeded random number generator, mulberry32: the same seed gives the same numbers everywhere */
function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Names sorted by path, then shuffled with the seed (Fisher and Yates' shuffle) */
export function shuffled(names: string[], seed: number): string[] {
  const out = [...names].sort(byCode)
  const next = random(seed)
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** How many of n go to each share: each its whole part, then what's left over one each by the largest remainder, earlier shares first on a tie */
export function shareOut(n: number, percents: number[]): number[] {
  const exact = percents.map((p) => (n * p) / 100)
  const counts = exact.map(Math.floor)
  const order = exact.map((e, i) => ({ i, rest: e - counts[i] })).sort((a, b) => b.rest - a.rest || a.i - b.i)
  for (let left = n - counts.reduce((a, b) => a + b, 0), k = 0; left > 0; left--, k++) counts[order[k % order.length].i]++
  return counts
}

// Keypoints files

/** A keypoints file as the window shows it, and its schema; throws with what's wrong with it */
export async function readKeypoints(file: string): Promise<{ schema: KeypointSchema; summary: KeypointsSummary }> {
  const { schema } = readPoseYaml(await fs.readFile(file, 'utf8'))
  return { schema, summary: summaryOf(file, schema) }
}

function summaryOf(file: string, schema: KeypointSchema): KeypointsSummary {
  const classes = Object.entries(schema.classes).map(([name, names]) => ({ name, keypoints: names.length }))
  const some = (pairs?: Record<string, unknown[]>) => !!pairs && Object.values(pairs).some((p) => p.length > 0)
  return { file, classes, flip: some(schema.flip), skeleton: some(schema.skeleton) }
}

// Names that say left, and the same names saying right: the keypoints a mirrored image swaps
function flipPairs(names: string[]): [string, string][] {
  const pairs: [string, string][] = []
  for (const a of names) {
    const b = a.replace(/left/gi, (m) => (m === 'LEFT' ? 'RIGHT' : m[0] === 'L' ? 'Right' : 'right'))
    if (b !== a && names.includes(b)) pairs.push([a, b])
  }
  return pairs
}

/**
 * A starting keypoints file from a folder's posed objects: each box's label (any closed shape)
 * with the labels of the points sharing its group_id, in the order first seen, and flip pairs
 * where names say left and right (left_eye, right_eye). Groups with no box, or boxes of two
 * labels, are passed over, as are AI proposals not yet accepted. Null when there are none.
 */
export function keypointSchemaOf(images: DatasetImage[]): KeypointSchema | null {
  const classes = new Map<string, string[]>()
  for (const image of images) {
    const groups = new Map<number, { boxes: Set<string>; points: string[] }>()
    image.shapes.forEach((s) => {
      if (s.group_id === null || isAi(s)) return
      const g = groups.get(s.group_id) ?? { boxes: new Set<string>(), points: [] }
      groups.set(s.group_id, g)
      if (CLOSED_TYPES.has(s.shape_type)) g.boxes.add(s.label)
      else if (s.shape_type === 'point') g.points.push(s.label)
    })
    for (const { boxes, points } of groups.values()) {
      if (boxes.size !== 1 || !points.length) continue
      const [label] = boxes
      const names = classes.get(label) ?? []
      classes.set(label, names)
      for (const p of points) if (!names.includes(p) && p !== label) names.push(p)
    }
  }
  if (!classes.size) return null
  const schema: KeypointSchema = { classes: Object.fromEntries(classes) }
  const flip = Object.fromEntries([...classes].map(([label, names]) => [label, flipPairs(names)] as const).filter(([, pairs]) => pairs.length))
  if (Object.keys(flip).length) schema.flip = flip
  return schema
}

/** The keypoints file keypointSchemaOf makes of an open folder's labels; null when it has no posed objects */
export async function keypointsFromFolder(root: string, prefer: LabelFormat, signal?: AbortSignal) {
  return keypointSchemaOf(await readDataset(root, { prefer, signal }))
}

/** Writes a keypoints file X-AnyLabeling reads, and says what's in it */
export async function saveKeypoints(file: string, schema: KeypointSchema): Promise<KeypointsSummary> {
  await writeAtomic(file, writePoseYaml(schema))
  return summaryOf(file, schema)
}

// The check

/** An image as the export takes it */
type Item = {
  image: DatasetImage
  /** The number in the label file of each of image's shapes, when AI proposals were left out of them */
  map: number[] | null
  split: string
  /** Its path in its split's folder of images */
  rel: string
  /** Its posed objects, for COCO with keypoints */
  poses?: PoseInstance[]
}

type Gathered = {
  options: ExportOptions
  items: Item[]
  /** Folder names in the dataset, in order: '' for a YOLO dataset that isn't split */
  splits: string[]
  classes: string[]
  schema: KeypointSchema | null
  keypoints: KeypointsSummary | null
  issues: Issue[]
  notes: string[]
}

// The writers number shapes in the list they were given. Where AI proposals were left out of it,
// issues name shapes by their number in the label file again, in `shape` and in messages ("shape
// 3"), but never inside a quoted label.
function renumbered(i: Issue, map: number[] | null): Issue {
  if (!map) return i
  const message = i.message.split(/(“[^”]*”)/).map((part, k) => (k % 2 ? part : part.replace(/\bshapes? \d+(?:(?:, | and )\d+)*/g,
    (m) => m.replace(/\d+/g, (d) => String((map[Number(d) - 1] ?? Number(d) - 1) + 1))))).join('')
  return { ...i, message, ...(i.shape !== undefined ? { shape: map[i.shape] ?? i.shape } : {}) }
}

function checkOptions(o: ExportOptions) {
  if (!EXPORT_FORMATS.includes(o.format)) throw new Error(`There's no dataset format called ${o.format}`)
  if (o.images === 'leave' && !o.format.startsWith('coco')) throw new Error('Only COCO can leave the images where they are')
  if (!['copy', 'link', 'leave'].includes(o.images)) throw new Error(`Images can't be ${o.images}`)
  const s = o.split
  if (s.mode === 'ratio') {
    const shares = [s.train, s.val, s.test]
    if (!shares.every((p) => Number.isInteger(p) && p >= 0) || shares.reduce((a, b) => a + b, 0) !== 100 || s.train === 0) {
      throw new Error('The split needs whole percentages adding up to 100, with some for train')
    }
    if (!Number.isInteger(s.seed) || s.seed < 0) throw new Error('The seed needs to be a whole number, 0 or more')
  } else if (s.mode !== 'none' && s.mode !== 'folders') throw new Error('Unknown split')
}

async function gather(root: string, options: ExportOptions, ctx: Context): Promise<Gathered> {
  checkOptions(options)
  const all = await readDataset(root, { prefer: options.prefer, signal: ctx.signal, onProgress: (done, total) => ctx.onProgress?.({ phase: 'read', done, total }) })
  const issues: Issue[] = []
  const notes: string[] = []
  const add = (severity: Severity, code: string, image: string | null, message: string, shape?: number) => issues.push(issue(severity, code, image, message, shape))
  const task = TASKS[options.format]
  const coco = !task
  const leave = options.images === 'leave'

  let schema: KeypointSchema | null = null, keypoints: KeypointsSummary | null = null
  if (needsKeypoints(options.format)) {
    if (!options.keypoints) add('error', 'export-no-keypoints', null, `${FORMAT_WORDS[options.format]} needs a keypoints file naming each class's keypoints: choose one, or make one from the folder's posed objects`)
    else {
      try {
        ;({ schema, summary: keypoints } = await readKeypoints(options.keypoints))
      } catch (e) {
        add('error', 'export-keypoints-unreadable', null, `${path.basename(options.keypoints)}: ${reason(e)}`)
      }
    }
  }

  // Which images go in, and what of them
  const chosen: Omit<Item, 'split' | 'rel'>[] = []
  let left = 0
  for (const image of all) {
    const { name } = image
    const never = image.file === null && !image.broken
    if (never && !options.backgrounds) {
      add('info', 'export-never-labelled', name, `${name} has never been labelled, so it's left out`)
      left++
      continue
    }
    if (task && !isYoloImage(name)) {
      add('warning', 'export-not-yolo-image', name, `${name} isn't an image Ultralytics reads, so it's left out`)
      continue
    }
    if (never) add('info', 'export-background', name, `${name} has never been labelled, so it goes in as a background, with no objects`)
    if (leave && isEmbedded(root, name)) add('error', 'export-in-label-file', name, `${name} is held inside its label file, so it can't be left in place: copy the images instead`)
    const ai: number[] = []
    image.shapes.forEach((s, i) => { if (isAi(s)) ai.push(i) })
    if (!ai.length) {
      chosen.push({ image, map: null })
      continue
    }
    const proposals = plural(ai.length, 'AI proposal')
    if (!options.leaveAi) {
      add('error', 'export-ai-pending', name, `${name} has ${proposals} not yet accepted: accept or reject ${ai.length === 1 ? 'it' : 'them'}, or leave unaccepted proposals out`, ai[0])
      chosen.push({ image, map: null })
      continue
    }
    const map = image.shapes.flatMap((s, i) => (isAi(s) ? [] : [i]))
    if (!map.length) {
      add('info', 'export-ai-only', name, `${name} holds only ${proposals} not yet accepted, so it's left out, as not yet labelled`, ai[0])
      continue
    }
    add('info', 'export-ai-left-out', name, `${name}: its ${proposals} not yet accepted ${ai.length === 1 ? 'is' : 'are'} left out`, ai[0])
    chosen.push({ image: { ...image, shapes: map.map((i) => image.shapes[i]), masks: map.map((i) => image.masks[i]) }, map })
  }

  // Splits
  const s = options.split
  let items: Item[]
  let splits: string[]
  if (s.mode === 'none') {
    // CVAT 2.83 calls the one subset of a dataset that isn't split default
    const folder = coco ? 'default' : ''
    items = chosen.map((c) => ({ ...c, split: folder, rel: c.image.name }))
    splits = [folder]
    if (!coco) notes.push("With no split, images/ and labels/ hold every image, and data.yaml's train and val both point at images, as Ultralytics needs a val entry.")
  } else if (s.mode === 'ratio') {
    const rank = new Map(shuffled(all.map((i) => i.name), s.seed).map((n, k) => [n, k]))
    const order = [...chosen].sort((a, b) => rank.get(a.image.name)! - rank.get(b.image.name)!)
    const counts = shareOut(order.length, [s.train, s.val, s.test])
    const at = new Map<string, string>()
    let k = 0
    SPLITS.forEach((split, i) => { for (let n = 0; n < counts[i]; n++) at.set(order[k++].image.name, split) })
    items = chosen.map((c) => ({ ...c, split: at.get(c.image.name)!, rel: c.image.name }))
    splits = SPLITS.filter((_, i) => [s.train, s.val, s.test][i] > 0)
  } else {
    items = chosen.map((c) => {
      const { split, rel } = splitFolder(c.image.name)
      if (!split) add('warning', 'export-no-split-folder', c.image.name, `${c.image.name} isn't under a train, val or test folder, so it goes to train`)
      return { ...c, split: split ?? 'train', rel }
    })
    splits = SPLITS.filter((split) => split === 'train' || items.some((i) => i.split === split))
  }
  // Images in the folder's order within each split
  items.sort((a, b) => splits.indexOf(a.split) - splits.indexOf(b.split))
  if (task && s.mode !== 'none' && items.length) {
    const train = items.some((i) => i.split === 'train')
    if (!train) add('warning', 'export-no-train', null, `No images go to train, so data.yaml's train points at the ${items[0].split} images`)
    if (!items.some((i) => i.split === 'val')) add('warning', 'export-no-val', null, `No images go to val, so data.yaml's val points at the ${train ? 'train' : items[0].split} images, as Ultralytics needs a val entry`)
  }

  // Images, and YOLO's label files, that would land on one path
  const clash = (code: string, paths: (i: Item) => string | null, say: (names: string[], at: string) => string) => {
    const by = new Map<string, { at: string; names: string[] }>()
    for (const i of items) {
      const at = paths(i)
      if (at === null) continue
      const known = by.get(fold(at))
      if (known) known.names.push(i.image.name)
      else by.set(fold(at), { at, names: [i.image.name] })
    }
    for (const { at, names } of by.values()) if (names.length > 1) add('error', code, names[0], say(names, at))
  }
  const imageAt = (i: Item) => posixJoin('images', i.split, i.rel)
  if (!leave) clash('export-same-path', imageAt, (names, at) => `${listed(names)} would ${names.length === 2 ? 'both' : 'all'} be written to ${at}: rename one of them`)
  if (task) clash('export-same-label', (i) => labelPathFor(imageAt(i)), (names, at) => `${listed(names)} would share one label file, ${at}, as Ultralytics finds label files by name: rename one of them`)
  const going = new Set(items.map((i) => i.image.name))
  for (const c of labelFileClashes(all)) {
    if (!c.images.some((n) => going.has(n))) continue
    add('error', 'export-shared-label-file', c.images[0], `${listed(c.images)} share one label file, ${path.basename(c.file)}, so whose labels are whose can't be told: give each image its own name`)
  }

  // Posed objects, and the order of the categories or classes
  if (options.format === 'coco-keypoints' && schema) {
    for (const i of items) {
      const found = poseInstances(i.image, schema)
      i.poses = found.instances
      for (const f of found.issues) issues.push(renumbered(f, i.map))
    }
  }
  let classes: string[]
  if (task === 'pose') {
    classes = schema ? poseYamlKeys(schema).names : []
    if (schema) issues.push(...poseYamlKeys(schema).issues)
  } else classes = cocoLabels(options.labels, items.map((i): CocoEntry => ({ image: i.image, fileName: '', poses: i.poses })))
  if (task && task !== 'pose' && !classes.length && items.length) {
    add('error', 'export-no-classes', null, "No shape has a label to make a class of, and data.yaml needs at least one: label some shapes, or add labels to the label list")
  }
  if (!items.length) {
    add('error', 'export-nothing', null, !all.length ? 'The folder has no images to export'
      : left ? `None of the images has labels to export: label some, or include the ${plural(left, 'never-labelled image')} as backgrounds`
        : 'None of the images can be exported')
  }
  if (coco) {
    notes.push(leave
      ? `Images stay where they are: each file_name is the image's path in the open folder, so use that folder (${root}) as the image root.`
      : 'Each file_name is the image\'s path in the dataset (images/<split>/…), so use the export folder as the image root.')
  }
  if (options.images === 'link') notes.push("A hard link shares an image's file with the open folder, so editing one edits both. Where a link can't be made, as across drives, the image is copied.")
  return { options, items, splits, classes, schema, keypoints, issues, notes }
}

/** What an export writes as it goes: an image, a text file, a chunk of a COCO file */
type Sink = {
  image(item: Item, at: string): Promise<void>
  text(at: string, text: string): Promise<void>
  coco(split: string, task: CocoTask, dataset: CocoDataset): Promise<void>
}

const CHUNK = 256

/**
 * The format's writers over every image, a split at a time and a chunk of images at a time,
 * counting what goes in and keeping what they report; with a sink, writing it all too
 */
async function convert(g: Gathered, ctx: Context, sink: Sink | null) {
  const { options, items, classes } = g
  const task = TASKS[options.format]
  const issues: Issue[] = []
  const splits = new Map(g.splits.map((folder) => [folder, { name: folder || 'all', folder, images: 0, objects: 0, backgrounds: 0, posed: 0 }]))
  const labels = new Map(classes.map((label) => [label, g.splits.map(() => 0)]))
  const count = (split: string, label: string) => {
    const at = g.splits.indexOf(split)
    const row = labels.get(label) ?? labels.set(label, g.splits.map(() => 0)).get(label)!
    row[at]++
    splits.get(split)!.objects++
  }
  const info: CocoInfo = {
    description: `Exported by Seamtag ${ctx.version ?? ''}`.trim(), version: ctx.version ?? '', year: new Date().getFullYear(),
    date_created: new Date().toISOString(), contributor: '', url: ''
  }
  const imageAt = (i: Item) => (options.images === 'leave' ? i.image.name : posixJoin('images', i.split, i.rel))
  const phase = sink ? 'write' : 'check'
  let done = 0, told = 0
  const progress = (n: number) => {
    done += n
    if (Date.now() - told >= 100 || done === items.length) {
      told = Date.now()
      ctx.onProgress?.({ phase, done, total: items.length })
    }
  }
  for (let start = 0; start < items.length;) {
    ctx.signal?.throwIfAborted()
    // A chunk is in one split: the items are in split order, and a chunk ends where its split does
    const chunk = items.slice(start, start + CHUNK)
    const end = chunk.findIndex((i) => i.split !== chunk[0].split)
    const part = end < 0 ? chunk : chunk.slice(0, end)
    start += part.length
    const split = part[0].split
    const stats = splits.get(split)!
    stats.images += part.length
    if (task) {
      const texts = part.map((i) => {
        const rows = labelRows(i.image, task, classes, g.schema ?? undefined)
        for (const f of rows.issues) issues.push(renumbered(f, i.map))
        if (!rows.rows) stats.backgrounds++
        for (const line of rows.text.split('\n')) if (line) count(split, classes[Number(line.slice(0, line.indexOf(' ')))])
        if (task === 'pose') stats.posed += rows.rows
        return rows.text
      })
      if (sink) {
        await pool(part.map((i, k) => [i, texts[k]] as const), async ([i, text]) => {
          const at = imageAt(i)
          await sink.image(i, at)
          await sink.text(labelPathFor(at), text)
        }, ctx.signal)
      }
    } else {
      const entries = part.map((i): CocoEntry => ({ image: i.image, fileName: imageAt(i), poses: i.poses }))
      const byName = new Map(part.map((i) => [i.image.name, i]))
      const tasks: CocoTask[] = options.format === 'coco-keypoints' ? ['instances', 'keypoints'] : ['instances']
      const objects = new Set<number>()
      for (const t of tasks) {
        if (t === 'keypoints' && !g.schema) continue
        const { dataset, issues: found } = writeCoco(entries, { task: t, labels: classes, info, schema: g.schema ?? undefined })
        for (const f of found) issues.push(renumbered(f, f.image === null ? null : byName.get(f.image)?.map ?? null))
        for (const a of dataset.annotations) {
          objects.add(a.image_id)
          if (t === 'instances') count(split, classes[a.category_id - 1] ?? String(a.category_id))
          else stats.posed++
        }
        if (sink) await sink.coco(split, t, dataset)
      }
      stats.backgrounds += part.length - objects.size
      if (sink && options.images !== 'leave') await pool(part, (i) => sink.image(i, imageAt(i)), ctx.signal)
    }
    progress(part.length)
    await tick()
  }
  // The same problem told twice, as both of COCO's files tell of an image that can't be read, is one
  const seen = new Set<string>()
  const unique = issues.filter((i) => {
    const key = `${i.code}\0${i.image}\0${i.shape}\0${i.message}`
    return !seen.has(key) && !!seen.add(key)
  })
  return { issues: unique, splits: [...splits.values()], labels: [...labels].map(([label, objects]) => ({ label, objects })) }
}

/** Runs one at a time per worker, four workers: enough to keep a disk busy */
async function pool<T>(list: readonly T[], fn: (t: T) => Promise<void>, signal?: AbortSignal) {
  let next = 0
  await Promise.all(Array.from({ length: Math.min(4, list.length) }, async () => {
    while (next < list.length) {
      signal?.throwIfAborted()
      await fn(list[next++])
    }
  }))
}

function grouped(issues: Issue[]) {
  const groups = new Map<string, Issue[]>()
  for (const i of issues) {
    const key = `${i.severity}:${i.code}`
    const list = groups.get(key)
    if (list) list.push(i)
    else groups.set(key, [i])
  }
  return groups
}

const SHOWN = 20
let lastCheck: { id: number; groups: Map<string, Issue[]> } | null = null
let checks = 0

/**
 * What exporting would do, without writing anything: the images and objects in each split and
 * of each label, and every issue the format's writers report, grouped by kind with the first few
 * of each. Errors stop the export.
 */
export async function preflight(root: string, options: ExportOptions, ctx: Context = {}): Promise<Preflight> {
  const g = await gather(root, options, ctx)
  const converted = await convert(g, ctx, null)
  const all = [...g.issues, ...converted.issues]
  const groups = grouped(all)
  const id = ++checks
  lastCheck = { id, groups }
  const rank = (s: Severity) => SEVERITIES.indexOf(s)
  const list: IssueGroup[] = [...groups].map(([key, issues]) => ({
    key, code: issues[0].code, severity: issues[0].severity, title: titleOf(issues[0].code, issues[0].severity), count: issues.length, issues: issues.slice(0, SHOWN)
  })).sort((a, b) => rank(a.severity) - rank(b.severity) || b.count - a.count || byCode(a.title, b.title))
  const counts = { error: 0, warning: 0, info: 0 }
  for (const i of all) counts[i.severity]++
  return { id, splits: converted.splits, labels: converted.labels, groups: list, counts, blocked: counts.error > 0, keypoints: g.keypoints, notes: g.notes }
}

/** More of a check's issues of one kind, from the one given on; none once a newer check has run */
export function moreIssues(id: number, key: string, from: number, count = 200): Issue[] {
  if (lastCheck?.id !== id) return []
  return (lastCheck.groups.get(key) ?? []).slice(from, from + count)
}

/** One COCO file written a chunk of images at a time, as writeCoco writes them all at once; its annotations wait in a file of their own until the last image is in */
class CocoFile {
  private images = 0
  private annotations = 0
  private categories: unknown[] = []
  private constructor(readonly file: string, private side: string, private head: fs.FileHandle, private rest: fs.FileHandle) {}
  static async open(file: string) {
    await fs.mkdir(path.dirname(file), { recursive: true })
    const side = `${file}.annotations`
    return new CocoFile(file, side, await fs.open(file, 'wx'), await fs.open(side, 'wx'))
  }
  async add(d: CocoDataset) {
    if (!this.images) {
      await this.head.write(`{"info":${JSON.stringify(d.info)},"licenses":${JSON.stringify(d.licenses)},"images":[`)
      this.categories = d.categories
    }
    if (d.images.length) await this.head.write((this.images ? ',' : '') + d.images.map((im) => JSON.stringify({ ...im, id: im.id + this.images })).join(','))
    if (d.annotations.length) {
      await this.rest.write((this.annotations ? ',' : '') + d.annotations.map((a) => JSON.stringify({ ...a, id: a.id + this.annotations, image_id: a.image_id + this.images })).join(','))
    }
    this.images += d.images.length
    this.annotations += d.annotations.length
  }
  async close() {
    await this.rest.close()
    await this.head.write('],"annotations":[')
    for await (const chunk of createReadStream(this.side)) await this.head.write(chunk)
    await this.head.write(`],"categories":${JSON.stringify(this.categories)}}`)
    await this.head.close()
    await fs.rm(this.side)
  }
  async abandon() {
    await Promise.allSettled([this.head.close(), this.rest.close()])
  }
}

// COCO's own names, which CVAT 2.83's COCO export also writes: annotations/instances_<split>.json beside images/<split>/
const COCO_FILES: Record<CocoTask, string> = { instances: 'instances', keypoints: 'person_keypoints' }

/**
 * Writes the dataset into `out`, a new or empty folder outside the image folder: built in a
 * hidden folder beside it and moved into place at the end, or, where `out` is a drive of its
 * own, in it, emptied again if the export stops. Refuses while the check finds errors. Stopping
 * leaves nothing behind and resolves with `stopped`.
 */
export async function runExport(root: string, options: ExportOptions, out: string, ctx: Context = {}): Promise<ExportResult> {
  const problem = await folderProblem(root, out)
  if (problem) throw new Error(problem)
  const stopped: ExportResult = { stopped: true, out, splits: [], images: { copied: 0, linked: 0, fellBack: 0, left: 0 }, files: 0, written: [], accepted: [], notes: [] }
  let g: Gathered
  try {
    g = await gather(root, options, ctx)
  } catch (e) {
    if (ctx.signal?.aborted) return stopped
    throw e
  }
  const blocking = g.issues.find((i) => i.severity === 'error')
  if (blocking) throw new Error(`The dataset can't be exported: ${blocking.message}`)
  const parent = path.dirname(out)
  const existing = await fs.stat(out).catch(() => null)
  // A folder that's a drive of its own can't take a folder moved in from beside it
  const direct = !!existing && existing.dev !== (await fs.stat(parent)).dev
  const work = direct ? out : await fs.mkdtemp(path.join(parent, `.${path.basename(out)}.export-`))
  const counts = { copied: 0, linked: 0, fellBack: 0, left: options.images === 'leave' ? g.items.length : 0 }
  let files = 0
  const made = new Set<string>()
  const dir = async (file: string) => {
    const d = path.dirname(file)
    if (!made.has(d)) {
      await fs.mkdir(d, { recursive: true })
      made.add(d)
    }
  }
  const cocoFiles = new Map<string, CocoFile>()
  const sink: Sink = {
    async image(item, at) {
      const to = path.join(work, ...at.split('/'))
      await dir(to)
      if (isEmbedded(root, item.image.name)) {
        // An image kept inside its label file is written from it
        const doc = await readDoc(imagePath(root, item.image.name))
        await fs.writeFile(to, Buffer.from(doc?.imageData ?? '', 'base64'), { flag: 'wx' })
        counts.copied++
      } else counts[await putImage(imagePath(root, item.image.name), to, options.images === 'link')]++
      files++
    },
    async text(at, text) {
      const to = path.join(work, ...at.split('/'))
      await dir(to)
      await fs.writeFile(to, text, { flag: 'wx' })
      files++
    },
    async coco(split, task, dataset) {
      const at = posixJoin('annotations', `${COCO_FILES[task]}_${split}.json`)
      let file = cocoFiles.get(at)
      if (!file) cocoFiles.set(at, (file = await CocoFile.open(path.join(work, ...at.split('/')))))
      await file.add(dataset)
    }
  }
  const cleanup = async () => {
    await Promise.allSettled([...cocoFiles.values()].map((f) => f.abandon()))
    if (!direct) await fs.rm(work, { recursive: true, force: true })
    else for (const e of await fs.readdir(work).catch(() => [] as string[])) await fs.rm(path.join(work, e), { recursive: true, force: true })
  }
  try {
    const converted = await convert(g, ctx, sink)
    const failed = converted.issues.find((i) => i.severity === 'error')
    if (failed) throw new Error(`The labels changed since they were checked, so the export stopped: ${failed.message}`)
    ctx.onProgress?.({ phase: 'finish', done: 0, total: 1 })
    const written: string[] = []
    for (const [at, file] of cocoFiles) {
      await file.close()
      written.push(at)
      files++
    }
    const task = TASKS[options.format]
    if (task) {
      // Splits with images; with no split, images/ itself is train and val both
      const used = g.splits.filter((s) => g.items.some((i) => i.split === s))
      const train = used.includes('train') ? 'train' : used[0] ?? ''
      const pose = task === 'pose' && g.schema ? poseYamlKeys(g.schema) : null
      const yamlText = writeDataYaml({
        names: g.classes, train: posixJoin('images', train), val: posixJoin('images', used.includes('val') ? 'val' : train),
        test: used.includes('test') ? 'images/test' : null, kptShape: pose?.kptShape ?? null, flipIdx: pose?.flipIdx ?? null, kptNames: pose?.kptNames ?? null
      })
      await fs.writeFile(path.join(work, 'data.yaml'), yamlText, { flag: 'wx' })
      written.push('data.yaml')
      files++
    }
    ctx.signal?.throwIfAborted()
    if (!direct) {
      // Only an empty folder goes: anything put in it since the check stops the export here
      if (existing) await fs.rmdir(out)
      await fs.rename(work, out)
    }
    ctx.onProgress?.({ phase: 'finish', done: 1, total: 1 })
    const accepted = [...grouped([...g.issues, ...converted.issues])].map(([, list]) => ({ severity: list[0].severity, title: titleOf(list[0].code, list[0].severity), count: list.length }))
      .sort((a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) || b.count - a.count)
    return { stopped: false, out, splits: converted.splits, images: counts, files, written: written.sort(byCode), accepted, notes: g.notes }
  } catch (e) {
    await cleanup()
    if (ctx.signal?.aborted) return stopped
    throw e
  }
}

/** An image copied, or hard-linked and copied where a link can't be made, as across drives or on file systems without links */
async function putImage(from: string, to: string, link: boolean): Promise<'copied' | 'linked' | 'fellBack'> {
  if (link) {
    try {
      await fs.link(from, to)
      return 'linked'
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT' || (e as NodeJS.ErrnoException).code === 'EEXIST') throw e
    }
  }
  // A copy shares the file's blocks until either changes where the file system can (APFS, Btrfs, XFS)
  await fs.copyFile(from, to, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE)
  return link ? 'fellBack' : 'copied'
}
