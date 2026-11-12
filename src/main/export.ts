// File > Export dataset…: an open folder's labels written as a COCO or Ultralytics YOLO dataset.
// Each image's own label file stays the working copy; the dataset is made from it, as X-AnyLabeling
// and CVAT make theirs. A check first reads every label file and runs the format's writers over it
// without writing anything, so the window shows what goes in and what the format can't hold. While
// the window is open, a check after a change of options uses that read and what the writers made
// of each image again, so only what the change touches is worked out again. The export reads the
// folder afresh and builds the dataset in a hidden folder, beside the folder chosen when that's
// new and inside it when it's an empty one, and moves it into place at the end, so a stopped or
// failed export leaves nothing half written. Nothing is ever written inside the image folder or
// the folder its label files are saved in.
import { randomBytes } from 'crypto'
import { constants, createReadStream, type Dirent, promises as fs } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { cocoLabels, type CocoDataset, type CocoEntry, type CocoInfo, type CocoTask, writeCoco } from '../shared/coco'
import { type DatasetImage, type Issue, type KeypointSchema, type PoseInstance, type Severity, SEVERITIES } from '../shared/dataset'
import type { LabelFormat } from '../shared/formats'
import { isAi } from '../shared/labelfile'
import { CLOSED_TYPES } from '../shared/outline'
import { poseInstances, readPoseYaml, writePoseYaml } from '../shared/pose'
import { isYoloImage, labelPathFor, labelRows, poseYamlKeys, type YoloTask, writeDataYaml } from '../shared/ultralytics'
import { labelFileClashes, readDataset } from './dataset'
import { nameKey } from './names'
import { folderOf, imagePath, isEmbedded, readDoc, renameRetrying, writeAtomic } from './project'

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
type Context = {
  signal?: AbortSignal
  onProgress?: (p: ExportProgress) => void
  version?: string
  /**
   * The read a check may use again: a check given the id the last check read the folder under
   * uses that read, rather than reading every label file again. The window gives a new id when
   * it opens and after an export fails. With none, the folder is read.
   */
  read?: string
}

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
  'export-empty-split': 'Splits with no images',
  'export-never-labelled': 'Never-labelled images left out',
  'export-background': 'Never-labelled images as backgrounds',
  'export-ai-left-out': 'AI proposals left out',
  'export-ai-only': 'Images with only AI proposals, left out',
  'export-not-a-class': 'Shapes labelled __ignore__ or _background_, left out',
  'export-size-mismatch': 'Label files that give another image size',
  'export-exif-orientation': 'Images turned by their EXIF orientation',
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
  'pose-two-classes': 'Pose groups with shapes of several classes',
  'pose-parts': 'Posed objects of several shapes',
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
const listed = (names: string[], others = 'other image') => names.length < 2 ? names.join('')
  : names.length > 4 ? `${names.slice(0, 3).join(', ')} and ${plural(names.length - 3, others)}` : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
const byCode = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)
const posixJoin = (...parts: string[]) => parts.filter(Boolean).join('/')
// The main process answers the window between chunks of work, so its saves never wait on an export
const tick = () => new Promise<void>((resolve) => setImmediate(resolve))
/**
 * A turn for the window in a long loop over images: awaited once an image, it waits a tick once
 * the loop has run 20 ms since the last, and a check or export stopped meanwhile stops there
 */
type Turn = () => Promise<void> | undefined
function turns(signal?: AbortSignal): Turn {
  let since = Date.now()
  return () => {
    if (Date.now() - since < 20) return
    return tick().then(() => {
      signal?.throwIfAborted()
      since = Date.now()
    })
  }
}
const reason = (e: unknown) => (e instanceof Error ? e.message : String(e))
const keyOf = (i: Issue) => `${i.code}\0${i.image}\0${i.shape}\0${i.message}`
// The same problem told twice, as both of COCO's files tell of an image that can't be read, is one
function unique(issues: Issue[]): Issue[] {
  const seen = new Set<string>()
  return issues.filter((i) => !seen.has(keyOf(i)) && !!seen.add(keyOf(i)))
}

// Paths compared as the file system compares them, as the check compares names: on Windows and
// macOS a.PNG is a.png
const same = (a: string, b: string) => nameKey(a) === nameKey(b)
const inside = (child: string, parent: string) => {
  const rel = path.relative(nameKey(parent), nameKey(child))
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

// What systems leave in folders of their own accord, so a folder holding only these counts as
// empty: Finder's .DS_Store, .localized, Icon\r and ._ files, Explorer's Thumbs.db and desktop.ini,
// and KDE's .directory; at the top of a drive, also the folders its system keeps there, Linux
// desktops' trash (.Trash-1000) among them. The export puts its files beside them and never
// touches them.
const SYSTEM_FILES = new Set(['.ds_store', '.localized', 'icon\r', 'thumbs.db', 'ehthumbs.db', 'desktop.ini', '.directory'])
const DRIVE_FOLDERS = new Set([
  '.spotlight-v100', '.fseventsd', '.trashes', '.temporaryitems', '.documentrevisions-v100', 'system volume information', '$recycle.bin', 'lost+found', '.trash'
])
const systemEntry = (e: Dirent, drive: boolean) => {
  const name = e.name.toLowerCase()
  return e.isFile() ? SYSTEM_FILES.has(name) || name.startsWith('._') : drive && e.isDirectory() && (DRIVE_FOLDERS.has(name) || /^\.trash-\d+$/.test(name))
}
// The hidden folder an export into an empty folder is built in. One that a quit or a crash cut
// off stays there, and is all that keeps the folder from being empty.
const WORK_PREFIX = '.seamtag-export-'
const leftover = (e: Dirent) => e.isDirectory() && e.name.startsWith(WORK_PREFIX) && /^[0-9a-f]{8}$/.test(e.name.slice(WORK_PREFIX.length))
// Whether new files can be made in a folder, as far as the system says before anything is written
const writable = (dir: string) => fs.access(dir, constants.W_OK | constants.X_OK).then(() => true, () => false)

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
  // The dataset would go into the folder a link points to, not the one the window names
  if (stat?.isSymbolicLink()) return "That's a link to a folder: choose the folder itself"
  if (stat && !stat.isDirectory()) return "That's a file: choose a new or empty folder"
  if (stat) {
    const entries = await fs.readdir(out, { withFileTypes: true }).catch(() => null)
    if (!entries) return "That folder can't be read: choose another"
    // The top of a drive: a folder that's its own parent, as E:\ is, or on another device than its parent
    const drive = path.dirname(out) === out || (await fs.stat(path.dirname(out)).catch(() => stat)).dev !== stat.dev
    const held = entries.filter((e) => !systemEntry(e, drive))
    // Named, as macOS and Linux hide it. It's left for the person to remove: it may hold a whole
    // dataset that couldn't be moved into place, or another window's export under way.
    if (held.length && held.every(leftover)) {
      const names = held.map((e) => e.name).sort(byCode)
      return names.length === 1 ? `That folder holds ${names[0]}, a hidden folder left by an export that didn't finish: delete it, or choose another folder`
        : `That folder holds ${listed(names, 'other folder')}, hidden folders left by exports that didn't finish: delete them, or choose another folder`
    }
    if (held.length) return "That folder isn't empty: choose a new or empty folder, so nothing in it is written over"
    if (!(await writable(out))) return "That folder can't be written to: choose another"
  } else if (!(await fs.stat(path.dirname(out)).catch(() => null))?.isDirectory()) {
    return "The folder it would go in doesn't exist"
  } else if (!(await writable(path.dirname(out)))) {
    return "The folder it would go in can't be written to: choose another"
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

/** Names sorted by path, then shuffled with the seed */
export function shuffled(names: string[], seed: number): string[] {
  return shuffle([...names].sort(byCode), seed)
}

/** A list shuffled in place with the seed (Fisher and Yates' shuffle) */
function shuffle<T>(out: T[], seed: number): T[] {
  const next = random(seed)
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

// What's worked out of a read's images as a whole, once: kept while reads give the very same
// images, as the checks after a change of options do, and the export after them while no label
// file or image changed (readDataset gives an image read before as it was). Dropped when the
// window closes.
let perRead: { images: DatasetImage[]; byPath?: number[]; clashes?: ReturnType<typeof labelFileClashes> } | null = null
function readOnce(all: DatasetImage[]) {
  if (perRead?.images.length !== all.length || perRead.images.some((image, k) => image !== all[k])) perRead = { images: all }
  return perRead
}

/** The read's images by path, as numbers in the read, for the ratio split to shuffle with its seed */
function byPath(all: DatasetImage[]): number[] {
  const memo = readOnce(all)
  memo.byPath ??= [...all.keys()].sort((a, b) => byCode(all[a].name, all[b].name))
  return [...memo.byPath]
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

// Images held inside their label files

// What an image file's first bytes say it is
const KINDS: [string, (head: string) => boolean][] = [
  ['.png', (h) => h.startsWith('\x89PNG')], ['.jpg', (h) => h.startsWith('\xff\xd8\xff')], ['.gif', (h) => h.startsWith('GIF8')],
  ['.webp', (h) => h.startsWith('RIFF') && h.slice(8, 12) === 'WEBP'], ['.tif', (h) => h.startsWith('II*\0') || h.startsWith('MM\0*')],
  ['.bmp', (h) => h.startsWith('BM')]
]
const namesIn = new WeakMap<DatasetImage, string>()

/**
 * The name an image goes into the dataset under: its own, or for one held inside its LabelMe
 * file (photo.json), that file's with the extension the image's bytes give (photo.png), so the
 * dataset holds an image file that COCO's and Ultralytics' readers take for one
 */
async function nameIn(root: string, image: DatasetImage): Promise<string> {
  if (!isEmbedded(root, image.name)) return image.name
  let name = namesIn.get(image)
  if (name === undefined) {
    const doc = await readDoc(imagePath(root, image.name)).catch(() => null)
    // 24 characters of base64 are its first 18 bytes
    const head = Buffer.from((doc?.imageData ?? '').slice(0, 24), 'base64').toString('latin1')
    name = `${image.name.replace(/\.[^./]*$/, '')}${KINDS.find(([, is]) => is(head))?.[0] ?? '.png'}`
    namesIn.set(image, name)
  }
  return name
}

// The check

/** An image as the export takes it */
type Item = {
  image: DatasetImage
  /** The image as read, before shapes were left out of it, which what checks made of it is kept by */
  read: DatasetImage
  /** The number in the label file of each of image's shapes, when shapes were left out of them */
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

// The writers number shapes in the list they were given. Where shapes were left out of it, issues
// name shapes by their number in the label file again, in `shape` and in messages ("shape 3"),
// but never inside a quoted label.
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

// The folder as the last check read it, under the id the window gave: a check after a change of
// options uses it again, rather than reading every label file again. Dropped when the window closes.
let lastRead: { id: string; root: string; prefer: LabelFormat; images: DatasetImage[] } | null = null

async function readFolder(root: string, prefer: LabelFormat, ctx: Context): Promise<DatasetImage[]> {
  if (ctx.read !== undefined && lastRead?.id === ctx.read && lastRead.root === root && lastRead.prefer === prefer) return lastRead.images
  const images = await readDataset(root, { prefer, signal: ctx.signal, onProgress: (done, total) => ctx.onProgress?.({ phase: 'read', done, total }) })
  if (ctx.read !== undefined) lastRead = { id: ctx.read, root, prefer, images }
  return images
}

// labelme's own names for what's no object: its converters, labelme2coco and labelme2voc, number
// the classes after them, and make nothing of shapes labelled __ignore__, the regions to ignore.
// The export leaves both out of the classes, and their shapes out of the dataset.
const NOT_CLASSES = new Map([['__ignore__', 'a region to ignore'], ['_background_', 'the background']])
const shapeList = (k: number[]) => k.length === 1 ? `shape ${k[0] + 1}`
  : k.length > 4 ? `shapes ${k.slice(0, 3).map((n) => n + 1).join(', ')} and ${k.length - 3} others` : `shapes ${k.slice(0, -1).map((n) => n + 1).join(', ')} and ${k[k.length - 1] + 1}`

async function gather(root: string, options: ExportOptions, ctx: Context): Promise<Gathered> {
  checkOptions(options)
  const all = await readFolder(root, options.prefer, ctx)
  const issues: Issue[] = []
  const notes: string[] = []
  const add = (severity: Severity, code: string, image: string | null, message: string, shape?: number) => issues.push(issue(severity, code, image, message, shape))
  const task = TASKS[options.format]
  const coco = !task
  const leave = options.images === 'leave'
  const turn = turns(ctx.signal)

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

  // Which images go in, what of them, and under what name; `at` is each one's place in the read
  type Chosen = Omit<Item, 'split' | 'rel'> & { as: string; at: number }
  const chosen: Chosen[] = []
  // A label file that gives another size than its image's may hold shapes drawn on the image at
  // that size, which the export would place on the image as it is: the Check calls it an error,
  // and labelme 7 won't open such a file
  const take = (c: Chosen) => {
    const { image } = c, size = image.labelSize
    if (image.file && size && !image.unreadable && image.width > 0 && image.height > 0 && (size[0] !== image.width || size[1] !== image.height)) {
      add('error', 'export-size-mismatch', image.name, `${path.basename(image.file)} gives the image size as ${size[0]} x ${size[1]}, but ${image.name} is ` +
        `${image.width} x ${image.height}, so its shapes may be out of place: open it, check its shapes and save it, which writes its own size`)
    }
    chosen.push(c)
  }
  let left = 0
  for (const [at, image] of all.entries()) {
    await turn()
    const { name } = image
    const never = image.file === null && !image.broken
    if (never && !options.backgrounds) {
      add('info', 'export-never-labelled', name, `${name} has never been labelled, so it's left out`)
      left++
      continue
    }
    const embedded = isEmbedded(root, name)
    const as = embedded ? await nameIn(root, image) : name
    if (task && !isYoloImage(as)) {
      add('warning', 'export-not-yolo-image', name, `${name} isn't an image Ultralytics reads, so it's left out`)
      continue
    }
    if (never) add('info', 'export-background', name, `${name} has never been labelled, so it goes in as a background, with no objects`)
    if (leave && embedded) add('error', 'export-in-label-file', name, `${name} is held inside its label file, so it can't be left in place: copy the images instead`)
    const ai: number[] = []
    let unclassed: Map<string, number[]> | null = null
    for (let i = 0; i < image.shapes.length; i++) {
      const s = image.shapes[i]
      if (isAi(s)) ai.push(i)
      else if (NOT_CLASSES.has(s.label)) {
        unclassed ??= new Map()
        unclassed.set(s.label, [...(unclassed.get(s.label) ?? []), i])
      }
    }
    const proposals = ai.length ? plural(ai.length, 'AI proposal') : ''
    if (ai.length && !options.leaveAi) {
      add('error', 'export-ai-pending', name, `${name} has ${proposals} not yet accepted: accept or reject ${ai.length === 1 ? 'it' : 'them'}, or leave unaccepted proposals out`, ai[0])
    }
    // What's left out of it: shapes labelled __ignore__ or _background_, and AI proposals when chosen
    if (!unclassed && !(options.leaveAi && ai.length)) {
      take({ image, read: image, map: null, as, at })
      continue
    }
    const dropped = new Set([...(unclassed?.values() ?? [])].flat())
    if (options.leaveAi) for (const i of ai) dropped.add(i)
    const map = image.shapes.flatMap((s, i) => (dropped.has(i) ? [] : [i]))
    if (options.leaveAi && ai.length) {
      // Shapes labelled __ignore__ or _background_ are someone's labels too, so an image holding
      // them goes in, with no objects if they're all that's left of it
      if (ai.length === image.shapes.length) {
        add('info', 'export-ai-only', name, `${name} holds only ${proposals} not yet accepted, so it's left out, as not yet labelled`, ai[0])
        continue
      }
      add('info', 'export-ai-left-out', name, `${name}: its ${proposals} not yet accepted ${ai.length === 1 ? 'is' : 'are'} left out`, ai[0])
    }
    for (const [label, shapes] of unclassed ?? []) {
      add('info', 'export-not-a-class', name, `${name}: ${shapeList(shapes)}, labelled “${label}”, ${shapes.length === 1 ? 'is' : 'are'} left out, ` +
        `as labelme's converters take ${label} for ${NOT_CLASSES.get(label)}, not an object`, shapes[0])
    }
    take({ image: { ...image, shapes: map.map((i) => image.shapes[i]), masks: map.map((i) => image.masks[i]) }, read: image, map, as, at })
  }

  // Posed objects, and the order of the categories or classes: the folder's, so that changing the
  // split never numbers them differently
  if (options.format === 'coco-keypoints' && schema) {
    for (const c of chosen) {
      await turn()
      const found = poseInstances(c.image, schema)
      c.poses = found.instances
      for (const f of found.issues) issues.push(renumbered(f, c.map))
    }
  }
  let classes: string[]
  if (task === 'pose') {
    const keys = schema ? poseYamlKeys(schema) : null
    classes = keys?.names ?? []
    if (keys) issues.push(...keys.issues)
  } else {
    const named = [...new Set(options.labels.filter((l) => NOT_CLASSES.has(l)))]
    if (named.length) {
      notes.push(`The label list's ${named.join(' and ')} ${named.length === 1 ? "isn't a class" : "aren't classes"}: labelme's converters number the classes after ${named.length === 1 ? 'it' : 'them'}.`)
    }
    // A chunk of images at a time, each adding the labels it finds past those before it, as one
    // call over them all would
    classes = cocoLabels(options.labels.filter((l) => !NOT_CLASSES.has(l)), [])
    for (let k = 0; k < chosen.length; k += 1000) {
      await turn()
      classes = cocoLabels(classes, chosen.slice(k, k + 1000).map((c): CocoEntry => ({ image: c.image, fileName: '', poses: c.poses })))
    }
  }

  // Splits: each image's, and its path in its split's folder of images
  const s = options.split
  let splits: string[]
  let place: (c: Chosen) => [string, string]
  if (s.mode === 'none') {
    // CVAT 2.83 calls the one subset of a dataset that isn't split default
    const folder = coco ? 'default' : ''
    splits = [folder]
    place = (c) => [folder, c.as]
    if (!coco) notes.push("With no split, images/ and labels/ hold every image, and data.yaml's train and val both point at images, as Ultralytics needs a val entry.")
  } else if (s.mode === 'ratio') {
    // Every image in the folder by path, shuffled with the seed: those going in take the shares in that order
    const counts = shareOut(chosen.length, [s.train, s.val, s.test])
    const going = new Uint8Array(all.length)
    for (const c of chosen) going[c.at] = 1
    const splitOf = new Array<string>(all.length).fill('')
    let n = 0
    for (const k of shuffle(byPath(all), s.seed)) {
      if (!going[k]) continue
      splitOf[k] = n < counts[0] ? 'train' : n < counts[0] + counts[1] ? 'val' : 'test'
      n++
    }
    splits = SPLITS.filter((_, i) => [s.train, s.val, s.test][i] > 0)
    place = (c) => [splitOf[c.at], c.as]
  } else {
    splits = [...SPLITS]
    place = (c) => {
      const { split, rel } = splitFolder(c.as)
      if (!split) add('warning', 'export-no-split-folder', c.image.name, `${c.image.name} isn't under a train, val or test folder, so it goes to train`)
      return [split ?? 'train', rel]
    }
  }
  // The images split by split, in the folder's order within each
  const bySplit = new Map(splits.map((split): [string, Item[]] => [split, []]))
  for (const c of chosen) {
    await turn()
    const [split, rel] = place(c)
    let list = bySplit.get(split)
    if (!list) bySplit.set(split, (list = []))
    list.push({ image: c.image, read: c.read, map: c.map, poses: c.poses, split, rel })
  }
  const has = (split: string) => !!bySplit.get(split)?.length
  if (s.mode === 'folders') splits = splits.filter((split) => split === 'train' || has(split))
  const items = [...bySplit.values()].flat()
  if (task && s.mode !== 'none' && items.length) {
    const train = has('train')
    if (!train) add('warning', 'export-no-train', null, `No images go to train, so data.yaml's train points at the ${items[0].split} images`)
    if (!has('val')) add('warning', 'export-no-val', null, `No images go to val, so data.yaml's val points at the ${train ? 'train' : items[0].split} images, as Ultralytics needs a val entry`)
  }
  // A split with a share, or train when the folders decide, that no image goes to
  if (s.mode !== 'none' && items.length) {
    for (const split of splits) {
      if (has(split)) continue
      if (coco) {
        const files = options.format === 'coco-keypoints' ? `instances_${split}.json and person_keypoints_${split}.json hold` : `instances_${split}.json holds`
        add('warning', 'export-empty-split', null, `No images go to ${split}, so ${files} none`)
      } else if (split === 'test') add('warning', 'export-empty-split', null, 'No images go to test, so data.yaml names no test folder')
    }
  }

  // Images, and YOLO's label files, that would land on one path
  const clash = async (code: string, paths: (i: Item) => string | null, say: (names: string[], at: string) => string) => {
    const by = new Map<string, { at: string; names: string[] }>()
    for (const i of items) {
      await turn()
      const at = paths(i)
      if (at === null) continue
      const known = by.get(nameKey(at))
      if (known) known.names.push(i.image.name)
      else by.set(nameKey(at), { at, names: [i.image.name] })
    }
    for (const { at, names } of by.values()) if (names.length > 1) add('error', code, names[0], say(names, at))
  }
  const imageAt = (i: Item) => posixJoin('images', i.split, i.rel)
  if (!leave) await clash('export-same-path', imageAt, (names, at) => `${listed(names)} would ${names.length === 2 ? 'both' : 'all'} be written to ${at}: rename one of them`)
  if (task) await clash('export-same-label', (i) => labelPathFor(imageAt(i)), (names, at) => `${listed(names)} would share one label file, ${at}, as Ultralytics finds label files by name: rename one of them`)
  const shared = (readOnce(all).clashes ??= labelFileClashes(all))
  if (shared.length) {
    const going = new Set(items.map((i) => i.image.name))
    for (const c of shared) {
      if (!c.images.some((n) => going.has(n))) continue
      add('error', 'export-shared-label-file', c.images[0], `${listed(c.images)} share one label file, ${path.basename(c.file)}, so whose labels are whose can't be told: give each image its own name`)
    }
  }

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

/** What the format's writers make of one image: its issues, and what it adds to its split */
type Tally = {
  issues: Issue[]
  /** The label of each object written */
  objects: string[]
  /** No object written */
  background: boolean
  posed: number
}

// COCO's own names, which CVAT 2.83's COCO export also writes: annotations/instances_<split>.json beside images/<split>/
const COCO_FILES: Record<CocoTask, string> = { instances: 'instances', keypoints: 'person_keypoints' }
const cocoTasks = (g: Gathered): CocoTask[] => (g.options.format === 'coco-keypoints' && g.schema ? ['instances', 'keypoints'] : ['instances'])
// A check's COCO files are never written: what they say of the export is nobody's
const NO_INFO: CocoInfo = { description: '', version: '', year: 0, date_created: '', contributor: '', url: '' }

/** The issues of no one image: COCO's skeleton lines to keypoints a class doesn't have */
const globalIssues = (g: Gathered): Issue[] => TASKS[g.options.format] ? []
  : cocoTasks(g).flatMap((task) => writeCoco([], { task, labels: g.classes, info: NO_INFO, schema: g.schema ?? undefined }).issues)

/** An image's YOLO label file, and its tally */
function yoloRows(g: Gathered, i: Item): { text: string; tally: Tally } {
  const task = TASKS[g.options.format]!
  // With no keypoints file there are no pose rows to make: the check says so, and nothing is exported
  if (task === 'pose' && !g.schema) return { text: '', tally: { issues: [], objects: [], background: false, posed: 0 } }
  const rows = labelRows(i.image, task, g.classes, g.schema ?? undefined)
  const objects: string[] = []
  for (const line of rows.text.split('\n')) if (line) objects.push(g.classes[Number(line.slice(0, line.indexOf(' ')))])
  return { text: rows.text, tally: { issues: unique(rows.issues.map((f) => renumbered(f, i.map))), objects, background: !rows.rows, posed: task === 'pose' ? rows.rows : 0 } }
}

/**
 * COCO's files for a chunk of images, each image going in under the file name given, and each
 * image's tally. Issues of no one image go in `global`.
 */
function cocoChunk(g: Gathered, part: Item[], names: string[], info: CocoInfo, global: Map<string, Issue>) {
  const entries = part.map((i, k): CocoEntry => ({ image: i.image, fileName: names[k], poses: i.poses }))
  const at = new Map(part.map((i, k) => [i.image.name, k]))
  const tallies: Tally[] = part.map(() => ({ issues: [], objects: [], background: true, posed: 0 }))
  const datasets: [CocoTask, CocoDataset][] = []
  for (const task of cocoTasks(g)) {
    const { dataset, issues } = writeCoco(entries, { task, labels: g.classes, info, schema: g.schema ?? undefined })
    for (const f of issues) {
      const k = f.image === null ? undefined : at.get(f.image)
      if (k === undefined) global.set(keyOf(f), f)
      else tallies[k].issues.push(renumbered(f, part[k].map))
    }
    // writeCoco lists the images it takes in the order given, leaving out those it can't read
    const of = new Map<number, number>()
    let k = 0
    for (const im of dataset.images) {
      while (k < names.length && names[k] !== im.file_name) k++
      of.set(im.id, k++)
    }
    for (const a of dataset.annotations) {
      const t = tallies[of.get(a.image_id) ?? -1]
      if (!t) continue
      t.background = false
      if (task === 'instances') t.objects.push(g.classes[a.category_id - 1] ?? String(a.category_id))
      else t.posed++
    }
    datasets.push([task, dataset])
  }
  for (const t of tallies) t.issues = unique(t.issues)
  return { datasets, tallies }
}

// EXIF orientations, by image as read, read once an image, of every format readDataset turns
// upright by its orientation: all but BMP, whose size it reads itself, and GIF, which holds none
const orientations = new WeakMap<DatasetImage, number>()
const HOW: Record<number, string> = { 2: 'mirrored', 3: 'turned', 4: 'mirrored', 5: 'turned and mirrored', 6: 'turned', 7: 'turned and mirrored', 8: 'turned' }
// labelme, OpenCV and Ultralytics all turn JPEG files upright, which is where cameras and phones
// write the orientation; on other formats they differ
const shownBy = (name: string) => (/\.(jpe?g|jfif)$/i.test(name) ? 'labelme, OpenCV and Ultralytics show it' : 'Seamtag shows it')

/**
 * COCO gives each image's width, height and shapes upright, as Seamtag shows it, while its file
 * goes into the dataset as it is: each image an EXIF orientation turns gets a note in its tally,
 * as a reader that ignores the orientation sees it otherwise
 */
async function orientationNotes(root: string, part: Item[], tallies: Tally[]) {
  // Eight headers read at a time, as readDataset reads them, so main never sets up many at once
  let next = 0
  await Promise.all(Array.from({ length: 8 }, async () => {
    while (next < part.length) {
      const k = next++
      const { image, read } = part[k]
      let o = orientations.get(read)
      if (o === undefined) {
        const file = imagePath(root, image.name)
        o = isEmbedded(root, image.name) || /\.(bmp|gif)$/i.test(file) ? 1
          : (await sharp(file, { limitInputPixels: false }).metadata().catch(() => null))?.orientation ?? 1
        orientations.set(read, o)
      }
      const how = HOW[o]
      if (how) {
        tallies[k].issues.push(issue('info', 'export-exif-orientation', image.name, `${image.name} is stored ${how}, with EXIF orientation ${o}: its width, height and ` +
          `shapes are for it upright, as ${shownBy(image.name)}, so a reader that ignores the orientation, such as torchvision's CocoDetection, sees it ${how}`))
      }
    }
  }))
}

/** The images and objects in each split and of each label, and the issues, from each item's tally */
async function summed(g: Gathered, tallies: Tally[], global: Issue[], turn: Turn) {
  const splits = new Map(g.splits.map((folder): [string, SplitCount] => [folder, { name: folder || 'all', folder, images: 0, objects: 0, backgrounds: 0, posed: 0 }]))
  const labels = new Map(g.classes.map((label) => [label, g.splits.map(() => 0)]))
  const issues: Issue[] = []
  for (const [k, item] of g.items.entries()) {
    await turn()
    const t = tallies[k], at = g.splits.indexOf(item.split), split = splits.get(item.split)!
    split.images++
    split.objects += t.objects.length
    split.posed += t.posed
    if (t.background) split.backgrounds++
    for (const label of t.objects) {
      let row = labels.get(label)
      if (!row) labels.set(label, (row = g.splits.map(() => 0)))
      row[at]++
    }
    for (const f of t.issues) issues.push(f)
  }
  for (const f of unique(global)) issues.push(f)
  return { splits: [...splits.values()], labels: [...labels].map(([label, objects]): LabelCount => ({ label, objects })), issues }
}

/** What checks made of each image under one setting of what the writers depend on, by image as read, and the issues of no one image */
type Memo = { images: WeakMap<DatasetImage, Tally>; global: Map<string, Issue> }
// The last few settings checks ran the writers under (format, AI proposals left out, classes and
// keypoints): a check after a change of split, folder, images or backgrounds only sums the tallies
// again, and only images read again since they changed go through the writers again. Dropped when
// the window closes.
const memos = new Map<string, Memo>()
const MEMOS = 4

function memoOf(g: Gathered): Memo {
  const key = JSON.stringify([g.options.format, g.options.leaveAi, g.classes, g.schema])
  const memo = memos.get(key) ?? { images: new WeakMap(), global: new Map() }
  memos.delete(key)
  memos.set(key, memo)
  for (const old of memos.keys()) if (memos.size > MEMOS) memos.delete(old)
  return memo
}

// Images a COCO file takes at a time as an export writes it; a check runs the writers over fewer
// at a time, so main answers the window more often between them
const CHUNK = 256
const CHECKED = 64

/** Each item's tally, from what an earlier check made of it where it can, and the issues of no one image */
async function tallied(root: string, g: Gathered, ctx: Context): Promise<{ tallies: Tally[]; global: Issue[] }> {
  const memo = memoOf(g)
  const turn = turns(ctx.signal)
  const todo: Item[] = []
  for (const i of g.items) {
    await turn()
    if (!memo.images.has(i.read)) todo.push(i)
  }
  let told = 0
  for (let start = 0; start < todo.length; start += CHECKED) {
    ctx.signal?.throwIfAborted()
    const part = todo.slice(start, start + CHECKED)
    let made: Tally[]
    if (TASKS[g.options.format]) made = part.map((i) => yoloRows(g, i).tally)
    else {
      made = cocoChunk(g, part, part.map((i) => i.image.name), NO_INFO, memo.global).tallies
      await orientationNotes(root, part, made)
    }
    part.forEach((i, k) => memo.images.set(i.read, made[k]))
    const done = start + part.length
    if (Date.now() - told >= 100 || done === todo.length) {
      told = Date.now()
      ctx.onProgress?.({ phase: 'check', done, total: todo.length })
    }
    await tick()
  }
  const tallies: Tally[] = []
  for (const i of g.items) {
    await turn()
    tallies.push(memo.images.get(i.read)!)
  }
  return { tallies, global: [...globalIssues(g), ...memo.global.values()] }
}

/** What an export writes as it goes: an image, a text file, a chunk of a COCO file */
type Sink = {
  image(item: Item, at: string): Promise<void>
  text(at: string, text: string): Promise<void>
  coco(split: string, task: CocoTask, dataset: CocoDataset): Promise<void>
}

/**
 * The format's writers over every image, a split at a time and a chunk of images at a time,
 * writing what they make. Labels that changed since the check, so that a writer finds an
 * error, stop it before that chunk is written.
 */
async function convert(root: string, g: Gathered, ctx: Context, sink: Sink) {
  const { options, items } = g
  const task = TASKS[options.format]
  const tallies: Tally[] = []
  const global = new Map(globalIssues(g).map((f) => [keyOf(f), f]))
  const info: CocoInfo = {
    description: `Exported by Seamtag ${ctx.version ?? ''}`.trim(), version: ctx.version ?? '', year: new Date().getFullYear(),
    date_created: new Date().toISOString(), contributor: '', url: ''
  }
  const imageAt = (i: Item) => (options.images === 'leave' ? i.image.name : posixJoin('images', i.split, i.rel))
  const stopAt = (made: Tally[]) => {
    const failed = made.flatMap((t) => t.issues).find((f) => f.severity === 'error')
    if (failed) throw new Error(`The labels changed since they were checked, so the export stopped: ${failed.message}`)
  }
  // Images written, told as each one is, and once before the first, so the window says what's
  // under way while a big image or a slow drive takes its time
  let done = 0, told = 0
  const tell = () => {
    if (Date.now() - told < 100 && done < items.length) return
    told = Date.now()
    ctx.onProgress?.({ phase: 'write', done, total: items.length })
  }
  tell()
  for (let start = 0; start < items.length;) {
    ctx.signal?.throwIfAborted()
    // A chunk is in one split: the items are in split order, and a chunk ends where its split does
    const chunk = items.slice(start, start + CHUNK)
    const end = chunk.findIndex((i) => i.split !== chunk[0].split)
    const part = end < 0 ? chunk : chunk.slice(0, end)
    start += part.length
    if (task) {
      const made = part.map((i) => yoloRows(g, i))
      stopAt(made.map((m) => m.tally))
      await pool(part.map((i, k) => [i, made[k].text] as const), async ([i, text]) => {
        const at = imageAt(i)
        await sink.image(i, at)
        await sink.text(labelPathFor(at), text)
        done++
        tell()
      }, ctx.signal)
      for (const m of made) tallies.push(m.tally)
    } else {
      const made = cocoChunk(g, part, part.map(imageAt), info, global)
      await orientationNotes(root, part, made.tallies)
      stopAt(made.tallies)
      for (const [t, dataset] of made.datasets) await sink.coco(part[0].split, t, dataset)
      if (options.images !== 'leave') {
        await pool(part, async (i) => {
          await sink.image(i, imageAt(i))
          done++
          tell()
        }, ctx.signal)
      } else {
        done += part.length
        tell()
      }
      for (const t of made.tallies) tallies.push(t)
    }
    await tick()
  }
  // A split no image goes to still gets its files, holding none, so whatever names them finds them
  if (!task) {
    for (const split of g.splits) {
      if (items.some((i) => i.split === split)) continue
      for (const t of cocoTasks(g)) await sink.coco(split, t, writeCoco([], { task: t, labels: g.classes, info, schema: g.schema ?? undefined }).dataset)
    }
  }
  return { tallies, global: [...global.values()] }
}

/**
 * Runs fn over the list, four at a time: enough to keep a disk busy. Once one fails or the
 * export stops, no more start, and it settles only once those under way have, so nothing is
 * still writing when what was written is removed. Throws the first failure.
 */
async function pool<T>(list: readonly T[], fn: (t: T) => Promise<void>, signal?: AbortSignal) {
  let next = 0
  const failures: unknown[] = []
  await Promise.all(Array.from({ length: Math.min(4, list.length) }, async () => {
    while (next < list.length && !failures.length && !signal?.aborted) {
      try {
        await fn(list[next++])
      } catch (e) {
        failures.push(e)
      }
    }
  }))
  signal?.throwIfAborted()
  if (failures.length) throw failures[0]
}

async function grouped(issues: Issue[], turn: Turn) {
  const groups = new Map<string, Issue[]>()
  for (const i of issues) {
    await turn()
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
  const turn = turns(ctx.signal)
  const g = await gather(root, options, ctx)
  const { tallies, global } = await tallied(root, g, ctx)
  const sums = await summed(g, tallies, global, turn)
  const all = [...g.issues, ...sums.issues]
  const groups = await grouped(all, turn)
  // A check that a newer one stopped never takes its place, so the window's Show more pages
  // through the issues of the check it shows
  ctx.signal?.throwIfAborted()
  const id = ++checks
  lastCheck = { id, groups }
  const rank = (s: Severity) => SEVERITIES.indexOf(s)
  const list: IssueGroup[] = [...groups].map(([key, issues]) => ({
    key, code: issues[0].code, severity: issues[0].severity, title: titleOf(issues[0].code, issues[0].severity), count: issues.length, issues: issues.slice(0, SHOWN)
  })).sort((a, b) => rank(a.severity) - rank(b.severity) || b.count - a.count || byCode(a.title, b.title))
  const counts = { error: 0, warning: 0, info: 0 }
  for (const i of all) counts[i.severity]++
  return { id, splits: sums.splits, labels: sums.labels, groups: list, counts, blocked: counts.error > 0, keypoints: g.keypoints, notes: g.notes }
}

/** More of a check's issues of one kind, from the one given on; none once a newer check has run */
export function moreIssues(id: number, key: string, from: number, count = 200): Issue[] {
  if (lastCheck?.id !== id) return []
  return (lastCheck.groups.get(key) ?? []).slice(from, from + count)
}

/** Drops what checks keep while the window is open: the folder's read, what the writers made of each image, and the last check's issues */
export function forgetExportChecks() {
  lastRead = null
  perRead = null
  memos.clear()
  lastCheck = null
}

/** One COCO file written a chunk of images at a time, as writeCoco writes them all at once; its annotations wait in a file of their own until the last image is in */
class CocoFile {
  private started = false
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
    if (!this.started) {
      await this.head.write(`{"info":${JSON.stringify(d.info)},"licenses":${JSON.stringify(d.licenses)},"images":[`)
      this.categories = d.categories
      this.started = true
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
    // Windows can hold a file just closed for a moment, while a scanner reads it: removing it is
    // tried again (maxRetries needs recursive, which takes a file too)
    await fs.rm(this.side, { recursive: true, maxRetries: 5 })
  }
  async abandon() {
    await Promise.allSettled([this.head.close(), this.rest.close()])
  }
}

/**
 * A new hidden folder in dir to build the dataset in, made as any folder is, so a new dataset
 * folder has the permissions folders there usually have, not mkdtemp's owner-only ones
 */
async function workFolder(dir: string, prefix: string): Promise<string> {
  for (;;) {
    const work = path.join(dir, `${prefix}${randomBytes(4).toString('hex')}`)
    try {
      await fs.mkdir(work)
      return work
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e
    }
  }
}

// Windows refuses to move a folder while a scanner or a sync client holds a file in it, which for
// a folder of thousands of images just written can take a while. The move comes once, at the end
// of a long export, so it's tried for up to a minute in all, as graceful-fs tries npm's renames.
const MOVE_PATIENCE = 60_000

/**
 * Moves a finished dataset into place: its hidden folder takes a new folder's name, or what it
 * holds moves into the empty folder chosen, one by one by name, never over anything put there
 * meanwhile. Should a move still fail, the dataset is kept where it is and the error says where.
 * Returns the emptied hidden folder if it couldn't be removed.
 */
async function intoPlace(work: string, out: string, existing: boolean, names: string[]): Promise<string | null> {
  const why = (e: unknown) => (e as NodeJS.ErrnoException).code ?? reason(e)
  const until = Date.now() + MOVE_PATIENCE
  const move = (from: string, to: string) => renameRetrying(from, to, Math.max(0, until - Date.now()))
  if (!existing) {
    try {
      await move(work, out)
    } catch (e) {
      throw new Error(`The dataset is written, but it couldn't be moved to ${out} (${why(e)}), so it's in ${work}: rename that folder ${path.basename(out)} yourself`, { cause: e })
    }
    return null
  }
  const left = [...names].sort(byCode)
  while (left.length) {
    try {
      if (await fs.lstat(path.join(out, left[0])).then(() => true, () => false)) throw new Error(`${left[0]} is there already`)
      await move(path.join(work, left[0]), path.join(out, left[0]))
      left.shift()
    } catch (e) {
      const one = left.length === 1
      throw new Error(`The dataset is written, but ${listed(left)} couldn't be moved into ${out} (${why(e)}), so ${one ? "it's" : "they're"} still in ${work}: ` +
        `move ${one ? 'it' : 'them'} there yourself`, { cause: e })
    }
  }
  return fs.rm(work, { recursive: true, force: true, maxRetries: 5 }).then(() => null, () => work)
}

/**
 * Writes the dataset into `out`, a new or empty folder outside the image folder. It's built in a
 * hidden folder beside `out`, or inside it when it's an empty folder, which so keeps itself and
 * what systems keep in it, then moved into place. Reads the folder afresh, and refuses while the
 * check finds errors. Stopping leaves nothing behind and resolves with `stopped`.
 */
export async function runExport(root: string, options: ExportOptions, out: string, ctx: Context = {}): Promise<ExportResult> {
  const problem = await folderProblem(root, out)
  if (problem) throw new Error(problem)
  const stopped: ExportResult = { stopped: true, out, splits: [], images: { copied: 0, linked: 0, fellBack: 0, left: 0 }, files: 0, written: [], accepted: [], notes: [] }
  let g: Gathered
  try {
    g = await gather(root, options, { ...ctx, read: undefined })
  } catch (e) {
    if (ctx.signal?.aborted) return stopped
    throw e
  }
  const blocking = g.issues.find((i) => i.severity === 'error')
  if (blocking) throw new Error(`The dataset can't be exported: ${blocking.message}`)
  const existing = !!(await fs.lstat(out).catch(() => null))
  const work = await workFolder(existing ? out : path.dirname(out), existing ? WORK_PREFIX : `.${path.basename(out)}.export-`)
  const counts = { copied: 0, linked: 0, fellBack: 0, left: options.images === 'leave' ? g.items.length : 0 }
  let files = 0
  // What the export put at the top of the dataset, which moves into an empty folder at the end
  const top = new Set<string>()
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
      top.add(at.split('/')[0])
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
      top.add(at.split('/')[0])
      const to = path.join(work, ...at.split('/'))
      await dir(to)
      await fs.writeFile(to, text, { flag: 'wx' })
      files++
    },
    async coco(split, task, dataset) {
      const at = posixJoin('annotations', `${COCO_FILES[task]}_${split}.json`)
      top.add('annotations')
      let file = cocoFiles.get(at)
      if (!file) cocoFiles.set(at, (file = await CocoFile.open(path.join(work, ...at.split('/')))))
      await file.add(dataset)
    }
  }
  let converted: { tallies: Tally[]; global: Issue[] }
  const written: string[] = []
  try {
    converted = await convert(root, g, ctx, sink)
    ctx.onProgress?.({ phase: 'finish', done: 0, total: 1 })
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
      top.add('data.yaml')
      await fs.writeFile(path.join(work, 'data.yaml'), yamlText, { flag: 'wx' })
      written.push('data.yaml')
      files++
    }
    ctx.signal?.throwIfAborted()
  } catch (e) {
    // Nothing the export started is still writing by now (pool waits for those under way), so what
    // it removes stays removed
    await Promise.allSettled([...cocoFiles.values()].map((f) => f.abandon()))
    const removed = await fs.rm(work, { recursive: true, force: true, maxRetries: 3 }).then(() => true, () => false)
    // The export's own failure is what's told, with what of it couldn't be removed
    const rest = removed ? '' : ` What it wrote couldn't all be removed: delete ${work} yourself.`
    if (ctx.signal?.aborted) {
      if (!rest) return stopped
      throw new Error(`The export stopped.${rest}`)
    }
    if (!rest) throw e
    throw new Error(`${reason(e).replace(/\.$/, '')}.${rest}`, { cause: e })
  }
  // The dataset is whole: from here on it's kept, whatever happens, and Stop no longer stops it
  const leftover = await intoPlace(work, out, existing, [...top])
  ctx.onProgress?.({ phase: 'finish', done: 1, total: 1 })
  const turn = turns()
  const sums = await summed(g, converted.tallies, converted.global, turn)
  const accepted = [...(await grouped([...g.issues, ...sums.issues], turn))].map(([, list]) => ({ severity: list[0].severity, title: titleOf(list[0].code, list[0].severity), count: list.length }))
    .sort((a, b) => SEVERITIES.indexOf(a.severity) - SEVERITIES.indexOf(b.severity) || b.count - a.count)
  const notes = leftover ? [...g.notes, `The empty folder ${leftover} couldn't be removed: delete it yourself.`] : g.notes
  return { stopped: false, out, splits: sums.splits, images: counts, files, written: written.sort(byCode), accepted, notes }
}

/**
 * An image copied, or hard-linked and copied where a link can't be made, as across drives or on
 * file systems without links. A link is to the file itself: an image that's a symbolic link (as
 * in git-annex and DVC folders) is followed first, so the dataset never holds the link.
 */
async function putImage(from: string, to: string, link: boolean): Promise<'copied' | 'linked' | 'fellBack'> {
  // Only a link is followed: on Windows, drives that can't tell a file's final path, as some RAM
  // disks, fail realpath for every file
  const file = (await fs.lstat(from)).isSymbolicLink() ? await fs.realpath(from) : from
  if (link) {
    try {
      await fs.link(file, to)
      return 'linked'
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT' || (e as NodeJS.ErrnoException).code === 'EEXIST') throw e
    }
  }
  // A copy shares the file's blocks until either changes where the file system can (APFS, Btrfs, XFS)
  await fs.copyFile(file, to, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE)
  return link ? 'fellBack' : 'copied'
}
