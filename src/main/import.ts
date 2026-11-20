/// <reference types="electron-vite/node" />
// File > Import dataset: a COCO file or an Ultralytics YOLO dataset written into the open folder's
// label files. Reading writes nothing: it matches the dataset's images to the folder's, measures
// them, looks at the labels they have and counts what each would get. Writing goes through
// saveLabels an image at a time, as LabelMe JSON, which holds every shape type, each image in one
// turn of the label files' queue. COCO and YOLO are converted in, never edited in place: each
// image's own label file stays the working copy.
import { constants, existsSync, promises as fs } from 'fs'
import path from 'path'
import type { Worker } from 'worker_threads'
import { MAX_GROUP } from '../shared/coco'
import { type Issue, type MaskBits, type Severity, sizeText } from '../shared/dataset'
import { LABEL_FORMATS, type LabelFormat } from '../shared/formats'
import type { Shape } from '../shared/labelfile'
import { readPoseYaml, short } from '../shared/pose'
import { countRows, labelText, newRowStats, readRows, YOLO_TASKS, type YoloTask } from '../shared/ultralytics'
import { imageSize } from './images'
import { type CocoAnswer, type CocoAsk, type CocoFound, type CocoProgress, CocoSession, type CocoShapes, tooBig } from './importWorker'
import { encodeMask } from './masks'
import { nameKey } from './names'
import {
  type Found, findLabels, folderOf, inTurn, labelPath, readLabels, renameRetrying, type Saved, saveLabels, type SaveRequest, sendToTrash, serial, summarizeAll
} from './project'
import { readYoloDataset, type YoloDataset } from './ultralytics'
import { folderKeypoints } from './keypoints'
import { keypointNames } from '../shared/skeleton'

/** The names of the folder's keypoints, which the image and label lists don't count as classes */
const folderKeypointNames = async (root: string) => keypointNames((await folderKeypoints(root)).schema)

/** What to import: a COCO file, or a YOLO dataset's data.yaml or folder */
export type ImportRequest = {
  kind: 'coco' | 'yolo'
  path: string
  /** The window's save format: an image with label files in several formats is read in this one */
  prefer: LabelFormat
  /** The label list, so labels new to it can be told */
  labels: string[]
  /** YOLO: the task its rows are read for; the guess when left out */
  task?: YoloTask
  /** YOLO pose: X-AnyLabeling's keypoint file, naming each class's keypoints when data.yaml doesn't */
  keypoints?: string | null
}

/** What happens to images that have labels already: left alone, the imported shapes added to theirs, or replaced */
export type ImportPolicy = 'skip' | 'add' | 'replace'

/** For the window's bar: what's being done, and how far it's got (0 of 0 when it can't tell) */
export type ImportProgress = { phase: string; done: number; total: number }

/** Issues of one kind: how many, and the first few; the rest come a page at a time from importRows */
export type IssueGroup = { code: string; severity: Severity; title: string; count: number; rows: string[] }

/**
 * A label's shapes in images without labels, which every choice writes, and in labelled ones,
 * which Replace writes too; of those, `added` are the ones Add writes, the rest being there already
 */
export type LabelCount = { label: string; fresh: number; labelled: number; added: number; listed: boolean }

/** What reading a dataset found, before anything is written */
export type ImportPreview = {
  kind: 'coco' | 'yolo'
  /** The file or folder read, and its name */
  source: string
  name: string
  /** Images in the dataset, and those found in the folder: by path, or by file name alone */
  images: number
  matched: number
  byName: number
  /** COCO: the folder its file names start from, as a path in the open folder ('' for the folder itself) */
  base: string | null
  missing: number
  ambiguous: number
  /** Matched images by what they have: no label file (every choice writes them), labels, labels in LabelImg's formats that Add and Replace convert */
  fresh: number
  labelled: number
  labelImg: number
  /** Images without labels that get an empty label file, their YOLO label file being empty */
  blank: number
  /** Labelled images the dataset gives no shapes (an empty YOLO label file): every choice leaves them alone */
  empty: number
  /** Labelled images that have every shape the dataset gives them already, which Add leaves as they are, and those of them in LabelImg's formats */
  unchanged: number
  unchangedLabelImg: number
  /** Label files that can't be read: never touched */
  unreadable: number
  /** Images that can't take labels from it: sizes differ, label file shared, image unreadable */
  blocked: number
  /** Images it has nothing for: no label file in a YOLO dataset, no annotations in a COCO file, or nothing that could be read */
  nothing: number
  labels: LabelCount[]
  groups: IssueGroup[]
  /** YOLO: the task the rows are read for, the guess and why, and whether the window should ask */
  task?: { task: YoloTask; guess: YoloTask; others: YoloTask[]; ask: boolean; reason: string }
  /** YOLO pose: where the keypoints' names come from */
  keypoints?: { from: 'yaml' | 'file' | 'generated'; file: string | null; count: number }
}

/** What writing did */
export type ImportReport = {
  stopped: boolean
  /** Images there were to write, those gone through (all but when stopped), and the label files written */
  total: number
  done: number
  written: number
  shapes: number
  labels: { label: string; shapes: number; listed: boolean }[]
  /** Add: shapes the images had already, so they weren't added twice */
  repeated: number
  /** Old label files that went to the trash */
  trashed: number
  /** Images left as they were, by why */
  skipped: IssueGroup[]
  /** The images whose label files changed, for the window to read again: those written, and any whose old file went to the trash before writing failed */
  names: string[]
}

type Summaries = Parameters<typeof summarizeAll>[1]

// An image of the folder that the dataset has labels for
type Entry = {
  name: string
  /** Its upright size; null when it can't be read */
  size: [number, number] | null
  /** COCO: the COCO images matched to it whose size is its own */
  ids: number[]
  /** YOLO: its label file in the dataset; null when it has none */
  label: string | null
  /** Matched by its file name alone, its path in the dataset not being its path in the folder */
  byName: boolean
  /** What the dataset has for it: objects (annotations or rows), the shapes they make, by label */
  annotations: number
  shapes: number
  labels: Record<string, number>
  /** COCO: how many pixels its mask shapes have, which writing makes, a byte each */
  pixels: number
  /** Why it can't take labels from this dataset */
  blocked: Issue | null
  /** Its label file now, in whichever format; null when it has none */
  has: { format: LabelFormat; file: string } | null
  /** Why its label file can't be read: such a file is never touched */
  broken: string | null
  /** The labels of the shapes it has, as looking at its label file found them */
  own: Set<string> | null
  /**
   * Add: the labels of the shapes it would get, those it has already being left out; null when
   * they weren't compared with its own, which share no label with them or couldn't be read again
   */
  adds: Record<string, number> | null
  /** Add: the dataset's shapes it has already */
  repeated: number
}

type CocoReader = { found: CocoFound; shapes: (ids: number[]) => Promise<CocoShapes[]>; close: () => void }

type Session = {
  root: string
  /** Where the folder's labels go: its -O folder or file, or beside the images */
  output: string | null
  kind: 'coco' | 'yolo'
  source: string
  prefer: LabelFormat
  listed: Set<string>
  entries: Entry[]
  images: number
  base: string | null
  missing: number
  ambiguous: number
  /** What reading the dataset and the folder found; `rows` is what the YOLO rows gave, read again when the task changes */
  issues: Issue[]
  rows: Issue[]
  coco: CocoReader | null
  /** COCO: each matched COCO image's size and counts */
  cocoImages: Map<number, CocoFound['matched'][number]>
  yolo: { dataset: YoloDataset; task: YoloTask; keypoints: string | null; names: Record<number, string[]> | null; from: 'yaml' | 'file' | 'generated' } | null
  /** After writing: the images left as they were, by why */
  skipped: Map<string, { severity: Severity; title: string; rows: string[] }>
  /** Written to: its idea of the folder's labels is out of date */
  used: boolean
}

// The worker's heap. Parsed, a COCO file takes about one and a half times its size as text, so
// with the text it peaks at about two and a half (COCO's train2017, 448 MB, about 1.1 GB); a
// chunk's shapes take a few megabytes more. Tests set less, to see a file too big for it.
const HEAP_MB = Number(process.env.SEAMTAG_IMPORT_HEAP_MB) || 2048
// A file this small is read in place when the worker can't start
const IN_PLACE_MB = 32
// Images written between asking the worker for their shapes, whose masks have at most
// BATCH_PIXELS pixels unless one image alone has more; no image's masks may have more than
// IMAGE_PIXELS. Masks' pixels take a byte each, in memory the worker's heap limit doesn't count.
const BATCH = 64
const BATCH_PIXELS = 2 ** 28
const IMAGE_PIXELS = 2 ** 30
// The most a keypoint file may hold, read whole here: a few kilobytes name a hundred keypoints
const KEYPOINT_FILE_BYTES = 4 * 2 ** 20
// Messages kept with each group in what the window is sent; the rest come a page at a time
const SHOWN = 20

// Shorter titles for the codes the readers and the import give their issues
const TITLES: Record<string, string> = {
  'coco-bad-image': "Images the file doesn't describe",
  'coco-bad-category': "Categories that can't be read",
  'coco-bad-annotation': "Annotations that can't be read",
  'coco-unknown-image': 'Annotations of images not in its list',
  'coco-unknown-category': 'Annotations of unknown categories',
  'coco-bad-polygon': 'Polygons left out',
  'coco-bad-mask': "Masks that can't be read",
  'coco-mask-size': 'Masks of another size',
  'coco-mask-too-big': 'Masks too big to hold',
  'coco-no-pixels': 'Annotations with no pixels',
  'coco-bad-attributes': "Seamtag's attributes not used",
  'coco-bad-keypoints': "Keypoints that can't be read",
  'coco-box-only': 'Boxes only',
  'coco-rotated-box': 'Rotated boxes',
  'coco-unlabelled-keypoints': 'Keypoints not labelled',
  'coco-no-keypoints': 'Posed objects without keypoints',
  'coco-crowd': 'Crowd regions',
  'coco-image-missing': 'Images not in this folder',
  'coco-image-ambiguous': 'Names that match several images',
  'coco-image-twice': 'Images listed twice',
  'yolo-missing': 'Dataset not found',
  'yolo-too-many-files': 'Too many files',
  'yolo-not-yaml': 'Not a data.yaml',
  'yolo-many-yaml': 'Several datasets',
  'yolo-yaml': 'In data.yaml',
  'yolo-yaml-name': 'Class names YAML reads as something else',
  'yolo-yaml-passed': 'YAML files passed over',
  'yolo-root': "data.yaml's path",
  'yolo-split-missing': 'Splits not found',
  'yolo-list-missing': 'Listed images not found',
  'yolo-list-fallback': 'Image lists not used',
  'yolo-label-beside': 'Label files beside their images',
  'yolo-label-outside': 'Label files outside the folder',
  'yolo-label-unreached': "Label files Ultralytics doesn't reach",
  'yolo-shared-label': 'Label files two images share',
  'yolo-two-splits': 'Images in two splits',
  'yolo-no-labels': 'No label files',
  'yolo-not-text': "Label files that aren't text",
  'yolo-no-yaml': 'No data.yaml',
  'yolo-no-images': 'No images',
  'yolo-corrupt': 'Files Ultralytics would leave out',
  'yolo-repeated': 'Rows repeated',
  'yolo-no-kpt-shape': 'No kpt_shape',
  'yolo-row': "Rows that can't be read",
  'yolo-keypoints-dropped': 'Keypoints the task leaves out',
  'yolo-box-row': 'Boxes among other rows',
  'yolo-outline-as-box': 'Outlines read as boxes',
  'yolo-obb-outline': 'Rows read as outlines',
  'import-image-missing': 'Images not in this folder',
  'import-image-ambiguous': 'Names that match several images',
  'import-size': 'Sizes that differ',
  'import-mask-pixels': 'Masks too big to write at once',
  'import-image-unreadable': "Images that can't be read",
  'import-shared-label': 'Images sharing a label file',
  'import-other-images': 'Label files of other images',
  'import-label-unreadable': "Label files that can't be read",
  'import-nothing': 'Nothing that could be read',
  'import-no-label-file': 'Images without a label file',
  'import-no-annotations': 'Images without annotations',
  'import-two-files': 'Images with two label files',
  'import-keypoint-names': 'Keypoint names',
  'import-keypoint-file': 'Keypoints file',
  // Why an image was left as it was, after writing
  labelled: 'Images that have labels, left as they are',
  unchanged: 'Images that have every shape already',
  'no-shapes': 'Images that have labels and get no shapes',
  failed: "Images that couldn't be written",
  kept: "Old label files that couldn't go to the trash"
}
const titleOf = (code: string) => TITLES[code] ?? code.replace(/^[a-z]+-/, '').replace(/-/g, ' ').replace(/^./, (c) => c.toUpperCase())
const RANK: Record<Severity, number> = { error: 0, warning: 1, info: 2 }
const SEVERE: Record<string, Severity> = { labelled: 'info', unchanged: 'info', 'no-shapes': 'info', failed: 'error', kept: 'warning' }

let session: Session | null = null
let job: AbortController | null = null

const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString('en-GB')} ${n === 1 ? one : many}`
const posix = (p: string) => p.split(path.sep).join('/')
const stemOf = (name: string) => name.replace(/\.[^./]+$/, '')
const say = (severity: Severity, code: string, image: string | null, text: string): Issue => ({ severity, code, image, message: text })

function begin() {
  job?.abort()
  job = new AbortController()
  return job.signal
}

/** Stops the read or write under way: a read is dropped, a write stops after the image it's on */
export function stopImport() {
  job?.abort()
}

/** Lets go of what was read, the COCO worker with it */
export function closeImport() {
  job?.abort()
  session?.coco?.close()
  session = null
}

// Runs fn on every item, a few at a time, telling how many are done at most ten times a second
async function pool<T>(items: T[], n: number, fn: (item: T) => Promise<void>, tick: (done: number) => void, signal: AbortSignal) {
  let next = 0, done = 0, told = 0
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length && !signal.aborted) {
      await fn(items[next++])
      done++
      if (done === items.length || Date.now() - told > 100) {
        told = Date.now()
        tick(done)
      }
    }
  }))
}

// What work gives, or null as soon as the signal stops it: for work that can't be stopped itself, whose answer is then dropped
function unlessStopped<T>(work: Promise<T>, signal: AbortSignal): Promise<T | null> {
  if (signal.aborted) return Promise.resolve(null)
  return new Promise<T | null>((resolve, reject) => {
    const stop = () => resolve(null)
    signal.addEventListener('abort', stop, { once: true })
    work.then(resolve, reject).finally(() => signal.removeEventListener('abort', stop))
  })
}

function checkRequest(req: ImportRequest) {
  if (!req || (req.kind !== 'coco' && req.kind !== 'yolo')) throw new Error('Pick a COCO file or a YOLO dataset')
  if (typeof req.path !== 'string' || !path.isAbsolute(req.path)) throw new Error('Pick the dataset again: its path is incomplete')
  if (!LABEL_FORMATS.includes(req.prefer)) throw new Error(`${req.prefer} isn't a label format`)
  if (!Array.isArray(req.labels) || !req.labels.every((l) => typeof l === 'string')) throw new Error('The label list must be names')
  if (req.task !== undefined && !YOLO_TASKS.includes(req.task)) throw new Error(`${req.task} isn't a YOLO task`)
  if (req.keypoints != null && (typeof req.keypoints !== 'string' || !path.isAbsolute(req.keypoints))) throw new Error('Pick the keypoints file again: its path is incomplete')
}

/**
 * Reads a dataset for importing into the open folder `root`: nothing is written. Reading it again
 * with another YOLO task or keypoint file reads only its rows again. Null when stopped.
 */
export async function readImport(root: string, req: ImportRequest, progress: (p: ImportProgress) => void): Promise<ImportPreview | null> {
  checkRequest(req)
  const folder = folderOf(root)
  const source = path.resolve(req.path)
  const s = session
  const same = s && !s.used && s.root === folder.dir && s.output === (folder.outputFile ?? folder.outputDir) && s.kind === req.kind && s.source === source && s.prefer === req.prefer
  if (!same) {
    s?.coco?.close()
    session = null
  }
  const signal = begin()
  try {
    if (!same) return await readNew(folder, req, source, progress, signal)
    s.listed = new Set(req.labels)
    if (s.yolo && ((req.task ?? s.yolo.task) !== s.yolo.task || (req.keypoints ?? null) !== s.yolo.keypoints)) {
      await readLabelRows(s, req.task ?? s.yolo.task, req.keypoints ?? null, progress, signal)
      if (!signal.aborted) await compare(s, progress, signal)
      if (signal.aborted) throw new Error('Stopped')
    }
    return preview(s)
  } catch (e) {
    const stopped = signal.aborted
    // Rows read halfway leave the images' counts half old: the dataset is read again from the start
    if (same && session === s) {
      s.coco?.close()
      session = null
    }
    if (stopped) return null
    throw e
  }
}

async function readNew(folder: ReturnType<typeof folderOf>, req: ImportRequest, source: string, progress: (p: ImportProgress) => void, signal: AbortSignal) {
  const s: Session = {
    root: folder.dir, output: folder.outputFile ?? folder.outputDir, kind: req.kind, source, prefer: req.prefer, listed: new Set(req.labels), entries: [], images: 0, base: null, missing: 0, ambiguous: 0,
    issues: [], rows: [], coco: null, cocoImages: new Map(), yolo: null, skipped: new Map(), used: false
  }
  const quit = () => {
    s.coco?.close()
    return null
  }
  // An image held inside its label file can't take another's labels
  const images = folder.images.filter((n) => !folder.embedded.has(n))
  if (req.kind === 'coco') await readCocoSource(s, images, progress, signal)
  else await readYoloSource(s, images, progress, signal)
  if (signal.aborted) return quit()
  // Two images whose LabelMe files are one file (a.jpg and a.png both have a.json, and on Windows
  // and macOS A.png too) can't both have their own
  const keyOf = (name: string) => (folder.outputFile ? '' : nameKey(stemOf(name)))
  const keys = new Map<string, string[]>()
  for (const name of folder.images) {
    const list = keys.get(keyOf(name))
    if (list) list.push(name)
    else keys.set(keyOf(name), [name])
  }
  for (const e of s.entries) {
    const others = keys.get(keyOf(e.name))!.filter((n) => n !== e.name)
    if (e.blocked || !others.length) continue
    const file = folder.outputFile ? path.basename(folder.outputFile) : `${path.basename(stemOf(e.name))}.json`
    e.blocked = say('error', 'import-shared-label', e.name, `${e.name} would share its label file, ${file}, with ${others.length > 3 ? `${others.slice(0, 3).join(', ')} and ${others.length - 3} more` : others.join(', ')}, so nothing is imported into it`)
  }
  if (s.yolo) await readLabelRows(s, s.yolo.task, null, progress, signal)
  if (signal.aborted) return quit()
  await look(s, s.entries.filter((e) => !e.blocked), progress, signal)
  if (signal.aborted) return quit()
  await compare(s, progress, signal)
  if (signal.aborted) return quit()
  // A size that differs was told for each COCO image already
  for (const e of s.entries) if (e.blocked && e.blocked.code !== 'import-size') s.issues.push(e.blocked)
  session = s
  return preview(s)
}

// COCO

// The worker itself failing, rather than answering that the file can't be read
class WorkerFailed extends Error {}

// The COCO file, read in a worker with a heap limit when the app runs, so a file too big for it
// can't take the app down; in place when tests run, and when the worker can't run for a small file
async function openCoco(file: string, folder: string[], root: string, progress: (p: ImportProgress) => void, signal: AbortSignal): Promise<CocoReader> {
  const tell: CocoProgress = (phase, done, total) => progress({ phase, done, total })
  const inPlace = () => {
    const session = new CocoSession(tell)
    const found = session.read(file, folder, root)
    return { found, shapes: async (ids: number[]) => session.shapes(ids), close: () => {} }
  }
  if (!process.versions.electron) return inPlace()
  const small = async () => ((await fs.stat(file).catch(() => null))?.size ?? Infinity) <= IN_PLACE_MB * 2 ** 20
  let worker: Worker
  try {
    // electron-vite builds importWorker.ts as a worker of its own
    const start = (await import('./importWorker?nodeWorker')).default
    worker = start({ workerData: { seamtagCoco: true }, resourceLimits: { maxOldGenerationSizeMb: HEAP_MB } })
  } catch (e) {
    if (await small()) return inPlace()
    throw new Error(`${path.basename(file)} can't be read: its reader couldn't start (${message(e)})`)
  }
  let waiting: { resolve: (a: CocoAnswer) => void; reject: (e: Error) => void } | null = null
  let failed: Error | null = null
  const fail = (e: Error) => {
    failed ??= e
    waiting?.reject(failed)
    waiting = null
  }
  worker.on('message', (a: CocoAnswer) => {
    if (a.type === 'progress') return tell(a.phase, a.done, a.total)
    const w = waiting
    waiting = null
    if (a.type === 'error') w?.reject(new Error(a.message))
    else w?.resolve(a)
  })
  worker.on('error', (e: NodeJS.ErrnoException) => fail(e.code === 'ERR_WORKER_OUT_OF_MEMORY' ? new Error(tooBig(path.basename(file), HEAP_MB * 2 ** 20)) : new WorkerFailed(e.message)))
  worker.on('exit', (code) => fail(new WorkerFailed(`The COCO reader stopped (${code})`)))
  const ask = (q: CocoAsk) => new Promise<CocoAnswer>((resolve, reject) => {
    if (failed) return reject(failed)
    waiting = { resolve, reject }
    worker.postMessage(q)
  })
  const close = () => void worker.terminate()
  signal.addEventListener('abort', close)
  try {
    const answer = await ask({ type: 'read', file, folder, root })
    if (answer.type !== 'read') throw new Error('The COCO reader answered out of turn')
    return {
      found: answer.found,
      shapes: async (ids) => {
        const a = await ask({ type: 'shapes', ids })
        if (a.type !== 'shapes') throw new Error('The COCO reader answered out of turn')
        return a.images
      },
      close
    }
  } catch (e) {
    close()
    if (!signal.aborted && e instanceof WorkerFailed && (await small())) return inPlace()
    throw e
  } finally {
    signal.removeEventListener('abort', close)
  }
}

async function readCocoSource(s: Session, folder: string[], progress: (p: ImportProgress) => void, signal: AbortSignal) {
  const reader = await openCoco(s.source, folder, s.root, progress, signal)
  s.coco = reader
  const { found } = reader
  s.images = found.images
  s.base = found.base
  // Issues name images by COCO file_name: matched ones by the folder's name too, where it differs
  const named = new Map<string, string>()
  for (const m of found.matched) named.set(m.fileName, m.name)
  s.issues = found.issues.map((i) => {
    const name = i.image === null ? undefined : named.get(i.image)
    if (!name) return i
    // Messages quote a long file name cut short, as readCoco does
    const quoted = short(i.image!, 120)
    const told = name !== i.image && i.message.startsWith(`${quoted}:`) ? `${quoted} (${short(name, 120)} in this folder)${i.message.slice(quoted.length)}` : i.message
    return { ...i, image: name, message: told }
  })
  s.missing = found.issues.filter((i) => i.code === 'coco-image-missing').length
  s.ambiguous = found.issues.filter((i) => i.code === 'coco-image-ambiguous').length
  const byName = new Map<string, Entry>()
  for (const m of found.matched) {
    s.cocoImages.set(m.id, m)
    let e = byName.get(m.name)
    if (!e) byName.set(m.name, (e = entry(m.name, null, false)))
    e.ids.push(m.id)
    e.byName ||= m.byName
  }
  s.entries = [...byName.values()]
  await measure(s, progress, signal)
  for (const e of s.entries) {
    if (e.size) {
      const [w, h] = e.size
      const wrong: Issue[] = []
      e.ids = e.ids.filter((id) => {
        const m = s.cocoImages.get(id)!
        if (m.width === w && m.height === h) return true
        const swapped = m.width === h && m.height === w
        wrong.push(say('error', 'import-size', e.name, `${e.name} is ${sizeText(w, h)} pixels upright, but ${short(m.fileName, 120)} in the COCO file is ${sizeText(m.width, m.height)}${swapped ? ': the sides are swapped, so the image\'s EXIF orientation probably differs from the one its annotations were made on' : ''}, so none of its annotations are imported`))
        return false
      })
      s.issues.push(...wrong)
      if (!e.ids.length) e.blocked = wrong[0]
    }
    for (const id of e.ids) {
      const m = s.cocoImages.get(id)!
      e.annotations += m.annotations
      e.shapes += m.shapes
      e.pixels += m.pixels
      for (const [label, n] of Object.entries(m.labels)) e.labels[label] = (e.labels[label] ?? 0) + n
    }
    // Writing makes an image's masks' pixels all at once, which its size, now known to be the
    // file's, doesn't bound: a file can give it any number of masks
    if (!e.blocked && e.pixels > IMAGE_PIXELS) {
      e.blocked = say('error', 'import-mask-pixels', e.name, `${e.name}: its masks in ${path.basename(s.source)} have ${plural(e.pixels, 'pixel')} in all, more than the ${IMAGE_PIXELS.toLocaleString('en-GB')} Seamtag makes for one image at a time, so nothing is imported into it`)
    }
    if (e.blocked) continue
    // A COCO file lists its images whatever they hold (a keypoints file lists those without people
    // too), so an image without annotations is left as it is, never given an empty label file
    if (!e.annotations) s.issues.push(say('info', 'import-no-annotations', e.name, `${e.name} has no annotations in ${path.basename(s.source)}, so it's left as it is`))
    else if (!e.shapes) s.issues.push(say('warning', 'import-nothing', e.name, `${e.name}: none of its ${plural(e.annotations, 'annotation')} could be read, so it's left as it is`))
  }
}

// Shapes by label, with no prototype, so a label called constructor or __proto__ counts like any other
const tally = (): Record<string, number> => Object.create(null)

const entry = (name: string, label: string | null, byName: boolean): Entry =>
  ({ name, size: null, ids: [], label, byName, annotations: 0, shapes: 0, labels: tally(), pixels: 0, blocked: null, has: null, broken: null, own: null, adds: null, repeated: 0 })

// YOLO

async function readYoloSource(s: Session, folder: string[], progress: (p: ImportProgress) => void, signal: AbortSignal) {
  progress({ phase: `Reading ${path.basename(s.source)}`, done: 0, total: 0 })
  // Stop doesn't wait for the dataset's files to be found: what's found is dropped, and the search
  // stops at its next folder
  const dataset = await unlessStopped(readYoloDataset(s.source, { signal }), signal)
  if (!dataset) return
  s.issues = [...dataset.issues]
  // Each image once, however many splits list it
  const items = new Map<string, string | null>()
  for (const split of dataset.splits) for (const it of split.items) if (!items.has(it.image)) items.set(it.image, it.label)
  s.images = items.size
  // By path when the image is in the open folder, else by its file name when one image has it on
  // each side. A path or a name the folder has only in another case or Unicode form, which the file
  // system takes for the same (nameKey), is matched when exactly one image on each side has its key.
  const names = new Set(folder)
  // Lists grow in place: a copy at each would take as long as the square of how many share a name
  const add = (lists: Map<string, string[]>, key: string, f: string) => {
    const list = lists.get(key)
    if (list) list.push(f)
    else lists.set(key, [f])
  }
  const ours = new Map<string, string[]>(), ourPaths = new Map<string, string[]>()
  for (const f of folder) {
    add(ours, nameKey(f.slice(f.lastIndexOf('/') + 1)), f)
    add(ourPaths, nameKey(f), f)
  }
  const placed = [...items].map(([image, label]) => {
    const rel = path.relative(s.root, image)
    return { image, label, at: rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? posix(rel) : null }
  })
  const theirs = new Map<string, number>(), theirPaths = new Map<string, number>()
  const count = (counts: Map<string, number>, key: string) => counts.set(key, (counts.get(key) ?? 0) + 1)
  for (const { image, at } of placed) {
    count(theirs, nameKey(path.basename(image)))
    if (at !== null) count(theirPaths, nameKey(at))
  }
  // The folder's image at a path: as it is, else the one image with its key, when no other image of
  // the dataset has that key
  const atPath = (at: string) => {
    if (names.has(at)) return at
    const same = ourPaths.get(nameKey(at))
    return same?.length === 1 && theirPaths.get(nameKey(at)) === 1 ? same[0] : null
  }
  const shown = (file: string) => posix(path.relative(dataset.root, file))
  for (const { image, label, at } of placed) {
    const found = at === null ? null : atPath(at)
    if (found !== null) {
      s.entries.push(entry(found, label, false))
      continue
    }
    const own = path.basename(image)
    const candidates = ours.get(nameKey(own)) ?? []
    if (candidates.length === 1 && theirs.get(nameKey(own)) === 1) {
      s.entries.push(entry(candidates[0], label, true))
      continue
    }
    if (candidates.length) {
      s.ambiguous++
      const why = candidates.length === 1 ? `other images in the dataset are named ${own} too` : `its name matches ${candidates.length > 3 ? `${candidates.slice(0, 3).join(', ')} and ${candidates.length - 3} more` : candidates.join(', ')}`
      s.issues.push(say('warning', 'import-image-ambiguous', null, `${shown(image)} isn't at that path in the folder and ${why}, so its labels are left out rather than guessed`))
    } else {
      s.missing++
      s.issues.push(say('warning', 'import-image-missing', null, `${shown(image)} isn't in the folder, so its labels are left out`))
    }
  }
  s.yolo = { dataset, task: dataset.guess.task, keypoints: null, names: null, from: 'yaml' }
  await measure(s, progress, signal)
}

// The rows of every matched label file, read for a task: what each image gets, and what's wrong
async function readLabelRows(s: Session, task: YoloTask, keypoints: string | null, progress: (p: ImportProgress) => void, signal: AbortSignal) {
  const y = s.yolo!
  const ds = y.dataset
  const issues: Issue[] = []
  // Keypoints' names: data.yaml's kpt_names, else the keypoint file's by class name, else kp0, kp1 and so on
  const kptShape = ds.kptShape ?? (task === 'pose' ? ds.guess.kptShape : null)
  let names = ds.kptNames
  let from: 'yaml' | 'file' | 'generated' = 'yaml'
  if (task === 'pose' && !names) {
    from = 'generated'
    if (keypoints) {
      try {
        // Read whole, here in the app: a file far larger than any keypoint file isn't
        const { size } = await fs.stat(keypoints)
        if (size > KEYPOINT_FILE_BYTES) throw new Error(`it's ${(size / 2 ** 20).toFixed(1)} MB, more than the ${KEYPOINT_FILE_BYTES / 2 ** 20} MB a keypoints file may be`)
        const schema = readPoseYaml(await fs.readFile(keypoints, 'utf8')).schema
        const mapped: Record<number, string[]> = {}
        ds.names.forEach((n, c) => {
          const list = Object.hasOwn(schema.classes, n) ? schema.classes[n] : null
          if (!list) return
          mapped[c] = list
          if (kptShape && list.length !== kptShape[0]) issues.push(say('warning', 'import-keypoint-names', null, `${path.basename(keypoints)} names ${list.length} keypoints for ${short(n)}, where each row has ${kptShape[0]}: ${list.length < kptShape[0] ? `the rest are called kp${list.length} and on` : 'the last are left unused'}`))
        })
        if (Object.keys(mapped).length) {
          names = mapped
          from = 'file'
        } else issues.push(say('error', 'import-keypoint-file', null, `${path.basename(keypoints)} names none of the dataset's classes (${ds.names.slice(0, 5).map((n) => short(n)).join(', ')}${ds.names.length > 5 ? ' …' : ''}), so its names aren't used`))
      } catch (e) {
        issues.push(say('error', 'import-keypoint-file', null, `${path.basename(keypoints)} can't be used: ${message(e)}`))
      }
    }
    if (from === 'generated' && kptShape) {
      issues.push(say('info', 'import-keypoint-names', null, `data.yaml has no kpt_names, so the keypoints are called kp0 to kp${kptShape[0] - 1}: pick a keypoints file to name them`))
    }
  }
  const live = s.entries.filter((e) => !e.blocked)
  const total = live.length
  progress({ phase: 'Reading the label files', done: 0, total })
  await pool(live, 16, async (e) => {
    e.annotations = 0
    e.shapes = 0
    e.labels = tally()
    if (!e.label) {
      issues.push(say('info', 'import-no-label-file', e.name, `${e.name} has no label file in the dataset (Ultralytics trains it as a background), so it's left as it is`))
      return
    }
    const file = posix(path.relative(ds.root, e.label))
    const text = labelText(await fs.readFile(e.label).catch(() => new Uint8Array([0xff])))
    if (text === null) {
      issues.push(say('warning', 'yolo-not-text', e.name, `${file} can't be read as UTF-8 text, so ${e.name} gets nothing from it`))
      e.annotations = 1
      return
    }
    const stats = newRowStats()
    countRows(stats, text)
    const read = readRows(text, { task, names: ds.names, width: e.size![0], height: e.size![1], kptShape, kptNames: names, image: e.name, file })
    e.annotations = stats.rows
    e.shapes = read.shapes.length
    for (const sh of read.shapes) e.labels[sh.label] = (e.labels[sh.label] ?? 0) + 1
    issues.push(...read.issues)
    if (e.annotations && !e.shapes) issues.push(say('warning', 'import-nothing', e.name, `${e.name}: none of the ${plural(e.annotations, 'row')} in ${file} could be read, so it's left as it is`))
  }, (done) => progress({ phase: 'Reading the label files', done, total }), signal)
  y.task = task
  y.keypoints = keypoints
  y.names = names
  y.from = from
  s.rows = issues
}

// Images: their sizes, and their labels

async function measure(s: Session, progress: (p: ImportProgress) => void, signal: AbortSignal) {
  const total = s.entries.length
  progress({ phase: 'Measuring the images', done: 0, total })
  await pool(s.entries, 8, async (e) => {
    try {
      e.size = await imageSize(s.root, e.name)
    } catch (err) {
      e.blocked = say('error', 'import-image-unreadable', e.name, `${e.name} can't be read (${message(err)}), so nothing is imported into it`)
    }
  }, (done) => progress({ phase: 'Measuring the images', done, total }), signal)
}

// An image whose labels are read from its VOC or YOLO file, the window's save format, while it has
// a LabelMe file too: writing the LabelMe file wouldn't change what the window shows, and saveLabels
// won't turn the other into LabelMe JSON over it, so neither is done
function twoFiles(root: string, name: string, found: Found): string | null {
  const json = labelPath(root, name)
  if (found.file === json || !existsSync(json)) return null
  return `${name} has labels in ${path.basename(found.file)} and in ${path.basename(json)}, so nothing is imported into it: keep one of them first`
}

async function look(s: Session, entries: Entry[], progress: (p: ImportProgress) => void, signal: AbortSignal) {
  const total = entries.length
  progress({ phase: 'Looking at their label files', done: 0, total })
  await pool(entries, 8, async (e) => {
    const found = await findLabels(s.root, e.name, s.prefer).catch(() => null)
    if (!found) return
    e.has = { format: found.format, file: found.file }
    const two = twoFiles(s.root, e.name, found)
    if (two) {
      e.blocked = say('warning', 'import-two-files', e.name, two)
      return
    }
    try {
      const read = await readLabels(s.root, e.name, found, async () => e.size ?? [1, 1])
      if (!read) e.blocked = say('error', 'import-other-images', e.name, `${e.name}'s label file, ${path.basename(found.file)}, is a CreateML list of other images, so nothing is imported into it`)
      else e.own = new Set(read.doc.shapes.map((sh) => sh.label))
    } catch (err) {
      e.broken = message(err)
      s.issues.push(say('warning', 'import-label-unreadable', e.name, `${path.basename(found.file)} can't be read, so it's never touched: ${message(err)}`))
    }
  }, (done) => progress({ phase: 'Looking at their label files', done, total }), signal)
}

/**
 * Add: what each labelled image would get, worked out as writing works it out (addShapes), so the
 * window counts what Add writes rather than every shape the dataset has for it. Only images whose
 * own shapes share a label with the dataset's can have any of them already; their labels are read
 * again, a batch at a time, rather than kept from looking at them, and the dataset's masks are
 * made PNGs only where the image has masks of its own to match them with.
 */
async function compare(s: Session, progress: (p: ImportProgress) => void, signal: AbortSignal) {
  for (const e of s.entries) {
    e.adds = null
    e.repeated = 0
  }
  const todo = s.entries.filter((e) => e.has && !e.blocked && !e.broken && e.shapes && Object.keys(e.labels).some((label) => e.own?.has(label)))
  const total = todo.length
  let done = 0
  if (total) progress({ phase: 'Comparing with the shapes they have', done, total })
  for (const batch of writeBatches(todo)) {
    if (signal.aborted) return
    const own = new Map<string, Shape[]>()
    await pool(batch, 8, async (e) => {
      const found = await findLabels(s.root, e.name, s.prefer).catch(() => null)
      const read = found && (await readLabels(s.root, e.name, found, async () => e.size ?? [1, 1]).catch(() => null))
      if (read) own.set(e.name, read.doc.shapes)
    }, () => {}, signal)
    if (signal.aborted) return
    const made = await shapesFor(s, batch, (e) => !!own.get(e.name)?.some((sh) => sh.shape_type === 'mask'))
    for (const e of batch) {
      const mine = own.get(e.name)
      if (!mine) continue
      const { added, repeated } = addShapes(mine, made.get(e.name) ?? [])
      e.adds = tally()
      for (const sh of added) e.adds[sh.label] = (e.adds[sh.label] ?? 0) + 1
      e.repeated = repeated
    }
    done += batch.length
    progress({ phase: 'Comparing with the shapes they have', done, total })
  }
}

// What the window is sent

function groupsOf(issues: Issue[]): IssueGroup[] {
  const groups = new Map<string, IssueGroup>()
  for (const i of issues) {
    let g = groups.get(i.code)
    if (!g) groups.set(i.code, (g = { code: i.code, severity: i.severity, title: titleOf(i.code), count: 0, rows: [] }))
    if (RANK[i.severity] < RANK[g.severity]) g.severity = i.severity
    g.count++
    if (g.rows.length < SHOWN) g.rows.push(i.message)
  }
  return [...groups.values()].sort((a, b) => RANK[a.severity] - RANK[b.severity] || b.count - a.count)
}

// Whether the import leaves an image alone whatever the choice: it gets no shapes. A YOLO label
// file without rows is the exception, being how a YOLO dataset marks an image with no objects:
// an image without labels gets an empty label file too.
const nothing = (s: Session, e: Entry) => !e.shapes && !(s.kind === 'yolo' && e.label && !e.annotations)
// Whether comparing found that a labelled image has every shape the dataset gives it already, so Add leaves it as it is
const hasEvery = (e: Entry) => !!e.adds && !Object.keys(e.adds).length && e.repeated > 0

function preview(s: Session): ImportPreview {
  let fresh = 0, blank = 0, labelled = 0, labelImg = 0, empty = 0, unchanged = 0, unchangedLabelImg = 0, unreadable = 0, blocked = 0, none = 0, byName = 0
  const counts = new Map<string, LabelCount>()
  for (const e of s.entries) {
    if (e.byName) byName++
    if (e.blocked) { blocked++; continue }
    if (nothing(s, e)) { none++; continue }
    if (e.broken) { unreadable++; continue }
    if (e.has) {
      labelled++
      if (!e.shapes) empty++
      else {
        const other = e.has.format !== 'labelme'
        if (other) labelImg++
        if (hasEvery(e)) {
          unchanged++
          if (other) unchangedLabelImg++
        }
      }
    } else {
      fresh++
      if (!e.shapes) blank++
    }
    for (const [label, n] of Object.entries(e.labels)) {
      let c = counts.get(label)
      if (!c) counts.set(label, (c = { label, fresh: 0, labelled: 0, added: 0, listed: s.listed.has(label) }))
      if (!e.has) c.fresh += n
      else {
        c.labelled += n
        c.added += e.adds ? e.adds[label] ?? 0 : n
      }
    }
  }
  const y = s.yolo
  const kpt = y?.dataset.kptShape ?? (y?.task === 'pose' ? y.dataset.guess.kptShape : null)
  return {
    kind: s.kind, source: s.source, name: path.basename(s.source), images: s.images, matched: s.entries.length,
    byName,
    base: s.base, missing: s.missing, ambiguous: s.ambiguous,
    fresh, labelled, labelImg, blank, empty, unchanged, unchangedLabelImg, unreadable, blocked, nothing: none,
    labels: [...counts.values()].sort((a, b) => b.fresh + b.labelled - a.fresh - a.labelled || a.label.localeCompare(b.label)),
    groups: groupsOf([...s.issues, ...s.rows]),
    ...(y ? {
      task: { task: y.task, guess: y.dataset.guess.task, others: y.dataset.guess.others, ask: y.dataset.guess.ask, reason: y.dataset.guess.reason },
      ...(y.task === 'pose' && kpt ? { keypoints: { from: y.from, file: y.keypoints, count: kpt[0] } } : {})
    } : {})
  }
}

/** More of a group's messages, a page at a time: the import's issues, or after writing, the images left as they were */
export function importRows(list: 'issues' | 'skipped', code: string, offset: number): string[] {
  const s = session
  if (!s || typeof code !== 'string' || !Number.isInteger(offset) || offset < 0) return []
  const rows = list === 'skipped' ? s.skipped.get(code)?.rows ?? [] : [...s.issues, ...s.rows].filter((i) => i.code === code).map((i) => i.message)
  return rows.slice(offset, offset + 200)
}

// Writing

/**
 * Shapes added to an image's own. Those it has already, exactly but for their group_id, are left
 * out and counted. The rest's groups move above its own, but a group that repeats a shape of one
 * of its groups joins that group instead, so a posed object's keypoints, imported after its
 * outline from an export's other file, join the outline.
 */
export function addShapes(own: Shape[], imported: Shape[]): { added: Shape[]; repeated: number } {
  const sorted = (o: Record<string, unknown>) => Object.keys(o).sort().map((k) => [k, o[k]])
  const key = (s: Shape) => JSON.stringify([s.label, s.shape_type, s.points, s.description, sorted(s.flags), s.mask, sorted(s.other), !!s.locked])
  const pool = new Map<string, (number | null)[]>()
  let top = 0
  for (const s of own) {
    const k = key(s)
    const list = pool.get(k)
    if (list) list.push(s.group_id)
    else pool.set(k, [s.group_id])
    // New groups count on from the highest below MAX_GROUP, as readCoco's do
    if (s.group_id !== null && s.group_id > top && s.group_id < MAX_GROUP) top = s.group_id
  }
  const kept: Shape[] = []
  const joins = new Map<number, Set<number | null>>()
  let repeated = 0
  for (const s of imported) {
    const list = pool.get(key(s))
    if (!list?.length) {
      kept.push(s)
      continue
    }
    repeated++
    const group = list.shift()!
    if (s.group_id === null) continue
    if (!joins.has(s.group_id)) joins.set(s.group_id, new Set())
    joins.get(s.group_id)!.add(group)
  }
  const moved = new Map<number, number>()
  const added = kept.map((s) => {
    if (s.group_id === null) return s
    let g = moved.get(s.group_id)
    if (g === undefined) {
      const join = joins.get(s.group_id)
      const one = join?.size === 1 ? [...join][0] : null
      g = one ?? ++top
      moved.set(s.group_id, g)
    }
    return { ...s, group_id: g }
  })
  return { added, repeated }
}

/**
 * Images in the batches writing asks the COCO reader for at once: at most BATCH images, whose
 * masks have at most BATCH_PIXELS pixels in all unless one image alone has more
 */
export function writeBatches<T extends { pixels: number }>(images: T[]): T[][] {
  const out: T[][] = []
  let batch: T[] = [], pixels = 0
  for (const im of images) {
    if (batch.length && (batch.length === BATCH || pixels + im.pixels > BATCH_PIXELS)) {
      out.push(batch)
      batch = []
      pixels = 0
    }
    batch.push(im)
    pixels += im.pixels
  }
  if (batch.length) out.push(batch)
  return out
}

// Shapes of several COCO images bound for one image of the folder: each one's groups go above the last's
function joined(parts: CocoShapes[]): { shapes: Shape[]; masks: (MaskBits | null)[] } {
  if (parts.length === 1) return parts[0]
  const shapes: Shape[] = [], masks: (MaskBits | null)[] = []
  let top = 0
  for (const part of parts) {
    const moved = new Map<number, number>()
    const base = top
    part.shapes.forEach((s, i) => {
      let g: number | null = s.group_id
      if (g !== null) {
        if (!moved.has(g)) moved.set(g, base + moved.size + 1)
        g = moved.get(g)!
        top = Math.max(top, g)
      }
      shapes.push({ ...s, group_id: g })
      masks.push(part.masks[i])
    })
  }
  return { shapes, masks }
}

// The shapes each image of a batch gets: COCO's from the worker, their masks' pixels made PNGs; YOLO's from its rows.
// Comparing them with an image's own needs no PNGs where it has no masks: `encode` says where.
async function shapesFor(s: Session, batch: Entry[], encode: (e: Entry) => boolean = () => true): Promise<Map<string, Shape[]>> {
  const out = new Map<string, Shape[]>()
  if (s.coco) {
    const read = new Map((await s.coco.shapes(batch.flatMap((e) => e.ids))).map((r) => [r.id, r]))
    for (const e of batch) {
      const { shapes, masks } = joined(e.ids.map((id) => read.get(id)).filter((r): r is CocoShapes => !!r))
      const ready: Shape[] = []
      const png = encode(e)
      for (const [i, sh] of shapes.entries()) {
        const bits = masks[i]
        if (sh.shape_type !== 'mask') ready.push(sh)
        else if (bits) ready.push(png ? { ...sh, mask: await encodeMask(bits.data, bits.width, bits.height) } : sh)
      }
      out.set(e.name, ready)
    }
    return out
  }
  const y = s.yolo!
  const ds = y.dataset
  const kptShape = ds.kptShape ?? (y.task === 'pose' ? ds.guess.kptShape : null)
  for (const e of batch) {
    const text = e.label ? labelText(await fs.readFile(e.label).catch(() => new Uint8Array([0xff]))) : null
    out.set(e.name, text === null ? [] : readRows(text, { task: y.task, names: ds.names, width: e.size![0], height: e.size![1], kptShape, kptNames: y.names }).shapes)
  }
  return out
}

// What the report says of an image Add leaves as it is, as it has every shape already
const hasAll = (name: string, shapes: number) => `${name} has ${shapes === 1 ? 'the imported shape' : `all ${plural(shapes, 'imported shape')}`} already, so it's left as it is`

type Outcome = {
  written?: Shape[]; repeated?: number; trashed?: number; skip?: { why: string; message: string }
  /** Nothing was written, but its label file changed: the old one went to the trash before writing failed */
  changed?: boolean
}

// One image, in one turn of the label files' queue, so no save or pre-labelling writes its labels
// between looking at them and writing them: its labels looked at again, as they may have changed
// since reading, then written as the choice says
function writeOne(s: Session, e: Entry, shapes: Shape[], policy: ImportPolicy): Promise<Outcome> {
  return serial(async () => {
    const size = e.size!
    const save = (req: SaveRequest) => saveLabels(s.root, e.name, req, {}, inTurn)
    const saved = (o: Saved, written: Shape[], repeated = 0): Outcome =>
      ({ written, repeated, trashed: o.replaced ? 1 : 0, ...(o.kept ? { skip: { why: 'kept', message: `${e.name}: ${o.kept}` } } : {}) })
    const found = await findLabels(s.root, e.name, s.prefer)
    if (!found) return saved(await save({ shapes, flags: {}, size, imageData: null, format: 'labelme' }), shapes)
    const old = path.basename(found.file)
    // saveLabels sends a CreateML file of this image alone to the trash before LabelMe's takes its name
    const failed = (err: unknown): Outcome => existsSync(found.file)
      ? { skip: { why: 'failed', message: `${e.name}: ${message(err)}` } }
      : { trashed: 1, changed: true, skip: { why: 'failed', message: `${old} went to the trash, but ${e.name}'s new labels couldn't be written: ${message(err)}` } }
    const two = twoFiles(s.root, e.name, found)
    if (two) return { skip: { why: 'import-two-files', message: two } }
    if (policy === 'skip') return { skip: { why: 'labelled', message: `${e.name} has labels, so it's left as it is` } }
    if (!shapes.length) return { skip: { why: 'no-shapes', message: `${e.name} gets no shapes from the dataset, so it's left as it is` } }
    let read
    try {
      read = await readLabels(s.root, e.name, found, async () => size)
    } catch (err) {
      return { skip: { why: 'import-label-unreadable', message: `${old} can't be read, so it's never touched: ${message(err)}` } }
    }
    if (!read) return { skip: { why: 'import-other-images', message: `${e.name}'s label file, ${old}, is a CreateML list of other images, so it's left as it is` } }
    if (policy === 'add') {
      const { added, repeated } = addShapes(read.doc.shapes, shapes)
      if (!added.length) return { repeated, skip: { why: 'unchanged', message: hasAll(e.name, shapes.length) } }
      const all = [...read.doc.shapes, ...added]
      try {
        return saved(found.format === 'labelme'
          ? await save({ shapes: all, flags: read.doc.flags, size, imageData: read.doc.imageData, verified: read.verified, format: 'labelme' })
          : await save({ shapes: all, flags: {}, size, imageData: null, verified: read.verified, format: 'labelme', from: found.format }), added, repeated)
      } catch (err) {
        return failed(err)
      }
    }
    if (found.format !== 'labelme') {
      try {
        return saved(await save({ shapes, flags: {}, size, imageData: null, format: 'labelme', from: found.format }), shapes)
      } catch (err) {
        return failed(err)
      }
    }
    // The new file is written beside the old one, the old one goes to the trash, where it can be
    // restored from, then the new one takes its name, as rewriteLabels does: if writing fails, the old one stays
    const tmp = `${found.file}.${process.pid}.${Date.now()}.tmp`
    try {
      await save({ shapes, flags: {}, size, imageData: read.doc.imageData, format: 'labelme', target: tmp })
    } catch (err) {
      await fs.rm(tmp, { force: true })
      return { skip: { why: 'failed', message: `${e.name}'s new labels couldn't be written, so ${old} is left unchanged: ${message(err)}` } }
    }
    try {
      await sendToTrash(found.file)
    } catch (err) {
      await fs.rm(tmp, { force: true })
      return { skip: { why: 'failed', message: `${old} couldn't go to the trash, so it's left unchanged: ${message(err)}` } }
    }
    try {
      await renameRetrying(tmp, found.file)
    } catch (err) {
      // A file something still holds, as a sync client such as OneDrive can for longer than the
      // retries wait, can most likely still be read: a copy of it takes the old one's name, so the
      // image isn't left without labels
      try {
        await fs.copyFile(tmp, found.file, constants.COPYFILE_EXCL)
      } catch {
        return { trashed: 1, changed: true, skip: { why: 'failed', message: `${old} went to the trash, but ${e.name}'s new labels couldn't take its name: they're in ${path.basename(tmp)} (${message(err)})` } }
      }
      await fs.rm(tmp, { force: true, maxRetries: 3 }).catch(() => {})
    }
    return { written: shapes, trashed: 1 }
  })
}

/**
 * Writes what was read into the folder's label files, as LabelMe JSON, an image at a time. Images
 * with labels are left alone (skip), get the imported shapes added (add), or have their label file
 * sent to the trash and the imported shapes take its place (replace); whatever the choice, an image
 * the dataset gives no shapes keeps what it has. The written images are then counted again for the
 * image list. Stopping keeps what's written.
 */
export async function writeImport(policy: ImportPolicy, progress: (p: ImportProgress) => void, summaries: Summaries): Promise<ImportReport> {
  if (!['skip', 'add', 'replace'].includes(policy)) throw new Error(`${policy} isn't a choice`)
  const s = session
  if (!s || s.used) throw new Error('Read the dataset again first')
  folderOf(s.root)
  const signal = begin()
  s.used = true
  const skipped = s.skipped
  const skip = (why: string, text: string, severity?: Severity) => {
    let g = skipped.get(why)
    if (!g) skipped.set(why, (g = { severity: severity ?? SEVERE[why] ?? 'warning', title: titleOf(why), rows: [] }))
    g.rows.push(text)
  }
  const todo: Entry[] = []
  let repeated = 0
  for (const e of s.entries) {
    if (e.blocked) skip(e.blocked.code, e.blocked.message, e.blocked.severity)
    else if (s.kind === 'yolo' && !e.label) skip('import-no-label-file', `${e.name} has no label file in the dataset, so it's left as it is`, 'info')
    else if (s.kind === 'coco' && !e.annotations) skip('import-no-annotations', `${e.name} has no annotations in ${path.basename(s.source)}, so it's left as it is`, 'info')
    else if (nothing(s, e)) skip('import-nothing', `${e.name}: none of what the dataset has for it could be read, so it's left as it is`)
    else if (e.broken) skip('import-label-unreadable', `${e.has ? path.basename(e.has.file) : e.name} can't be read, so it's never touched: ${e.broken}`)
    else if (e.has && policy === 'skip') skip('labelled', `${e.name} has labels, so it's left as it is`)
    else if (e.has && !e.shapes) skip('no-shapes', `${e.name} gets no shapes from the dataset, so it's left as it is`)
    // Reading found it has every shape already, and the window said it's left as it is: it isn't
    // looked at again, so the count and the progress are of the images Add writes
    else if (e.has && policy === 'add' && hasEvery(e)) {
      repeated += e.repeated
      skip('unchanged', hasAll(e.name, e.repeated))
    } else todo.push(e)
  }
  const names: string[] = []
  const labels = new Map<string, number>()
  let written = 0, shapes = 0, trashed = 0, done = 0, told = 0
  const total = todo.length
  progress({ phase: 'Writing label files', done: 0, total })
  const batches = writeBatches(todo)
  for (const [k, batch] of batches.entries()) {
    if (signal.aborted) break
    let made: Map<string, Shape[]>
    try {
      made = await shapesFor(s, batch)
    } catch (err) {
      // The COCO reader is gone, so the rest can't be read either
      for (const e of batches.slice(k).flat()) skip('failed', `${e.name}: ${message(err)}`)
      break
    }
    for (const e of batch) {
      if (signal.aborted) break
      try {
        const o = await writeOne(s, e, made.get(e.name) ?? [], policy)
        repeated += o.repeated ?? 0
        trashed += o.trashed ?? 0
        if (o.skip) skip(o.skip.why, o.skip.message)
        if (o.written || o.changed) names.push(e.name)
        if (o.written) {
          written++
          shapes += o.written.length
          for (const sh of o.written) labels.set(sh.label, (labels.get(sh.label) ?? 0) + 1)
        }
      } catch (err) {
        skip('failed', `${e.name}: ${message(err)}`)
      }
      done++
      if (done === total || Date.now() - told > 100) {
        told = Date.now()
        progress({ phase: 'Writing label files', done, total })
      }
    }
  }
  // The image list and the label list count the images whose label files changed again
  if (names.length) await summarizeAll(s.root, summaries, s.prefer, false, names, await folderKeypointNames(s.root)).catch(() => {})
  return {
    stopped: signal.aborted, total, done, written, shapes, repeated, trashed,
    labels: [...labels].map(([label, n]) => ({ label, shapes: n, listed: s.listed.has(label) })).sort((a, b) => b.shapes - a.shapes || a.label.localeCompare(b.label)),
    skipped: [...skipped].map(([code, g]) => ({ code, severity: g.severity, title: g.title, count: g.rows.length, rows: g.rows.slice(0, SHOWN) }))
      .sort((a, b) => RANK[a.severity] - RANK[b.severity] || b.count - a.count),
    names
  }
}

