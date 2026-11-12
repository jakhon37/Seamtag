// The COCO half of File > Import dataset, read off the window's thread. src/main/import.ts runs
// it in a worker with a heap limit, so a file too big for it fails there with a plain message
// instead of taking the app down; tests use CocoSession in place. A file is parsed once and
// matched to the folder's images, then read into shapes a few thousand annotations at a time,
// as readCoco reads a whole file (src/shared/coco.ts): memory holds the parsed file and one
// chunk's shapes, never every shape of COCO's 118,000 train2017 images at once. Nothing here
// touches the folder.
import { constants } from 'buffer'
import { readFileSync, statSync } from 'fs'
import path from 'path'
import { getHeapSpaceStatistics, getHeapStatistics } from 'v8'
import { parentPort, resourceLimits, workerData } from 'worker_threads'
import { type CocoImage, type CocoImageResult, matchCocoImages, readCoco } from '../shared/coco'
import type { Issue, MaskBits } from '../shared/dataset'
import type { Shape } from '../shared/labelfile'
import { shapesFromPose } from '../shared/pose'

/** A COCO image matched to an image in the folder, and what its annotations give */
export type CocoMatched = {
  id: number
  /** The image in the folder it's matched to */
  name: string
  fileName: string
  /** The size the COCO file gives it */
  width: number
  height: number
  /** Matched by its file name alone, its path not being in the folder */
  byName: boolean
  /** Its annotations, and the shapes they make, by label */
  annotations: number
  shapes: number
  labels: Record<string, number>
}

/** A COCO file as read for importing */
export type CocoFound = {
  /** Images in its images list */
  images: number
  /** The folder its file names start from, as a path in the open folder; '' for the folder itself */
  base: string
  matched: CocoMatched[]
  /** What can't be imported and why, each naming its image by COCO file_name */
  issues: Issue[]
}

/** One COCO image's shapes, ready to write but for its masks' pixels, which the main process encodes */
export type CocoShapes = { id: number; shapes: Shape[]; masks: (MaskBits | null)[] }

/** Told as the file is read: what's being done, and how far it's got (0 of 0 when it can't tell) */
export type CocoProgress = (phase: string, done: number, total: number) => void

/** What the main process asks the worker, and what it answers */
export type CocoAsk = { type: 'read'; file: string; folder: string[]; root: string } | { type: 'shapes'; ids: number[] }
export type CocoAnswer =
  | { type: 'progress'; phase: string; done: number; total: number }
  | { type: 'read'; found: CocoFound }
  | { type: 'shapes'; images: CocoShapes[] }
  | { type: 'error'; message: string }

// Annotations read into shapes at a time: a few megabytes of JSON
const CHUNK = 20000
// Images whose paths are tried under each folder the file names could start from
const SAMPLE = 2000

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const plural = (n: number, one: string, many: string) => (n === 1 ? one : many)

/**
 * A COCO image's shapes as they're written: the annotations' shapes, then each posed object's
 * keypoints, named after them, in its group. A posed object whose outline is among the shapes
 * adds only its keypoints; one whose box is all there is of it adds a rectangle too.
 */
export function shapesOf(result: CocoImageResult): { shapes: Shape[]; masks: (MaskBits | null)[] } {
  const shapes = [...result.shapes], masks = [...result.masks]
  for (const pose of result.poses) {
    const made = shapesFromPose(pose.label, pose.box, pose.keypoints, pose.names, pose.group)
    for (const s of pose.outlined ? made.slice(1) : made) {
      shapes.push(s)
      masks.push(null)
    }
  }
  return { shapes, masks }
}

// A path with / between folders and no . or .. left in it, as matchCocoImages tidies one; null when it climbs out
function tidy(p: string): string | null {
  const out: string[] = []
  for (const part of p.replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue
    if (part !== '..') out.push(part)
    else if (!out.pop()) return null
  }
  return out.join('/')
}

/**
 * The folder a COCO file's names start from: the open folder, the COCO file's own folder or one
 * above it, or a folder its first images turn up in by name (val2017 for COCO's own layout). The
 * one under which most images are found by their paths, tried on a sample; none when two tie.
 */
export function startFolder(images: CocoImage[], folder: string[], at: string | null): string {
  const names = new Set(folder)
  const candidates = new Set<string>([''])
  if (at !== null) for (let p = at; p; p = p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '') candidates.add(p)
  const byName = new Map<string, string[]>()
  for (const f of folder) {
    const own = f.slice(f.lastIndexOf('/') + 1)
    const list = byName.get(own)
    if (list) list.push(f)
    else byName.set(own, [f])
  }
  for (const im of images.slice(0, 50)) {
    const name = tidy(im.file_name)
    if (!name) continue
    for (const f of byName.get(name.slice(name.lastIndexOf('/') + 1)) ?? []) {
      if (f.length > name.length && f.endsWith(`/${name}`)) candidates.add(f.slice(0, f.length - name.length - 1))
    }
  }
  if (candidates.size === 1) return ''
  const step = Math.max(1, Math.floor(images.length / SAMPLE))
  const sample = images.filter((_, i) => i % step === 0)
  let best = '', most = 0, tied = false
  for (const base of candidates) {
    let found = 0
    for (const im of sample) {
      const p = tidy(base ? `${base}/${im.file_name}` : im.file_name)
      if (p !== null && names.has(p)) found++
    }
    if (found > most) [best, most, tied] = [base, found, false]
    else if (found === most && found > 0) tied = true
  }
  return tied ? '' : best
}

// readCoco sums these up once per file: read a chunk at a time, they're summed again here
const TALLIES = new Set(['coco-box-only', 'coco-unlabelled-keypoints', 'coco-no-keypoints', 'coco-crowd'])
// readCoco's own words for each sum (src/shared/coco.ts), so a file read in chunks says what it says read whole
function tallyMessage(code: string, file: string, n: number) {
  const one = n === 1
  if (code === 'coco-box-only') return `${file}: ${n} ${plural(n, 'annotation holds', 'annotations hold')} only a box, so ${plural(n, 'it becomes a rectangle', 'they become rectangles')}`
  if (code === 'coco-unlabelled-keypoints') return `${file}: ${n} ${plural(n, 'keypoint is', 'keypoints are')} marked not labelled (visibility 0), so ${plural(n, "it's", "they're")} left out`
  if (code === 'coco-no-keypoints') return `${file}: ${n} ${one ? 'annotation has' : 'annotations have'} no keypoints labelled, so ${one ? "it's read as a plain object" : "they're read as plain objects"}`
  return `${file}: ${n} ${one ? 'annotation is a crowd region' : 'annotations are crowd regions'} (iscrowd 1): the shapes made from ${one ? 'it' : 'them'} keep iscrowd, so an export writes ${one ? 'it as a crowd' : 'them as crowds'} again`
}

/** What a file too big for the reader's memory is told: `limit` the most it may use, in bytes */
export function tooBig(name: string, limit: number) {
  const size = limit >= 2 ** 30 ? `${+(limit / 2 ** 30).toFixed(1)} GB` : `${Math.round(limit / 2 ** 20)} MB`
  return `${name} is too big to import here: reading it needs more than the ${size} of memory its reader may use, so nothing is imported. A COCO file with fewer images reads in less.`
}

// Thrown when what's left of the heap can't take the next step, which is safer than finding out
class NoRoom extends Error {}

// JSON.parse can't be stopped halfway: a worker that runs out of memory inside it takes the whole
// app down, where one that runs out between steps is stopped on its own (Node's
// ERR_WORKER_OUT_OF_MEMORY). So nothing big is parsed in one go: the images and annotations a slice
// of a couple of megabytes at a time, and anything else only when the heap has room for it, about
// four times the text: the text, and what it parses to.
const SLICE = 2 * 2 ** 20
// The most the heap may hold for long-lived and big things: the worker's old generation, whose
// limit the main process sets (the young one is small, and holds no big strings); else the heap's
const oldLimit = () => (resourceLimits.maxOldGenerationSizeMb ?? 0) * 2 ** 20 || getHeapStatistics().heap_size_limit
function headroom() {
  if (!resourceLimits.maxOldGenerationSizeMb) return getHeapStatistics().total_available_size
  let used = 0
  for (const space of getHeapSpaceStatistics()) if (!space.space_name.startsWith('new_')) used += space.space_used_size
  return oldLimit() - used
}
function room(length: number) {
  if (headroom() < length * 4 + 32 * 2 ** 20) throw new NoRoom()
}
function parse(text: string): unknown {
  room(text.length)
  return JSON.parse(text)
}
/** readCoco on part of a file, when the heap has room for it */
function readPart(part: unknown, name: string) {
  const json = JSON.stringify(part)
  room(json.length)
  return readCoco(json, name)
}

// Where the JSON value starting at i ends: past its closing bracket or quote, or at the character
// after a number, true, false or null. Strings are skipped whole, escapes and all; -1 when the text
// ends inside it.
function valueEnd(text: string, i: number): number {
  let depth = 0
  for (let k = i; k < text.length; k++) {
    const c = text.charCodeAt(k)
    if (c === 34) {
      for (k++; k < text.length; k++) {
        const d = text.charCodeAt(k)
        if (d === 92) k++
        else if (d === 34) break
      }
      if (k >= text.length) return -1
      if (!depth) return k + 1
    } else if (c === 123 || c === 91) depth++
    else if (c === 125 || c === 93) {
      if (!depth) return k
      if (!--depth) return k + 1
    } else if (!depth && (c === 44 || c === 58 || c === 32 || c === 10 || c === 13 || c === 9)) return k
  }
  return depth ? -1 : text.length
}
const space = (text: string, i: number) => {
  while (i < text.length) {
    const c = text.charCodeAt(i)
    if (c !== 32 && c !== 10 && c !== 13 && c !== 9) break
    i++
  }
  return i
}

class Invalid extends Error {}

// A JSON array from text[i] ('['), parsed a slice of whole elements at a time
function sliced(text: string, i: number, out: unknown[]): number {
  let k = space(text, i + 1)
  if (text.charCodeAt(k) === 93) return k + 1
  let from = k
  for (;;) {
    const end = valueEnd(text, k)
    if (end < 0) throw new Invalid('it ends inside a list')
    k = space(text, end)
    const c = text.charCodeAt(k)
    if (c !== 44 && c !== 93) throw new Invalid(`there's ${k < text.length ? JSON.stringify(text[k]) : 'nothing'} at character ${k + 1}, where a comma or ] belongs`)
    if (c === 93 || k - from >= SLICE) {
      const items = parse(`[${text.slice(from, k)}]`) as unknown[]
      for (const item of items) out.push(item)
      from = k + 1
    }
    if (c === 93) return k + 1
    k = space(text, k + 1)
  }
}

/**
 * A COCO file's text as the object JSON.parse gives, but for keys other than images, annotations
 * and categories, which aren't read: a big file's lists are parsed a slice at a time. Throws
 * readCoco's own words when the text isn't JSON or isn't COCO.
 */
export function readTop(text: string, name: string): Record<string, unknown> {
  const invalid = (why: string) => new Error(`${name} isn't valid JSON (${why}), so nothing is imported`)
  let i = space(text, 0)
  if (text.charCodeAt(i) !== 123) {
    // Not an object: valid JSON of another kind, or not JSON at all
    try {
      parse(text)
    } catch (e) {
      throw invalid(e instanceof NoRoom ? "it doesn't start with {" : (e as Error).message)
    }
    throw new Error(`${name} isn't a COCO file: the top level must be an object, so nothing is imported`)
  }
  const out: Record<string, unknown> = {}
  try {
    i = space(text, i + 1)
    if (text.charCodeAt(i) === 125) i++
    else {
      for (;;) {
        if (text.charCodeAt(i) !== 34) throw new Invalid(`there's no key in quotes at character ${i + 1}`)
        const keyEnd = valueEnd(text, i)
        if (keyEnd < 0) throw new Invalid('it ends inside a key')
        const key = JSON.parse(text.slice(i, keyEnd)) as string
        i = space(text, keyEnd)
        if (text.charCodeAt(i) !== 58) throw new Invalid(`there's no colon after "${key}"`)
        i = space(text, i + 1)
        if ((key === 'images' || key === 'annotations') && text.charCodeAt(i) === 91) {
          const list: unknown[] = []
          i = sliced(text, i, list)
          out[key] = list
        } else {
          const end = valueEnd(text, i)
          if (end < 0 || end === i) throw new Invalid(`"${key}" has no value that ends`)
          // Only these are read: the rest, such as info and licenses, is passed over
          if (key === 'images' || key === 'annotations' || key === 'categories') out[key] = parse(text.slice(i, end))
          i = end
        }
        i = space(text, i)
        const c = text.charCodeAt(i)
        if (c === 125) { i++; break }
        if (c !== 44) throw new Invalid(`there's ${i < text.length ? JSON.stringify(text[i]) : 'nothing'} at character ${i + 1}, where a comma or } belongs`)
        i = space(text, i + 1)
      }
    }
    if (space(text, i) < text.length) throw new Invalid(`there's more after its end, at character ${space(text, i) + 1}`)
  } catch (e) {
    if (e instanceof Invalid) throw invalid(e.message)
    if (e instanceof SyntaxError) throw invalid(e.message)
    throw e
  }
  return out
}

/**
 * A COCO file's text, refused with a plain message when JavaScript can't hold it as one string
 * (512 MB of plain text)
 */
function readText(file: string): string {
  const name = path.basename(file)
  let size: number
  try {
    const info = statSync(file)
    if (!info.isFile()) throw new Error(`${name} isn't a file`)
    size = info.size
  } catch (e) {
    throw new Error((e as NodeJS.ErrnoException).code === 'ENOENT' ? `${name} isn't there` : (e as Error).message)
  }
  const max = constants.MAX_STRING_LENGTH
  if (size > max) throw new Error(`${name} is ${Math.round(size / 2 ** 20)} MB, more than the ${Math.floor(max / 2 ** 20)} MB Seamtag can read as one file, so nothing is imported`)
  // Text beyond Latin-1 takes two bytes a character
  if (headroom() < size * 2 + 32 * 2 ** 20) throw new NoRoom()
  return readFileSync(file, 'utf8')
}

/**
 * A COCO file being imported: read() parses and matches it and counts what each image would get;
 * shapes() reads the shapes of the images asked for. `chunk` is how many annotations are read at a time.
 */
export class CocoSession {
  private name = ''
  private categories: unknown[] = []
  // The images matched, and their annotations as the file has them
  private images = new Map<number, CocoImage>()
  private annotations = new Map<number, unknown[]>()

  constructor(private progress: CocoProgress = () => {}, private chunk = CHUNK) {}

  /** Reads the file, matches its images to the folder's (`root` the open folder's path) and counts what each image gets */
  read(file: string, folder: string[], root: string): CocoFound {
    this.name = path.basename(file)
    this.images.clear()
    this.annotations.clear()
    return this.roomy(() => this.readAll(file, folder, root))
  }

  /** The shapes of these images, as read() counted them */
  shapes(ids: number[]): CocoShapes[] {
    const out: CocoShapes[] = []
    return this.roomy(() => {
      this.each(ids.filter((id) => this.images.has(id)), (id, result) => out.push({ id, ...shapesOf(result) }))
      return out
    })
  }

  // A step the heap hadn't room for is a file too big, told plainly
  private roomy<T>(fn: () => T): T {
    try {
      return fn()
    } catch (e) {
      if (e instanceof NoRoom) throw new Error(tooBig(this.name, oldLimit()))
      throw e
    }
  }

  private readAll(file: string, folder: string[], root: string): CocoFound {
    this.progress(`Reading ${this.name}`, 0, 0)
    let text = readText(file)
    let raw: Record<string, unknown> | null = readTop(text, this.name)
    // The text can go once it's parsed: a big file is as big again as text
    text = ''
    if (!Array.isArray(raw.images)) throw new Error(`${this.name} isn't a COCO file: it has no images list, so nothing is imported`)
    this.categories = Array.isArray(raw.categories) ? raw.categories : []
    // The images and categories first, read as readCoco reads them, with what's wrong with them
    const head = readPart({ images: raw.images, categories: this.categories }, this.name)
    const issues: Issue[] = [...head.issues]
    this.progress('Matching images', 0, 0)
    const rel = path.relative(root, path.dirname(file))
    const at = rel.startsWith('..') || path.isAbsolute(rel) ? null : rel.split(path.sep).join('/')
    const base = startFolder(head.images, folder, at)
    const match = matchCocoImages(head.images, folder, base)
    issues.push(...match.issues)
    const known = new Set(head.images.map((im) => im.id))
    for (const im of head.images) if (match.names.has(im.id)) this.images.set(im.id, im)
    for (const id of this.images.keys()) this.annotations.set(id, [])
    // Annotations of images not matched are left out quietly: their images are listed as missing
    const strays: unknown[] = []
    const annotations: unknown[] = Array.isArray(raw.annotations) ? raw.annotations : []
    annotations.forEach((a, k) => {
      if (!isObject(a)) return issues.push({ severity: 'warning', code: 'coco-bad-annotation', image: null, message: `${this.name}: annotation ${k + 1} in its annotations list isn't an object, so it's left out` })
      const list = this.annotations.get(a.image_id as number)
      if (list) list.push(a)
      else if (!known.has(a.image_id as number)) strays.push(a)
    })
    raw = null
    // Each one of an image not in the images list is told, as readCoco tells them
    if (strays.length) issues.push(...readPart({ images: [], categories: this.categories, annotations: strays }, this.name).issues.filter((i) => i.code !== 'coco-bad-category'))
    const ids = [...this.images.keys()]
    const matched = new Map<number, CocoMatched>()
    const sums = new Map<string, number>()
    let done = 0, told = 0
    this.progress(`Reading ${this.name}'s annotations`, 0, ids.length)
    this.each(ids, (id, result, chunk) => {
      const im = this.images.get(id)!
      const { shapes } = shapesOf(result)
      const labels: Record<string, number> = {}
      for (const s of shapes) labels[s.label] = (labels[s.label] ?? 0) + 1
      const name = match.names.get(id)!
      matched.set(id, {
        id, name, fileName: im.file_name, width: im.width, height: im.height, byName: tidy(base ? `${base}/${im.file_name}` : im.file_name) !== name,
        annotations: this.annotations.get(id)!.length, shapes: shapes.length, labels
      })
      if (chunk) {
        for (const i of chunk.issues) {
          if (i.code === 'coco-bad-category') continue
          const sum = i.image === null && TALLIES.has(i.code) ? /^: (\d+) /.exec(i.message.slice(this.name.length)) : null
          if (sum) sums.set(i.code, (sums.get(i.code) ?? 0) + Number(sum[1]))
          else issues.push(i)
        }
      }
      // Ten times a second is enough for a progress bar
      if (++done === ids.length || Date.now() - told > 100) {
        told = Date.now()
        this.progress(`Reading ${this.name}'s annotations`, done, ids.length)
      }
    })
    for (const [code, n] of sums) issues.push({ severity: 'info', code, image: null, message: tallyMessage(code, this.name, n) })
    return { images: head.images.length, base, matched: ids.map((id) => matched.get(id)!), issues }
  }

  // Reads images a chunk of annotations at a time, as readCoco reads a whole file: each image's
  // annotations stay together and in order, so its shapes, groups and order come out the same.
  // `chunk` is passed with the first image of each chunk read.
  private each(ids: number[], fn: (id: number, result: CocoImageResult, chunk: { issues: Issue[] } | null) => void) {
    for (let at = 0; at < ids.length;) {
      const batch: number[] = []
      let count = 0
      while (at < ids.length && (!batch.length || count + this.annotations.get(ids[at])!.length <= this.chunk)) {
        count += this.annotations.get(ids[at])!.length
        batch.push(ids[at++])
      }
      const read = readPart({
        images: batch.map((id) => this.images.get(id)), categories: this.categories, annotations: batch.flatMap((id) => this.annotations.get(id)!)
      }, this.name)
      batch.forEach((id, k) => fn(id, read.results.get(id)!, k === 0 ? read : null))
    }
  }
}

// The worker: one session, asked by src/main/import.ts, which ends the worker when it's done
if (parentPort && workerData?.seamtagCoco) {
  const port = parentPort
  const session = new CocoSession((phase, done, total) => port.postMessage({ type: 'progress', phase, done, total } satisfies CocoAnswer))
  port.on('message', (ask: CocoAsk) => {
    try {
      port.postMessage(ask.type === 'read'
        ? { type: 'read', found: session.read(ask.file, ask.folder, ask.root) } satisfies CocoAnswer
        : { type: 'shapes', images: session.shapes(ask.ids) } satisfies CocoAnswer)
    } catch (e) {
      port.postMessage({ type: 'error', message: (e as Error).message } satisfies CocoAnswer)
    }
  })
}
