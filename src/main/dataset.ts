// A whole folder's labels, read against each image's real size for exporting, checking and
// counting them. Each image's own label file stays the working copy: nothing here writes a
// label file except rewriteLabels and rewriteList, which put the old one in the trash.
import { createHash } from 'crypto'
import { createReadStream, promises as fs, type Stats } from 'fs'
import path from 'path'
import type { Clash, OrphanFile } from '../shared/checks'
import type { DatasetImage, MaskBits } from '../shared/dataset'
import {
  boxOf, type LabelFormat, looksLikeCreateML, parseXml, pyRound, readClasses, readCreateML, readVoc, readYolo, writeClasses, writeCreateML,
  writeCreateMLEntries, writeVoc, writeYolo, type XmlElement, YOLO_CLASSES
} from '../shared/formats'
import { LabelFileError, parseLabelFile, serializeLabelFile, type Shape } from '../shared/labelfile'
import { summarize } from '../shared/labels'
import { imageSize } from './images'
import { decodeMask } from './masks'
import { nameKey } from './names'
import {
  findLabels, folderOf, type Found, type ImageItem, imagePath, isEmbedded, keypointsOf, labelPath, type Labels, nameOrder, opening, outputFileOf, readLabels, readText,
  serial, sharesStem, writeAtomic, writeTrashing
} from './project'

const posix = (p: string) => p.split(path.sep).join('/')
const stemOf = (name: string) => name.replace(/\.[^./]+$/, '')
const reason = (e: unknown) => (e instanceof Error ? e.message : String(e))
const turn = () => new Promise<void>((go) => setImmediate(go))
// Size and modified time can stay the same over a quick rewrite where file systems keep times to
// the second (HFS+) or two (FAT); the change time and inode still move, as they do when a new
// file takes an old one's name
const stamp = (s: Stats | null) => (s ? `${s.size}:${s.mtimeMs}:${s.ctimeMs}:${s.ino}` : '-')
// An image's change time moves too when it's only linked to again, as an export that hard-links
// images does, which changes nothing in it: its birth time and inode tell a new file instead
const imageStamp = (s: Stats | null) => (s ? `${s.size}:${s.mtimeMs}:${s.birthtimeMs}:${s.ino}` : '-')
const outside = (rel: string) => !rel || rel === '..' || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)
const shown = (root: string, file: string) => {
  const rel = path.relative(root, file)
  return outside(rel) ? file : posix(rel)
}
const digest = (text: string) => createHash('sha1').update(text).digest('hex')

// Windows and macOS find a.PNG by the name a.png, and IMG_1.json for img_1.jpg; macOS also finds a
// name in either Unicode form
const fold = nameKey

export type ReadOptions = {
  /** The format read first when an image has label files in two, as the window's format picker sets it */
  prefer?: LabelFormat
  /** Called as images are read: at most every 100 ms, and once at the end */
  onProgress?: (done: number, total: number) => void
  /** Stops the read, which then rejects with the signal's reason. What was read stays cached. */
  signal?: AbortSignal
  /**
   * Only these images are read again, as after the window saved their labels: the folder's
   * others are as the last read found them, their files not looked at. Images never read are read.
   */
  only?: ReadonlySet<string>
}

type Cached = { key: string; image: DatasetImage }
/** A CreateML list as one read gives it to all its images: each image's entries as text, by the name they give it, or why it can't be read */
type List = { entries: Map<string, string>; error?: string }
type Lists = Map<string, { stamp: string; list: Promise<List> }>
// By folder, as this opening of it (project.ts' opening), so a read still going when the folder
// closes keeps nothing of it once it ends, not even the reads that failed as the folder closed:
// images by name, and CreateML lists by file. A re-run reads only label files and images whose
// size or times changed, and of a list that every image shares, only the images whose own
// entries changed.
const caches = new WeakMap<object, { images: Map<string, Cached>; lists: Lists }>()

/** Drops what was read of a folder, to read it afresh; closing it drops it too */
export function forgetDataset(root: string) {
  const opened = opening(root)
  if (opened) caches.delete(opened)
}

/** One readDataset: what every image of it shares */
type Run = {
  root: string
  prefer: LabelFormat
  outputFile: string | null
  embedded: Map<string, string>
  lists: Lists
  /** The CreateML lists this run read, which the cache keeps */
  used: Set<string>
  /** One -O labels.json in LabelMe's format holds every image's labels: parsed once, by its stamp */
  docs: Map<string, Promise<Labels | null>>
}

/**
 * Every image of an open folder with its labels, read as the window reads them (findLabels,
 * then readLabels) but against the image's real upright size from its header: never through
 * openImage, whose tile pyramids would thrash over a folder. Eight at a time, as the
 * summaries are, and in the folder's list order. A CreateML list, or the one -O labels.json
 * every image shares, is read once a run, not once an image.
 */
export async function readDataset(root: string, { prefer = 'labelme', onProgress, signal, only }: ReadOptions = {}): Promise<DatasetImage[]> {
  const folder = folderOf(root)
  const names = folder.images
  const opened = opening(root)!
  const old = caches.get(opened)
  const run: Run = {
    root, prefer, outputFile: folder.outputFile, embedded: folder.embedded, lists: new Map(old?.lists), used: new Set(), docs: new Map()
  }
  const fresh = new Map<string, Cached>()
  const out: DatasetImage[] = new Array(names.length)
  let next = 0, done = 0, told = 0, kept = 0
  await Promise.all(Array.from({ length: 8 }, async () => {
    while (next < names.length && !signal?.aborted) {
      const at = next++
      const name = names[at]
      const cached = old?.images.get(name)
      let read: Cached
      if (only && cached && !only.has(name)) {
        read = cached
        // Taken as they are, the rest of a folder of 100,000 would go by in one stretch: the window gets a turn now and then
        if (++kept % 2000 === 0) await turn()
      } else read = await readImage(run, name, cached)
      fresh.set(name, read)
      out[at] = read.image
      done++
      if (onProgress && (done === names.length || Date.now() - told >= 100)) {
        told = Date.now()
        onProgress(done, names.length)
      }
    }
  }))
  // A read of a few images keeps every list: the images it didn't look at still read theirs
  const lists: Lists = only ? run.lists : new Map([...run.lists].filter(([file]) => run.used.has(file)))
  if (signal?.aborted) {
    caches.set(opened, { images: new Map([...(old?.images ?? []), ...fresh]), lists: run.lists })
    signal.throwIfAborted()
  }
  // Images gone from the folder drop out of the cache, and lists no image read
  caches.set(opened, { images: fresh, lists })
  return out
}

/** A CreateML list, read once for every image in it however many there are, and again only once it changes */
function listOf(lists: Lists, file: string, at: string): Promise<List> {
  const known = lists.get(file)
  if (known?.stamp === at) return known.list
  const list = readText(file).then((text) => indexed(text ?? '[]'), (e): List => ({ entries: new Map(), error: reason(e) }))
  lists.set(file, { stamp: at, list })
  return list
}

/** A CreateML list's entries grouped by the image they name, each group as text that readCreateML reads as it reads the whole list */
function indexed(text: string): List {
  try {
    // Every entry is checked whatever image it names: one that's wrong makes the list unreadable for all, as it is image by image
    readCreateML(text, '\0')
  } catch (e) {
    return { entries: new Map(), error: reason(e) }
  }
  const groups = new Map<string, unknown[]>()
  for (const entry of JSON.parse(text) as { image: string }[]) {
    const group = groups.get(entry.image)
    if (group) group.push(entry)
    else groups.set(entry.image, [entry])
  }
  return { entries: new Map([...groups].map(([image, group]) => [image, JSON.stringify(group)])) }
}

/** The name an image has in a CreateML list: its path in the folder in one -O file for all, its file name in its own */
const entryName = (outputFile: string | null, name: string) => (outputFile ? name : path.posix.basename(name))

/**
 * What an image's labels were read from, to know when they've changed: the file's size and
 * times, with classes.txt's for YOLO, whose names come from it, or for a CreateML list, the
 * image's own entries in it, so a change to another image's entries leaves this one be
 */
const versionOf = (format: LabelFormat, file: Stats | null, classes: Stats | null, entries: string | undefined) =>
  format === 'createml' ? `entries:${entries === undefined ? '-' : digest(entries)}` : format === 'yolo' ? `${stamp(file)}|${stamp(classes)}` : stamp(file)

const once = <T>(memo: Map<string, Promise<T>>, key: string, make: () => Promise<T>) => {
  let made = memo.get(key)
  if (!made) memo.set(key, (made = make()))
  return made
}

/** An image's labels as one read gives them */
type Read = Found & { shapes: Shape[]; flags: Record<string, boolean>; verified: boolean; labelSize: [number, number] | null; labelImage?: string | null }

async function readImage(run: Run, name: string, cached?: Cached): Promise<Cached> {
  const { root } = run
  const file = imagePath(root, name)
  let found: Found | null = null
  let broken: string | undefined
  try {
    found = await findLabels(root, name, run.prefer)
  } catch (e) {
    // A .json that's there but won't open: findLabels reads its first bytes to tell LabelMe from CreateML
    found = { format: 'labelme', file: labelPath(root, name) }
    broken = `It can't be read (${(e as NodeJS.ErrnoException).code ?? reason(e)})`
  }
  const [labelStat, classesStat, imageStat] = await Promise.all([
    found ? fs.stat(found.file).catch(() => null) : null,
    found?.format === 'yolo' ? fs.stat(path.join(path.dirname(found.file), YOLO_CLASSES)).catch(() => null) : null,
    fs.stat(file).catch(() => null)
  ])
  let list: List | null = null
  if (found?.format === 'createml' && labelStat && !broken) {
    run.used.add(found.file)
    list = await listOf(run.lists, found.file, stamp(labelStat))
  }
  const entry = entryName(run.outputFile, name)
  const entries = list?.entries.get(entry)
  const version = !found || !labelStat ? '-' : list?.error ? `${stamp(labelStat)}!` : versionOf(found.format, labelStat, classesStat, entries)
  const key = [found?.format, found?.file, version, imageStamp(imageStat)].join('\0')
  if (cached?.key === key) return cached
  let width = 0, height = 0, unreadable: string | undefined
  try {
    ;[width, height] = await imageSize(root, name)
  } catch (e) {
    unreadable = reason(e)
  }
  let read: Read | null = null
  if (found && labelStat && !broken) {
    try {
      read = list ? listed(list, entry, found) : await fromFile(run, name, found, version, width, height, !!unreadable)
    } catch (e) {
      broken = reason(e)
    }
  }
  const shapes = read?.shapes ?? []
  const image: DatasetImage = {
    name, width, height,
    file: read?.file ?? (broken && found ? found.file : null),
    format: read?.format ?? (broken && found ? found.format : null),
    shapes,
    // One mask that can't be placed is one mask left out, never the image
    masks: await Promise.all(shapes.map((s) => (s.shape_type === 'mask' ? placeMask(s, width, height).catch(() => null) : null))),
    flags: read?.flags ?? {},
    verified: read?.verified ?? false,
    ...(broken ? { broken } : {}),
    ...(unreadable ? { unreadable } : {}),
    labelSize: read?.labelSize ?? null,
    ...(read?.labelImage !== undefined ? { labelImage: read.labelImage } : {}),
    ...(read ? { stamp: version } : {}),
    ...(!run.embedded.has(name) && imageStat ? { bytes: imageStat.size } : {})
  }
  return { key, image }
}

/** An image's entries in a CreateML list; null when the list names it nowhere, so it holds no labels of it */
function listed(list: List, entry: string, found: Found): Read | null {
  if (list.error) throw new LabelFileError(list.error)
  const text = list.entries.get(entry)
  if (text === undefined) return null
  const { shapes, verified } = readCreateML(text, entry)
  return { ...found, shapes, flags: {}, verified, labelSize: null }
}

async function fromFile(run: Run, name: string, found: Found, version: string, width: number, height: number, unreadable: boolean): Promise<Read | null> {
  // An image that can't be read gives YOLO's shares a 1 x 1 frame, as the summaries read them
  const read = () => readLabels(run.root, name, found, async () => (unreadable ? [1, 1] : [width, height]))
  // A LabelMe file reads the same for every image that has it. Only the one all of them share
  // is kept for the run: others can hold megabytes of imageData each.
  const labels = found.format === 'labelme' && found.file === run.outputFile ? await once(run.docs, `${found.file}\0${version}`, read) : await read()
  if (!labels) return null
  return {
    file: labels.file, format: labels.format, shapes: [...labels.doc.shapes], flags: labels.doc.flags, verified: labels.verified,
    labelSize: await sizeIn(labels, width, height), ...imageNamed(run.root, name, labels)
  }
}

/**
 * A mask shape's pixels where labelme 7.9.1's shapes_to_label pastes them
 * (examples/utils.py): at its first corner rounded half to even, cropped to the box between
 * its corners when both are whole pixels, and clipped to the image, so only what the image
 * holds is kept or counted. An image of unknown size clips at the mask's own pixels. A mask
 * wholly outside keeps no pixels. Null for a mask with no pixels that decode.
 */
async function placeMask(s: Shape, width: number, height: number): Promise<MaskBits | null> {
  if (!s.mask || !s.points.length) return null
  const m = await decodeMask(s.mask).catch(() => null)
  if (!m) return null
  const [x, y] = s.points[0].map(pyRound)
  const whole = s.points.length >= 2 && s.points.flat().every(Number.isInteger)
  const w = whole ? s.points[1][0] - x + 1 : m.width, h = whole ? s.points[1][1] - y + 1 : m.height
  const [right, bottom] = width > 0 && height > 0 ? [width, height] : [x + m.width, y + m.height]
  // labelme 7.9.1 pastes rows max(y1, 0) to min(y1 + patch height, height), and columns the same way
  const x0 = Math.max(x, 0), y0 = Math.max(y, 0), x1 = Math.min(x + w, right), y1 = Math.min(y + h, bottom)
  if (x0 >= x1 || y0 >= y1) return new PackedMask(Math.min(x0, right), Math.min(y0, bottom), 0, 0, new Uint8Array(0))
  const data = new Uint8Array((x1 - x0) * (y1 - y0))
  // labelme can't paste a mask smaller than its box at all: past its pixels, the box is left empty here
  const c0 = x0 - x, c1 = Math.min(x1, x + m.width) - x
  for (let row = y0; row < Math.min(y1, y + m.height) && c1 > c0; row++) {
    const from = (row - y) * m.width
    data.set(m.data.subarray(from + c0, from + c1), (row - y0) * (x1 - x0))
  }
  return new PackedMask(x0, y0, x1 - x0, y1 - y0, data)
}

/**
 * A mask's pixels, 0s and 1s as decodeMask gives them, as the lengths of their runs along its
 * rows, in turn, starting with 0s: a few hundred numbers for an object's mask of a hundred
 * thousand pixels. A mask so speckled that its runs would take more room keeps its pixels.
 */
function pack(data: Uint8Array): Uint32Array | Uint8Array {
  const runs: number[] = []
  for (let from = 0, value = 0; from < data.length; value ^= 1) {
    const to = data.indexOf(value ^ 1, from)
    runs.push((to < 0 ? data.length : to) - from)
    from = to < 0 ? data.length : to
  }
  return runs.length * 4 < data.length ? Uint32Array.from(runs) : data
}

function unpack(runs: Uint32Array, size: number) {
  const data = new Uint8Array(size)
  for (let k = 0, at = 0; k < runs.length; at += runs[k++]) if (k % 2) data.fill(1, at, at + runs[k])
  return data
}

// Pixels kept unpacked at once, past the two masks unpacked last
const UNPACKED = 64 * 1024 * 1024

/**
 * A mask shape's pixels kept packed (pack), and unpacked when `data` is read: a byte a pixel, the
 * masks of a folder of segmentation labels would come to gigabytes, held for as long as the
 * folder is open. The masks unpacked last keep their pixels until the event loop's next turn,
 * so code that reads a mask pixel by pixel unpacks it once, and none stay unpacked between the
 * slices of a pass. `data` is each mask's own property, as it is a plain MaskBits', so a copy
 * made by spreading one has the pixels.
 */
class PackedMask implements MaskBits {
  declare readonly data: Uint8Array
  /** Its pixels' runs; null for a mask kept as its pixels */
  #runs: Uint32Array | null = null
  /** Its pixels, while they're unpacked */
  #pixels: Uint8Array | null = null

  constructor(readonly x: number, readonly y: number, readonly width: number, readonly height: number, data: Uint8Array) {
    const packed = pack(data)
    if (packed instanceof Uint32Array) this.#runs = packed
    else this.#pixels = packed
    Object.defineProperty(this, 'data', { enumerable: true, get: PackedMask.#data })
  }

  // The masks unpacked since the last turn, oldest first, and their pixels in all
  static #recent: PackedMask[] = []
  static #bytes = 0

  // Read at each pixel by code that goes through them one by one: pixels unpacked come first
  static #data = function (this: PackedMask): Uint8Array {
    return this.#pixels ?? this.#unpack()
  }

  #unpack() {
    const recent = PackedMask.#recent
    if (!recent.length) {
      setImmediate(() => {
        for (const m of recent) m.#pixels = null
        recent.length = 0
        PackedMask.#bytes = 0
      })
    }
    const pixels = (this.#pixels = unpack(this.#runs!, this.width * this.height))
    recent.push(this)
    PackedMask.#bytes += pixels.length
    while (PackedMask.#bytes > UNPACKED && recent.length > 2) {
      const old = recent.shift()!
      PackedMask.#bytes -= old.#pixels!.length
      old.#pixels = null
    }
    return pixels
  }
}

const childOf = (e: XmlElement | undefined, name: string) => e?.children.find((c) => c.name === name)

/** Pascal VOC's <size>: LabelImg always writes it; a 0 is a tool that didn't know, and says nothing */
function vocSize(xml: XmlElement): [number, number] | null {
  const size = childOf(xml, 'size')
  const side = (tag: string) => Number(childOf(size, tag)?.text.trim() || NaN)
  const width = side('width'), height = side('height')
  return width > 0 && height > 0 ? [width, height] : null
}

/**
 * The size a label file gives: LabelMe's imageWidth and imageHeight, or Pascal VOC's <size>.
 * labelme 7.9.1 checks each side on its own (_check_image_dimensions), so a side left out
 * is taken as the image's own and can't differ.
 */
async function sizeIn(labels: Labels, width: number, height: number): Promise<[number, number] | null> {
  if (labels.format === 'labelme') {
    const { imageWidth: w, imageHeight: h } = labels.doc
    return w === null && h === null ? null : [w ?? width, h ?? height]
  }
  if (labels.format !== 'voc') return null
  const text = await readText(labels.file).catch(() => null)
  try {
    return text === null ? null : vocSize(parseXml(text))
  } catch {
    return null
  }
}

/**
 * The image a LabelMe file names in imagePath, as a path in the folder; null for one outside
 * it. A file that holds its own image says nothing: labelme shows its imageData, not imagePath.
 */
function imageNamed(root: string, name: string, labels: Labels): { labelImage?: string | null } {
  const { imagePath: named, imageData } = labels.doc
  if (labels.format !== 'labelme' || !named || imageData) return {}
  const rel = path.relative(root, path.resolve(path.dirname(labels.file), named))
  if (outside(rel)) return { labelImage: null }
  const image = posix(rel)
  return { labelImage: fold(image) === fold(name) ? name : image }
}

/**
 * Images whose label file is one and the same file: a.jpg and a.png beside one a.json or
 * a.xml (and on Windows and macOS, a.jpg and A.png), or every image in one -O labels.json.
 * A CreateML list keeps each image's entry apart, so it never counts.
 */
export function labelFileClashes(images: DatasetImage[]): Clash[] {
  const byFile = new Map<string, Clash>()
  for (const i of images) {
    if (!i.file || i.format === 'createml') continue
    const clash = byFile.get(fold(i.file))
    if (clash) clash.images.push(i.name)
    else byFile.set(fold(i.file), { file: i.file, images: [i.name] })
  }
  return [...byFile.values()].filter((c) => c.images.length > 1)
}

// sha256 by file, size and times: a re-run hashes only new and changed files
const hashes = new Map<string, { stamp: string; hash: string }>()

async function hashFile(file: string) {
  const at = imageStamp(await fs.stat(file))
  const known = hashes.get(file)
  if (known?.stamp === at) return known.hash
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  const hex = hash.digest('hex')
  hashes.set(file, { stamp: at, hash: hex })
  return hex
}

/** Adds a name to its group, made when it's the first */
function grouped<K>(groups: Map<K, string[]>, key: K, name: string) {
  const group = groups.get(key)
  if (group) group.push(name)
  else groups.set(key, [name])
}

/**
 * Image files with the same bytes, in groups in the folder's order. Only files that share a
 * byte size are read, as Datumaro and fiftyone find repeated images, so most never are. A
 * camera's uncompressed frames all share one: progress is told as files are read.
 */
export async function findDuplicateImages(root: string, images: DatasetImage[], { signal, onProgress }: {
  signal?: AbortSignal
  /** Called as files are read, with how many there are to read */
  onProgress?: (done: number, total: number) => void
} = {}): Promise<string[][]> {
  const bySize = new Map<number, string[]>()
  // An empty file can't be read as an image, which is reported on its own
  for (const i of images) if (i.bytes) grouped(bySize, i.bytes, i.name)
  const todo = [...bySize].flatMap(([bytes, names]) => (names.length > 1 ? names.map((name) => [name, bytes] as const) : []))
  const byHash = new Map<string, string[]>()
  let next = 0, done = 0
  onProgress?.(0, todo.length)
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (next < todo.length && !signal?.aborted) {
      const [name, bytes] = todo[next++]
      const hash = await hashFile(imagePath(root, name)).catch(() => null)
      if (hash) grouped(byHash, `${bytes}:${hash}`, name)
      onProgress?.(++done, todo.length)
    }
  }))
  signal?.throwIfAborted()
  const groups = [...byHash.values()].filter((g) => g.length > 1)
  if (!groups.length) return []
  const order = new Map(images.map((i, k) => [i.name, k]))
  const at = (name: string) => order.get(name)!
  return groups.map((g) => g.sort((a, b) => at(a) - at(b))).sort((a, b) => at(a[0]) - at(b[0]))
}

/** Goes through a folder's worth of items a few thousand at a time, with a turn for the window between */
async function inSlices<T>(items: T[], each: (item: T) => void) {
  for (let at = 0; at < items.length; at += 5000) {
    if (at) await turn()
    for (let k = at; k < Math.min(at + 5000, items.length); k++) each(items[k])
  }
}

const LABEL_EXTS = new Set(['.json', '.xml', '.txt'])
// A COCO file can run to hundreds of megabytes; one image's label file, even with imageData, doesn't
const MAX_LABEL_FILE = 64 * 1024 * 1024

/**
 * Files that may be labels under a folder, by path in it, in the order the disk keeps them;
 * hidden folders are skipped, as the image list skips them
 */
async function filesUnder(dir: string, skip: string[], signal?: AbortSignal): Promise<string[]> {
  const found: string[] = []
  const skipped = new Set(skip.map(fold))
  let seen = 0
  const walk = async (sub: string) => {
    const full = path.join(dir, sub)
    if (skipped.has(fold(full))) return
    signal?.throwIfAborted()
    let entries
    try {
      entries = await fs.readdir(full, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      // A folder of 100,000 images lists 200,000 files: the window gets a turn now and then
      if (++seen % 5000 === 0) await turn()
      if (e.name.startsWith('.')) continue
      const rel = sub ? `${sub}/${e.name}` : e.name
      if (e.isDirectory()) await walk(rel)
      else if (LABEL_EXTS.has(path.extname(e.name).toLowerCase())) found.push(rel)
    }
  }
  await walk('')
  return found
}

/**
 * Is this a label file? LabelMe JSON, CreateML, Pascal VOC, or a .txt that's YOLO's because
 * classes.txt sits beside it. Anything else, such as a COCO file, a caption or settings,
 * isn't. With the image a LabelMe file names, when that's in the folder, and whether it holds
 * its own image: labelme opens such a file on its own, but no image of the folder shows it.
 */
async function labelFileKind(root: string, file: string): Promise<Omit<OrphanFile, 'file'> | null> {
  const ext = path.extname(file).toLowerCase()
  let classes: string[] = []
  if (ext === '.txt') {
    if (path.basename(file) === YOLO_CLASSES) return null
    const text = await readText(path.join(path.dirname(file), YOLO_CLASSES)).catch(() => null)
    if (text === null) return null
    classes = text.trim() ? readClasses(text) : []
  }
  const stat = await fs.stat(file).catch(() => null)
  if (!stat || stat.size > MAX_LABEL_FILE) return null
  const text = await readText(file).catch(() => null)
  if (text === null) return null
  try {
    if (ext === '.xml') return parseXml(text).name === 'annotation' ? { names: null } : null
    // Each throws on what isn't a label file of its kind
    if (ext === '.txt') readYolo(text, classes, 1, 1)
    else if (looksLikeCreateML(text)) readCreateML(text, '')
    if (ext === '.txt' || looksLikeCreateML(text)) return { names: null }
    const doc = parseLabelFile(text)
    const rel = doc.imagePath ? path.relative(root, path.resolve(path.dirname(file), doc.imagePath)) : ''
    return { names: outside(rel) ? null : posix(rel), ...(doc.imageData ? { holdsImage: true } : {}) }
  } catch {
    return null
  }
}

/** Is this .json a CreateML list? Its first characters say, as findLabels tells them apart. */
async function isList(file: string) {
  const handle = await fs.open(file, 'r')
  try {
    const head = Buffer.alloc(256)
    const { bytesRead } = await handle.read(head, 0, head.length, 0)
    return looksLikeCreateML(head.subarray(0, bytesRead).toString('utf8'))
  } finally {
    await handle.close()
  }
}

/**
 * Label files no image reads, beside the images or in the -O folder: no image of the same
 * name sits where they'd belong, by labelPath's rules (folders mirrored, or flat in the output
 * folder as labelme and LabelImg write them). A LabelMe file that holds its own image is one
 * too: it opens on its own, but never with the folder, nor goes into an export. CreateML lists
 * that do belong to images are looked inside, for entries no image reads. `images` is what
 * readDataset gave, which says which label files are lists. Shown by path in the folder, or
 * in full when outside it, and in the image list's order.
 */
export async function orphanLabels(root: string, images: DatasetImage[], { signal, onProgress }: {
  signal?: AbortSignal
  /** Called as the label files found are looked at, with how many there are */
  onProgress?: (done: number, total: number) => void
} = {}): Promise<OrphanFile[]> {
  const folder = folderOf(root)
  const stems = new Set<string>(), bases = new Set<string>()
  await inSlices(folder.images, (n) => {
    stems.add(fold(stemOf(n)))
    bases.add(fold(stemOf(path.posix.basename(n))))
  })
  const out = folder.outputDir ?? (folder.outputFile ? path.dirname(folder.outputFile) : null)
  const within = (dir: string, parent: string) => fold(dir).startsWith(fold(parent) + path.sep)
  const places = [{ dir: root, flat: !!out && fold(out) === fold(root), skip: out && within(out, root) ? [out] : [] }]
  if (out && fold(out) !== fold(root)) places.push({ dir: out, flat: true, skip: within(root, out) ? [root] : [] })
  const embedded = new Set([...folder.embedded.values()].map(fold))
  const outputFile = folder.outputFile && fold(folder.outputFile)
  const readAs = new Map<string, LabelFormat>()
  await inSlices(images, (i) => {
    if (i.file && i.format) readAs.set(fold(i.file), i.format)
  })
  const listed: { dir: string; flat: boolean; files: string[] }[] = []
  for (const { dir, flat, skip } of places) listed.push({ dir, flat, files: await filesUnder(dir, skip, signal) })
  const total = listed.reduce((n, place) => n + place.files.length, 0)
  let done = 0
  const orphans: OrphanFile[] = []
  const lists = new Map<string, string>()
  for (const { dir, flat, files } of listed) {
    const found: { rel: string; orphan: OrphanFile }[] = []
    for (const rel of files) {
      // Before each file: one that isn't an image's own is read through
      signal?.throwIfAborted()
      if (++done % 1000 === 0) {
        onProgress?.(done, total)
        await turn()
      }
      const file = path.join(dir, rel)
      const key = fold(file), stem = fold(stemOf(rel))
      // The image open from inside its own label file
      if (embedded.has(key)) continue
      if (key === outputFile || stems.has(stem) || (flat && !rel.includes('/') && bases.has(stem))) {
        // An image's label file: a CreateML list in it can still hold other images' labels
        const format = readAs.get(key)
        if (path.extname(rel).toLowerCase() === '.json' && (format === 'createml' || (!format && await isList(file).catch(() => false)))) lists.set(key, file)
        continue
      }
      const kind = await labelFileKind(root, file)
      if (kind) found.push({ rel, orphan: { file: shown(root, file), ...kind } })
    }
    // Sorted once found, as the image list is, not as the files were listed: there are few
    for (const { orphan } of found.sort((a, b) => nameOrder(a.rel, b.rel))) orphans.push(orphan)
  }
  onProgress?.(total, total)
  if (lists.size) orphans.push(...await strayEntries(root, lists))
  return orphans
}

/** Entries of CreateML lists that no image reads: an image reads only those under its own name, from the one file labelPath gives it */
async function strayEntries(root: string, lists: Map<string, string>): Promise<OrphanFile[]> {
  const { images, outputFile } = folderOf(root)
  const readers = new Map<string, Set<string>>()
  for (const name of images) {
    const key = fold(labelPath(root, name, 'createml'))
    if (lists.has(key)) readers.set(key, (readers.get(key) ?? new Set()).add(entryName(outputFile, name)))
  }
  const opened = opening(root)
  const cache = (opened && caches.get(opened)?.lists) || new Map()
  const stray: OrphanFile[] = []
  for (const [key, file] of lists) {
    const at = await fs.stat(file).catch(() => null)
    if (!at) continue
    const list = await listOf(cache, file, stamp(at))
    const read = readers.get(key) ?? new Set()
    const entries = [...list.entries.keys()].filter((e) => !read.has(e))
    if (entries.length) stray.push({ file: shown(root, file), names: null, entries })
  }
  return stray.sort((a, b) => nameOrder(a.file, b.file))
}

const unreadable = (file: string, e: unknown, as = '') =>
  new Error(`${path.basename(file)} can't be read${as}, so it's left unchanged: ${reason(e)}`)

/** Another image whose labels go to the same file: a.jpg and a.png, or one -O labels.json for all */
function sharesFile(root: string, name: string, found: Found) {
  const outputFile = outputFileOf(root)
  const stem = fold(stemOf(name)), file = fold(found.file)
  const all = !!outputFile && fold(outputFile) === file
  // Most images have a name of their own: renaming a label in all of a big folder's files would
  // otherwise go through the whole folder once a file
  if (!all && !sharesStem(root, name)) return false
  return folderOf(root).images.some((other) => other !== name && (all || fold(stemOf(other)) === stem) && fold(labelPath(root, other, found.format)) === file)
}

/** New shapes, or a change to the shapes the file holds when it's written */
export type ShapeChange = Shape[] | ((current: Shape[]) => Shape[])

// Which of the file's shapes a changed one was made from: spreading a shape, as renaming and
// the fixes do, keeps it
const ORIGIN = Symbol('origin')
const samePoints = (a: number[][], b: number[][]) => a.length === b.length && a.every((p, k) => p[0] === b[k][0] && p[1] === b[k][1])
const sameShape = (a: Shape, b: Shape) => a.label === b.label && a.shape_type === b.shape_type && samePoints(a.points, b.points) &&
  a.group_id === b.group_id && a.description === b.description && a.mask === b.mask && JSON.stringify(a.flags) === JSON.stringify(b.flags)

/**
 * The new shapes, and which of the file's shapes each one is, so what the file keeps of it
 * beyond a Shape (a VOC object's pose, say) can stay: for a change function, the shape it was
 * made from; for a list, the shape in its place when it's as long as the file's, otherwise
 * the first one left that it equals. Null for a new shape.
 */
function applied(change: ShapeChange, before: Shape[]): { shapes: Shape[]; origins: (number | null)[] } {
  if (Array.isArray(change)) {
    if (change.length === before.length) return { shapes: change, origins: change.map((_, k) => k) }
    let next = 0
    const origins = change.map((s) => {
      for (let k = next; k < before.length; k++) {
        if (!sameShape(before[k], s)) continue
        next = k + 1
        return k
      }
      return null
    })
    return { shapes: change, origins }
  }
  const shapes = change(before.map((s, k) => ({ ...s, [ORIGIN]: k })))
  const used = new Set<number>()
  const origins = shapes.map((s) => {
    const k = (s as Shape & { [ORIGIN]?: number })[ORIGIN]
    if (k === undefined || used.has(k)) return null
    used.add(k)
    return k
  })
  return { shapes, origins }
}

/** An element's place in an XML file: its start tag from `start`, what's inside from `open` to `close`, its end tag to `end` */
type Span = { name: string; start: number; open: number; close: number; end: number; children: Span[] }
const selfClosed = (e: Span) => e.close === e.end
const spanOf = (e: Span, name: string) => e.children.find((c) => c.name === name)

/**
 * Where each element of an XML file sits, found as formats.ts' parseXml finds them (comments,
 * CDATA, declarations, a DOCTYPE, quoted attributes), so its elements are the ones readVoc read
 */
function spans(text: string): Span {
  const stack: Span[] = []
  let root: Span | null = null
  let i = 0
  const skipTo = (end: string, from: number) => {
    const at = text.indexOf(end, from)
    if (at < 0) throw new LabelFileError("It isn't valid XML")
    return at + end.length
  }
  while (i < text.length) {
    const lt = text.indexOf('<', i)
    if (lt < 0) break
    if (text.startsWith('<!--', lt)) i = skipTo('-->', lt + 4)
    else if (text.startsWith('<![CDATA[', lt)) i = skipTo(']]>', lt + 9)
    else if (text.startsWith('<?', lt)) i = skipTo('?>', lt + 2)
    else if (text.startsWith('<!', lt)) {
      let j = lt + 2, depth = 0
      for (; j < text.length; j++) {
        if (text[j] === '[') depth++
        else if (text[j] === ']') depth--
        else if (text[j] === '>' && depth <= 0) break
      }
      i = j + 1
    } else {
      // The tag ends at the first ">" outside quotes
      let gt = lt + 1, quote = ''
      for (; gt < text.length; gt++) {
        const c = text[gt]
        if (quote) {
          if (c === quote) quote = ''
        } else if (c === '"' || c === "'") quote = c
        else if (c === '>') break
      }
      if (gt >= text.length) throw new LabelFileError("It isn't valid XML")
      const body = text.slice(lt + 1, gt)
      i = gt + 1
      if (body.startsWith('/')) {
        const open = stack.pop()
        if (!open) throw new LabelFileError("It isn't valid XML")
        open.close = lt
        open.end = i
        continue
      }
      const closed = body.endsWith('/')
      const span: Span = { name: /^[^\s/>]+/.exec(closed ? body.slice(0, -1) : body)?.[0] ?? '', start: lt, open: i, close: i, end: i, children: [] }
      if (stack.length) stack[stack.length - 1].children.push(span)
      else root ??= span
      if (!closed) stack.push(span)
    }
  }
  if (!root) throw new LabelFileError("It isn't valid XML")
  return root
}

type Edit = { from: number; to: number; text: string }
/** The text with each edit's span replaced. Edits don't overlap; at one place, what's put in goes before what's taken out. */
function edited(text: string, edits: Edit[]) {
  let out = '', at = 0
  for (const e of [...edits].sort((a, b) => a.from - b.from || a.to - b.to)) {
    out += text.slice(at, e.from) + e.text
    at = e.to
  }
  return out + text.slice(at)
}

const isDifficult = (s: Shape) => s.flags.difficult === true
const escapeXml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** A new object as LabelImg 1.8.6 writes it (formats.ts' writeVoc): whole pixels, never 0 at the left or top */
function labelImgObject(s: Shape, [width, height]: [number, number]) {
  const lines = writeVoc({ folder: '', filename: '', path: '', width, height, depth: 3 }, [s], false).split('\n')
  return lines.slice(lines.indexOf('\t<object>'), lines.lastIndexOf('\t</object>') + 1).join('\n')
}

/**
 * Pascal VOC changed in place. Each object that stays keeps its text as it is, with its
 * <pose>, <truncated>, <occluded>, CVAT's <attributes>, VOC2012's <part> and <actions>, and
 * whatever else the tool that wrote it put there; only its <name>, <difficult> and <bndbox>
 * numbers are rewritten, where they changed, and a moved box takes its numbers exactly rather
 * than in LabelImg's whole pixels. Objects left out are taken out with their line, and new
 * ones are written as LabelImg writes them, after the shape before them. The rest of the
 * file, <source> and <size> too, stays as it was. `size` is the image's, for new objects.
 */
function vocInPlace(base: string, text: string, before: Shape[], verified: boolean, after: Shape[], origins: (number | null)[], size: [number, number] | null) {
  const root = spans(text)
  const objects = root.children.filter((c) => c.name === 'object')
  const edits: Edit[] = []
  // Back over the whitespace before a place: an object taken out takes its line with it
  const lineStart = (at: number) => {
    while (at > 0 && /\s/.test(text[at - 1])) at--
    return at
  }
  const replace = (e: Span, inner: string) =>
    edits.push(selfClosed(e) ? { from: e.start, to: e.end, text: `<${e.name}>${inner}</${e.name}>` } : { from: e.open, to: e.close, text: inner })
  const kept = new Map<number, number>()
  origins.forEach((k, i) => {
    if (k !== null) kept.set(k, i)
  })
  objects.forEach((o, k) => {
    const i = kept.get(k)
    if (i === undefined) {
      edits.push({ from: lineStart(o.start), to: o.end, text: '' })
      return
    }
    const s = after[i], was = before[k]
    if (s.label !== was.label) replace(spanOf(o, 'name')!, escapeXml(s.label))
    if (isDifficult(s) !== isDifficult(was)) {
      const difficult = spanOf(o, 'difficult')
      const box = spanOf(o, 'bndbox')!
      if (difficult) replace(difficult, isDifficult(s) ? '1' : '0')
      // Indented as the <bndbox> it goes before
      else edits.push({ from: box.start, to: box.start, text: `<difficult>1</difficult>${text.slice(lineStart(box.start), box.start)}` })
    }
    if (!samePoints(s.points, was.points)) {
      const box = spanOf(o, 'bndbox')!
      boxOf(s).forEach((v, n) => replace(spanOf(box, ['xmin', 'ymin', 'xmax', 'ymax'][n])!, String(v)))
    }
  })
  // New objects go after the object of the shape before them, or before the first object
  const added = new Map<number, Shape[]>()
  let anchor = -1
  after.forEach((s, i) => {
    const k = origins[i]
    if (k !== null) anchor = k
    else added.set(anchor, [...(added.get(anchor) ?? []), s])
  })
  for (const [k, shapes] of added) {
    const blocks = shapes.map((s) => labelImgObject(s, size!)).join('\n')
    if (k >= 0) edits.push({ from: objects[k].end, to: objects[k].end, text: `\n${blocks}` })
    else if (objects.length) {
      const at = lineStart(objects[0].start)
      edits.push({ from: at, to: at, text: `\n${blocks}` })
    } else if (selfClosed(root)) {
      edits.push({ from: root.start, to: root.end, text: `${text.slice(root.start, root.end).replace(/\s*\/>$/, '>')}\n${blocks}\n</${root.name}>` })
    } else edits.push({ from: root.close, to: root.close, text: `${blocks}\n` })
  }
  const out = edited(text, edits)
  // It must read back as asked, in the file's order: anything else is a file this can't change safely
  const order = [...(added.get(-1) ?? []), ...objects.flatMap((_, k) => (kept.has(k) ? [after[kept.get(k)!], ...(added.get(k) ?? [])] : []))]
  let back: ReturnType<typeof readVoc> | null = null
  try {
    back = readVoc(out)
  } catch {}
  if (!back || back.verified !== verified || back.shapes.length !== order.length || back.shapes.some((s, j) => s.label !== order[j].label.trim())) {
    throw new Error(`${base} couldn't be changed where its objects are, so it's left unchanged`)
  }
  return out
}

/** The image's labels in its file's own format, written by the rules saveLabels follows */
async function rewritten(root: string, name: string, found: Found, text: string, change: ShapeChange, flags: Record<string, boolean> | null) {
  const base = path.basename(found.file)
  if (found.format === 'labelme') {
    let doc
    try {
      doc = parseLabelFile(text)
    } catch (e) {
      throw unreadable(base, e)
    }
    const { shapes } = applied(change, doc.shapes)
    // Only the shapes and flags change: imagePath, imageData, the size, version and unknown keys stay
    return { text: serializeLabelFile({ ...doc, flags: flags ?? doc.flags, shapes }), shapes, verified: doc.other.verified === true }
  }
  if (found.format === 'createml') {
    const entry = entryName(outputFileOf(root), name)
    let list
    try {
      list = readCreateML(text, entry)
    } catch (e) {
      throw unreadable(base, e, ' as CreateML')
    }
    const { shapes } = applied(change, list.shapes)
    return { text: writeCreateML(text, entry, shapes, list.verified), shapes, verified: list.verified }
  }
  // LabelImg's formats measure boxes against the image: YOLO's numbers are shares of it, and a
  // new VOC object's truncated mark says whether it reaches an edge
  const size = async () => {
    try {
      return await imageSize(root, name)
    } catch (e) {
      throw new Error(`${name} can't be read, so ${base} is left unchanged: ${reason(e)}`)
    }
  }
  if (found.format === 'voc') {
    let voc
    try {
      voc = readVoc(text)
    } catch (e) {
      throw unreadable(base, e, ' as Pascal VOC')
    }
    const { shapes, origins } = applied(change, voc.shapes)
    const out = vocInPlace(base, text, voc.shapes, voc.verified, shapes, origins, origins.includes(null) ? await size() : null)
    return { text: out, shapes, verified: voc.verified }
  }
  const classesFile = path.join(path.dirname(found.file), YOLO_CLASSES)
  const classesText = await readText(classesFile)
  if (classesText === null) throw new Error(`${base} isn't a YOLO file (there's no ${YOLO_CLASSES} beside it), so it's left unchanged`)
  const known = classesText.trim() ? readClasses(classesText) : []
  const [width, height] = await size()
  let before: Shape[]
  try {
    before = readYolo(text, known, width, height)
  } catch (e) {
    throw unreadable(base, e, ' as YOLO')
  }
  const { shapes, origins } = applied(change, before)
  // readYolo numbers the lines that aren't blank, in order
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  let classes = [...known]
  const out = shapes.map((s, i) => {
    const k = origins[i]
    // A box that didn't move keeps its numbers as written: reading them rounds to whole pixels
    if (k !== null && s.shape_type === 'rectangle' && samePoints(s.points, before[k].points)) {
      if (!classes.includes(s.label)) classes.push(s.label)
      return `${classes.indexOf(s.label)} ${lines[k].trim().split(/\s+/).slice(1).join(' ')}\n`
    }
    const one = writeYolo([s], classes, width, height)
    classes = one.classes
    return one.text
  }).join('')
  // Class numbers never move: new labels go after the ones classes.txt has
  return { text: out, shapes, verified: false, classes: classes.length !== known.length ? { file: classesFile, text: writeClasses(classes) } : undefined }
}

/** The stamp readDataset gave an image's labels, worked out again from the file as it is now */
async function versionNow(root: string, name: string, found: Found, text: string) {
  if (found.format === 'createml') {
    const list = indexed(text)
    return list.error ? null : versionOf('createml', null, null, list.entries.get(entryName(outputFileOf(root), name)))
  }
  const [file, classes] = await Promise.all([
    fs.stat(found.file).catch(() => null),
    found.format === 'yolo' ? fs.stat(path.join(path.dirname(found.file), YOLO_CLASSES)).catch(() => null) : null
  ])
  return versionOf(found.format, file, classes, undefined)
}

export type Rewritten = { file: string; format: LabelFormat; item: ImageItem }
/** The label file an image's labels were read from, as readDataset gives it (a DatasetImage will do): rewriteLabels writes there and nowhere else */
export type ReadFrom = Pick<DatasetImage, 'file' | 'format' | 'stamp'>

/** Where readDataset found an image's labels, if they're there still: a file other images share is never written */
async function stillIn(root: string, name: string, read: ReadFrom): Promise<Found> {
  const { file, format } = read
  if (!file || !format) throw new Error(`${name} had no label file when its labels were read, so there's nothing to rewrite`)
  const base = path.basename(file)
  const found = await findLabels(root, name, format).catch((e) => {
    throw unreadable(file, e)
  })
  if (!found || fold(found.file) !== fold(file) || found.format !== format) {
    throw new Error(`${base} was read as ${name}'s labels, but ${found ? `they're in ${path.basename(found.file)} now` : "it's gone"}, so nothing is rewritten: check the folder again`)
  }
  if (found.format !== 'createml' && sharesFile(root, name, found)) throw new Error(`${base} holds the labels of other images too, so it's left unchanged`)
  return found
}

/** A label file's text, to rewrite it */
async function textOf(file: string) {
  let text: string | null
  try {
    text = await readText(file)
  } catch (e) {
    throw unreadable(file, e)
  }
  if (text === null) throw new Error(`${path.basename(file)} is gone, so there's nothing to rewrite`)
  return text
}

/**
 * Saves an image's labels in the format its label file is in, by the rules saveLabels
 * follows, with the old file in the trash to restore it from: the new file is written beside
 * the old one, the old one goes to the trash, then the new one takes its name (writeTrashing).
 * A file that can't be read, or that other images share, is never written over.
 *
 * `read` is where readDataset found the labels: only that file is written, and when the
 * image's labels are elsewhere now (the window's format picker can read either of two files),
 * nothing is. `change` is a change made to what the file holds as it's written, so a rename
 * planned from an earlier read can't undo edits made since: with checks.ts' renaming,
 * `rewriteLabels(root, image.name, renaming(from, to), null, image)`. It can also be a new
 * list made from the shapes as read, such as a fix gives (`fixIssue(image, issue)`): that is
 * refused once the file has changed since the read, as it would undo what changed.
 * Image flags (LabelMe's only) are replaced, or kept with null.
 */
export async function rewriteLabels(root: string, name: string, change: ShapeChange, flags: Record<string, boolean> | null, read: ReadFrom): Promise<Rewritten> {
  return serial(async () => {
    const found = await stillIn(root, name, read)
    const text = await textOf(found.file)
    const next = await rewritten(root, name, found, text, change, flags)
    // After reading, so a change made meanwhile shows as one
    if (Array.isArray(change) && (await versionNow(root, name, found, text)) !== read.stamp) {
      throw new Error(`${path.basename(found.file)} has changed since ${name}'s labels were read, so it's left unchanged: check the folder again`)
    }
    if ('classes' in next && next.classes) await writeAtomic(next.classes.file, next.classes.text)
    await writeTrashing(found.file, next.text)
    const item: ImageItem = { ...summarize(next.shapes, true, await keypointsOf(root)), name, embedded: isEmbedded(root, name) || undefined, verified: next.verified || undefined }
    return { file: found.file, format: found.format, item }
  })
}

/**
 * rewriteLabels for the images of one CreateML list, as every image has its labels in one -O
 * labels.json or Roboflow's _annotations.createml.json, with a change such as a rename: the
 * list is read and written once, with one copy of it in the trash, where an image at a time
 * would read and write it whole, and put a copy of it in the trash, once an image. An image
 * whose labels are elsewhere now, or whose entries can't be read, fails alone; a list that
 * can't be read or written fails them all. Gives, image by image, what rewriteLabels gives, or
 * why it failed; null when `signal` stopped it before the list was written, so none was.
 */
export async function rewriteList(root: string, file: string, images: { name: string; read: ReadFrom }[], change: (current: Shape[]) => Shape[],
  signal?: AbortSignal): Promise<(Rewritten | Error)[] | null> {
  return serial(async () => {
    const base = path.basename(file)
    const results: (Rewritten | Error)[] = []
    const failure = (e: unknown) => (e instanceof Error ? e : new Error(reason(e)))
    // The images whose labels are in the list still, by the names it gives them
    const listed = new Map<string, number>()
    for (const [k, { name, read }] of images.entries()) {
      // A list of tens of thousands of images takes seconds to look through, and the window's saves wait for it
      if (signal?.aborted) return null
      try {
        const found = await stillIn(root, name, read)
        if (found.format !== 'createml' || fold(found.file) !== fold(file)) throw new Error(`${name}'s labels weren't read from ${base}, so nothing is rewritten`)
        listed.set(entryName(outputFileOf(root), name), k)
      } catch (e) {
        results[k] = failure(e)
      }
    }
    if (!listed.size) return results
    try {
      const text = await textOf(file)
      const list = indexed(text)
      if (list.error) throw unreadable(base, list.error, ' as CreateML')
      const changes = new Map<string, { shapes: Shape[]; verified: boolean }>()
      for (const [entry, k] of listed) {
        const { name } = images[k]
        const own = list.entries.get(entry)
        if (own === undefined) {
          results[k] = new Error(`${base} no longer lists ${name}, so nothing is rewritten: check the folder again`)
          continue
        }
        try {
          const { shapes, verified } = readCreateML(own, entry)
          changes.set(entry, { shapes: change(shapes), verified })
        } catch (e) {
          results[k] = new Error(`${name}'s entries in ${base} can't be read, so they're left unchanged: ${reason(e)}`)
        }
      }
      // The last moment a stop leaves the list as it was
      if (signal?.aborted) return null
      if (changes.size) await writeTrashing(file, writeCreateMLEntries(text, changes))
      const keypoints = await keypointsOf(root)
      for (const [entry, { shapes, verified }] of changes) {
        const k = listed.get(entry)!
        const { name } = images[k]
        results[k] = { file, format: 'createml', item: { ...summarize(shapes, true, keypoints), name, embedded: isEmbedded(root, name) || undefined, verified: verified || undefined } }
      }
    } catch (e) {
      for (const k of listed.values()) results[k] ??= failure(e)
    }
    return results
  })
}
