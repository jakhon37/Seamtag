// A whole folder's labels, read against each image's real size for exporting, checking and
// counting them. Each image's own label file stays the working copy: nothing here writes a
// label file except rewriteLabels, which sends the old one to the trash first.
import { createHash } from 'crypto'
import { createReadStream, promises as fs, type Stats } from 'fs'
import path from 'path'
import type { Clash, OrphanFile } from '../shared/checks'
import type { DatasetImage, MaskBits } from '../shared/dataset'
import {
  type LabelFormat, looksLikeCreateML, parseXml, pyRound, readClasses, readCreateML, readVoc, readYolo, writeClasses, writeCreateML, writeVoc,
  writeYolo, type XmlElement, YOLO_CLASSES
} from '../shared/formats'
import { parseLabelFile, serializeLabelFile, type Shape } from '../shared/labelfile'
import { summarize } from '../shared/labels'
import { imageDepth, imageSize } from './images'
import { decodeMask } from './masks'
import {
  findLabels, folderOf, type Found, type ImageItem, imagePath, labelPath, type Labels, readLabels, readText, sendToTrash, serial, writeAtomic
} from './project'

const posix = (p: string) => p.split(path.sep).join('/')
const stemOf = (name: string) => name.replace(/\.[^./]+$/, '')
const reason = (e: unknown) => (e instanceof Error ? e.message : String(e))
const stamp = (s: Stats | null) => (s ? `${s.size}:${s.mtimeMs}` : '-')
const outside = (rel: string) => !rel || rel === '..' || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)
// Windows and macOS find a.PNG by the name a.png
const foldsCase = process.platform === 'win32' || process.platform === 'darwin'

export type ReadOptions = {
  /** The format read first when an image has label files in two, as the window's format picker sets it */
  prefer?: LabelFormat
  /** Called as images are read: at most every 100 ms, and once at the end */
  onProgress?: (done: number, total: number) => void
  /** Stops the read, which then rejects with the signal's reason. What was read stays cached. */
  signal?: AbortSignal
}

type Cached = { key: string; image: DatasetImage }
// By folder, then image: a re-run reads only label files and images whose size or time changed
const caches = new Map<string, Map<string, Cached>>()

/** Drops what was read of a folder, as when it's closed */
export function forgetDataset(root: string) {
  caches.delete(root)
}

/**
 * Every image of an open folder with its labels, read as the window reads them (findLabels,
 * then readLabels) but against the image's real upright size from its header: never through
 * openImage, whose tile pyramids would thrash over a folder. Eight at a time, as the
 * summaries are, and in the folder's list order.
 */
export async function readDataset(root: string, { prefer = 'labelme', onProgress, signal }: ReadOptions = {}): Promise<DatasetImage[]> {
  const folder = folderOf(root)
  const names = folder.images
  const old = caches.get(root) ?? new Map<string, Cached>()
  const fresh = new Map<string, Cached>()
  const out: DatasetImage[] = new Array(names.length)
  let next = 0, done = 0, told = 0
  await Promise.all(Array.from({ length: 8 }, async () => {
    while (next < names.length && !signal?.aborted) {
      const at = next++
      const name = names[at]
      const read = await readImage(root, name, prefer, folder.embedded.has(name), old.get(name))
      fresh.set(name, read)
      out[at] = read.image
      done++
      if (onProgress && (done === names.length || Date.now() - told >= 100)) {
        told = Date.now()
        onProgress(done, names.length)
      }
    }
  }))
  if (signal?.aborted) {
    for (const [name, read] of fresh) old.set(name, read)
    caches.set(root, old)
    signal.throwIfAborted()
  }
  // Images gone from the folder drop out of the cache
  caches.set(root, fresh)
  return out
}

async function readImage(root: string, name: string, prefer: LabelFormat, embedded: boolean, cached?: Cached): Promise<Cached> {
  const file = imagePath(root, name)
  let found: Found | null = null
  let broken: string | undefined
  try {
    found = await findLabels(root, name, prefer)
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
  // YOLO's names come from classes.txt, so a change there counts too
  const key = [found?.format, found?.file, stamp(labelStat), stamp(classesStat), stamp(imageStat)].join('\0')
  if (cached?.key === key) return cached
  let width = 0, height = 0, unreadable: string | undefined
  try {
    ;[width, height] = await imageSize(root, name)
  } catch (e) {
    unreadable = reason(e)
  }
  let labels: Labels | null = null
  if (found && labelStat && !broken) {
    try {
      // An image that can't be read gives YOLO's shares a 1 x 1 frame, as the summaries read them
      labels = await readLabels(root, name, found, async () => (unreadable ? [1, 1] : [width, height]))
    } catch (e) {
      broken = reason(e)
    }
  }
  const shapes = labels?.doc.shapes ?? []
  const image: DatasetImage = {
    name, width, height,
    file: labels?.file ?? (broken && found ? found.file : null),
    format: labels?.format ?? (broken && found ? found.format : null),
    shapes,
    masks: await Promise.all(shapes.map((s) => (s.shape_type === 'mask' ? placeMask(s) : null))),
    flags: labels?.doc.flags ?? {},
    verified: labels?.verified ?? false,
    ...(broken ? { broken } : {}),
    ...(unreadable ? { unreadable } : {}),
    labelSize: labels ? await sizeIn(labels, width, height) : null,
    ...(labels ? imageNamed(root, name, labels) : {}),
    ...(!embedded && imageStat ? { bytes: imageStat.size } : {})
  }
  return { key, image }
}

/**
 * A mask shape's pixels where labelme 7.9.1's shapes_to_label pastes them
 * (examples/utils.py): at its first corner rounded half to even, cropped to the box between
 * its corners when both are whole pixels. Null for a mask with no pixels that decode.
 */
async function placeMask(s: Shape): Promise<MaskBits | null> {
  if (!s.mask || !s.points.length) return null
  const m = await decodeMask(s.mask).catch(() => null)
  if (!m) return null
  const [x, y] = s.points[0].map(pyRound)
  if (s.points.length < 2 || !s.points.flat().every(Number.isInteger)) return { x, y, width: m.width, height: m.height, data: m.data }
  const width = Math.max(0, s.points[1][0] - x + 1), height = Math.max(0, s.points[1][1] - y + 1)
  if (width === m.width && height === m.height) return { x, y, width, height, data: m.data }
  // labelme can't paste a mask smaller than its box at all: what's missing is left empty here
  const data = new Uint8Array(width * height)
  for (let row = 0; row < Math.min(height, m.height); row++) {
    data.set(m.data.subarray(row * m.width, row * m.width + Math.min(width, m.width)), row * width)
  }
  return { x, y, width, height, data }
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
  return { labelImage: foldsCase && image.toLowerCase() === name.toLowerCase() ? name : image }
}

/**
 * Images whose label file is one and the same file: a.jpg and a.png beside one a.json or
 * a.xml, or every image in one -O labels.json. A CreateML list keeps each image's entry
 * apart, so it never counts.
 */
export function labelFileClashes(images: DatasetImage[]): Clash[] {
  const byFile = new Map<string, string[]>()
  for (const i of images) {
    if (!i.file || i.format === 'createml') continue
    byFile.set(i.file, [...(byFile.get(i.file) ?? []), i.name])
  }
  return [...byFile].filter(([, names]) => names.length > 1).map(([file, names]) => ({ file, images: names }))
}

// sha256 by file, size and time: a re-run hashes only new and changed files
const hashes = new Map<string, { stamp: string; hash: string }>()

async function hashFile(file: string) {
  const at = stamp(await fs.stat(file))
  const known = hashes.get(file)
  if (known?.stamp === at) return known.hash
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  const hex = hash.digest('hex')
  hashes.set(file, { stamp: at, hash: hex })
  return hex
}

/**
 * Image files with the same bytes, in groups in the folder's order. Only files that share a
 * byte size are read, as Datumaro and fiftyone find repeated images, so most never are.
 */
export async function findDuplicateImages(root: string, images: DatasetImage[], { signal }: { signal?: AbortSignal } = {}): Promise<string[][]> {
  const bySize = new Map<number, string[]>()
  // An empty file can't be read as an image, which is reported on its own
  for (const i of images) if (i.bytes) bySize.set(i.bytes, [...(bySize.get(i.bytes) ?? []), i.name])
  const todo = [...bySize.values()].filter((names) => names.length > 1).flat()
  const bytes = new Map(images.map((i) => [i.name, i.bytes]))
  const byHash = new Map<string, string[]>()
  let next = 0
  await Promise.all(Array.from({ length: 4 }, async () => {
    while (next < todo.length && !signal?.aborted) {
      const name = todo[next++]
      const hash = await hashFile(imagePath(root, name)).catch(() => null)
      const key = `${bytes.get(name)}:${hash}`
      if (hash) byHash.set(key, [...(byHash.get(key) ?? []), name])
    }
  }))
  signal?.throwIfAborted()
  const order = new Map(images.map((i, k) => [i.name, k]))
  const at = (name: string) => order.get(name)!
  return [...byHash.values()].filter((g) => g.length > 1).map((g) => g.sort((a, b) => at(a) - at(b))).sort((a, b) => at(a[0]) - at(b[0]))
}

const LABEL_EXTS = new Set(['.json', '.xml', '.txt'])
// A COCO file can run to hundreds of megabytes; one image's label file, even with imageData, doesn't
const MAX_LABEL_FILE = 64 * 1024 * 1024

/** Files that may be labels under a folder, by path in it; hidden folders are skipped, as the image list skips them */
async function filesUnder(dir: string, skip: string[]): Promise<string[]> {
  const found: string[] = []
  const walk = async (sub: string) => {
    const full = path.join(dir, sub)
    if (skip.includes(full)) return
    let entries
    try {
      entries = await fs.readdir(full, { withFileTypes: true })
    } catch {
      return
    }
    // In the image list's order, whatever order the disk keeps them in
    for (const e of entries.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))) {
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
 * Is this a label file? LabelMe JSON (one holding its own image is its own image, so not an
 * orphan), CreateML, Pascal VOC, or a .txt that's YOLO's because classes.txt sits beside it.
 * Anything else, such as a COCO file, a caption or settings, isn't. With the image a LabelMe
 * file names, when that's in the folder.
 */
async function labelFileKind(root: string, file: string): Promise<{ names: string | null } | null> {
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
    if (doc.imageData) return null
    if (!doc.imagePath) return { names: null }
    const rel = path.relative(root, path.resolve(path.dirname(file), doc.imagePath))
    return { names: outside(rel) ? null : posix(rel) }
  } catch {
    return null
  }
}

/**
 * Label files no image has, beside the images or in the -O folder: no image of the same name
 * sits where they'd belong, by labelPath's rules (folders mirrored, or flat in the output
 * folder as labelme and LabelImg write them). Shown by path in the folder, or in full when
 * outside it.
 */
export async function orphanLabels(root: string): Promise<OrphanFile[]> {
  const folder = folderOf(root)
  const stems = new Set(folder.images.map(stemOf))
  const bases = new Set(folder.images.map((n) => stemOf(path.posix.basename(n))))
  const out = folder.outputDir ?? (folder.outputFile ? path.dirname(folder.outputFile) : null)
  const within = (dir: string, parent: string) => dir.startsWith(parent + path.sep)
  const places = [{ dir: root, flat: out === root, skip: out && within(out, root) ? [out] : [] }]
  if (out && out !== root) places.push({ dir: out, flat: true, skip: within(root, out) ? [root] : [] })
  const known = new Set<string>([...folder.embedded.values(), ...(folder.outputFile ? [folder.outputFile] : [])])
  const orphans: OrphanFile[] = []
  for (const { dir, flat, skip } of places) {
    for (const rel of await filesUnder(dir, skip)) {
      const file = path.join(dir, rel)
      const stem = stemOf(rel)
      if (known.has(file) || stems.has(stem) || (flat && !rel.includes('/') && bases.has(stem))) continue
      const kind = await labelFileKind(root, file)
      const shown = path.relative(root, file)
      if (kind) orphans.push({ file: outside(shown) ? file : posix(shown), names: kind.names })
    }
  }
  return orphans
}

const unreadable = (file: string, e: unknown, as = '') =>
  new Error(`${path.basename(file)} can't be read${as}, so it's left unchanged: ${reason(e)}`)

/** Pascal VOC's <depth>, kept as the file has it */
function vocDepth(text: string): number | null {
  const depth = Number(childOf(childOf(parseXml(text), 'size'), 'depth')?.text.trim() || NaN)
  return Number.isInteger(depth) && depth > 0 ? depth : null
}

/** Another image whose labels go to the same file: a.jpg and a.png, or one -O labels.json for all */
function sharesFile(root: string, name: string, found: Found) {
  const folder = folderOf(root)
  const stem = stemOf(name)
  return folder.images.some((other) => other !== name && (found.file === folder.outputFile || stemOf(other) === stem) &&
    labelPath(root, other, found.format) === found.file)
}

/** New shapes, or a change to the shapes the file holds when it's written */
export type ShapeChange = Shape[] | ((current: Shape[]) => Shape[])
const applied = (change: ShapeChange, current: Shape[]) => (typeof change === 'function' ? change(current) : change)

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
    const shapes = applied(change, doc.shapes)
    // Only the shapes and flags change: imagePath, imageData, the size, version and unknown keys stay
    return { text: serializeLabelFile({ ...doc, flags: flags ?? doc.flags, shapes }), shapes, verified: doc.other.verified === true }
  }
  if (found.format === 'createml') {
    // One output file lists every image by its path in the folder; one file per image, by file name
    const entry = folderOf(root).outputFile ? name : path.posix.basename(name)
    let list
    try {
      list = readCreateML(text, entry)
    } catch (e) {
      throw unreadable(base, e, ' as CreateML')
    }
    const shapes = applied(change, list.shapes)
    return { text: writeCreateML(text, entry, shapes, list.verified), shapes, verified: list.verified }
  }
  // LabelImg's formats need the image's size: VOC writes it, YOLO's numbers are shares of it
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
    const image = imagePath(root, name)
    const [width, height] = await size()
    const depth = vocDepth(text) ?? await imageDepth(root, name)
    const shapes = applied(change, voc.shapes)
    const out = writeVoc({ folder: path.basename(path.dirname(image)), filename: path.basename(image), path: image, width, height, depth }, shapes, voc.verified)
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
  const shapes = applied(change, before)
  // readYolo numbers the lines that aren't blank, in order
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  const used = before.map(() => false)
  let classes = [...known]
  const out = shapes.map((s) => {
    const j = s.shape_type === 'rectangle'
      ? before.findIndex((b, k) => !used[k] && b.points.every((p, n) => p[0] === s.points[n]?.[0] && p[1] === s.points[n]?.[1]))
      : -1
    if (j < 0) {
      const one = writeYolo([s], classes, width, height)
      classes = one.classes
      return one.text
    }
    // A box that didn't move keeps its numbers as written: reading them rounds to whole pixels
    used[j] = true
    if (!classes.includes(s.label)) classes.push(s.label)
    return `${classes.indexOf(s.label)} ${lines[j].trim().split(/\s+/).slice(1).join(' ')}\n`
  }).join('')
  // Class numbers never move: new labels go after the ones classes.txt has
  return { text: out, shapes, verified: false, classes: classes.length !== known.length ? { file: classesFile, text: writeClasses(classes) } : undefined }
}

export type Rewritten = { file: string; format: LabelFormat; item: ImageItem }

/**
 * Saves an image's labels in the format its label file is in, by the rules saveLabels
 * follows, with the old file in the trash to restore it from: the new file is written beside
 * the old one, the old one goes to the trash, then the new one takes its name. A file that
 * can't be read, or that other images share, is never written over.
 *
 * `shapes` is the new list, or a change made to what the file holds as it's written, so a
 * rename planned from an earlier read can't undo edits made since: with checks.ts' renaming,
 * `rewriteLabels(root, name, renaming(from, to), null)`.
 * Image flags (LabelMe's only) are replaced, or kept with null. `prefer` picks the file as
 * readDataset did when an image has two.
 */
export async function rewriteLabels(root: string, name: string, shapes: ShapeChange, flags: Record<string, boolean> | null, prefer: LabelFormat = 'labelme'): Promise<Rewritten> {
  return serial(async () => {
    const found = await findLabels(root, name, prefer).catch((e) => {
      throw unreadable(labelPath(root, name), e)
    })
    if (!found) throw new Error(`${name} has no label file, so there's nothing to rewrite`)
    const base = path.basename(found.file)
    if (found.format !== 'createml' && sharesFile(root, name, found)) throw new Error(`${base} holds the labels of other images too, so it's left unchanged`)
    let text: string | null
    try {
      text = await readText(found.file)
    } catch (e) {
      throw unreadable(base, e)
    }
    if (text === null) throw new Error(`${base} is gone, so there's nothing to rewrite`)
    const next = await rewritten(root, name, found, text, shapes, flags)
    if ('classes' in next && next.classes) await writeAtomic(next.classes.file, next.classes.text)
    const tmp = `${found.file}.${process.pid}.${Date.now()}.tmp`
    try {
      await fs.writeFile(tmp, next.text)
    } catch (e) {
      await fs.rm(tmp, { force: true })
      throw new Error(`${base}'s new labels couldn't be written, so it's left unchanged: ${reason(e)}`)
    }
    try {
      await sendToTrash(found.file)
    } catch (e) {
      await fs.rm(tmp, { force: true })
      throw new Error(`${base} couldn't go to the trash, so it's left unchanged: ${reason(e)}`)
    }
    try {
      await fs.rename(tmp, found.file)
    } catch (e) {
      throw new Error(`${base} is in the trash, but its new labels couldn't take its name: they're in ${tmp} (${reason(e)})`)
    }
    const item: ImageItem = { ...summarize(next.shapes, true), name, embedded: folderOf(root).embedded.has(name) || undefined, verified: next.verified || undefined }
    return { file: found.file, format: found.format, item }
  })
}
