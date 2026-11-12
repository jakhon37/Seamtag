// Ultralytics YOLO datasets on disk, for import: a data.yaml, a folder holding one, or a folder of
// images and labels without one. Splits are found as Ultralytics 8.4.211 finds them
// (check_det_dataset and BaseDataset.get_img_files in ultralytics/data/), each image's label file
// by img2label_paths, and every label file's rows are counted for guessing the task. Roboflow's
// ../train/images and CVAT's data/images/... lists are read as those tools mean them, and so are
// darknet's files of class names, in CVAT's YOLO 1.1 and Roboflow's YOLO Darknet downloads.
import { promises as fs } from 'fs'
import path from 'path'
import type { Issue, Severity } from '../shared/dataset'
import { readClasses, YOLO_CLASSES } from '../shared/formats'
import { pyFloat, short } from '../shared/pose'
import {
  countRows, type DataYaml, guessTask, isDatasetYaml, isYoloImage, labelPathFor, labelText, MOST_CLASSES, newRowStats, readDataYaml, type TaskGuess
} from '../shared/ultralytics'

/** An image of the dataset and its label file (null when it has none, a background to Ultralytics); paths are absolute */
export type YoloItem = { image: string; label: string | null; split: string }
/** train, val, test, or for a folder without data.yaml its own subfolder's name ('' when it has none) */
export type YoloSplit = { name: string; items: YoloItem[] }

/** A dataset as found on disk: its classes, keypoints and splits, a guess at its task, and what's wrong with it */
export type YoloDataset = {
  /** The data.yaml read; null for a folder without one */
  yaml: string | null
  /** The folder splits are found from */
  root: string
  names: string[]
  kptShape: [number, number] | null
  kptNames: Record<number, string[]> | null
  flipIdx: number[] | null
  splits: YoloSplit[]
  /** The task the rows look made for, and why */
  guess: TaskGuess
  issues: Issue[]
}

// Files looked at together
const BATCH = 64
// The most files the folders of one dataset may hold, all told: COCO's train2017 has 118,000
// images and Objects365's 1.7 million, each with a label file. A data.yaml can name any folder,
// the disk's root too, and every file listed is kept while the dataset is read.
const MOST_FILES = 5_000_000

/** What reading one dataset shares: Stop, and how many files may still be listed */
type Reading = { signal?: AbortSignal; left: number }
// Thrown when a dataset's folders hold more files than Reading allows
class TooMany extends Error {}

const stat = (p: string) => fs.stat(p).catch(() => null)
const isDir = async (p: string) => (await stat(p))?.isDirectory() ?? false
const isFile = async (p: string) => (await stat(p))?.isFile() ?? false

// A tidy folder and a path inside it, joined without path.join's tidying, which takes ten times as long
const join = (dir: string, rel: string) => (dir.endsWith(path.sep) ? dir + rel : dir + path.sep + rel)

/**
 * Every file under a folder, as Python's glob('**') lists them: names starting with a dot are left
 * out. Files are added to `out` one at a time: a folder can hold more of them than a call takes
 * arguments (about 125,000 with Node's stack), so a list of files is never spread into one. Stop
 * ends it at the next folder, and a dataset's files past what `reading` allows end it with TooMany.
 */
async function walk(dir: string, reading: Reading, out: string[] = [], seen = new Set<string>()): Promise<string[]> {
  reading.signal?.throwIfAborted()
  const real = await fs.realpath(dir).catch(() => dir)
  // A link back up the tree would go round for ever
  if (seen.has(real)) return out
  seen.add(real)
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  for (const e of entries) {
    if (e.name.startsWith('.')) continue
    const full = join(dir, e.name)
    if (e.isDirectory() || (e.isSymbolicLink() && (await isDir(full)))) await walk(full, reading, out, seen)
    else {
      if (--reading.left < 0) throw new TooMany()
      out.push(full)
    }
  }
  return out
}

/**
 * A check of whether files are there that lists each folder once: a stat of every label file
 * takes ten seconds for COCO's 118,000. A name a listing holds only in another case or Unicode
 * form, or as a link or a folder, is left to stat, so the file system decides as Ultralytics'
 * os.path.isfile would (macOS's and Windows' ignore case), and so are names beyond ASCII that
 * aren't listed as they are, and every name in a folder that can't be listed.
 */
function fileCheck(): (file: string) => Promise<boolean> {
  type Listing = { files: Set<string>; names: Set<string> } | null
  const listings = new Map<string, Promise<Listing>>()
  const ascii = (name: string) => /^[\x00-\x7f]*$/.test(name)
  // Generous, so anything a file system that ignores case could take for an ASCII name folds alike (a Kelvin sign for k)
  const fold = (name: string) => (ascii(name) ? name : name.normalize('NFKC').toUpperCase()).toLowerCase()
  const list = (dir: string) => {
    let listing = listings.get(dir)
    if (!listing) {
      listing = fs.readdir(dir, { withFileTypes: true }).then((entries) => {
        const files = new Set<string>(), names = new Set<string>()
        for (const e of entries) {
          if (e.isFile()) files.add(e.name)
          names.add(fold(e.name))
        }
        return { files, names }
      }, (e: NodeJS.ErrnoException) => (e.code === 'ENOENT' || e.code === 'ENOTDIR' ? { files: new Set<string>(), names: new Set<string>() } : null))
      listings.set(dir, listing)
    }
    return listing
  }
  return async (file) => {
    const listing = await list(path.dirname(file))
    const name = path.basename(file)
    if (!listing) return isFile(file)
    if (listing.files.has(name)) return true
    return (!ascii(name) || listing.names.has(fold(name))) && isFile(file)
  }
}

const sortPaths = (files: string[]) => files.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
const YAML_FILE = /\.ya?ml$/i
const stem = (file: string) => path.basename(file, path.extname(file))

/**
 * A folder's data.yaml, looked for as Ultralytics' find_dataset_yaml looks when check_det_dataset
 * is given a folder: at the top, and anywhere below only when the top has none; of several, .yaml
 * before .yml, then the one named after the folder, and here data.yaml after that. Only a
 * dataset's YAML counts (isDatasetYaml, or a data.yaml at the top, whose problems are then told):
 * X-AnyLabeling's keypoint file, a model's settings or a training run's args.yaml beside the
 * images are passed over, so a folder without a dataset's YAML is read as images and labels.
 * `files` is every file under the folder when it had to be listed.
 */
async function findDatasetYaml(dir: string, reading: Reading): Promise<{ found: string[]; passed: string[]; files: string[] | null }> {
  const passed: string[] = []
  const pick = async (candidates: string[], top: boolean) => {
    const taken: string[] = []
    for (let i = 0; i < candidates.length; i += BATCH) {
      reading.signal?.throwIfAborted()
      const batch = candidates.slice(i, i + BATCH)
      const texts = await Promise.all(batch.map((f) => fs.readFile(f, 'utf8').catch(() => null)))
      batch.forEach((f, j) => {
        const text = texts[j]
        ;((top && stem(f).toLowerCase() === 'data') || (text !== null && isDatasetYaml(text)) ? taken : passed).push(f)
      })
    }
    if (taken.length < 2) return taken
    const yamls = taken.filter((f) => path.extname(f).toLowerCase() === '.yaml')
    const pool = yamls.length ? yamls : taken
    for (const named of [pool.filter((f) => stem(f) === path.basename(dir)), pool.filter((f) => stem(f).toLowerCase() === 'data')]) {
      if (named.length === 1) return named
    }
    return pool
  }
  const top: string[] = []
  for (const e of await fs.readdir(dir, { withFileTypes: true }).catch(() => [])) {
    const full = path.join(dir, e.name)
    if (!e.name.startsWith('.') && YAML_FILE.test(e.name) && (e.isFile() || (e.isSymbolicLink() && (await isFile(full))))) top.push(full)
  }
  let found = await pick(sortPaths(top), true)
  let files: string[] | null = null
  if (!found.length) {
    files = await walk(dir, reading)
    found = await pick(files.filter((f) => YAML_FILE.test(f) && path.dirname(f) !== dir), false)
  }
  return { found, passed, files }
}

/**
 * The images of the dataset in a folder without data.yaml, by split: the folder after the last
 * images folder in their path (images/train/a.jpg), or before it (Roboflow's train/images/a.jpg)
 */
function splitOf(root: string, image: string): string {
  const parts = inside(root, image).split(path.sep)
  const folders = parts.slice(0, -1)
  const k = folders.lastIndexOf('images')
  if (k < 0) return ''
  return folders[k + 1] ?? folders[k - 1] ?? ''
}

// A file's path inside a folder it was listed from, without path.relative, which takes far longer
const inside = (root: string, file: string) => (file.startsWith(root + path.sep) ? file.slice(root.length + 1) : path.relative(root, file))

const SPLIT_ORDER = ['train', 'val', 'valid', 'validation', 'test']
const splitRank = (name: string) => (SPLIT_ORDER.includes(name) ? SPLIT_ORDER.indexOf(name) : SPLIT_ORDER.length)

// "a.txt", or "a.txt and 2 more label files", for messages about many files
const some = (first: string, n: number, kind: string) => (n > 1 ? `${first} and ${n - 1} more ${kind}${n > 2 ? 's' : ''}` : first)

// A darknet .data file is a few lines of key = value: a bigger one is something else
const DATA_FILE_BYTES = 64 * 2 ** 10
// Roboflow's YOLO Darknet labelmap, one in each split's folder: _darknet.labels, as its downloads
// name it, or darknet.labels, as its page on the format does
const DARKNET_LABELS = new Set(['_darknet.labels', 'darknet.labels'])

/**
 * The class names of a darknet dataset, one a line, for a folder with neither data.yaml nor
 * classes.txt, found as datumaro, which CVAT reads and writes them with, finds them: the file the
 * .data file's names line gives, inside the dataset (CVAT's YOLO 1.1 writes obj.data with
 * names = data/obj.names, a path from darknet's own folder, which datumaro takes from the .data
 * file's folder without its data/); else obj.names, or the one .names file, at the top; else
 * Roboflow's labelmap, _darknet.labels in each split's folder, when they're all the same.
 * `files` is every file under `root`.
 */
async function darknetNames(root: string, files: string[], say: (s: Severity, code: string, m: string) => void): Promise<{ names: string[]; file: string } | null> {
  const listed = new Set(files)
  const top = files.filter((f) => path.dirname(f) === root)
  const named = (ext: string, name: string) => {
    const all = top.filter((f) => path.extname(f).toLowerCase() === ext)
    return all.find((f) => path.basename(f).toLowerCase() === name) ?? (all.length === 1 ? all[0] : null)
  }
  const read = async (file: string) => {
    const text = await fs.readFile(file, 'utf8').catch(() => null)
    return text === null ? null : { names: readClasses(text), file }
  }
  const data = named('.data', 'obj.data')
  if (data && ((await stat(data))?.size ?? Infinity) <= DATA_FILE_BYTES) {
    let given = ''
    for (const line of (await fs.readFile(data, 'utf8').catch(() => '')).split(/\r\n|\r|\n/)) {
      const m = /^\s*(\w+)\s*=\s*(.+)$/.exec(line)
      if (m?.[1] === 'names') given = m[2].trim().replace(/\\/g, '/')
    }
    for (const p of given.startsWith('data/') ? [given, given.slice(5)] : given ? [given] : []) {
      const file = path.resolve(path.dirname(data), p)
      if (listed.has(file)) return read(file)
    }
  }
  const names = named('.names', 'obj.names')
  if (names) return read(names)
  const maps = sortPaths(files.filter((f) => DARKNET_LABELS.has(path.basename(f))))
  if (!maps.length) return null
  const lists = (await Promise.all(maps.map(read))).filter((l) => l !== null)
  if (lists.every((l) => l.names.join('\n') === lists[0].names.join('\n'))) return lists[0] ?? null
  say('warning', 'yolo-no-yaml', `${some(path.relative(root, maps[0]), maps.length, 'labelmap')} name the classes differently, so ${maps.length > 2 ? 'none' : 'neither'} is used`)
  return null
}

/**
 * A dataset from a data.yaml file, or a folder: its own data.yaml (found as Ultralytics finds it),
 * or, without one, the images under it, split by their folders, with names from a classes.txt at
 * the top or in labels/, else from darknet's files (darknetNames), or class_0, class_1 and so on.
 * A data.yaml's images have their labels where Ultralytics looks for them, and nowhere else; a
 * folder without one is read in place: the images folder rule counts from the folder (so folders
 * above it named images don't count), then a label beside its image is taken, then one where
 * Ultralytics' rule finds it. Nothing is written.
 * `signal` stops it, which throws its reason; `most` is how many files its folders may hold.
 */
export async function readYoloDataset(target: string, { signal, most = MOST_FILES }: { signal?: AbortSignal; most?: number } = {}): Promise<YoloDataset> {
  const issues: Issue[] = []
  const say = (severity: Severity, code: string, message: string, image: string | null = null) => issues.push({ severity, code, image, message })
  target = path.resolve(target)
  const empty = (root: string, yaml: string | null): YoloDataset =>
    ({ yaml, root, names: [], kptShape: null, kptNames: null, flipIdx: null, splits: [], guess: guessTask(newRowStats()), issues })
  const reading: Reading = { signal, left: most }
  // Folders holding more files than that are told, and nothing is read: a dataset found only in
  // part would be imported only in part
  const tooMany = (folder: string) => (e: unknown) => {
    if (!(e instanceof TooMany)) throw e
    say('error', 'yolo-too-many-files', `${folder} more than ${most.toLocaleString('en-GB')} files, more than Seamtag reads for one dataset, so none are read`)
    return null
  }
  const info = await stat(target)
  if (!info) {
    say('error', 'yolo-missing', `${target} isn't there`)
    return empty(target, null)
  }
  let yamlFile: string | null = null
  let files: string[] | null = null
  if (info.isFile()) {
    if (!YAML_FILE.test(target)) {
      say('error', 'yolo-not-yaml', `${path.basename(target)} isn't a data.yaml: pick the dataset's YAML file or its folder`)
      return empty(path.dirname(target), null)
    }
    yamlFile = target
  } else {
    const search = await findDatasetYaml(target, reading).catch(tooMany(`${path.basename(target)} holds`))
    if (!search) return empty(target, null)
    if (search.found.length > 1) {
      say('error', 'yolo-many-yaml', `${target} holds ${search.found.length} datasets' YAML files (${search.found.map((f) => path.relative(target, f)).join(', ')}): pick the dataset's own`)
      return empty(target, null)
    }
    yamlFile = search.found[0] ?? null
    files = search.files
    if (!yamlFile && search.passed.length) {
      const n = search.passed.length
      say('info', 'yolo-yaml-passed', `${some(path.relative(target, search.passed[0]), n, 'YAML file')} ${n > 1 ? "aren't datasets'" : "isn't a dataset's"} (there's no train or val in ${n > 1 ? 'them' : 'it'}), so ${path.basename(target)} is read as images and labels`)
    }
  }

  let data: DataYaml
  let root: string
  const splits: YoloSplit[] = []
  const exists = fileCheck()
  if (yamlFile) {
    // Ultralytics reads it as UTF-8
    const text = await fs.readFile(yamlFile, 'utf8').catch(() => null)
    if (text === null) {
      say('error', 'yolo-yaml', `${path.basename(yamlFile)} can't be read, so the dataset can't be`)
      return empty(path.dirname(yamlFile), yamlFile)
    }
    const read = readDataYaml(text, path.basename(yamlFile))
    data = read.data
    issues.push(...read.issues)
    root = await rootOf(data.path, path.dirname(yamlFile), path.basename(yamlFile), say)
    for (const name of ['train', 'val', 'test'] as const) {
      const spec = data[name]
      if (!spec) continue
      const images = await splitImages(root, name, spec, say, exists, reading).catch(tooMany(`The splits ${path.basename(yamlFile)} names hold`))
      if (!images) return empty(root, yamlFile)
      splits.push({ name, items: images.map((image) => ({ image, label: null, split: name })) })
    }
  } else {
    root = target
    data = { path: null, train: null, val: null, test: null, names: [], kptShape: null, kptNames: null, flipIdx: null }
    const bySplit = new Map<string, string[]>()
    for (const image of sortPaths((files ?? []).filter(isYoloImage))) {
      const split = splitOf(root, image)
      if (!bySplit.has(split)) bySplit.set(split, [])
      bySplit.get(split)!.push(image)
    }
    for (const name of [...bySplit.keys()].sort((a, b) => splitRank(a) - splitRank(b) || (a < b ? -1 : 1))) {
      splits.push({ name, items: bySplit.get(name)!.map((image) => ({ image, label: null, split: name })) })
    }
    for (const where of [root, path.join(root, 'labels')]) {
      const classes = path.join(where, YOLO_CLASSES)
      if (await isFile(classes)) {
        data.names = readClasses(await fs.readFile(classes, 'utf8'))
        say('info', 'yolo-no-yaml', `${path.basename(target)} has no data.yaml, so the class names come from ${path.relative(target, classes)}`)
        break
      }
    }
    const darknet = data.names.length ? null : await darknetNames(root, files ?? [], say)
    if (darknet) {
      data.names = darknet.names
      say('info', 'yolo-no-yaml', `${path.basename(target)} has no data.yaml, so the class names come from ${path.relative(target, darknet.file)}`)
    }
  }

  // Each image's label file, looked for once however many splits hold it
  const items = splits.flatMap((s) => s.items)
  const labelOf = new Map<string, string | null>()
  for (const item of items) labelOf.set(item.image, null)
  const beside = new Set<string>(), outside = new Set<string>()
  const unreached = new Map<string, { near: string; rule: string }>()
  const lookUp = async (image: string) => {
    // img2label_paths on the whole path, as Ultralytics trains a data.yaml's images; inside the folder for one without
    const rule = yamlFile ? labelPathFor(image, path.sep) : join(root, labelPathFor(inside(root, image), path.sep))
    if (await exists(rule)) {
      labelOf.set(image, rule)
      return
    }
    // An image's name ends with its format after a dot, so this is labelPathFor with no images folder
    const near = `${image.slice(0, image.lastIndexOf('.'))}.txt`
    if (near !== rule && (await exists(near))) {
      if (yamlFile) unreached.set(image, { near, rule })
      else {
        labelOf.set(image, near)
        beside.add(near)
      }
      return
    }
    if (yamlFile) return
    // A dataset's images folder picked on its own has its labels beside it, in ../labels
    const whole = labelPathFor(image, path.sep)
    if (whole !== rule && whole !== near && (await exists(whole))) {
      labelOf.set(image, whole)
      outside.add(whole)
    }
  }
  const images = [...labelOf.keys()]
  for (let i = 0; i < images.length; i += BATCH) {
    signal?.throwIfAborted()
    await Promise.all(images.slice(i, i + BATCH).map(lookUp))
  }
  for (const item of items) item.label = labelOf.get(item.image) ?? null
  const rel = (f: string) => path.relative(root, f)
  if (beside.size) {
    const n = beside.size
    say('info', 'yolo-label-beside', `${some(rel([...beside][0]), n, 'label file')} ${n > 1 ? 'are beside their images' : 'is beside its image'} in an images folder, so ${n > 1 ? "they're" : "it's"} read from there, though Ultralytics would look in labels/`)
  }
  if (outside.size) {
    const n = outside.size
    say('info', 'yolo-label-outside', `${some(rel([...outside][0]), n, 'label file')} ${n > 1 ? 'are' : 'is'} outside ${path.basename(root)}, where Ultralytics' rule puts ${n > 1 ? 'them' : 'it'} (the last images folder in an image's path becomes labels), so ${n > 1 ? "they're" : "it's"} read from there`)
  }
  if (unreached.size) {
    const n = unreached.size
    const [image, { near, rule }] = [...unreached][0]
    say('warning', 'yolo-label-unreached', `${some(rel(near), n, 'label file')} ${n > 1 ? 'are beside their images' : 'is beside its image'}, but Ultralytics looks for ${n > 1 ? 'their labels' : 'its labels'} in ${rel(path.dirname(rule))} (the last images folder in the path becomes labels), so it would train ${n > 1 ? 'those images as backgrounds' : 'that image as a background'}, and ${n > 1 ? "they're" : "it's"} read as such here: move the label files there`, image)
  }
  const owner = new Map<string, string>()
  const seen = new Map<string, string>()
  const both = new Map<string, number>()
  for (const item of items) {
    const twice = seen.get(item.image)
    if (twice && twice !== item.split) both.set(`${twice} and ${item.split}`, (both.get(`${twice} and ${item.split}`) ?? 0) + 1)
    else seen.set(item.image, item.split)
    const first = item.label && owner.get(item.label)
    if (item.label && !first) owner.set(item.label, item.image)
    else if (first && first !== item.image) {
      say('warning', 'yolo-shared-label', `${path.relative(root, first)} and ${path.relative(root, item.image)} share ${path.relative(root, item.label!)}, so both get its labels`, item.image)
    }
  }
  for (const [pair, n] of both) say('info', 'yolo-two-splits', `${n === 1 ? '1 image is' : `${n} images are`} in both ${pair}`)
  if (seen.size && !owner.size && !unreached.size) {
    const all = seen.size === 1 ? 'The one image has no' : `None of the ${seen.size} images has a`
    say('warning', 'yolo-no-labels', yamlFile
      ? `${all} label file where Ultralytics looks (labels for the last images folder in its path, else beside it), so ${seen.size === 1 ? 'it' : 'they'}'d train as ${seen.size === 1 ? 'a background' : 'backgrounds'}`
      : `${all} label file beside it or in labels/, so ${seen.size === 1 ? 'it' : 'they'}'d read as ${seen.size === 1 ? 'a background' : 'backgrounds'}`)
  }

  // What the rows look like, and the highest class number for a folder with no names: below
  // MOST_CLASSES, as each gets a name; a row of a class past it is told and left out
  const stats = newRowStats()
  const labels = [...owner.keys()]
  let top = -1
  // Set in the callbacks below, which TypeScript doesn't follow
  let past = null as { file: string; c: number } | null
  let unreadable = 0
  for (let i = 0; i < labels.length; i += BATCH) {
    signal?.throwIfAborted()
    const texts = await Promise.all(labels.slice(i, i + BATCH).map(async (f) => labelText(await fs.readFile(f).catch(() => new Uint8Array([0xff])))))
    texts.forEach((text, j) => {
      if (text === null) {
        unreadable++
        if (unreadable <= 3) say('warning', 'yolo-not-text', `${path.relative(root, labels[i + j])} can't be read as UTF-8 text, which Ultralytics needs`, owner.get(labels[i + j]))
        return
      }
      countRows(stats, text)
      for (const m of text.matchAll(/^[^\S\r\n]*(\S+)/gm)) {
        const c = pyFloat(m[1])
        if (c === null || !Number.isInteger(c)) continue
        if (c >= MOST_CLASSES) past ??= { file: labels[i + j], c }
        else if (c > top) top = c
      }
    })
  }
  if (unreadable > 3) say('warning', 'yolo-not-text', `${unreadable} label files in all can't be read as UTF-8 text`)
  if (!yamlFile && !data.names.length) {
    data.names = Array.from({ length: top + 1 }, (_, i) => `class_${i}`)
    if (top >= 0) say('info', 'yolo-no-yaml', `${path.basename(target)} has neither data.yaml nor ${YOLO_CLASSES}, so the classes are named class_0 to class_${top}`)
    if (past) say('warning', 'yolo-no-yaml', `${path.relative(root, past.file)} has a row of class ${past.c}, and without data.yaml Seamtag names at most ${MOST_CLASSES.toLocaleString('en-GB')} classes, so rows past class_${MOST_CLASSES - 1} are left out`)
  }
  if (!seen.size) say('error', 'yolo-no-images', `No images were found${yamlFile ? ` for ${path.basename(yamlFile)}'s splits` : ` in ${target}`}`)
  return {
    yaml: yamlFile, root, names: data.names, kptShape: data.kptShape, kptNames: data.kptNames, flipIdx: data.flipIdx, splits,
    guess: guessTask(stats, data.kptShape), issues
  }
}

/**
 * The dataset's root: path, or the YAML's own folder without one. Ultralytics takes a relative path
 * from the folder it runs in, and Roboflow and Colab write ones from other computers, so a path
 * that isn't there falls back to the YAML's folder.
 */
async function rootOf(given: string | null, yamlDir: string, file: string, say: (s: Severity, code: string, m: string) => void) {
  if (!given) return yamlDir
  const root = path.resolve(yamlDir, given)
  if (await isDir(root)) return root
  say('info', 'yolo-root', `${file} gives path ${given}, which isn't a folder here, so its own folder is used`)
  return yamlDir
}

/**
 * A split's images, as get_img_files lists them: a folder's images at any depth, an image, or a
 * text file listing images, where ./ lines are relative to the list. Ultralytics takes other
 * lines from the folder it runs in; here they're tried from the list's folder and the root, and
 * CVAT's data/ prefix is dropped. When most of a list isn't there, images/<split> is read. Lists
 * are added a file at a time, never spread into a call, as walk explains.
 */
async function splitImages(
  root: string, split: string, spec: string | string[], say: (s: Severity, code: string, m: string) => void, exists: (file: string) => Promise<boolean>, reading: Reading
): Promise<string[]> {
  const entries: string[] = []
  if (typeof spec === 'string') {
    // Roboflow writes ../train/images beside its train folder; Ultralytics retries without the ../
    let p = path.resolve(root, spec)
    if (!(await stat(p)) && spec.startsWith('../')) p = path.resolve(root, spec.slice(3))
    entries.push(p)
  } else for (const s of spec) entries.push(path.resolve(root, s))
  const images: string[] = []
  const add = (files: string[]) => {
    for (const f of files) if (isYoloImage(f)) images.push(f)
  }
  for (const p of entries) {
    const info = await stat(p)
    if (!info) {
      say('warning', 'yolo-split-missing', `${split}: ${path.relative(root, p) || '.'} isn't there`)
      continue
    }
    if (info.isDirectory()) {
      add(await walk(p, reading))
      continue
    }
    if (isYoloImage(p)) {
      images.push(p)
      continue
    }
    const text = await fs.readFile(p, 'utf8').catch(() => null)
    if (text === null) {
      say('warning', 'yolo-split-missing', `${split}: ${path.relative(root, p)} can't be read, so its images are left out`)
      continue
    }
    const listed = text.trim().split(/\r\n|\r|\n/).map((l) => l.trim()).filter(Boolean)
    const dir = path.dirname(p)
    const found: string[] = []
    const missing: string[] = []
    const where = async (line: string) => {
      // get_img_files turns / into the system's separator first, so C:/data/a.jpg is C:\data\a.jpg on Windows
      const tries = line.startsWith('./') ? [path.resolve(dir, line)]
        : path.isAbsolute(line) ? [path.resolve(line)]
          : [path.resolve(dir, line), path.resolve(root, line), ...(line.startsWith('data/') ? [path.resolve(root, line.slice(5))] : [])]
      for (const t of tries) if (await exists(t)) return t
      return null
    }
    for (let i = 0; i < listed.length; i += BATCH) {
      reading.signal?.throwIfAborted()
      const hits = await Promise.all(listed.slice(i, i + BATCH).map(where))
      hits.forEach((hit, j) => (hit ? found.push(hit) : missing.push(listed[i + j])))
    }
    const fallback = path.join(root, 'images', split)
    if (missing.length && missing.length * 2 >= listed.length && (await isDir(fallback))) {
      say('info', 'yolo-list-fallback', `${split}: ${missing.length} of the ${listed.length} images ${path.basename(p)} lists aren't there, so ${path.relative(root, fallback)} is read instead`)
      add(await walk(fallback, reading))
      continue
    }
    if (missing.length) say('warning', 'yolo-list-missing', `${split}: ${missing.length} of the ${listed.length} images ${path.basename(p)} lists aren't there, such as ${short(missing[0], 120)}`)
    add(found)
  }
  return sortPaths(images)
}
