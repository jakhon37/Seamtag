// Ultralytics YOLO datasets on disk, for import: a data.yaml, a folder holding one, or a folder of
// images and labels without one. Splits are found as Ultralytics 8.4.211 finds them
// (check_det_dataset and BaseDataset.get_img_files in ultralytics/data/), each image's label file
// by img2label_paths, and every label file's rows are counted for guessing the task. Roboflow's
// ../train/images and CVAT's data/images/... lists are read as those tools mean them.
import { promises as fs } from 'fs'
import path from 'path'
import type { Issue, Severity } from '../shared/dataset'
import { readClasses, YOLO_CLASSES } from '../shared/formats'
import { pyFloat } from '../shared/pose'
import { countRows, type DataYaml, guessTask, isYoloImage, labelPathFor, labelText, newRowStats, readDataYaml, type TaskGuess } from '../shared/ultralytics'

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

const stat = (p: string) => fs.stat(p).catch(() => null)
const isDir = async (p: string) => (await stat(p))?.isDirectory() ?? false
const isFile = async (p: string) => (await stat(p))?.isFile() ?? false

/** Every file under a folder, as Python's glob('**') lists them: names starting with a dot are left out */
async function walk(dir: string, seen = new Set<string>()): Promise<string[]> {
  const real = await fs.realpath(dir).catch(() => dir)
  // A link back up the tree would go round for ever
  if (seen.has(real)) return []
  seen.add(real)
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  const out: string[] = []
  for (const e of entries) {
    if (e.name.startsWith('.')) continue
    const full = path.join(dir, e.name)
    const directory = e.isDirectory() || (e.isSymbolicLink() && (await isDir(full)))
    if (directory) out.push(...(await walk(full, seen)))
    else out.push(full)
  }
  return out
}

/**
 * Ultralytics' find_dataset_yaml: .yaml files at the top, else anywhere below, else .yml the same
 * way; of several, those named after the folder. Hidden folders aren't searched here.
 */
function datasetYamls(dir: string, files: string[]): string[] {
  for (const ext of ['.yaml', '.yml']) {
    const all = files.filter((f) => path.extname(f).toLowerCase() === ext)
    if (!all.length) continue
    const top = all.filter((f) => path.dirname(f) === dir)
    const found = top.length ? top : all
    if (found.length === 1) return found
    const named = found.filter((f) => path.basename(f, path.extname(f)) === path.basename(dir))
    return named.length ? named : found
  }
  return []
}

const sortPaths = (files: string[]) => files.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))

/**
 * The images of the dataset in a folder without data.yaml, by split: the folder after the last
 * images folder in their path (images/train/a.jpg), or before it (Roboflow's train/images/a.jpg)
 */
function splitOf(root: string, image: string): string {
  const parts = path.relative(root, image).split(path.sep)
  const folders = parts.slice(0, -1)
  const k = folders.lastIndexOf('images')
  if (k < 0) return ''
  return folders[k + 1] ?? folders[k - 1] ?? ''
}

const SPLIT_ORDER = ['train', 'val', 'valid', 'validation', 'test']
const splitRank = (name: string) => (SPLIT_ORDER.includes(name) ? SPLIT_ORDER.indexOf(name) : SPLIT_ORDER.length)

/**
 * A dataset from a data.yaml file, or a folder: its own data.yaml (found as Ultralytics finds it),
 * or, without one, the images under it, split by their folders, with names from a classes.txt at
 * the top or in labels/, or class_0, class_1 and so on. Nothing is written.
 */
export async function readYoloDataset(target: string): Promise<YoloDataset> {
  const issues: Issue[] = []
  const say = (severity: Severity, code: string, message: string, image: string | null = null) => issues.push({ severity, code, image, message })
  target = path.resolve(target)
  const empty = (root: string, yaml: string | null): YoloDataset =>
    ({ yaml, root, names: [], kptShape: null, kptNames: null, flipIdx: null, splits: [], guess: guessTask(newRowStats()), issues })
  const info = await stat(target)
  if (!info) {
    say('error', 'yolo-missing', `${target} isn't there`)
    return empty(target, null)
  }
  let yamlFile: string | null = null
  let files: string[] | null = null
  if (info.isFile()) {
    if (!/\.ya?ml$/i.test(target)) {
      say('error', 'yolo-not-yaml', `${path.basename(target)} isn't a data.yaml: pick the dataset's YAML file or its folder`)
      return empty(path.dirname(target), null)
    }
    yamlFile = target
  } else {
    files = await walk(target)
    const found = datasetYamls(target, files)
    if (found.length > 1) {
      say('error', 'yolo-many-yaml', `${target} holds ${found.length} YAML files (${found.map((f) => path.relative(target, f)).join(', ')}): pick the dataset's own`)
      return empty(target, null)
    }
    yamlFile = found[0] ?? null
  }

  let data: DataYaml
  let root: string
  const splits: YoloSplit[] = []
  if (yamlFile) {
    // Ultralytics reads it as UTF-8, leaving out what isn't
    const read = readDataYaml(await fs.readFile(yamlFile, 'utf8'), path.basename(yamlFile))
    data = read.data
    issues.push(...read.issues)
    root = await rootOf(data.path, path.dirname(yamlFile), path.basename(yamlFile), say)
    for (const name of ['train', 'val', 'test'] as const) {
      const spec = data[name]
      if (!spec) continue
      const images = await splitImages(root, name, spec, say)
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
  }

  // Each image's label file, looked for a few at a time
  const items = splits.flatMap((s) => s.items)
  for (let i = 0; i < items.length; i += BATCH) {
    await Promise.all(items.slice(i, i + BATCH).map(async (item) => {
      const label = labelPathFor(item.image, path.sep)
      if (await isFile(label)) item.label = label
    }))
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

  // What the rows look like, and the highest class number for a folder with no names
  const stats = newRowStats()
  const labels = [...owner.keys()]
  let top = -1
  let unreadable = 0
  for (let i = 0; i < labels.length; i += BATCH) {
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
        if (c !== null && Number.isInteger(c) && c > top) top = c
      }
    })
  }
  if (unreadable > 3) say('warning', 'yolo-not-text', `${unreadable} label files in all can't be read as UTF-8 text`)
  if (!yamlFile && !data.names.length) {
    data.names = Array.from({ length: top + 1 }, (_, i) => `class_${i}`)
    if (top >= 0) say('info', 'yolo-no-yaml', `${path.basename(target)} has neither data.yaml nor ${YOLO_CLASSES}, so the classes are named class_0 to class_${top}`)
  }
  const images = splits.reduce((n, s) => n + s.items.length, 0)
  if (!images) say('error', 'yolo-no-images', `No images were found${yamlFile ? ` for ${path.basename(yamlFile)}'s splits` : ` in ${target}`}`)
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
 * CVAT's data/ prefix is dropped. When most of a list isn't there, images/<split> is read.
 */
async function splitImages(root: string, split: string, spec: string | string[], say: (s: Severity, code: string, m: string) => void): Promise<string[]> {
  const entries: string[] = []
  if (typeof spec === 'string') {
    // Roboflow writes ../train/images beside its train folder; Ultralytics retries without the ../
    let p = path.resolve(root, spec)
    if (!(await stat(p)) && spec.startsWith('../')) p = path.resolve(root, spec.slice(3))
    entries.push(p)
  } else entries.push(...spec.map((s) => path.resolve(root, s)))
  const images: string[] = []
  for (const p of entries) {
    const info = await stat(p)
    if (!info) {
      say('warning', 'yolo-split-missing', `${split}: ${path.relative(root, p) || '.'} isn't there`)
      continue
    }
    if (info.isDirectory()) {
      images.push(...(await walk(p)).filter(isYoloImage))
      continue
    }
    if (isYoloImage(p)) {
      images.push(p)
      continue
    }
    const listed = (await fs.readFile(p, 'utf8')).trim().split(/\r\n|\r|\n/).map((l) => l.trim()).filter(Boolean)
    const dir = path.dirname(p)
    const found: string[] = []
    const missing: string[] = []
    const where = async (line: string) => {
      const tries = line.startsWith('./') ? [path.resolve(dir, line)]
        : path.isAbsolute(line) ? [line]
          : [path.resolve(dir, line), path.resolve(root, line), ...(line.startsWith('data/') ? [path.resolve(root, line.slice(5))] : [])]
      for (const t of tries) if (await isFile(t)) return t
      return null
    }
    for (let i = 0; i < listed.length; i += BATCH) {
      const hits = await Promise.all(listed.slice(i, i + BATCH).map(where))
      hits.forEach((hit, j) => (hit ? found.push(hit) : missing.push(listed[i + j])))
    }
    const fallback = path.join(root, 'images', split)
    if (missing.length && missing.length * 2 >= listed.length && (await isDir(fallback))) {
      say('info', 'yolo-list-fallback', `${split}: ${missing.length} of the ${listed.length} images ${path.basename(p)} lists aren't there, so ${path.relative(root, fallback)} is read instead`)
      images.push(...(await walk(fallback)).filter(isYoloImage))
      continue
    }
    if (missing.length) say('warning', 'yolo-list-missing', `${split}: ${missing.length} of the ${listed.length} images ${path.basename(p)} lists aren't there, such as ${missing[0]}`)
    images.push(...found)
  }
  return sortPaths(images.filter(isYoloImage))
}
