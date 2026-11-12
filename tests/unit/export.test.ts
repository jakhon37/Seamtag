import { existsSync, promises as fsp, lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import yaml from 'js-yaml'
import { afterAll, afterEach, describe, expect, test, vi } from 'vitest'
import { readDataset } from '../../src/main/dataset'
import {
  type ExportOptions, type ExportProgress, folderProblem, forgetChecks, keypointSchemaOf, moreIssues, preflight, runExport, shareOut, shuffled, splitFolder
} from '../../src/main/export'
import { setCacheDir } from '../../src/main/images'
import { encodeMask } from '../../src/main/masks'
import { openProject } from '../../src/main/project'
import { type CocoEntry, writeCoco } from '../../src/shared/coco'
import { writePoseYaml } from '../../src/shared/pose'
import { verifyRows } from '../../src/shared/ultralytics'

// Whole folders exported with real images, made by sharp as e2e/helpers.ts makes them
const cache = mkdtempSync(path.join(tmpdir(), 'seamtag-cache-'))
setCacheDir(cache)
const made: string[] = [cache]
const folder = () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-export-'))
  made.push(dir)
  return dir
}
afterAll(() => {
  for (const dir of made) {
    try {
      rmSync(dir, { recursive: true, force: true, maxRetries: 5 })
    } catch {}
  }
})
afterEach(() => {
  vi.restoreAllMocks()
})

let shade = 0
async function picture(file: string, [width, height]: [number, number] = [40, 30]) {
  mkdirSync(path.dirname(file), { recursive: true })
  shade += 7
  const pixels = sharp({ create: { width, height, channels: 3, background: { r: shade % 256, g: (shade * 5) % 256, b: 80 } } })
  const ext = path.extname(file).toLowerCase()
  await (ext === '.jpg' ? pixels.jpeg() : ext === '.gif' ? pixels.gif() : pixels.png()).toFile(file)
}
const shape = (label: string, shape_type: string, points: number[][], extra: Record<string, unknown> = {}) =>
  ({ label, points, group_id: null, description: '', shape_type, flags: {}, mask: null, ...extra })
const ai = (label: string, points: number[][]) => shape(label, 'polygon', points, { flags: { ai: true }, score: 0.8 })
const square = (label: string, x: number, y: number, side = 10, extra: Record<string, unknown> = {}) =>
  shape(label, 'polygon', [[x, y], [x + side, y], [x + side, y + side], [x, y + side]], extra)
const labelme = (image: string, shapes: unknown[]) =>
  JSON.stringify({ version: '5.5.0', flags: {}, shapes, imagePath: image, imageData: null, imageHeight: 30, imageWidth: 40 })

/** A folder of images, each with a LabelMe file of the shapes given, or none (null), opened as the window opens it */
async function folderWith(files: Record<string, unknown[] | null>, size: [number, number] = [40, 30]) {
  const dir = folder()
  for (const [name, shapes] of Object.entries(files)) {
    await picture(path.join(dir, name), size)
    if (shapes) writeFileSync(path.join(dir, name.replace(/\.[^./]+$/, '.json')), labelme(path.basename(name), shapes))
  }
  await openProject(dir, null)
  return dir
}
const options = (o: Partial<ExportOptions> = {}): ExportOptions =>
  ({ format: 'coco', split: { mode: 'none' }, images: 'copy', backgrounds: false, leaveAi: false, keypoints: null, prefer: 'labelme', labels: [], ...o })
const newFolder = () => path.join(folder(), 'dataset')
const text = (dir: string, file: string) => readFileSync(path.join(dir, file), 'utf8')
const json = (dir: string, file: string) => JSON.parse(text(dir, file))
/** Every file under a folder, as paths in it */
const filesIn = (dir: string): string[] => readdirSync(dir, { recursive: true, withFileTypes: true }).filter((e) => e.isFile())
  .map((e) => path.relative(dir, path.join(e.parentPath, e.name)).split(path.sep).join('/')).sort()
const codes = (p: { groups: { code: string; severity: string; count: number }[] }) => p.groups.map((g) => [g.severity, g.code, g.count])

describe('COCO', () => {
  test('instances_default.json holds every labelled image, copied under images/default with its subfolders, categories from the label list first', async () => {
    const dir = await folderWith({
      'a.png': [square('cat', 2, 2), shape('dog', 'rectangle', [[20, 5], [30, 25]]), shape('cat', 'circle', [[10, 20], [14, 20]])],
      'sub/b.png': [square('bird', 5, 5)], 'n.png': null
    })
    const out = newFolder()
    const result = await runExport(dir, options({ labels: ['dog', 'cat'] }), out, { version: '9.9.9' })
    expect(result).toMatchObject({ stopped: false, out, images: { copied: 2, linked: 0, fellBack: 0, left: 0 }, files: 3, written: ['annotations/instances_default.json'] })
    expect(result.splits).toEqual([{ name: 'default', folder: 'default', images: 2, objects: 4, backgrounds: 0, posed: 0 }])
    expect(filesIn(out)).toEqual(['annotations/instances_default.json', 'images/default/a.png', 'images/default/sub/b.png'])
    expect(readFileSync(path.join(out, 'images/default/sub/b.png'))).toEqual(readFileSync(path.join(dir, 'sub/b.png')))
    const coco = json(out, 'annotations/instances_default.json')
    expect(coco.images.map((i: any) => [i.id, i.file_name, i.width, i.height])).toEqual([[1, 'images/default/a.png', 40, 30], [2, 'images/default/sub/b.png', 40, 30]])
    expect(coco.categories.map((c: any) => [c.id, c.name])).toEqual([[1, 'dog'], [2, 'cat'], [3, 'bird']])
    expect(coco.annotations.map((a: any) => [a.id, a.image_id, a.category_id])).toEqual([[1, 1, 2], [2, 1, 1], [3, 1, 2], [4, 2, 3]])
    expect(coco.info).toEqual({ description: 'Exported by Seamtag 9.9.9', url: '', version: '9.9.9', year: new Date().getFullYear(), contributor: '', date_created: expect.any(String) })
    expect(Date.parse(coco.info.date_created)).toBeGreaterThan(Date.now() - 60000)
    // The never-labelled image is left out, and said to be
    expect(result.accepted).toEqual([{ severity: 'info', title: 'Never-labelled images left out', count: 1 }])
    // Nothing is left beside the dataset
    expect(readdirSync(path.dirname(out))).toEqual(['dataset'])
  })

  test("a split of hundreds of images is written a chunk at a time, and the file is writeCoco's for all of them at once, byte for byte", async () => {
    const dir = folder()
    for (let i = 0; i < 300; i++) {
      const name = `img${String(i).padStart(3, '0')}.png`
      await picture(path.join(dir, name), [8, 6])
      const shapes = i % 7 === 3 ? [] : [square(i % 2 ? 'cat' : 'dog', 1, 1, 3), shape('cat', 'rectangle', [[0, 0], [7, 5]])]
      writeFileSync(path.join(dir, name.replace('.png', '.json')), labelme(name, shapes))
    }
    await openProject(dir, null)
    const out = newFolder()
    const result = await runExport(dir, options(), out, { version: '1.0.0' })
    expect(result.splits[0]).toMatchObject({ images: 300, backgrounds: 43 })
    const written = text(out, 'annotations/instances_default.json')
    const parsed = JSON.parse(written)
    const images = await readDataset(dir)
    const entries: CocoEntry[] = images.map((image) => ({ image, fileName: `images/default/${image.name}` }))
    expect(written).toBe(writeCoco(entries, { task: 'instances', labels: ['dog', 'cat'], info: parsed.info }).json)
    expect(parsed.annotations.length).toBe(514)
  })

  test("with keypoints: person_keypoints files beside the instances files, the keypoints file's names and flip pairs, and the instances leave the points to them", async () => {
    const dir = await folderWith({
      'a.png': [
        shape('person', 'rectangle', [[5, 5], [30, 25]], { group_id: 1 }), shape('nose', 'point', [[12, 10]], { group_id: 1 }),
        shape('left_eye', 'point', [[10, 8]], { group_id: 1, flags: { difficult: true } }), square('cat', 30, 2, 5)
      ]
    })
    const schema = keypointSchemaOf(await readDataset(dir))
    expect(schema).toEqual({ classes: { person: ['nose', 'left_eye'] } })
    const file = path.join(folder(), 'keypoints.yaml')
    writeFileSync(file, writePoseYaml({ classes: { person: ['nose', 'left_eye', 'right_eye'] }, flip: { person: [['left_eye', 'right_eye']] } }))
    const blocked = await preflight(dir, options({ format: 'coco-keypoints' }))
    expect(codes(blocked)).toContainEqual(['error', 'export-no-keypoints', 1])
    const check = await preflight(dir, options({ format: 'coco-keypoints', keypoints: file }))
    expect(check.blocked).toBe(false)
    expect(check.keypoints).toEqual({ file, classes: [{ name: 'person', keypoints: 3 }], flip: true, skeleton: false })
    expect(check.splits).toEqual([{ name: 'default', folder: 'default', images: 1, objects: 2, backgrounds: 0, posed: 1 }])
    const out = newFolder()
    const result = await runExport(dir, options({ format: 'coco-keypoints', keypoints: file }), out)
    expect(result.written).toEqual(['annotations/instances_default.json', 'annotations/person_keypoints_default.json'])
    const posed = json(out, 'annotations/person_keypoints_default.json')
    expect(posed.categories).toEqual([{ id: 1, name: 'person', supercategory: '', keypoints: ['nose', 'left_eye', 'right_eye'], skeleton: [] }])
    expect(posed.annotations.map((a: any) => [a.category_id, a.keypoints, a.num_keypoints])).toEqual([[1, [12, 10, 2, 10, 8, 1, 0, 0, 0], 2]])
    const objects = json(out, 'annotations/instances_default.json')
    expect(objects.annotations.map((a: any) => a.category_id)).toEqual([1, 2])
  })

  test("left in place: no images are written, file_name is each image's path in the open folder, and the summary says to use it as the root", async () => {
    const dir = await folderWith({ 'x/a.png': [square('cat', 2, 2)] })
    const out = newFolder()
    const result = await runExport(dir, options({ images: 'leave' }), out)
    expect(filesIn(out)).toEqual(['annotations/instances_default.json'])
    expect(json(out, 'annotations/instances_default.json').images[0].file_name).toBe('x/a.png')
    expect(result.images).toEqual({ copied: 0, linked: 0, fellBack: 0, left: 1 })
    expect(result.notes[0]).toBe(`Images stay where they are: each file_name is the image's path in the open folder, so use that folder (${dir}) as the image root.`)
    await expect(runExport(dir, options({ format: 'yolo-detect', images: 'leave' }), newFolder())).rejects.toThrow('Only COCO can leave the images where they are')
  })

  test('by ratio, a split with a share but no images still gets its file, holding none, and the check says so first', async () => {
    const files: Record<string, unknown[]> = {}
    for (let i = 0; i < 5; i++) files[`e${i}.png`] = [square('cat', 1, 1)]
    const dir = await folderWith(files)
    const split = { mode: 'ratio', train: 80, val: 10, test: 10, seed: 3 } as const
    const check = await preflight(dir, options({ split }))
    expect(check.splits.map((s) => [s.name, s.images])).toEqual([['train', 4], ['val', 1], ['test', 0]])
    expect(check.groups.find((g) => g.code === 'export-empty-split')!.issues.map((i) => i.message)).toEqual(['No images go to test, so instances_test.json holds none'])
    const out = newFolder()
    const result = await runExport(dir, options({ split }), out, { version: '1.0.0' })
    expect(result.written).toEqual(['annotations/instances_test.json', 'annotations/instances_train.json', 'annotations/instances_val.json'])
    const empty = json(out, 'annotations/instances_test.json')
    expect(empty).toMatchObject({ images: [], annotations: [], categories: [{ id: 1, name: 'cat', supercategory: '' }] })
    expect(text(out, 'annotations/instances_test.json')).toBe(writeCoco([], { task: 'instances', labels: ['cat'], info: empty.info }).json)
    expect(result.accepted).toContainEqual({ severity: 'warning', title: 'Splits with no images', count: 1 })
    // YOLO says so too, and data.yaml names no test folder
    const yolo = await preflight(dir, options({ format: 'yolo-detect', split }))
    expect(yolo.groups.find((g) => g.code === 'export-empty-split')!.issues.map((i) => i.message)).toEqual(['No images go to test, so data.yaml names no test folder'])
  })

  test("an image held inside its LabelMe file goes in under that file's name, with the extension its bytes give", async () => {
    const base = folder()
    const file = path.join(base, 'photo.json')
    const hold = (bytes: Buffer) => writeFileSync(file, JSON.stringify({
      version: '5.5.0', flags: {}, shapes: [square('cat', 2, 2)], imagePath: 'gone/photo.bmp', imageData: bytes.toString('base64'), imageHeight: 30, imageWidth: 40
    }))
    const pixels = () => sharp({ create: { width: 40, height: 30, channels: 3, background: '#336699' } })
    const png = await pixels().png().toBuffer()
    hold(png)
    const { root } = await openProject(file, null)
    const coco = newFolder()
    await runExport(root, options(), coco)
    expect(filesIn(coco)).toEqual(['annotations/instances_default.json', 'images/default/photo.png'])
    expect(json(coco, 'annotations/instances_default.json').images[0].file_name).toBe('images/default/photo.png')
    expect(readFileSync(path.join(coco, 'images/default/photo.png'))).toEqual(png)
    const yolo = newFolder()
    await runExport(root, options({ format: 'yolo-detect' }), yolo)
    expect(filesIn(yolo)).toEqual(['data.yaml', 'images/photo.png', 'labels/photo.txt'])
    expect(text(yolo, 'labels/photo.txt')).toBe('0 0.175000 0.233333 0.250000 0.333333\n')
    // A JPEG held inside it goes in as photo.jpg
    hold(await pixels().jpeg().toBuffer())
    await openProject(file, null)
    const jpeg = newFolder()
    await runExport(root, options({ format: 'yolo-detect' }), jpeg)
    expect(filesIn(jpeg)).toEqual(['data.yaml', 'images/photo.jpg', 'labels/photo.txt'])
  })
})

describe('YOLO', () => {
  const shapes = () => [
    square('cat', 2, 2), shape('dog', 'rectangle', [[20, 5], [30, 25]]), shape('cat', 'oriented_rectangle', [[5, 15], [15, 15], [15, 20], [5, 20]]),
    shape('dog', 'circle', [[30, 22], [33, 22]]), shape('cat', 'mask', [[1, 20], [4, 22]], { mask: null }), shape('cat', 'line', [[1, 1], [5, 5]])
  ]
  test.each(['yolo-detect', 'yolo-segment', 'yolo-obb'] as const)('%s with no split: images/, labels/ and data.yaml whose train and val both point at images', async (format) => {
    const mask = await encodeMask(Uint8Array.from([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]), 4, 3)
    const list = shapes()
    list[4] = { ...list[4], mask }
    const dir = await folderWith({ 'a.png': list, 'sub/b.png': [], 'n.png': null })
    const out = newFolder()
    const result = await runExport(dir, options({ format, labels: ['dog'] }), out)
    expect(filesIn(out)).toEqual(['data.yaml', 'images/a.png', 'images/sub/b.png', 'labels/a.txt', 'labels/sub/b.txt'])
    const task = format.slice(5) as 'detect' | 'segment' | 'obb'
    const rows = text(out, 'labels/a.txt').trim().split('\n')
    // One row an object: the mask's two parts are one, the line is left out
    expect(rows.map((r) => r.split(' ')[0])).toEqual(['1', '0', '1', '0', '1'])
    expect(verifyRows(text(out, 'labels/a.txt'), task, 2)).toMatchObject({ status: 'ok', repeated: [] })
    expect(text(out, 'labels/sub/b.txt')).toBe('')
    expect(yaml.load(text(out, 'data.yaml'))).toEqual({ train: 'images', val: 'images', nc: 2, names: { 0: 'dog', 1: 'cat' } })
    expect(result.notes).toContain("With no split, images/ and labels/ hold every image, and data.yaml's train and val both point at images, as Ultralytics needs a val entry.")
    expect(result.splits).toEqual([{ name: 'all', folder: '', images: 2, objects: 5, backgrounds: 1, posed: 0 }])
  })

  test("pose: a row for each posed object with the keypoints file's classes, and data.yaml's kpt_shape, kpt_names and flip_idx", async () => {
    const dir = await folderWith({
      'a.png': [
        shape('person', 'rectangle', [[4, 4], [20, 24]], { group_id: 2 }), shape('right_eye', 'point', [[10, 8]], { group_id: 2 }),
        shape('nose', 'point', [[12, 10]], { group_id: 2 })
      ]
    })
    const file = path.join(folder(), 'pose.yaml')
    writeFileSync(file, writePoseYaml({ classes: { person: ['nose', 'left_eye', 'right_eye'] }, flip: { person: [['left_eye', 'right_eye']] } }))
    const out = newFolder()
    const result = await runExport(dir, options({ format: 'yolo-pose', keypoints: file, labels: ['cat'] }), out)
    expect(text(out, 'labels/a.txt')).toBe('0 0.300000 0.466667 0.400000 0.666667 0.300000 0.333333 2 0 0 0 0.250000 0.266667 2\n')
    expect(yaml.load(text(out, 'data.yaml'))).toEqual({ train: 'images', val: 'images', kpt_shape: [3, 3], flip_idx: [0, 2, 1], nc: 1, names: { 0: 'person' }, kpt_names: { 0: ['nose', 'left_eye', 'right_eye'] } })
    expect(result.splits[0]).toMatchObject({ images: 1, objects: 1, posed: 1 })
  })

  test("an image Ultralytics doesn't read is left out with a warning, and two images whose label files would be one stop the export", async () => {
    const dir = await folderWith({ 'a.jpg': null, 'a.png': null, 'c.gif': [square('cat', 1, 1)], 'd.png': [square('cat', 1, 1)] })
    const check = await preflight(dir, options({ format: 'yolo-detect', backgrounds: true }))
    expect(check.blocked).toBe(true)
    const same = check.groups.find((g) => g.code === 'export-same-label')!
    expect(same.issues[0].message).toBe("a.jpg and a.png would share one label file, labels/a.txt, as Ultralytics finds label files by name: rename one of them")
    expect(check.groups.find((g) => g.code === 'export-not-yolo-image')!.issues[0]).toMatchObject({ severity: 'warning', image: 'c.gif' })
    await expect(runExport(dir, options({ format: 'yolo-detect', backgrounds: true }), newFolder())).rejects.toThrow(/a\.jpg and a\.png would share one label file/)
    // COCO's images keep their names whole, so it can take both
    expect((await preflight(dir, options({ backgrounds: true }))).blocked).toBe(false)
  })

  test('pose with no keypoints file: the check says the file is needed, rather than failing', async () => {
    const dir = await folderWith({ 'a.png': [shape('person', 'rectangle', [[4, 4], [20, 24]], { group_id: 2 }), shape('nose', 'point', [[12, 10]], { group_id: 2 })] })
    const check = await preflight(dir, options({ format: 'yolo-pose' }))
    expect(check.blocked).toBe(true)
    expect(codes(check)).toEqual([['error', 'export-no-keypoints', 1]])
  })

  test("classes are numbered in the folder's order, whatever split the seed makes", async () => {
    const dir = await folderWith({ 'p0.png': [square('bird', 1, 1)], 'p1.png': [square('ant', 1, 1)] })
    // A seed that puts p1.png in train and p0.png in val
    const seed = Array.from({ length: 50 }, (_, k) => k).find((k) => shuffled(['p0.png', 'p1.png'], k)[0] === 'p1.png')!
    const out = newFolder()
    await runExport(dir, options({ format: 'yolo-detect', split: { mode: 'ratio', train: 50, val: 50, test: 0, seed } }), out)
    expect(filesIn(out).filter((f) => f.startsWith('labels/'))).toEqual(['labels/train/p1.txt', 'labels/val/p0.txt'])
    expect(yaml.load(text(out, 'data.yaml'))).toMatchObject({ names: { 0: 'bird', 1: 'ant' } })
    expect(text(out, 'labels/train/p1.txt').split(' ')[0]).toBe('1')
  })
})

describe('splits', () => {
  test('by ratio: sorted by path, shuffled with the seed, then cut by the shares; the same seed gives the same split', () => {
    const names = Array.from({ length: 30 }, (_, i) => `f${i}.png`)
    expect(shuffled(names, 4)).toEqual(shuffled([...names].reverse(), 4))
    expect(shuffled(names, 4)).not.toEqual(shuffled(names, 5))
    expect([...shuffled(names, 4)].sort()).toEqual([...names].sort())
    expect(shareOut(30, [80, 10, 10])).toEqual([24, 3, 3])
    expect(shareOut(7, [70, 20, 10])).toEqual([5, 1, 1])
    expect(shareOut(3, [80, 10, 10])).toEqual([3, 0, 0])
    expect(shareOut(2, [50, 50, 0])).toEqual([1, 1, 0])
    expect(shareOut(1, [10, 90, 0])).toEqual([0, 1, 0])
    expect(shareOut(0, [80, 10, 10])).toEqual([0, 0, 0])
  })

  test('by ratio: each split has its share of the images, the same seed writes the same split, and data.yaml names the splits', async () => {
    const files: Record<string, unknown[]> = {}
    for (let i = 0; i < 20; i++) files[`p${i}.png`] = [square('cat', 1, 1)]
    const dir = await folderWith(files)
    const split = { mode: 'ratio', train: 60, val: 25, test: 15, seed: 11 } as const
    const one = newFolder(), two = newFolder()
    const result = await runExport(dir, options({ format: 'yolo-detect', split }), one)
    await runExport(dir, options({ format: 'yolo-detect', split }), two)
    expect(result.splits.map((s) => [s.name, s.images])).toEqual([['train', 12], ['val', 5], ['test', 3]])
    expect(filesIn(two)).toEqual(filesIn(one))
    const other = await preflight(dir, options({ format: 'yolo-detect', split: { ...split, seed: 12 } }))
    expect(other.splits.map((s) => s.images)).toEqual([12, 5, 3])
    expect(yaml.load(text(one, 'data.yaml'))).toMatchObject({ train: 'images/train', val: 'images/val', test: 'images/test' })
    // Another seed puts some image elsewhere
    const three = newFolder()
    await runExport(dir, options({ format: 'yolo-detect', split: { ...split, seed: 12 } }), three)
    expect(filesIn(three)).not.toEqual(filesIn(one))
  })

  test("by folders: the first train, val, valid, validation or test folder in any case decides and leaves the path; images under none go to train, with a warning", async () => {
    expect(splitFolder('x/Valid/y/a.png')).toEqual({ split: 'val', rel: 'x/y/a.png' })
    expect(splitFolder('train/test/a.png')).toEqual({ split: 'train', rel: 'test/a.png' })
    expect(splitFolder('a/train.png')).toEqual({ split: null, rel: 'a/train.png' })
    const dir = await folderWith({
      'Train/a.png': [square('cat', 1, 1)], 'x/valid/b.png': [square('cat', 1, 1)], 'validation/c.png': [square('dog', 1, 1)],
      'TEST/d.png': [square('cat', 1, 1)], 'e.png': [square('cat', 1, 1)]
    })
    const out = newFolder()
    const result = await runExport(dir, options({ split: { mode: 'folders' } }), out)
    expect(filesIn(out)).toEqual([
      'annotations/instances_test.json', 'annotations/instances_train.json', 'annotations/instances_val.json',
      'images/test/d.png', 'images/train/a.png', 'images/train/e.png', 'images/val/c.png', 'images/val/x/b.png'
    ])
    expect(result.accepted).toEqual([{ severity: 'warning', title: 'Images under no train, val or test folder', count: 1 }])
    expect(json(out, 'annotations/instances_val.json').images.map((i: any) => i.file_name)).toEqual(['images/val/c.png', 'images/val/x/b.png'])
  })

  test('folders that put two images on one path stop the export', async () => {
    const dir = await folderWith({ 'train/a.png': [square('cat', 1, 1)], 'a.png': [square('cat', 1, 1)] })
    const check = await preflight(dir, options({ split: { mode: 'folders' } }))
    expect(check.groups.find((g) => g.severity === 'error')!.issues[0].message).toBe('a.png and train/a.png would both be written to images/train/a.png: rename one of them')
  })
})

describe('the output folder', () => {
  test('must be new or empty, outside the image folder and the folder the label files are saved in', async () => {
    const dir = await folderWith({ 'a.png': [square('cat', 1, 1)] })
    const base = folder()
    const labels = path.join(base, 'labels')
    mkdirSync(labels)
    await openProject(dir, labels)
    mkdirSync(path.join(base, 'full'))
    writeFileSync(path.join(base, 'full', 'x.txt'), 'x')
    mkdirSync(path.join(base, 'empty'))
    writeFileSync(path.join(base, 'file'), 'x')
    symlinkSync(dir, path.join(base, 'link'), 'dir')
    expect(await folderProblem(dir, dir)).toBe("That's the image folder: choose a new or empty folder outside it")
    expect(await folderProblem(dir, path.join(dir, 'new'))).toBe("That's inside the image folder: choose a new or empty folder outside it")
    expect(await folderProblem(dir, path.join(base, 'link', 'new'))).toBe("That's inside the image folder: choose a new or empty folder outside it")
    expect(await folderProblem(dir, labels)).toBe("That's the folder the label files are saved in: choose a new or empty folder outside it")
    expect(await folderProblem(dir, path.join(labels, 'x'))).toBe("That's inside the folder the label files are saved in: choose a new or empty folder outside it")
    expect(await folderProblem(dir, path.join(base, 'full'))).toBe("That folder isn't empty: choose a new or empty folder, so nothing in it is written over")
    expect(await folderProblem(dir, path.join(base, 'file'))).toBe("That's a file: choose a new or empty folder")
    symlinkSync(path.join(base, 'empty'), path.join(base, 'to-empty'), 'dir')
    expect(await folderProblem(dir, path.join(base, 'to-empty'))).toBe("That's a link to a folder: choose the folder itself")
    expect(await folderProblem(dir, path.join(base, 'no', 'such'))).toBe("The folder it would go in doesn't exist")
    expect(await folderProblem(dir, 'relative')).toBe('Choose a folder to export to')
    expect(await folderProblem(dir, path.join(base, 'new'))).toBeNull()
    expect(await folderProblem(dir, path.join(base, 'empty'))).toBeNull()
    await expect(runExport(dir, options(), path.join(dir, 'new'))).rejects.toThrow("That's inside the image folder")
    // An empty folder takes the dataset, and nothing else is left
    await openProject(dir, null)
    await runExport(dir, options(), path.join(base, 'empty'))
    expect(filesIn(path.join(base, 'empty'))).toEqual(['annotations/instances_default.json', 'images/default/a.png'])
    expect(readdirSync(base).sort()).toEqual(['empty', 'file', 'full', 'labels', 'link', 'to-empty'])
  })
})

describe('images', () => {
  test("hard links where they can be made, and copies, counted, where linking fails, as across drives", async () => {
    const dir = await folderWith({ 'a.png': [square('cat', 1, 1)], 'b.png': [square('cat', 1, 1)] })
    const out = newFolder()
    const linked = await runExport(dir, options({ images: 'link' }), out)
    expect(linked.images).toEqual({ copied: 0, linked: 2, fellBack: 0, left: 0 })
    expect(statSync(path.join(out, 'images/default/a.png')).ino).toBe(statSync(path.join(dir, 'a.png')).ino)
    expect(linked.notes).toContain("A hard link shares an image's file with the open folder, so editing one edits both. Where a link can't be made, as across drives, the image is copied.")
    vi.spyOn(fsp, 'link').mockRejectedValue(Object.assign(new Error('cross-device link not permitted'), { code: 'EXDEV' }))
    const other = newFolder()
    const copied = await runExport(dir, options({ images: 'link' }), other)
    expect(copied.images).toEqual({ copied: 0, linked: 0, fellBack: 2, left: 0 })
    expect(statSync(path.join(other, 'images/default/a.png')).ino).not.toBe(statSync(path.join(dir, 'a.png')).ino)
    expect(readFileSync(path.join(other, 'images/default/a.png'))).toEqual(readFileSync(path.join(dir, 'a.png')))
  })

  test('an image that is a symbolic link is hard-linked as the file it points to, so the dataset never holds the link', async () => {
    const base = folder()
    await picture(path.join(base, 'pool', 'x.png'))
    const dir = path.join(base, 'imgs')
    mkdirSync(dir)
    // A relative link, as git-annex and DVC make them
    symlinkSync(path.join('..', 'pool', 'x.png'), path.join(dir, 'a.png'), 'file')
    writeFileSync(path.join(dir, 'a.json'), labelme('a.png', [square('cat', 1, 1)]))
    await openProject(dir, null)
    const out = newFolder()
    const result = await runExport(dir, options({ format: 'yolo-detect', images: 'link' }), out)
    expect(result.images).toEqual({ copied: 0, linked: 1, fellBack: 0, left: 0 })
    const file = path.join(out, 'images', 'a.png')
    expect(lstatSync(file).isSymbolicLink()).toBe(false)
    expect(statSync(file).ino).toBe(statSync(path.join(base, 'pool', 'x.png')).ino)
    expect(readFileSync(file)).toEqual(readFileSync(path.join(base, 'pool', 'x.png')))
  })

  test('never-labelled images are left out unless chosen as backgrounds; images labelled with no shapes always go in', async () => {
    const dir = await folderWith({ 'a.png': [square('cat', 1, 1)], 'empty.png': [], 'never.png': null })
    const out = newFolder()
    const result = await runExport(dir, options({ format: 'yolo-segment' }), out)
    expect(filesIn(out)).toEqual(['data.yaml', 'images/a.png', 'images/empty.png', 'labels/a.txt', 'labels/empty.txt'])
    expect(result.splits[0]).toMatchObject({ images: 2, backgrounds: 1 })
    const all = newFolder()
    const withBackgrounds = await runExport(dir, options({ format: 'yolo-segment', backgrounds: true }), all)
    expect(text(all, 'labels/never.txt')).toBe('')
    expect(withBackgrounds.splits[0]).toMatchObject({ images: 3, backgrounds: 2 })
    expect(withBackgrounds.accepted).toEqual([{ severity: 'info', title: 'Never-labelled images as backgrounds', count: 1 }])
  })
})

describe('what stops an export', () => {
  test("AI proposals not yet accepted stop it unless left out; then an image of proposals only is left out, and issues keep the label file's shape numbers", async () => {
    const dir = await folderWith({
      'a.png': [ai('cat', [[1, 1], [9, 1], [9, 9]]), shape('cat', 'polygon', [[1, 1], [5, 5]]), square('dog', 10, 10)],
      'b.png': [ai('cat', [[1, 1], [9, 1], [9, 9]])]
    })
    const blocked = await preflight(dir, options())
    expect(codes(blocked)).toContainEqual(['error', 'export-ai-pending', 2])
    expect(blocked.groups[0].issues[0]).toMatchObject({ image: 'a.png', shape: 0, message: 'a.png has 1 AI proposal not yet accepted: accept or reject it, or leave unaccepted proposals out' })
    await expect(runExport(dir, options(), newFolder())).rejects.toThrow("The dataset can't be exported: a.png has 1 AI proposal")
    const check = await preflight(dir, options({ leaveAi: true }))
    expect(check.blocked).toBe(false)
    expect(codes(check)).toEqual([['warning', 'coco-too-few-points', 1], ['info', 'export-ai-left-out', 1], ['info', 'export-ai-only', 1]])
    const few = check.groups[0].issues[0]
    expect(few).toMatchObject({ image: 'a.png', shape: 1 })
    expect(few.message).toBe("a.png: shape 2, a polygon labelled “cat”, has 2 points, and COCO needs 3, so it's left out")
    expect(check.groups[2].issues[0].message).toBe("b.png holds only 1 AI proposal not yet accepted, so it's left out, as not yet labelled")
    const out = newFolder()
    await runExport(dir, options({ leaveAi: true }), out)
    expect(json(out, 'annotations/instances_default.json').annotations.map((a: any) => a.category_id)).toEqual([2])
    expect(filesIn(out)).toEqual(['annotations/instances_default.json', 'images/default/a.png'])
  })

  test("images sharing one label file, label files that can't be read, and a folder with nothing to export", async () => {
    const dir = await folderWith({ 'a.png': [square('cat', 1, 1)], 'b.png': null, 'c.png': null })
    await picture(path.join(dir, 'a.jpg'))
    writeFileSync(path.join(dir, 'b.json'), '{ not json')
    await openProject(dir, null)
    const check = await preflight(dir, options())
    expect(codes(check).filter(([s]) => s === 'error')).toEqual([['error', 'export-shared-label-file', 1], ['error', 'coco-broken-labels', 1]])
    expect(check.groups.find((g) => g.code === 'export-shared-label-file')!.issues[0].message).toBe("a.jpg and a.png share one label file, a.json, so whose labels are whose can't be told: give each image its own name")
    // Every image sharing one -O labels.json is one problem, its images named a few at a time
    const shared = folder()
    for (const name of ['p.png', 'q.png', 'r.png', 's.png', 't.png']) await picture(path.join(shared, name))
    const all = path.join(folder(), 'labels.json')
    writeFileSync(all, labelme('p.png', [square('cat', 1, 1)]))
    await openProject(shared, all)
    const one = (await preflight(shared, options())).groups.find((g) => g.code === 'export-shared-label-file')!
    expect(one.issues.map((i) => i.message)).toEqual(["p.png, q.png, r.png and 2 other images share one label file, labels.json, so whose labels are whose can't be told: give each image its own name"])
    const none = await folderWith({ 'x.png': null, 'y.png': null })
    const nothing = await preflight(none, options())
    expect(nothing.groups[0]).toMatchObject({ severity: 'error', code: 'export-nothing', title: 'Nothing to export' })
    expect(nothing.groups[0].issues[0].message).toBe('None of the images has labels to export: label some, or include the 2 never-labelled images as backgrounds')
  })

  test('Stop leaves nothing behind, in a new folder or an empty one', async () => {
    const files: Record<string, unknown[]> = {}
    for (let i = 0; i < 40; i++) files[`s${i}.png`] = [square('cat', 1, 1)]
    const dir = await folderWith(files)
    const copy = fsp.copyFile.bind(fsp)
    for (const [format, existing] of [['yolo-detect', false], ['coco', true], ['coco-keypoints', false]] as const) {
      const base = folder()
      const out = path.join(base, 'dataset')
      if (existing) mkdirSync(out)
      const stop = new AbortController()
      // Stopped with some images and label files written, and a COCO file begun
      let copies = 0, during: string[] = []
      vi.spyOn(fsp, 'copyFile').mockImplementation(async (...args: Parameters<typeof fsp.copyFile>) => {
        await copy(...args)
        if (++copies === 5) {
          during = filesIn(base)
          stop.abort()
        }
      })
      const file = path.join(base, 'pose.yaml')
      writeFileSync(file, writePoseYaml({ classes: { cat: ['a'] } }))
      const result = await runExport(dir, options({ format, keypoints: file }), out, { signal: stop.signal })
      expect(result.stopped).toBe(true)
      expect(during.length).toBeGreaterThan(5)
      expect(during.every((f) => f === 'pose.yaml' || f.startsWith('.dataset.export-'))).toBe(true)
      expect(readdirSync(base).sort()).toEqual(existing ? ['dataset', 'pose.yaml'] : ['pose.yaml'])
      if (existing) expect(readdirSync(out)).toEqual([])
      vi.restoreAllMocks()
    }
    // A stop while the labels are read leaves nothing either
    const stop = new AbortController()
    stop.abort()
    const base = folder()
    expect(await runExport(dir, options(), path.join(base, 'dataset'), { signal: stop.signal })).toMatchObject({ stopped: true })
    expect(readdirSync(base)).toEqual([])
  })

  describe('while images are being written', () => {
    // One image in each of 40 folders, so each image written makes folders of its own; copies
    // under way end over the next 100 ms, as on a slow disk
    const nested = async () => {
      const files: Record<string, unknown[]> = {}
      for (let i = 0; i < 40; i++) files[`d${i}/s.png`] = [square('cat', 1, 1)]
      return folderWith(files)
    }
    const slowCopies = (then: (k: number) => void) => {
      const copy = fsp.copyFile.bind(fsp)
      const made = { copies: 0 }
      vi.spyOn(fsp, 'copyFile').mockImplementation(async (...args: Parameters<typeof fsp.copyFile>) => {
        const k = ++made.copies
        await copy(...args)
        then(k)
        await new Promise((resolve) => setTimeout(resolve, 10 + (k % 4) * 30))
      })
      return made
    }
    const awhile = () => new Promise((resolve) => setTimeout(resolve, 400))

    test("a failure stops the other writers before what was written is removed, nothing is left, and the failure is what's told", async () => {
      const dir = await nested()
      const made = slowCopies((k) => {
        if (k === 6) throw Object.assign(new Error('ENOSPC: no space left on device, copyfile'), { code: 'ENOSPC' })
      })
      const base = folder()
      await expect(runExport(dir, options({ format: 'yolo-detect' }), path.join(base, 'dataset'))).rejects.toThrow('ENOSPC: no space left on device')
      const copies = made.copies
      await awhile()
      expect(made.copies).toBe(copies)
      expect(readdirSync(base)).toEqual([])
    })

    test('Stop waits for the images under way, then leaves nothing behind', async () => {
      const dir = await nested()
      const stop = new AbortController()
      slowCopies((k) => { if (k === 6) stop.abort() })
      const base = folder()
      expect(await runExport(dir, options({ format: 'yolo-detect' }), path.join(base, 'dataset'), { signal: stop.signal })).toMatchObject({ stopped: true })
      await awhile()
      expect(readdirSync(base)).toEqual([])
    })

    test('in a drive of its own, a failure or Stop removes only what the export made there, not what came in meanwhile', async () => {
      const dir = await nested()
      for (const how of ['fail', 'stop'] as const) {
        const out = path.join(folder(), 'drive')
        mkdirSync(out)
        // The folder is a drive's root: a device of its own
        const stat = fsp.stat.bind(fsp)
        vi.spyOn(fsp, 'stat').mockImplementation(async (...args: Parameters<typeof fsp.stat>) => {
          const s = await stat(...args)
          return path.resolve(String(args[0])) === out ? Object.assign(s, { dev: Number(s.dev) + 1 }) : s
        })
        const stop = new AbortController()
        slowCopies((k) => {
          // Something else saves a file there meanwhile
          if (k === 2) writeFileSync(path.join(out, 'notes.txt'), 'mine')
          if (k === 6 && how === 'stop') stop.abort()
          if (k === 6 && how === 'fail') throw Object.assign(new Error('ENOSPC: no space left on device, copyfile'), { code: 'ENOSPC' })
        })
        const run = runExport(dir, options({ format: 'yolo-detect' }), out, { signal: stop.signal })
        if (how === 'fail') await expect(run).rejects.toThrow('ENOSPC')
        else expect(await run).toMatchObject({ stopped: true })
        await awhile()
        expect(readdirSync(out)).toEqual(['notes.txt'])
        vi.restoreAllMocks()
      }
    })
  })
})

describe('the check', () => {
  test('counts images and objects per split and per label, groups issues by kind, and pages through them', async () => {
    const files: Record<string, unknown[]> = {}
    for (let i = 0; i < 30; i++) files[`l${i}.png`] = [square('cat', 1, 1), shape('cat', 'line', [[1, 1], [5, 5]])]
    files['m.png'] = [square('dog', 1, 1), square('dog', 12, 12)]
    const dir = await folderWith(files)
    const check = await preflight(dir, options({ format: 'yolo-detect', labels: ['dog', 'bird'] }))
    expect(check.splits).toEqual([{ name: 'all', folder: '', images: 31, objects: 32, backgrounds: 0, posed: 0 }])
    expect(check.labels).toEqual([{ label: 'dog', objects: [2] }, { label: 'bird', objects: [0] }, { label: 'cat', objects: [30] }])
    expect(check.counts).toEqual({ error: 0, warning: 30, info: 0 })
    const lines = check.groups[0]
    expect(lines).toMatchObject({ key: 'warning:yolo-open-shape', title: "Lines and points YOLO can't hold", count: 30 })
    expect(lines.issues.length).toBe(20)
    const rest = moreIssues(check.id, lines.key, 20)
    expect(rest.length).toBe(10)
    expect(new Set([...lines.issues, ...rest].map((i) => i.image)).size).toBe(30)
    await preflight(dir, options({ format: 'yolo-detect' }))
    expect(moreIssues(check.id, lines.key, 20)).toEqual([])
  })

  test("a check with the last check's read id uses its read, and what the writers made of each image, again; a new id reads the folder again", async () => {
    const dir = await folderWith({ 'train/a.png': [square('cat', 1, 1), shape('cat', 'line', [[1, 1], [5, 5]])], 'val/b.png': [square('dog', 2, 2)], 'c.png': null })
    const phases: ExportProgress[] = []
    const ctx = (read: string) => ({ read, onProgress: (p: ExportProgress) => phases.push(p) })
    // What a check read and ran through the writers, by its progress: each phase and how many it went through
    const work = () => [...new Set(phases.splice(0).map((p) => `${p.phase} ${p.total}`))]
    const segment = options({ format: 'yolo-segment' })
    await preflight(dir, segment, ctx('one'))
    expect(work()).toEqual(['read 3', 'check 2'])
    // Another split: nothing is read or run through the writers again, and the check is the one made afresh
    const folders = { ...segment, split: { mode: 'folders' } } as const
    const again = await preflight(dir, folders, ctx('one'))
    expect(work()).toEqual([])
    expect(again.splits.map((s) => [s.name, s.images, s.objects])).toEqual([['train', 1, 1], ['val', 1, 1]])
    // Backgrounds: only the never-labelled image is new to the writers
    await preflight(dir, { ...segment, backgrounds: true }, ctx('one'))
    expect(work()).toEqual(['check 1'])
    // Another format, and back: what the writers made under each is kept
    await preflight(dir, options(), ctx('one'))
    expect(work()).toEqual(['check 2'])
    await preflight(dir, segment, ctx('one'))
    expect(work()).toEqual([])
    // A label file changed since: under the same id the check doesn't see it; under a new one the folder is read,
    // and only that image goes through the writers again
    writeFileSync(path.join(dir, 'val/b.json'), labelme('b.png', [square('dog', 2, 2), square('dog', 20, 10)]))
    expect((await preflight(dir, segment, ctx('one'))).splits[0].objects).toBe(2)
    expect(work()).toEqual([])
    const fresh = await preflight(dir, folders, ctx('two'))
    expect(work()).toEqual(['read 3', 'check 1'])
    expect(fresh.splits.map((s) => s.objects)).toEqual([1, 2])
    // Once the window closes nothing is kept, and the check made afresh is the same
    forgetChecks()
    expect(await preflight(dir, folders, ctx('two'))).toEqual({ ...fresh, id: expect.any(Number) })
    expect(work()).toEqual(['read 3', 'check 2'])
  })

  test('a keypoints file made from the posed objects: each box with its points in the order first seen, and flip pairs where names say left and right', async () => {
    const dir = await folderWith({
      'a.png': [
        shape('person', 'rectangle', [[1, 1], [9, 9]], { group_id: 1 }), shape('nose', 'point', [[2, 2]], { group_id: 1 }), shape('Left eye', 'point', [[3, 3]], { group_id: 1 }),
        shape('hand', 'polygon', [[20, 20], [25, 20], [25, 25]], { group_id: 2 }), shape('wrist', 'point', [[21, 21]], { group_id: 2 })
      ],
      'b.png': [
        shape('person', 'rectangle', [[1, 1], [9, 9]], { group_id: 4 }), shape('Right eye', 'point', [[3, 3]], { group_id: 4 }), shape('nose', 'point', [[2, 2]], { group_id: 4 }),
        // No box, two boxes, and a proposal: none of them say which class the points are of
        shape('ear', 'point', [[5, 5]], { group_id: 5 }), square('cat', 1, 1, 3, { group_id: 6 }), square('dog', 1, 1, 3, { group_id: 6 }), shape('tail', 'point', [[2, 2]], { group_id: 6 }),
        shape('fox', 'rectangle', [[1, 1], [9, 9]], { group_id: 7, flags: { ai: true } }), shape('snout', 'point', [[2, 2]], { group_id: 7 })
      ]
    })
    expect(keypointSchemaOf(await readDataset(dir))).toEqual({
      classes: { person: ['nose', 'Left eye', 'Right eye'], hand: ['wrist'] }, flip: { person: [['Left eye', 'Right eye']] }
    })
    expect(keypointSchemaOf([])).toBeNull()
  })
})

test('nothing is ever written in the image folder', async () => {
  const dir = await folderWith({ 'a.png': [square('cat', 1, 1)], 'sub/b.png': [square('cat', 2, 2)] })
  const before = filesIn(dir)
  for (const format of ['coco', 'yolo-segment'] as const) await runExport(dir, options({ format, images: 'link' }), newFolder())
  expect(filesIn(dir)).toEqual(before)
  expect(existsSync(path.join(dir, 'labels'))).toBe(false)
})
