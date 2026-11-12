import { cpSync, existsSync, promises as fsp, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import { afterAll, afterEach, describe, expect, test, vi } from 'vitest'
import { addShapes, closeImport, type ImportPolicy, type ImportPreview, type ImportRequest, importRows, readImport, stopImport, writeBatches, writeImport } from '../../src/main/import'
import { CocoSession, readTop, shapesOf, startFolder } from '../../src/main/importWorker'
import { setCacheDir } from '../../src/main/images'
import { decodeMask } from '../../src/main/masks'
import { setFolding } from '../../src/main/names'
import { openProject, serial, setTrash } from '../../src/main/project'
import { type CocoEntry, readCoco, writeCoco } from '../../src/shared/coco'
import type { DatasetImage, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, parseLabelFile, type Shape, shapeRecord } from '../../src/shared/labelfile'
import { poseInstances, writePoseYaml } from '../../src/shared/pose'
import { rleToMaskBits } from '../../src/shared/rle'
import { labelRows, writeDataYaml } from '../../src/shared/ultralytics'

// Masks' pixels are made by rleToMaskBits: counted, to see which images' are made, and when
vi.mock('../../src/shared/rle', async (original) => {
  const rle = await original<typeof import('../../src/shared/rle')>()
  return { ...rle, rleToMaskBits: vi.fn(rle.rleToMaskBits) }
})

// A heap as small as a test wants, to see what a file too big for the COCO reader's does
const heap = vi.hoisted(() => ({ free: 0 }))
vi.mock('v8', async (original) => {
  const v8 = await original<typeof import('v8')>()
  const getHeapStatistics = () => (heap.free ? { ...v8.getHeapStatistics(), total_available_size: heap.free, heap_size_limit: heap.free } : v8.getHeapStatistics())
  return { ...v8, default: { ...v8, getHeapStatistics }, getHeapStatistics }
})

// File > Import dataset, read and written in temporary folders of real images made by sharp, from
// COCO files writeCoco writes and YOLO datasets labelRows and writeDataYaml write, and the
// fixtures in tests/fixtures/coco and tests/fixtures/ultralytics. The trash is a folder of the
// test's own, as in project.test.ts, and records the order things went into it.
const bin = mkdtempSync(path.join(tmpdir(), 'seamtag-trash-'))
let trashed: string[] = []
let onTrash: (file: string) => void = () => {}
let afterTrash: (file: string) => void = () => {}
setTrash(async (file) => {
  onTrash(file)
  renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
  trashed.push(file)
  afterTrash(file)
})
const cache = mkdtempSync(path.join(tmpdir(), 'seamtag-cache-'))
setCacheDir(cache)

const made: string[] = [bin, cache]
const folder = () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-import-'))
  made.push(dir)
  return dir
}
afterEach(() => {
  closeImport()
  onTrash = () => {}
  afterTrash = () => {}
  vi.restoreAllMocks()
})
afterAll(() => {
  for (const dir of made) {
    try {
      rmSync(dir, { recursive: true, force: true, maxRetries: 5 })
    } catch {}
  }
})

const INFO = { description: 'Seamtag test', version: '3.3.0', year: 2026, date_created: '2026-11-12 00:00:00' }
const W = 40, H = 30
let shade = 0
/** A PNG of its own colour, so no two are one file by chance */
async function png(file: string, width = W, height = H) {
  mkdirSync(path.dirname(file), { recursive: true })
  shade += 7
  await sharp({ create: { width, height, channels: 3, background: { r: shade % 256, g: 90, b: (shade * 5) % 256 } } }).png().toFile(file)
}
function write(dir: string, files: Record<string, string>) {
  for (const [name, text] of Object.entries(files)) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true })
    writeFileSync(path.join(dir, name), text)
  }
}
const text = (dir: string, name: string) => readFileSync(path.join(dir, name), 'utf8')
const doc = (dir: string, name: string) => parseLabelFile(text(dir, name))
const brief = (shapes: Shape[]) => shapes.map((s) => [s.label, s.shape_type, s.group_id, s.points])
const labelme = (image: string, shapes: Shape[], extra: Record<string, unknown> = {}) =>
  JSON.stringify({ version: '5.5.0', flags: {}, shapes, imagePath: image, imageData: null, imageHeight: H, imageWidth: W, ...extra })
const image = (name: string, shapes: Shape[], masks: (MaskBits | null)[] = [], width = W, height = H): DatasetImage =>
  ({ name, width, height, file: null, format: null, shapes, masks: shapes.map((_, i) => masks[i] ?? null), flags: {}, verified: false })
/** A COCO file of these images, written where asked */
function cocoFile(file: string, entries: CocoEntry[], labels: string[], task: 'instances' | 'keypoints' = 'instances', schema?: KeypointSchema) {
  mkdirSync(path.dirname(file), { recursive: true })
  writeFileSync(file, writeCoco(entries, { task, labels, info: INFO, schema }).json)
  return file
}
const entry = (name: string, shapes: Shape[], masks: (MaskBits | null)[] = [], width = W, height = H): CocoEntry =>
  ({ image: image(name, shapes, masks, width, height), fileName: name })

type Batch = { root: string; items: { name: string }[]; labels: string[] }
let batches: Batch[] = []
const counted = () => batches.flatMap((b) => b.items.map((i) => i.name)).sort()
async function open(dir: string, output: string | null = null) {
  batches = []
  return (await openProject(dir, output)).root
}
async function read(root: string, req: Partial<ImportRequest> & Pick<ImportRequest, 'kind' | 'path'>): Promise<ImportPreview> {
  const p = await readImport(root, { prefer: 'labelme', labels: [], ...req }, () => {})
  expect(p).not.toBeNull()
  return p!
}
const save = (policy: ImportPolicy = 'skip') => writeImport(policy, () => {}, (b) => batches.push(b as Batch))
const group = <G extends { code: string }>(p: { groups?: G[]; skipped?: G[] }, code: string) => (p.groups ?? p.skipped)!.find((g) => g.code === code)

/** A BMP file that is only its header, of any size: images are measured by their headers */
function bmp(file: string, width: number, height: number) {
  const head = Buffer.alloc(54)
  head.write('BM', 0, 'ascii')
  head.writeInt32LE(width, 18)
  head.writeInt32LE(height, 22)
  mkdirSync(path.dirname(file), { recursive: true })
  writeFileSync(file, head)
}

// Pixels of a mask shape, rows of 0 and 1, its top-left pixel at x, y
function bits(x: number, y: number, rows: string[]): MaskBits {
  const height = rows.length, width = rows[0].length
  const data = new Uint8Array(width * height)
  rows.forEach((row, j) => { for (let i = 0; i < width; i++) data[j * width + i] = row[i] === '1' ? 1 : 0 })
  return { x, y, width, height, data }
}

describe('a COCO file', () => {
  test("comes back as each image's LabelMe file: its shapes exactly, at the image's size, masks as labelme's PNGs", async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'sub/c.png']) await png(path.join(dir, n))
    const mask = bits(5, 6, ['01110', '11111', '11011', '01110'])
    const a = [
      newShape('cat', 'polygon', [[2, 3], [20, 4.5], [15.25, 18]]),
      newShape('dog', 'rectangle', [[22, 5], [35, 25]], { description: 'brown', flags: { difficult: true } })
    ]
    const b = [newShape('blob', 'mask', [[5, 6], [9, 9]]), newShape('ball', 'circle', [[30, 15], [34, 15]])]
    const c = [newShape('cat', 'oriented_rectangle', [[10, 5], [20, 10], [15, 20], [5, 15]])]
    const file = cocoFile(path.join(folder(), 'instances.json'), [entry('a.png', a), entry('b.png', b, [mask]), entry('sub/c.png', c)], ['cat', 'dog'])
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file, labels: ['cat'] })
    expect([p.images, p.matched, p.byName, p.base, p.missing, p.fresh, p.labelled, p.blocked]).toEqual([3, 3, 0, '', 0, 3, 0, 0])
    expect(p.labels).toEqual([
      { label: 'cat', fresh: 2, labelled: 0, added: 0, listed: true },
      { label: 'ball', fresh: 1, labelled: 0, added: 0, listed: false },
      { label: 'blob', fresh: 1, labelled: 0, added: 0, listed: false },
      { label: 'dog', fresh: 1, labelled: 0, added: 0, listed: false }
    ])
    const r = await save()
    expect([r.written, r.total, r.done, r.shapes, r.trashed, r.stopped]).toEqual([3, 3, 3, 5, 0, false])
    expect(r.names.sort()).toEqual(['a.png', 'b.png', 'sub/c.png'])
    const da = doc(dir, 'a.json')
    expect([da.imageWidth, da.imageHeight, da.imagePath]).toEqual([W, H, 'a.png'])
    expect(da.shapes.map((s) => [s.label, s.shape_type, s.points, s.description, s.flags])).toEqual(a.map((s) => [s.label, s.shape_type, s.points, s.description, s.flags]))
    const db = doc(dir, 'b.json')
    expect(brief(db.shapes)).toEqual(brief(b))
    const pixels = await decodeMask(db.shapes[0].mask!)
    expect([pixels.width, pixels.height, [...pixels.data]]).toEqual([5, 4, [...mask.data]])
    expect(doc(dir, 'sub/c.json').imagePath).toBe('c.png')
    expect(brief(doc(dir, 'sub/c.json').shapes)).toEqual(brief(c))
    // The image list counts the written images again, and no others
    expect(counted()).toEqual(['a.png', 'b.png', 'sub/c.png'])
    expect(batches.flatMap((b) => b.labels).sort()).toEqual(['ball', 'blob', 'cat', 'dog'])
  })

  test('images are matched by path under the folder the names start from, then by a name both sides have once; others are listed, never guessed', async () => {
    const dir = folder()
    for (const n of ['train/x.png', 'val/x.png', 'val/y.png', 'z.png', 'w.png']) await png(path.join(dir, n))
    const shape = [newShape('cat', 'rectangle', [[1, 1], [9, 9]])]
    // In the folder, beside val/: x.png is in train/ and val/, so only val/ finds both of its images
    const inside = cocoFile(path.join(dir, 'annotations', 'val.json'), ['x.png', 'y.png', 'q.png', 'deep/z.png'].map((n) => entry(n, shape)), ['cat'])
    // deep/z.png's second annotation is of a category the file doesn't have
    const coco = JSON.parse(text(dir, 'annotations/val.json'))
    coco.annotations.push({ ...coco.annotations[3], id: 99, category_id: 7 })
    writeFileSync(inside, JSON.stringify(coco))
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: inside })
    expect([p.images, p.matched, p.byName, p.base, p.missing, p.ambiguous]).toEqual([4, 3, 1, 'val', 1, 0])
    expect(group(p, 'coco-image-missing')!.rows).toEqual(["q.png isn't in the folder, so its annotations are left out"])
    // Its messages say which image of the folder it is
    expect(group(p, 'coco-unknown-category')!.rows).toEqual(["deep/z.png (z.png in this folder): annotation 99 is of category 7, which isn't in the categories list, so it's left out"])
    const r = await save()
    expect(r.names.sort()).toEqual(['val/x.png', 'val/y.png', 'z.png'])
    // Outside the folder, x.png could be either: it's left out
    const outside = cocoFile(path.join(folder(), 'b.json'), ['x.png', 'w.png'].map((n) => entry(n, shape)), ['cat'])
    const q = await read(root, { kind: 'coco', path: outside })
    expect([q.matched, q.byName, q.base, q.ambiguous]).toEqual([1, 0, '', 1])
    expect(group(q, 'coco-image-ambiguous')!.rows[0]).toContain('its name matches train/x.png, val/x.png')
  })

  test("names in another case or Unicode form are matched as the file system matches them: by path, and only where it would", async () => {
    const nfc = '사과.png'.normalize('NFC'), nfd = '사과.png'.normalize('NFD')
    const dir = folder()
    for (const n of [nfd, 'images/photo.png', 'images/plain.png']) await png(path.join(dir, n))
    const shape = [newShape('cat', 'rectangle', [[1, 1], [9, 9]])]
    const file = cocoFile(path.join(folder(), 'c.json'), [nfc, 'images/PHOTO.PNG', 'images/plain.png'].map((n) => entry(n, shape)), ['cat'])
    const root = await open(dir)
    try {
      // As Linux compares names: only the one named as it is
      setFolding('exact')
      let p = await read(root, { kind: 'coco', path: file })
      expect([p.matched, p.missing]).toEqual([1, 2])
      closeImport()
      // As macOS compares them, case and Unicode form aside
      setFolding('case-and-form')
      p = await read(root, { kind: 'coco', path: file })
      expect([p.matched, p.byName, p.missing]).toEqual([3, 0, 0])
      const r = await save()
      expect(r.names.sort()).toEqual(['images/photo.png', 'images/plain.png', nfd].sort())
      expect(brief(doc(dir, nfd.replace(/\.png$/, '.json')).shapes)).toEqual(brief(shape))
    } finally {
      setFolding()
    }
  })

  test('the folder names start from is one that finds images by path; none when two find as many', () => {
    const im = (file_name: string, id = 1) => ({ id, width: 1, height: 1, file_name })
    expect(startFolder([im('a.jpg'), im('b.jpg', 2)], ['val2017/a.jpg', 'val2017/b.jpg', 'train2017/c.jpg'], null)).toBe('val2017')
    expect(startFolder([im('images/a.jpg')], ['images/a.jpg'], 'annotations')).toBe('')
    expect(startFolder([im('../images/a.jpg')], ['data/images/a.jpg'], 'data/annotations')).toBe('data/annotations')
    expect(startFolder([im('a.jpg')], ['x/a.jpg', 'y/a.jpg'], null)).toBe('')
  })

  test("an image whose size isn't the COCO file's gets none of its annotations; swapped sides blame its EXIF orientation", async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'c.png']) await png(path.join(dir, n))
    // Turned by EXIF: 40 x 30 as stored, 30 x 40 upright, as the window shows it
    await sharp({ create: { width: W, height: H, channels: 3, background: '#808080' } }).jpeg().withMetadata({ orientation: 6 }).toFile(path.join(dir, 'e.jpg'))
    const shape = [newShape('cat', 'rectangle', [[1, 1], [9, 9]])]
    const file = cocoFile(path.join(folder(), 'c.json'), [entry('a.png', shape, [], H, W), entry('b.png', shape, [], 50, H), entry('c.png', shape), entry('e.jpg', shape, [], H, W)], ['cat'])
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.matched, p.blocked, p.fresh]).toEqual([4, 2, 2])
    const sizes = group(p, 'import-size')!
    expect([sizes.severity, sizes.count]).toEqual(['error', 2])
    expect(sizes.rows[0]).toBe("a.png is 40 × 30 pixels upright, but a.png in the COCO file is 30 × 40: the sides are swapped, so the image's EXIF orientation probably differs from the one its annotations were made on, so none of its annotations are imported")
    expect(sizes.rows[1]).toBe('b.png is 40 × 30 pixels upright, but b.png in the COCO file is 50 × 30, so none of its annotations are imported')
    const r = await save()
    expect(r.names.sort()).toEqual(['c.png', 'e.jpg'])
    expect(group(r, 'import-size')!.count).toBe(2)
    expect([existsSync(path.join(dir, 'a.json')), doc(dir, 'e.json').imageWidth, doc(dir, 'e.json').imageHeight]).toEqual([false, H, W])
  })

  test('two images whose label files would be one file get nothing, as with one -O labels.json for all', async () => {
    const dir = folder()
    for (const n of ['a.png', 'a.jpg', 'b.png']) await png(path.join(dir, n))
    const shape = [newShape('cat', 'rectangle', [[1, 1], [9, 9]])]
    const file = cocoFile(path.join(folder(), 'c.json'), [entry('a.png', shape), entry('b.png', shape)], ['cat'])
    let root = await open(dir)
    let p = await read(root, { kind: 'coco', path: file })
    expect([p.blocked, p.fresh]).toEqual([1, 1])
    expect(group(p, 'import-shared-label')!.rows).toEqual(['a.png would share its label file, a.json, with a.jpg, so nothing is imported into it'])
    const out = folder()
    root = await open(dir, path.join(out, 'labels.json'))
    p = await read(root, { kind: 'coco', path: file })
    expect([p.blocked, p.fresh]).toEqual([2, 0])
  })

  test('a file read a chunk of annotations at a time gives what it gives read whole', () => {
    const dir = folder()
    const cat = { id: 1, name: 'cat', supercategory: '' }
    const person = { id: 2, name: 'person', supercategory: '', keypoints: ['nose', 'eye'] }
    const images = Array.from({ length: 6 }, (_, k) => ({ id: k + 1, width: W, height: H, file_name: `i${k}.png` }))
    let id = 0
    const ann = (image_id: number, extra: Record<string, unknown>) => ({ id: ++id, image_id, category_id: 1, iscrowd: 0, area: 1, bbox: [1, 1, 8, 8], segmentation: [[1, 1, 9, 1, 9, 9]], ...extra })
    const annotations = [
      ann(1, {}), ann(1, { segmentation: [] }), ann(2, { segmentation: [] }), ann(2, { segmentation: [[1, 2, 3]] }),
      ann(3, { iscrowd: 1, segmentation: { size: [H, W], counts: [31, 5, 25, 5, 1134] } }), ann(4, { category_id: 9 }),
      ann(5, { category_id: 2, keypoints: [3, 3, 2, 0, 0, 0], num_keypoints: 1 }), ann(5, { category_id: 2, keypoints: [0, 0, 0, 0, 0, 0] }),
      ann(6, { category_id: 2, keypoints: [4, 4, 1, 5, 5, 2] }), ann(77, {}), 'not an annotation',
      // CVAT's rotated boxes, in two chunks
      ann(2, { segmentation: [], attributes: { occluded: false, rotation: 45 } }), ann(6, { segmentation: [], attributes: { occluded: false, rotation: 30 } })
    ]
    const file = path.join(dir, 'c.json')
    writeFileSync(file, JSON.stringify({ images, annotations, categories: [cat, person] }))
    const names = images.map((i) => i.file_name)
    const whole = new CocoSession(() => {}, 1e9), chunked = new CocoSession(() => {}, 2)
    const a = whole.read(file, names, dir), b = chunked.read(file, names, dir)
    const ids = images.map((i) => i.id)
    expect(b.matched).toEqual(a.matched)
    expect(b.issues.map((i) => i.message).sort()).toEqual(a.issues.map((i) => i.message).sort())
    expect(chunked.shapes(ids)).toEqual(whole.shapes(ids))
    // Read whole by readCoco itself, the same messages and shapes
    const direct = readCoco(readFileSync(file, 'utf8'), 'c.json')
    expect(a.issues.map((i) => i.message).sort()).toEqual(direct.issues.map((i) => i.message).sort())
    expect(whole.shapes(ids)).toEqual(ids.map((n) => ({ id: n, ...shapesOf(direct.results.get(n)!) })))
    expect(a.issues.map((i) => i.code)).toEqual(expect.arrayContaining(['coco-box-only', 'coco-rotated-box', 'coco-crowd', 'coco-no-keypoints', 'coco-unlabelled-keypoints', 'coco-unknown-image', 'coco-bad-annotation']))
    expect(b.issues.find((i) => i.code === 'coco-rotated-box')!.message).toBe('c.json: 2 annotations are boxes CVAT rotated (attributes.rotation), so they become rotated boxes, as CVAT shows them')
  })

  test("its lists are parsed a slice at a time, giving what JSON.parse gives for the keys read", () => {
    const tricky = {
      info: { note: ['}', { x: ']' }] },
      images: [{ id: 1, file_name: 'a "b" [c] {d}\\.png', width: 2, height: 3 }],
      annotations: Array.from({ length: 30000 }, (_, i) => ({ id: i, image_id: 1, segmentation: [[1.5e2, -2, 3, 4, 5, 6.25]], bbox: [0, 0, 1, 1], note: 'x]},"' })),
      categories: [{ id: 1, name: 'ü ✓' }]
    }
    const { info: _info, ...read } = tricky
    for (const text of [JSON.stringify(tricky), JSON.stringify(tricky, null, 2)]) {
      expect(text.length).toBeGreaterThan(2 * 2 ** 20)
      expect(readTop(text, 'c.json')).toEqual(JSON.parse(JSON.stringify(read)))
    }
    expect(readTop(' { "images" : [ ] , "categories":[] } ', 'c.json')).toEqual({ images: [], categories: [] })
    expect(readTop('{}', 'c.json')).toEqual({})
    for (const bad of ['{"images": [1, 2', '{"images": [1 2]}', '{"images" [1]}', '{"images": [1]} x', '{images: []}', '{"images": [{"a": "b]}]}', '{"images": [1],}', '[1, 2'])
      expect(() => readTop(bad, 'c.json'), bad).toThrow(/^c\.json isn't valid JSON \(/)
    expect(() => readTop('[1, 2]', 'c.json')).toThrow("c.json isn't a COCO file: the top level must be an object, so nothing is imported")
    // A key of megabytes is quoted cut short
    const key = 'k'.repeat(5_000_000), cut = `${'k'.repeat(29)}…${'k'.repeat(30)}`
    expect(() => readTop(`{"${key}" 1}`, 'c.json')).toThrow(`c.json isn't valid JSON (there's no colon after "${cut}"), so nothing is imported`)
    expect(() => readTop(`{"${key}": [1, 2`, 'c.json')).toThrow(`c.json isn't valid JSON ("${cut}" has no value that ends), so nothing is imported`)
  })

  test('a file too big for the memory its reader may use is refused plainly before it gets near the limit', () => {
    const dir = folder()
    const file = path.join(dir, 'c.json')
    writeFileSync(file, JSON.stringify({ images: [{ id: 1, width: W, height: H, file_name: 'a.png' }], annotations: [], categories: [], pad: 'x'.repeat(2 ** 20) }))
    heap.free = 20 * 2 ** 20
    try {
      expect(() => new CocoSession().read(file, ['a.png'], dir)).toThrow('c.json is too big to import here: reading it needs more than the 20 MB of memory its reader may use, so nothing is imported. A COCO file with fewer images reads in less.')
    } finally {
      heap.free = 0
    }
    expect(new CocoSession().read(file, ['a.png'], dir).matched).toHaveLength(1)
  })

  test("isn't imported when it isn't COCO, with readCoco's own words", async () => {
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    write(dir, { 'x.json': '{"images": 5}', 'y.json': '[1, 2' })
    await expect(readImport(root, { kind: 'coco', path: path.join(dir, 'x.json'), prefer: 'labelme', labels: [] }, () => {}))
      .rejects.toThrow("x.json isn't a COCO file: it has no images list, so nothing is imported")
    await expect(readImport(root, { kind: 'coco', path: path.join(dir, 'y.json'), prefer: 'labelme', labels: [] }, () => {}))
      .rejects.toThrow(/^y\.json isn't valid JSON/)
    await expect(readImport(root, { kind: 'coco', path: 'relative.json', prefer: 'labelme', labels: [] }, () => {})).rejects.toThrow('Pick the dataset again')
    await expect(readImport(folder(), { kind: 'coco', path: path.join(dir, 'x.json'), prefer: 'labelme', labels: [] }, () => {})).rejects.toThrow('That folder is not open')
  })
  test("an image the file lists without annotations is left as it is, labelled or not, whatever the choice", async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'c.png']) await png(path.join(dir, n))
    write(dir, { 'a.json': labelme('a.png', [newShape('cat', 'polygon', [[1, 1], [9, 1], [5, 8]])]) })
    const before = text(dir, 'a.json')
    // A keypoints file lists every image, those without people too
    const file = cocoFile(path.join(folder(), 'person_keypoints.json'), [entry('a.png', []), entry('b.png', []), entry('c.png', [newShape('dog', 'rectangle', [[1, 1], [9, 9]])])], ['dog'])
    const root = await open(dir)
    trashed = []
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.matched, p.fresh, p.blank, p.labelled, p.empty, p.nothing]).toEqual([3, 1, 0, 0, 0, 2])
    expect(group(p, 'import-no-annotations')!.rows).toEqual([
      "a.png has no annotations in person_keypoints.json, so it's left as it is", "b.png has no annotations in person_keypoints.json, so it's left as it is"
    ])
    const r = await save('replace')
    expect([r.names, r.written, r.trashed, trashed]).toEqual([['c.png'], 1, 0, []])
    expect([text(dir, 'a.json'), existsSync(path.join(dir, 'b.json'))]).toEqual([before, false])
    expect(group(r, 'import-no-annotations')!.count).toBe(2)
  })

  test("labelme 7.9.1's labelme2coco file comes in: its polygons as they were, objects' parts in one group, masks pixel for pixel", async () => {
    // tests/fixtures/coco/make_labelme2coco.py ran labelme2coco on these LabelMe files
    const fx = JSON.parse(readFileSync(path.join(__dirname, '../fixtures/coco/labelme2coco.json'), 'utf8'))
    const dir = folder()
    for (const im of fx.coco.images) {
      mkdirSync(path.join(dir, path.dirname(im.file_name)), { recursive: true })
      await sharp({ create: { width: im.width, height: im.height, channels: 3, background: '#808080' } }).jpeg().toFile(path.join(dir, im.file_name))
    }
    const file = path.join(folder(), 'annotations.json')
    writeFileSync(file, JSON.stringify(fx.coco))
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.images, p.matched, p.byName, p.fresh, p.blocked, p.groups]).toEqual([2, 2, 0, 2, 0, []])
    const r = await save()
    expect([r.written, r.shapes]).toEqual([2, 13])
    const a = doc(dir, 'JPEGImages/a.json').shapes, b = doc(dir, 'JPEGImages/b.json').shapes
    // Without Seamtag's attributes, a shape is what its segmentation is: rectangles, circles and rotated boxes come in as polygons
    expect(a.map((s) => [s.label, s.shape_type])).toEqual([
      ['cat', 'polygon'], ['dog', 'polygon'], ['ball', 'polygon'], ['plate', 'polygon'], ['dog', 'polygon'], ['dog', 'polygon'], ['cat', 'polygon'], ['blob', 'mask']
    ])
    expect(b.map((s) => [s.label, s.shape_type])).toEqual([['ball', 'polygon'], ['blob', 'mask'], ['ball', 'polygon'], ['plate', 'polygon'], ['plate', 'polygon']])
    expect(a[0].points).toEqual(fx.files['a.json'].shapes[0].points)
    // labelme2coco makes one object of a label's shapes in a group: its polygons share a group here
    expect([a[4].group_id, a[4].group_id === a[5].group_id, a[6].group_id === a[4].group_id, b[3].group_id === b[4].group_id]).toEqual([expect.any(Number), true, false, true])
    // A mask on its own is labelme's own pixels, in its own box
    const blob = fx.files['a.json'].shapes[7]
    const pixels = await decodeMask(a[7].mask!)
    expect([a[7].points, pixels.width, pixels.height]).toEqual([blob.points, 10, 8])
    expect(Array.from({ length: 8 }, (_, y) => Array.from(pixels.data.slice(y * 10, y * 10 + 10)).join(''))).toEqual(fx.masks['a.json']['7'])
  })

  test("pycocotools' masks come in pixel for pixel, as compressed counts or as runs", async () => {
    // What pycocotools 2.0.11 makes of them: tests/fixtures/coco/make_rle_cases.py
    const ref = JSON.parse(readFileSync(path.join(__dirname, '../fixtures/coco/rle_cases.json'), 'utf8'))
    type Case = { name: string; h: number; w: number; counts: string; runs?: number[]; rows?: string[] }
    const cases = [...ref.masks, ...ref.runs].filter((c: Case) => c.rows).map((c: Case) => ({ ...c, segmentation: { size: [c.h, c.w], counts: c.runs ?? c.counts } }))
    expect(cases.length).toBeGreaterThan(20)
    const dir = folder()
    for (const [k, c] of cases.entries()) await png(path.join(dir, `m${k}.png`), c.w, c.h)
    const file = path.join(folder(), 'masks.json')
    writeFileSync(file, JSON.stringify({
      images: cases.map((c, k) => ({ id: k + 1, file_name: `m${k}.png`, width: c.w, height: c.h })),
      annotations: cases.map((c, k) => ({ id: k + 1, image_id: k + 1, category_id: 1, iscrowd: 0, area: 1, bbox: [0, 0, 1, 1], segmentation: c.segmentation })),
      categories: [{ id: 1, name: 'blob' }]
    }))
    const root = await open(dir)
    await read(root, { kind: 'coco', path: file })
    const r = await save()
    const set = cases.filter((c) => c.rows!.join('').includes('1'))
    expect(r.written).toBe(set.length)
    for (const [k, c] of cases.entries()) {
      const json = path.join(dir, `m${k}.json`)
      // A mask without pixels is no shape, so the image gets nothing
      if (!set.includes(c)) {
        expect(existsSync(json), c.name).toBe(false)
        continue
      }
      const [shape] = parseLabelFile(readFileSync(json, 'utf8')).shapes
      const pixels = await decodeMask(shape.mask!)
      // labelme's mask: its box's corners, both inside it, and the pixels in that box
      const [[x0, y0], [x1, y1]] = shape.points
      expect([pixels.width, pixels.height], c.name).toEqual([x1 - x0 + 1, y1 - y0 + 1])
      const at = (x: number, y: number) => x >= x0 && x <= x1 && y >= y0 && y <= y1 && pixels.data[(y - y0) * pixels.width + x - x0] ? '1' : '0'
      expect(Array.from({ length: c.h }, (_, y) => Array.from({ length: c.w }, (_, x) => at(x, y)).join('')), c.name).toEqual(c.rows)
    }
  })

  test("masks' pixels are made as images are written, so never at a size the file gives wrongly, nor for an image whose masks have too many", async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png']) await png(path.join(dir, n))
    bmp(path.join(dir, 'big.bmp'), 16000, 16000)
    const all = (h: number, w: number) => ({ size: [h, w], counts: [0, h * w] })
    const file = path.join(folder(), 'masks.json')
    writeFileSync(file, JSON.stringify({
      // a.png is 40 x 30, not the 4000 x 3000 the file says, and big.bmp has 5 masks of 256 million pixels
      images: [{ id: 1, file_name: 'a.png', width: 4000, height: 3000 }, { id: 2, file_name: 'b.png', width: W, height: H }, { id: 3, file_name: 'big.bmp', width: 16000, height: 16000 }],
      categories: [{ id: 1, name: 'blob' }],
      annotations: [
        { id: 1, image_id: 1, category_id: 1, segmentation: all(3000, 4000), bbox: [0, 0, 4000, 3000] },
        { id: 2, image_id: 2, category_id: 1, segmentation: { size: [H, W], counts: [31, 5, 25, 5, 1134] }, bbox: [1, 1, 2, 5] },
        ...Array.from({ length: 5 }, (_, k) => ({ id: 3 + k, image_id: 3, category_id: 1, segmentation: all(16000, 16000), bbox: [0, 0, 16000, 16000] }))
      ]
    }))
    const root = await open(dir)
    vi.mocked(rleToMaskBits).mockClear()
    const p = await read(root, { kind: 'coco', path: file })
    expect(rleToMaskBits).not.toHaveBeenCalled()
    expect([p.matched, p.blocked, p.fresh]).toEqual([3, 2, 1])
    expect(group(p, 'import-mask-pixels')!.rows).toEqual(['big.bmp: its masks in masks.json have 1,280,000,000 pixels in all, more than the 1,073,741,824 Seamtag makes for one image at a time, so nothing is imported into it'])
    const r = await save()
    expect([r.names, vi.mocked(rleToMaskBits).mock.calls.length]).toEqual([['b.png'], 1])
    expect(brief(doc(dir, 'b.json').shapes)).toEqual([['blob', 'mask', null, [[1, 1], [2, 5]]]])
  })

  test('images are written a batch at a time: at most 64, whose masks have at most 2^28 pixels unless one alone has more', () => {
    const batches = (pixels: number[]) => writeBatches(pixels.map((n) => ({ pixels: n }))).map((b) => b.map((im) => im.pixels))
    expect(batches(Array(130).fill(0)).map((b) => b.length)).toEqual([64, 64, 2])
    expect(batches([2 ** 27, 2 ** 27, 1, 2 ** 29, 5, 2 ** 28])).toEqual([[2 ** 27, 2 ** 27], [1], [2 ** 29], [5], [2 ** 28]])
    expect(batches([])).toEqual([])
  })

  test('labels named after what every object has, such as constructor or __proto__, are counted and written as they are', async () => {
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const names = ['constructor', 'toString', '__proto__', 'dog']
    const file = path.join(folder(), 'c.json')
    writeFileSync(file, JSON.stringify({
      images: [{ id: 1, file_name: 'a.png', width: W, height: H }], categories: names.map((name, i) => ({ id: i + 1, name })),
      annotations: names.map((_, i) => ({ id: i + 1, image_id: 1, category_id: i + 1, bbox: [1 + i, 1, 5, 5] }))
    }))
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file })
    expect(p.labels.map((l) => [l.label, l.fresh, l.labelled]).sort()).toEqual(names.map((n) => [n, 1, 0]).sort())
    const r = await save()
    expect(r.labels.map((l) => [l.label, l.shapes]).sort()).toEqual(names.map((n) => [n, 1]).sort())
    expect(doc(dir, 'a.json').shapes.map((s) => s.label)).toEqual(names)
  })

  test("a box too large to place on the image is left out, so the label file written, old shapes and all, can be read again", async () => {
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    write(dir, { 'a.json': labelme('a.png', [newShape('kept', 'rectangle', [[1, 1], [5, 5]])]) })
    const file = path.join(folder(), 'c.json')
    writeFileSync(file, JSON.stringify({
      images: [{ id: 1, file_name: 'a.png', width: W, height: H }], categories: [{ id: 1, name: 'car' }],
      annotations: [
        { id: 1, image_id: 1, category_id: 1, bbox: [1e308, 2, 1e308, 10] }, { id: 2, image_id: 1, category_id: 1, bbox: [2, 2, 10, 10] },
        // CVAT's rotated box, whose corners come to Infinity only once turned
        { id: 3, image_id: 1, category_id: 1, bbox: [0, 0, 1.7e308, 1.7e308], attributes: { occluded: false, rotation: 45 } }
      ]
    }))
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file })
    expect(group(p, 'coco-no-pixels')!.rows).toEqual([
      "a.png: annotation 1 (“car”) has neither a shape nor a box with an area, so it's left out",
      "a.png: annotation 3 (“car”) has a rotated box too large to place on the image, so it's left out"
    ])
    await save('add')
    expect(brief(doc(dir, 'a.json').shapes)).toEqual([['kept', 'rectangle', null, [[1, 1], [5, 5]]], ['car', 'rectangle', null, [[2, 2], [12, 12]]]])
  })
})

describe('images that have labels', () => {
  const own = () => [newShape('cat', 'polygon', [[1, 1], [9, 1], [5, 8]], { group_id: 1 }), newShape('cat', 'rectangle', [[20, 2], [30, 12]])]
  const imported = () => [newShape('dog', 'rectangle', [[2, 12], [12, 22]]), newShape('person', 'polygon', [[20, 15], [35, 16], [30, 28]], { group_id: 1 })]
  async function setUp() {
    const dir = folder()
    for (const n of ['a.png', 'n.png']) await png(path.join(dir, n))
    write(dir, { 'a.json': labelme('a.png', own(), { imageData: 'aGVsbG8=', flags: { checked: true }, verified: true }) })
    const file = cocoFile(path.join(folder(), 'c.json'), [entry('a.png', imported()), entry('n.png', [newShape('dog', 'rectangle', [[1, 1], [5, 5]])])], ['dog', 'person'])
    return { dir, file, root: await open(dir) }
  }

  test('Skip leaves them as they are', async () => {
    const { dir, file, root } = await setUp()
    const before = text(dir, 'a.json')
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.fresh, p.labelled, p.labelImg]).toEqual([1, 1, 0])
    expect(p.labels.map((l) => [l.label, l.fresh, l.labelled])).toEqual([['dog', 1, 1], ['person', 0, 1]])
    const r = await save('skip')
    expect(r.names).toEqual(['n.png'])
    expect(text(dir, 'a.json')).toBe(before)
    expect(group(r, 'labelled')!.rows).toEqual(["a.png has labels, so it's left as it is"])
  })

  test("Add puts the imported shapes after theirs, groups moved above theirs, keeps the file's image data, flags and mark, and adds nothing twice", async () => {
    const { dir, file, root } = await setUp()
    await read(root, { kind: 'coco', path: file })
    const r = await save('add')
    expect([r.written, r.shapes, r.repeated]).toEqual([2, 3, 0])
    const d = doc(dir, 'a.json')
    expect(brief(d.shapes)).toEqual(brief([...own(), imported()[0], { ...imported()[1], group_id: 2 }]))
    expect([d.imageData, d.flags, d.other.verified]).toEqual(['aGVsbG8=', { checked: true }, true])
    // The same again: everything is there already
    const before = text(dir, 'a.json')
    await read(root, { kind: 'coco', path: file })
    const again = await save('add')
    expect([again.written, again.repeated]).toEqual([0, 3])
    expect(text(dir, 'a.json')).toBe(before)
    expect(group(again, 'unchanged')!.count).toBe(2)
  })

  test('Add counts what it writes: the shapes an image has already, and images that have every one, which it leaves as they are without writing or counting them', async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'c.png', 'd.png']) await png(path.join(dir, n))
    // a has every imported shape, b the dog only, d none of them; c has no labels. Written as the app writes shapes, which the imported ones are compared with.
    const saved = (image: string, shapes: Shape[]) => labelme(image, shapes.map((sh) => shapeRecord(sh)) as unknown as Shape[])
    write(dir, { 'a.json': saved('a.png', [...own(), ...imported()]), 'b.json': saved('b.png', [...own(), imported()[0]]), 'd.json': saved('d.png', own()) })
    const before = text(dir, 'a.json')
    const file = cocoFile(path.join(folder(), 'c.json'), ['a.png', 'b.png', 'c.png', 'd.png'].map((n) => entry(n, imported())), ['dog', 'person'])
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.fresh, p.labelled, p.empty, p.unchanged]).toEqual([1, 3, 0, 1])
    // Replace writes every shape into the labelled images; Add only those they haven't
    expect(p.labels.map((l) => [l.label, l.fresh, l.labelled, l.added])).toEqual([['dog', 1, 3, 1], ['person', 1, 3, 2]])
    const told: [number, number][] = []
    const r = await writeImport('add', (at) => told.push([at.done, at.total]), (b) => batches.push(b as Batch))
    expect([r.total, r.written, r.shapes, r.repeated]).toEqual([3, 3, 5, 3])
    expect([told[0], told[told.length - 1]]).toEqual([[0, 3], [3, 3]])
    expect(group(r, 'unchanged')!.rows).toEqual(["a.png has all 2 imported shapes already, so it's left as it is"])
    expect(text(dir, 'a.json')).toBe(before)
    expect(brief(doc(dir, 'b.json').shapes)).toEqual(brief([...own(), imported()[0], { ...imported()[1], group_id: 2 }]))
  })

  test('Replace writes the new file beside the old one, sends the old one to the trash, then gives the new one its name', async () => {
    const { dir, file, root } = await setUp()
    const before = text(dir, 'a.json')
    trashed = []
    // When the old file goes, the new one is written beside it, under another name
    let seen: unknown = null
    onTrash = (f) => {
      const beside = readdirSync(dir).filter((n) => /^a\.json\.\d+\.\d+\.tmp$/.test(n))
      seen = [existsSync(f) && text(dir, 'a.json') === before, beside.length, beside.length && brief(doc(dir, beside[0]).shapes)]
    }
    await read(root, { kind: 'coco', path: file })
    const r = await save('replace')
    expect([r.written, r.trashed, seen]).toEqual([2, 1, [true, 1, brief(imported())]])
    expect(trashed).toEqual([path.join(dir, 'a.json')])
    expect(readFileSync(path.join(bin, `0-a.json`), 'utf8')).toBe(before)
    expect(brief(doc(dir, 'a.json').shapes)).toEqual(brief(imported()))
    expect([doc(dir, 'a.json').other.verified, doc(dir, 'a.json').imagePath, doc(dir, 'a.json').imageData]).toEqual([undefined, 'a.png', 'aGVsbG8='])
    expect(readdirSync(dir).filter((n) => n.endsWith('.tmp'))).toEqual([])
  })

  test("when Replace's new file can't take the old one's name, the report says where each is, and the image is counted again", async () => {
    const { dir, file, root } = await setUp()
    await read(root, { kind: 'coco', path: file })
    // Something takes a.json's name as soon as the old file is in the trash
    afterTrash = (f) => mkdirSync(f)
    const r = await save('replace')
    expect([r.written, r.trashed, r.names.sort()]).toEqual([1, 1, ['a.png', 'n.png']])
    const failed = group(r, 'failed')!
    expect(failed.rows).toEqual([expect.stringMatching(/^a\.json went to the trash, but a\.png's new labels couldn't take its name: they're in a\.json\.\d+\.\d+\.tmp \(/)])
    expect(brief(doc(dir, /(a\.json\.\d+\.\d+\.tmp)/.exec(failed.rows[0])![1]).shapes)).toEqual(brief(imported()))
    // Its labels are gone from where the image list read them, so it counts them again
    expect(counted()).toEqual(['a.png', 'n.png'])
  })

  test("when Replace's new file is still held after the retries, as OneDrive can hold one, a copy of it takes the old one's name", async () => {
    const { dir, file, root } = await setUp()
    await read(root, { kind: 'coco', path: file })
    // Windows refuses to rename a file something holds; it can still be read
    const rename = fsp.rename.bind(fsp)
    vi.spyOn(fsp, 'rename').mockImplementation((async (from: string, to: string) => {
      if (/a\.json\.\d+\.\d+\.tmp$/.test(String(from)) && path.basename(String(to)) === 'a.json') throw Object.assign(new Error('EPERM: operation not permitted'), { code: 'EPERM' })
      return rename(from, to)
    }) as typeof rename)
    const r = await save('replace')
    expect([r.written, r.trashed, group(r, 'failed')]).toEqual([2, 1, undefined])
    expect(brief(doc(dir, 'a.json').shapes)).toEqual(brief(imported()))
    expect(readdirSync(dir).filter((n) => n.endsWith('.tmp'))).toEqual([])
  })

  test("each image is looked at and written in one turn of the label files' queue, so labels written meanwhile aren't lost", async () => {
    for (const policy of ['skip', 'add'] as const) {
      const dir = folder()
      for (const n of ['a.png', 'b.png']) await png(path.join(dir, n))
      write(dir, { 'b.json': labelme('b.png', own()) })
      const file = cocoFile(path.join(folder(), 'c.json'), ['a.png', 'b.png'].map((n) => entry(n, imported())), ['dog', 'person'])
      const root = await open(dir)
      await read(root, { kind: 'coco', path: file })
      // A pre-labelling batch's turn, taken as the import starts, writes both images' labels a moment later
      const kite = [newShape('kite', 'rectangle', [[30, 2], [38, 9]], { flags: { ai: true }, other: { score: 0.9 } })]
      const writing = save(policy)
      const meanwhile = serial(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50))
        write(dir, { 'a.json': labelme('a.png', kite), 'b.json': labelme('b.png', [...own(), ...kite]) })
      })
      await Promise.all([writing, meanwhile])
      // The import's turns come after it: the image it labelled has labels when the import looks
      expect(brief(doc(dir, 'a.json').shapes), policy).toEqual(brief(policy === 'skip' ? kite : [...kite, ...imported()]))
      expect(brief(doc(dir, 'b.json').shapes), policy).toEqual(brief(policy === 'skip' ? [...own(), ...kite] : [...own(), ...kite, imported()[0], { ...imported()[1], group_id: 2 }]))
      closeImport()
    }
  })

  test('an image with labels in its VOC file and in a LabelMe file is left alone when VOC is the format read', async () => {
    const dir = folder()
    await png(path.join(dir, 'c.png'))
    const box = '<annotation><size><width>40</width><height>30</height><depth>3</depth></size><object><name>cat</name><difficult>0</difficult>' +
      '<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>15</ymax></bndbox></object></annotation>'
    write(dir, { 'c.xml': box, 'c.json': labelme('c.png', own()) })
    const before = text(dir, 'c.json')
    const file = cocoFile(path.join(folder(), 'c.json'), [entry('c.png', imported())], ['dog', 'person'])
    const root = await open(dir)
    trashed = []
    const p = await read(root, { kind: 'coco', path: file, prefer: 'voc' })
    expect([p.labelled, p.labelImg, p.blocked]).toEqual([0, 0, 1])
    expect(group(p, 'import-two-files')!.rows).toEqual(['c.png has labels in c.xml and in c.json, so nothing is imported into it: keep one of them first'])
    for (const policy of ['add', 'replace'] as const) {
      await read(root, { kind: 'coco', path: file, prefer: 'voc' })
      const r = await save(policy)
      expect([r.written, group(r, 'import-two-files')!.count, trashed, text(dir, 'c.xml'), text(dir, 'c.json')]).toEqual([0, 1, [], box, before])
    }
    // Read as LabelMe JSON, the LabelMe file takes them
    await read(root, { kind: 'coco', path: file, prefer: 'labelme' })
    expect((await save('add')).written).toBe(1)
    expect(brief(doc(dir, 'c.json').shapes)).toEqual(brief([...own(), imported()[0], { ...imported()[1], group_id: 2 }]))
  })

  test("importing an export's instances file, then its keypoints file, adds the keypoints to the outline's group without doubling it", async () => {
    const schema: KeypointSchema = { classes: { person: ['nose', 'eye'] } }
    const shapes = [
      newShape('person', 'polygon', [[5, 5], [22, 6], [20, 25], [6, 24]], { group_id: 1 }),
      newShape('nose', 'point', [[10, 10]], { group_id: 1 }),
      newShape('eye', 'point', [[12, 9]], { group_id: 1, flags: { difficult: true } }),
      newShape('dog', 'rectangle', [[25, 3], [35, 13]]),
      // A polygon lying on its own box, which isn't taken for one
      newShape('person', 'polygon', [[24, 16], [36, 16], [36, 28], [24, 28]], { group_id: 2 }),
      newShape('nose', 'point', [[30, 20]], { group_id: 2 })
    ]
    const im = image('p.png', shapes)
    const poses = poseInstances(im, schema).instances
    const src = folder()
    const objects = cocoFile(path.join(src, 'instances_x.json'), [{ image: im, fileName: 'p.png', poses }], ['person', 'dog'])
    const keypoints = cocoFile(path.join(src, 'person_keypoints_x.json'), [{ image: im, fileName: 'p.png', poses }], ['person', 'dog'], 'keypoints', schema)
    const dir = folder()
    await png(path.join(dir, 'p.png'))
    const root = await open(dir)
    await read(root, { kind: 'coco', path: objects })
    await save('skip')
    expect(brief(doc(dir, 'p.json').shapes)).toEqual(brief([shapes[0], shapes[3], shapes[4]]))
    await read(root, { kind: 'coco', path: keypoints })
    const r = await save('add')
    expect([r.shapes, r.repeated]).toEqual([3, 2])
    const out = doc(dir, 'p.json').shapes
    expect(brief(out)).toEqual(brief([shapes[0], shapes[3], shapes[4], shapes[1], shapes[2], shapes[5]]))
    expect(out[4].flags).toEqual({ difficult: true })
    // The keypoints file alone gives back the outlines as they were
    const alone = folder()
    await png(path.join(alone, 'p.png'))
    await read(await open(alone), { kind: 'coco', path: keypoints })
    await save('skip')
    expect(brief(doc(alone, 'p.json').shapes)).toEqual(brief([shapes[0], shapes[4], shapes[1], shapes[2], shapes[5]]))
  })

  test("Pascal VOC, YOLO and CreateML labels become LabelMe JSON, their boxes kept by Add and their old files sent to the trash", async () => {
    const dir = folder()
    for (const n of ['v.png', 'y.png', 'c.png']) await png(path.join(dir, n))
    write(dir, {
      'v.xml': '<annotation verified="yes"><size><width>40</width><height>30</height><depth>3</depth></size><object><name>cat</name><difficult>0</difficult>' +
        '<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>15</ymax></bndbox></object></annotation>',
      'y.txt': '0 0.5 0.5 0.25 0.2\n', 'classes.txt': 'cat\n',
      'c.json': JSON.stringify([{ image: 'c.png', verified: false, annotations: [{ label: 'cat', coordinates: { x: 20, y: 15, width: 10, height: 6 } }] }])
    })
    const dog = [newShape('dog', 'polygon', [[1, 20], [8, 20], [4, 28]])]
    const file = cocoFile(path.join(folder(), 'c.json'), ['v.png', 'y.png', 'c.png'].map((n) => entry(n, dog)), ['dog'])
    const root = await open(dir)
    trashed = []
    const p = await read(root, { kind: 'coco', path: file, prefer: 'voc' })
    expect([p.labelled, p.labelImg]).toEqual([3, 3])
    const r = await save('add')
    expect([r.written, r.trashed]).toEqual([3, 3])
    expect(trashed.sort()).toEqual(['c.json', 'v.xml', 'y.txt'].map((f) => path.join(dir, f)))
    expect(brief(doc(dir, 'v.json').shapes)).toEqual([['cat', 'rectangle', null, [[2, 3], [20, 15]]], ['dog', 'polygon', null, dog[0].points]])
    expect(doc(dir, 'v.json').other.verified).toBe(true)
    expect(brief(doc(dir, 'y.json').shapes)).toEqual([['cat', 'rectangle', null, [[15, 12], [25, 18]]], ['dog', 'polygon', null, dog[0].points]])
    expect(brief(doc(dir, 'c.json').shapes)).toEqual([['cat', 'rectangle', null, [[15, 12], [25, 18]]], ['dog', 'polygon', null, dog[0].points]])
    expect([existsSync(path.join(dir, 'classes.txt')), existsSync(path.join(dir, 'v.xml'))]).toEqual([true, false])
  })

  test("a label file that can't be read is never touched, whatever the choice", async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png']) await png(path.join(dir, n))
    write(dir, { 'a.json': '{ not json' })
    const file = cocoFile(path.join(folder(), 'c.json'), ['a.png', 'b.png'].map((n) => entry(n, [newShape('cat', 'rectangle', [[1, 1], [5, 5]])])), ['cat'])
    const root = await open(dir)
    trashed = []
    const p = await read(root, { kind: 'coco', path: file })
    expect([p.unreadable, p.fresh, p.labelled]).toEqual([1, 1, 0])
    expect(group(p, 'import-label-unreadable')!.rows[0]).toMatch(/^a\.json can't be read, so it's never touched: /)
    const r = await save('replace')
    expect([r.names, trashed, text(dir, 'a.json')]).toEqual([['b.png'], [], '{ not json'])
    expect(group(r, 'import-label-unreadable')!.count).toBe(1)
  })

  test('with an output folder, label files go there, as the window saves them', async () => {
    const dir = folder(), out = folder()
    for (const n of ['a.png', 'sub/c.png']) await png(path.join(dir, n))
    const file = cocoFile(path.join(folder(), 'c.json'), ['a.png', 'sub/c.png'].map((n) => entry(n, [newShape('cat', 'rectangle', [[1, 1], [5, 5]])])), ['cat'])
    const root = await open(dir, out)
    await read(root, { kind: 'coco', path: file })
    const r = await save()
    expect(r.written).toBe(2)
    expect([existsSync(path.join(dir, 'a.json')), existsSync(path.join(dir, 'sub/c.json'))]).toEqual([false, false])
    expect(doc(out, 'sub/c.json').imagePath).toBe(path.relative(path.join(out, 'sub'), path.join(dir, 'sub/c.png')).split(path.sep).join('/'))
    expect(brief(doc(out, 'a.json').shapes)).toEqual([['cat', 'rectangle', null, [[1, 1], [5, 5]]]])
    // Replace writes where the old file is: sub/c.png's labels straight in the output folder, as labelme keeps them
    renameSync(path.join(out, 'sub/c.json'), path.join(out, 'c.json'))
    const again = cocoFile(path.join(folder(), 'd.json'), [entry('sub/c.png', [newShape('dog', 'rectangle', [[2, 2], [8, 8]])])], ['dog'])
    trashed = []
    await read(root, { kind: 'coco', path: again })
    expect((await save('replace')).written).toBe(1)
    expect([trashed, existsSync(path.join(out, 'sub/c.json')), readdirSync(out).filter((n) => n.endsWith('.tmp'))]).toEqual([[path.join(out, 'c.json')], false, []])
    expect([brief(doc(out, 'c.json').shapes), doc(out, 'c.json').imagePath]).toEqual([[['dog', 'rectangle', null, [[2, 2], [8, 8]]]], path.relative(out, path.join(dir, 'sub/c.png')).split(path.sep).join('/')])
  })

  test('Stop: writing stops after the image it is on, keeps what it wrote, and only that is counted again', async () => {
    const dir = folder()
    const names = Array.from({ length: 12 }, (_, i) => `i${String(i).padStart(2, '0')}.png`)
    for (const n of names) {
      await png(path.join(dir, n))
      write(dir, { [n.replace('.png', '.json')]: labelme(n, own()) })
    }
    const file = cocoFile(path.join(folder(), 'c.json'), names.map((n) => entry(n, imported())), ['dog', 'person'])
    const root = await open(dir)
    await read(root, { kind: 'coco', path: file })
    // Replace sends each old file to the trash before writing: the third one stops it
    let n = 0
    onTrash = () => { if (++n === 3) stopImport() }
    const r = await save('replace')
    expect([r.stopped, r.written, r.done, r.total]).toEqual([true, 3, 3, 12])
    expect(counted()).toEqual(names.slice(0, 3))
    expect(brief(doc(dir, 'i03.json').shapes)).toEqual(brief(own()))
    // Writing again needs the dataset read again, as the folder has changed
    await expect(save('replace')).rejects.toThrow('Read the dataset again first')
  })

  test('Stop while reading drops what was read', async () => {
    const dir = folder()
    for (const n of ['a.png', 'b.png']) await png(path.join(dir, n))
    const file = cocoFile(path.join(folder(), 'c.json'), ['a.png', 'b.png'].map((n) => entry(n, [newShape('cat', 'rectangle', [[1, 1], [5, 5]])])), ['cat'])
    const root = await open(dir)
    const p = await readImport(root, { kind: 'coco', path: file, prefer: 'labelme', labels: [] }, (at) => { if (at.phase === 'Measuring the images') stopImport() })
    expect(p).toBeNull()
    await expect(save()).rejects.toThrow('Read the dataset again first')
  })
})

describe('adding shapes to an image', () => {
  const rect = (label: string, x: number, group: number | null = null) => newShape(label, 'rectangle', [[x, 0], [x + 5, 5]], { group_id: group })
  test('repeats are left out once each, new groups go above its own, and a group that repeats one of its shapes joins that group', () => {
    const own = [rect('a', 0, 3), rect('b', 10), rect('b', 10)]
    const { added, repeated } = addShapes(own, [rect('a', 0, 1), rect('kp', 50, 1), rect('b', 10), rect('b', 10), rect('b', 10), rect('c', 20, 1), rect('d', 30, 2), rect('e', 40, 9)])
    expect(repeated).toBe(3)
    // Group 1 repeats own shape a of group 3, so its new shapes join 3; 2 and 9 go above 3
    expect(added.map((s) => [s.label, s.group_id])).toEqual([['kp', 3], ['b', null], ['c', 3], ['d', 4], ['e', 5]])
  })
  test('new groups count on from the highest below 2^31, so an own group near 2^53 keeps imported objects apart', () => {
    const own = [rect('a', 0, Number.MAX_SAFE_INTEGER), rect('b', 10, 3)]
    const { added } = addShapes(own, [rect('c', 20, 1), rect('d', 30, 1), rect('e', 40, 2)])
    // 2^53 + 1 is 2^53: counted on from it, both objects were one group
    expect(added.map((s) => [s.label, s.group_id])).toEqual([['c', 4], ['d', 4], ['e', 5]])
  })
  test("shapes that differ in anything but their group aren't repeats", () => {
    const own = [newShape('a', 'polygon', [[0, 0], [5, 0], [0, 5]], { flags: { difficult: true } })]
    expect(addShapes(own, [newShape('a', 'polygon', [[0, 0], [5, 0], [0, 5]])]).repeated).toBe(0)
    expect(addShapes(own, [newShape('a', 'polygon', [[0, 0], [5, 0], [0, 5]], { flags: { difficult: true }, group_id: 4 })]).repeated).toBe(1)
  })
})

describe('a YOLO dataset', () => {
  /** A data.yaml dataset of these images' label rows, outside the folder */
  async function dataset(rows: Record<string, string>, names: Record<number, string>, extra = '') {
    const ds = folder()
    for (const [n, r] of Object.entries(rows)) {
      await png(path.join(ds, 'images', 'train', n))
      write(ds, { [`labels/train/${n.replace(/\.\w+$/, '.txt')}`]: r })
    }
    write(ds, { 'data.yaml': `train: images/train\nval: images/train\nnames:\n${Object.entries(names).map(([k, v]) => `  ${k}: ${v}`).join('\n')}\n${extra}` })
    return ds
  }

  test('rows are shares of the image they match in the folder, by its file name when the dataset is elsewhere', async () => {
    const ds = await dataset({ 'a.png': '0 0.5 0.5 0.5 0.5\n', 'b.png': '1 0.25 0.25 0.2 0.2\n1 0.75 0.75 0.2 0.2\n', 'c.png': '' }, { 0: 'cat', 1: 'dog' })
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'c.png', 'd.png']) await png(path.join(dir, n))
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: path.join(ds, 'data.yaml') })
    expect([p.images, p.matched, p.byName, p.fresh]).toEqual([3, 3, 3, 3])
    expect(p.task).toEqual({ task: 'detect', guess: 'detect', others: [], ask: false, reason: 'Every row is a box: a class and 4 numbers' })
    expect(p.labels.map((l) => [l.label, l.fresh])).toEqual([['dog', 2], ['cat', 1]])
    const r = await save()
    expect(r.written).toBe(3)
    expect(brief(doc(dir, 'a.json').shapes)).toEqual([['cat', 'rectangle', null, [[10, 7.5], [30, 22.5]]]])
    // An empty label file: labelled, with nothing in it
    expect(doc(dir, 'c.json').shapes).toEqual([])
    expect(existsSync(path.join(dir, 'd.json'))).toBe(false)
  })

  test('rows that fit two tasks are asked about, and read again for the task chosen', async () => {
    // A rectangle turned 30 degrees, as shares of a 40 x 30 image
    const corners = [[20, 5], [32.99, 12.5], [25.49, 25.49], [12.5, 18]].map(([x, y]) => `${(x / W).toFixed(6)} ${(y / H).toFixed(6)}`).join(' ')
    const ds = await dataset({ 'a.png': `0 ${corners}\n` }, { 0: 'plate' })
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: ds })
    expect([p.task!.task, p.task!.ask, p.task!.others]).toEqual(['obb', true, ['segment']])
    const q = await read(root, { kind: 'yolo', path: ds, task: 'segment' })
    expect([q.task!.task, q.task!.guess, q.fresh]).toEqual(['segment', 'obb', 1])
    await save()
    expect(doc(dir, 'a.json').shapes.map((s) => s.shape_type)).toEqual(['polygon'])
    // Read again as the guess, OBB, the image hasn't the rotated box; as Segment it has the outline, which Add leaves as it is
    const obb = await read(root, { kind: 'yolo', path: ds })
    expect([obb.task!.task, obb.labelled, obb.unchanged, obb.labels.map((l) => [l.label, l.labelled, l.added])]).toEqual(['obb', 1, 0, [['plate', 1, 1]]])
    const outline = await read(root, { kind: 'yolo', path: ds, task: 'segment' })
    expect([outline.unchanged, outline.labels.map((l) => [l.label, l.labelled, l.added])]).toEqual([1, [['plate', 1, 0]]])
  })

  test("pose keypoints are named by data.yaml's kpt_names, else a keypoint file, else kp0, kp1 and on, which the window is told", async () => {
    // tests/fixtures/ultralytics/make_layouts.py's Ultralytics layout, opened as the folder itself
    const dir = folder()
    cpSync(path.join(__dirname, '../fixtures/ultralytics/layouts/ultralytics'), dir, { recursive: true })
    let root = await open(dir)
    let p = await read(root, { kind: 'yolo', path: dir })
    expect([p.task!.task, p.keypoints, p.byName, p.matched]).toEqual(['pose', { from: 'yaml', file: null, count: 3 }, 0, 3])
    await save()
    expect(brief(doc(dir, 'images/train/a.json').shapes)).toEqual([
      ['person', 'rectangle', 1, [[4, 4], [12, 12]]], ['nose', 'point', 1, [[8, 6.4]]], ['left_eye', 'point', 1, [[7.2, 5.6]]], ['right_eye', 'point', 1, [[8.8, 5.6]]]
    ])
    expect(doc(dir, 'images/train/a.json').shapes[3].flags).toEqual({ difficult: true })
    // Without kpt_names
    const bare = folder()
    cpSync(path.join(__dirname, '../fixtures/ultralytics/layouts/ultralytics'), bare, { recursive: true })
    write(bare, { 'data.yaml': text(bare, 'data.yaml').replace(/kpt_names:[\s\S]*$/, '') })
    root = await open(bare)
    p = await read(root, { kind: 'yolo', path: path.join(bare, 'data.yaml') })
    expect(p.keypoints).toEqual({ from: 'generated', file: null, count: 3 })
    expect(group(p, 'import-keypoint-names')!.rows).toEqual(["data.yaml has no kpt_names, so the keypoints are called kp0 to kp2: pick a keypoints file to name them"])
    const keypoints = path.join(folder(), 'pose.yaml')
    writeFileSync(keypoints, writePoseYaml({ classes: { person: ['top', 'left', 'right'] } }))
    p = await read(root, { kind: 'yolo', path: path.join(bare, 'data.yaml'), keypoints })
    expect([p.keypoints, group(p, 'import-keypoint-names')]).toEqual([{ from: 'file', file: keypoints, count: 3 }, undefined])
    await save()
    expect(doc(bare, 'images/train/a.json').shapes.map((s) => s.label)).toEqual(['person', 'top', 'left', 'right'])
  })

  test("Roboflow's and CVAT's layouts, read from their own folders", async () => {
    // Roboflow's test image and CVAT's val image have no label file: Ultralytics trains them as backgrounds
    for (const [layout, expected] of [['roboflow', ['train/images/a.jpg', 'train/images/b.jpg', 'valid/images/c.jpg']], ['cvat', ['images/train/frame_000000.png', 'images/train/frame_000001.png']]] as const) {
      const dir = folder()
      cpSync(path.join(__dirname, `../fixtures/ultralytics/layouts/${layout}`), dir, { recursive: true })
      const root = await open(dir)
      const p = await read(root, { kind: 'yolo', path: dir })
      expect([p.fresh, p.nothing]).toEqual([expected.length, 1])
      const r = await save()
      expect(r.names.sort()).toEqual([...expected])
    }
    // Their darknet downloads, without data.yaml, name the classes in their own files
    for (const [layout, labels] of [['cvat11', { 'obj_train_data/frame_000000': 'tree', 'obj_train_data/frame_000001': 'car' }], ['darknet', { 'train/e': 'helmet', 'valid/f': 'head' }]] as const) {
      const dir = folder()
      cpSync(path.join(__dirname, `../fixtures/ultralytics/layouts/${layout}`), dir, { recursive: true })
      const p = await read(await open(dir), { kind: 'yolo', path: dir })
      expect(p.labels.map((l) => l.label).sort()).toEqual(Object.values(labels).sort())
      await save()
      for (const [image, label] of Object.entries(labels)) expect(doc(dir, `${image}.json`).shapes.map((sh) => sh.label)).toEqual([label])
    }
  })

  test('a dataset written by labelRows and writeDataYaml comes back as it was, for segment', async () => {
    const shapes = [newShape('cat', 'polygon', [[4, 3], [20, 4], [12, 20]]), newShape('dog', 'rectangle', [[22, 2], [38, 28]])]
    const rows = labelRows(image('a.png', shapes), 'segment', ['cat', 'dog'])
    const ds = folder()
    await png(path.join(ds, 'images/train/a.png'))
    write(ds, { 'labels/train/a.txt': rows.text, 'data.yaml': writeDataYaml({ names: ['cat', 'dog'], train: 'images/train', val: 'images/train' }) })
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: ds })
    expect(p.task!.task).toBe('segment')
    await save()
    const out = doc(dir, 'a.json').shapes
    expect(out.map((s) => [s.label, s.shape_type])).toEqual([['cat', 'polygon'], ['dog', 'polygon']])
    // Back within the millionth of the image's side that YOLO's six decimals keep
    const near = (got: number[][], want: number[][]) => want.forEach(([x, y], i) => expect([Math.abs(got[i][0] - x), Math.abs(got[i][1] - y)].every((d) => d < 1e-4)).toBe(true))
    near(out[0].points, shapes[0].points)
    near(out[1].points, [[22, 2], [38, 2], [38, 28], [22, 28]])
  })

  test("class names such as constructor, which data.yaml holds without quotes, come back as they are", async () => {
    const names = ['car', 'constructor', 'valueOf']
    const shapes = names.map((n, i) => newShape(n, 'rectangle', [[2 + 12 * i, 2], [8 + 12 * i, 8]]))
    const ds = folder()
    await png(path.join(ds, 'images/train/a.png'))
    write(ds, { 'labels/train/a.txt': labelRows(image('a.png', shapes), 'detect', names).text, 'data.yaml': writeDataYaml({ names, train: 'images/train', val: 'images/train' }) })
    expect(text(ds, 'data.yaml')).toContain('\n  1: constructor\n')
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: ds })
    expect([p.labels.map((l) => l.label).sort(), p.groups.map((g) => g.code)]).toEqual([[...names].sort(), ['yolo-two-splits']])
    await save()
    expect(doc(dir, 'a.json').shapes.map((s) => s.label)).toEqual(names)
  })

  test("a row too large to place on the image is left out, so the label file written, old shapes and all, can be read again", async () => {
    const ds = await dataset({ 'c.png': '0 1e308 0.5 0.2 0.2\n0 0.5 0.5 0.5 0.5\n' }, { 0: 'car' })
    const dir = folder()
    await png(path.join(dir, 'c.png'))
    write(dir, { 'c.json': labelme('c.png', [newShape('kept', 'rectangle', [[1, 1], [5, 5]])]) })
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: ds })
    expect(group(p, 'yolo-row')!.rows).toEqual(["labels/train/c.txt: line 1 has a value too large to place on the image, so it's left out"])
    await save('add')
    expect(brief(doc(dir, 'c.json').shapes)).toEqual([['kept', 'rectangle', null, [[1, 1], [5, 5]]], ['car', 'rectangle', null, [[10, 7.5], [30, 22.5]]]])
  })

  test("Stop ends the search for the dataset's files at its next folder", async () => {
    const ds = folder()
    for (let i = 0; i < 30; i++) write(ds, { [`images/f${i}/a.png`]: '' })
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    const listed: string[] = []
    const readdir = fsp.readdir.bind(fsp)
    vi.spyOn(fsp, 'readdir').mockImplementation(((p: string, o: object) => {
      if (String(p).startsWith(ds)) {
        listed.push(String(p))
        // Stopped while the third of the dataset's folders is listed
        if (listed.length === 3) stopImport()
      }
      return readdir(p, o as never)
    }) as never)
    expect(await readImport(root, { kind: 'yolo', path: ds, prefer: 'labelme', labels: [] }, () => {})).toBeNull()
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect(listed.length).toBe(3)
  })

  test('a keypoint file far larger than any is refused before it is read', async () => {
    const bare = folder()
    cpSync(path.join(__dirname, '../fixtures/ultralytics/layouts/ultralytics'), bare, { recursive: true })
    write(bare, { 'data.yaml': text(bare, 'data.yaml').replace(/kpt_names:[\s\S]*$/, '') })
    const root = await open(bare)
    await read(root, { kind: 'yolo', path: path.join(bare, 'data.yaml') })
    const keypoints = path.join(folder(), 'pose.yaml')
    writeFileSync(keypoints, `${writePoseYaml({ classes: { person: ['top', 'left', 'right'] } })}# ${'x'.repeat(5 * 2 ** 20)}\n`)
    const p = await read(root, { kind: 'yolo', path: path.join(bare, 'data.yaml'), keypoints })
    expect([p.keypoints, group(p, 'import-keypoint-file')!.rows]).toEqual([{ from: 'generated', file: keypoints, count: 3 }, ["pose.yaml can't be used: it's 5.0 MB, more than the 4 MB a keypoints file may be"]])
  })

  test('an empty label file gives an image without labels an empty one; an image with labels keeps them, whatever the choice', async () => {
    const ds = await dataset({ 'a.png': '', 'b.png': '', 'c.png': '0 0.5 0.5 0.5 0.5\n' }, { 0: 'cat' })
    const dir = folder()
    for (const n of ['a.png', 'b.png', 'c.png']) await png(path.join(dir, n))
    write(dir, { 'b.json': labelme('b.png', [newShape('dog', 'rectangle', [[1, 1], [9, 9]])]), 'c.json': labelme('c.png', []) })
    const before = text(dir, 'b.json')
    const root = await open(dir)
    trashed = []
    const p = await read(root, { kind: 'yolo', path: ds })
    expect([p.fresh, p.blank, p.labelled, p.empty, p.nothing]).toEqual([1, 1, 2, 1, 0])
    const r = await save('replace')
    expect([r.names.sort(), r.trashed, trashed]).toEqual([['a.png', 'c.png'], 1, [path.join(dir, 'c.json')]])
    expect([doc(dir, 'a.json').shapes, text(dir, 'b.json'), brief(doc(dir, 'c.json').shapes)]).toEqual([[], before, [['cat', 'rectangle', null, [[10, 7.5], [30, 22.5]]]]])
    expect(group(r, 'no-shapes')!.rows).toEqual(["b.png gets no shapes from the dataset, so it's left as it is"])
  })

  test("Stop answers at once while the dataset's files are still being found, and what they find is dropped", async () => {
    const ds = await dataset({ 'a.png': '0 0.5 0.5 0.5 0.5\n' }, { 0: 'cat' })
    const dir = folder()
    await png(path.join(dir, 'a.png'))
    const root = await open(dir)
    // data.yaml can't be read until the test lets it, as if the dataset were on a slow disk
    let release = () => {}
    const held = new Promise<void>((resolve) => (release = resolve))
    const real = fsp.readFile
    vi.spyOn(fsp, 'readFile').mockImplementation((async (...args: Parameters<typeof real>) => {
      if (String(args[0]).endsWith('data.yaml')) await held
      return real(...args)
    }) as typeof real)
    try {
      const reading = readImport(root, { kind: 'yolo', path: path.join(ds, 'data.yaml'), prefer: 'labelme', labels: [] }, (at) => {
        if (at.phase === 'Reading data.yaml') setTimeout(stopImport, 20)
      })
      expect(await Promise.race([reading, new Promise((resolve) => setTimeout(() => resolve('still reading'), 2000))])).toBeNull()
    } finally {
      release()
    }
    await expect(save()).rejects.toThrow('Read the dataset again first')
  })

  test('images not in the folder, and names that match two of its images, are listed and left out', async () => {
    const ds = await dataset({ 'a.png': '0 0.5 0.5 0.5 0.5\n', 'q.png': '0 0.5 0.5 0.5 0.5\n' }, { 0: 'cat' })
    const dir = folder()
    for (const n of ['x/a.png', 'y/a.png']) await png(path.join(dir, n))
    const root = await open(dir)
    const p = await read(root, { kind: 'yolo', path: ds })
    expect([p.matched, p.missing, p.ambiguous]).toEqual([0, 1, 1])
    expect(group(p, 'import-image-ambiguous')!.rows).toEqual(["images/train/a.png isn't at that path in the folder and its name matches x/a.png, y/a.png, so its labels are left out rather than guessed"])
    expect(group(p, 'import-image-missing')!.rows).toEqual(["images/train/q.png isn't in the folder, so its labels are left out"])
  })

  test("an image named in another case or Unicode form is matched by its name as the file system matches it", async () => {
    const nfc = '사과.png'.normalize('NFC'), nfd = '사과.png'.normalize('NFD')
    const ds = await dataset({ [nfc]: '0 0.5 0.5 0.5 0.5\n', 'PHOTO.PNG': '0 0.5 0.5 0.5 0.5\n' }, { 0: 'cat' })
    const dir = folder()
    for (const n of [nfd, 'sub/photo.png']) await png(path.join(dir, n))
    const root = await open(dir)
    try {
      setFolding('exact')
      expect((await read(root, { kind: 'yolo', path: ds })).missing).toBe(2)
      closeImport()
      setFolding('case-and-form')
      const p = await read(root, { kind: 'yolo', path: ds })
      expect([p.matched, p.byName, p.missing]).toEqual([2, 2, 0])
      expect((await save()).names.sort()).toEqual(['sub/photo.png', nfd].sort())
    } finally {
      setFolding()
    }
  })

  test('a folder of 40,000 images of one name, one to a folder, is matched in a time their number gives, not its square', async () => {
    const ds = await dataset({ 'rgb.png': '0 0.5 0.5 0.5 0.5\n' }, { 0: 'cat' })
    const dir = folder()
    // The folders are listed, not made: making them would take longer than the test
    const readdir = fsp.readdir.bind(fsp)
    const listed = (name: string, isDir: boolean) => ({ name, isDirectory: () => isDir, isFile: () => !isDir, isSymbolicLink: () => false })
    vi.spyOn(fsp, 'readdir').mockImplementation((async (p: string, o: unknown) => {
      if (p === dir) return Array.from({ length: 40_000 }, (_, i) => listed(`s${i}`, true))
      if (path.dirname(p) === dir) return [listed('rgb.png', false)]
      return readdir(p, o as never)
    }) as never)
    const root = await open(dir)
    const started = performance.now()
    const p = await read(root, { kind: 'yolo', path: ds })
    expect(performance.now() - started).toBeLessThan(3000)
    expect([p.matched, p.ambiguous]).toEqual([0, 1])
    expect(group(p, 'import-image-ambiguous')!.rows[0]).toContain('its name matches s0/rgb.png, s1/rgb.png, s2/rgb.png and 39997 more')
  })

  test('a thousand label files are read 16 at a time, and messages come a page at a time', async () => {
    const ds = folder(), dir = folder()
    const names = Array.from({ length: 1000 }, (_, i) => `f${i}.png`)
    const tiny = await sharp({ create: { width: W, height: H, channels: 3, background: '#405060' } }).png().toBuffer()
    for (const n of names) {
      mkdirSync(path.join(ds, 'images/train'), { recursive: true })
      writeFileSync(path.join(ds, 'images/train', n), tiny)
      writeFileSync(path.join(dir, n), tiny)
      write(ds, { [`labels/train/${n.replace('.png', '.txt')}`]: '0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.5 0.5\n' })
    }
    write(ds, { 'data.yaml': 'train: images/train\nval: images/train\nnames: [cat]\n' })
    const root = await open(dir)
    // The label files open at once while their rows are read
    const real = fsp.readFile
    let reading = false, now = 0, most = 0
    vi.spyOn(fsp, 'readFile').mockImplementation((async (...args: Parameters<typeof real>) => {
      if (!reading || !String(args[0]).endsWith('.txt')) return real(...args)
      most = Math.max(most, ++now)
      try {
        return await real(...args)
      } finally {
        now--
      }
    }) as typeof real)
    const started = Date.now()
    const p = (await readImport(root, { kind: 'yolo', path: ds, prefer: 'labelme', labels: [] }, (at) => { reading = at.phase === 'Reading the label files' }))!
    expect(Date.now() - started).toBeLessThan(20000)
    expect(most).toBe(16)
    expect([p.matched, p.fresh]).toEqual([1000, 1000])
    // Each file's repeated row is a note: 1000 of them, 20 sent, the rest a page at a time
    const repeated = group(p, 'yolo-repeated')!
    expect([repeated.count, repeated.rows.length]).toEqual([1000, 20])
    expect(importRows('issues', 'yolo-repeated', 20)).toHaveLength(200)
    expect(importRows('issues', 'yolo-repeated', 990)).toHaveLength(10)
  })
})
