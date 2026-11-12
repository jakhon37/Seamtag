import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import { afterAll, afterEach, describe, expect, test, vi } from 'vitest'
import { addShapes, closeImport, type ImportPolicy, type ImportPreview, type ImportRequest, importRows, readImport, stopImport, writeImport } from '../../src/main/import'
import { CocoSession, readTop, shapesOf, startFolder } from '../../src/main/importWorker'
import { setCacheDir } from '../../src/main/images'
import { decodeMask } from '../../src/main/masks'
import { openProject, setTrash } from '../../src/main/project'
import { type CocoEntry, readCoco, writeCoco } from '../../src/shared/coco'
import type { DatasetImage, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, parseLabelFile, type Shape } from '../../src/shared/labelfile'
import { poseInstances, writePoseYaml } from '../../src/shared/pose'
import { labelRows, writeDataYaml } from '../../src/shared/ultralytics'

// A heap as small as a test wants, to see what a file too big for the COCO reader's does
const heap = vi.hoisted(() => ({ free: 0 }))
vi.mock('v8', async (original) => {
  const v8 = await original<typeof import('v8')>()
  const getHeapStatistics = () => (heap.free ? { ...v8.getHeapStatistics(), total_available_size: heap.free, heap_size_limit: heap.free } : v8.getHeapStatistics())
  return { ...v8, default: { ...v8, getHeapStatistics }, getHeapStatistics }
})

// File > Import dataset, read and written in temporary folders of real images made by sharp, from
// COCO files writeCoco writes and YOLO datasets labelRows and writeDataYaml write, and the
// fixtures in tests/fixtures/ultralytics. The trash is a folder of the test's own, as in
// project.test.ts, and records the order things went into it.
const bin = mkdtempSync(path.join(tmpdir(), 'seamtag-trash-'))
let trashed: string[] = []
let onTrash: (file: string) => void = () => {}
setTrash(async (file) => {
  onTrash(file)
  renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
  trashed.push(file)
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
const group = (p: { groups?: { code: string }[]; skipped?: { code: string }[] }, code: string) => (p.groups ?? p.skipped)!.find((g) => g.code === code)

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
      { label: 'cat', fresh: 2, labelled: 0, listed: true },
      { label: 'ball', fresh: 1, labelled: 0, listed: false },
      { label: 'blob', fresh: 1, labelled: 0, listed: false },
      { label: 'dog', fresh: 1, labelled: 0, listed: false }
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
    const root = await open(dir)
    const p = await read(root, { kind: 'coco', path: inside })
    expect([p.images, p.matched, p.byName, p.base, p.missing, p.ambiguous]).toEqual([4, 3, 1, 'val', 1, 0])
    expect(group(p, 'coco-image-missing')!.rows).toEqual(["q.png isn't in the folder, so its annotations are left out"])
    const r = await save()
    expect(r.names.sort()).toEqual(['val/x.png', 'val/y.png', 'z.png'])
    // Outside the folder, x.png could be either: it's left out
    const outside = cocoFile(path.join(folder(), 'b.json'), ['x.png', 'w.png'].map((n) => entry(n, shape)), ['cat'])
    const q = await read(root, { kind: 'coco', path: outside })
    expect([q.matched, q.byName, q.base, q.ambiguous]).toEqual([1, 0, '', 1])
    expect(group(q, 'coco-image-ambiguous')!.rows[0]).toContain('its name matches train/x.png, val/x.png')
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
    expect(sizes.rows[0]).toBe("a.png is 40 x 30 pixels upright, but a.png in the COCO file is 30 x 40: the sides are swapped, so the image's EXIF orientation probably differs from the one its annotations were made on, so none of its annotations are imported")
    expect(sizes.rows[1]).toBe('b.png is 40 x 30 pixels upright, but b.png in the COCO file is 50 x 30, so none of its annotations are imported')
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
      ann(6, { category_id: 2, keypoints: [4, 4, 1, 5, 5, 2] }), ann(77, {}), 'not an annotation'
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
    expect(a.issues.map((i) => i.code)).toEqual(expect.arrayContaining(['coco-box-only', 'coco-crowd', 'coco-no-keypoints', 'coco-unlabelled-keypoints', 'coco-unknown-image', 'coco-bad-annotation']))
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

  test('Replace sends the old file to the trash first, and the imported shapes take its place', async () => {
    const { dir, file, root } = await setUp()
    const before = text(dir, 'a.json')
    trashed = []
    // When the old file goes, the new one isn't written yet
    let seen: boolean | null = null
    onTrash = (f) => { seen = existsSync(f) && text(dir, 'a.json') === before }
    await read(root, { kind: 'coco', path: file })
    const r = await save('replace')
    expect([r.written, r.trashed, seen]).toEqual([2, 1, true])
    expect(trashed).toEqual([path.join(dir, 'a.json')])
    expect(readFileSync(path.join(bin, `0-a.json`), 'utf8')).toBe(before)
    expect(brief(doc(dir, 'a.json').shapes)).toEqual(brief(imported()))
    expect(doc(dir, 'a.json').other.verified).toBeUndefined()
  })

  test("importing an export's instances file, then its keypoints file, adds the keypoints to the outline's group without doubling it", async () => {
    const schema: KeypointSchema = { classes: { person: ['nose', 'eye'] } }
    const shapes = [
      newShape('person', 'polygon', [[5, 5], [22, 6], [20, 25], [6, 24]], { group_id: 1 }),
      newShape('nose', 'point', [[10, 10]], { group_id: 1 }),
      newShape('eye', 'point', [[12, 9]], { group_id: 1, flags: { difficult: true } }),
      newShape('dog', 'rectangle', [[25, 3], [35, 13]])
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
    expect(brief(doc(dir, 'p.json').shapes)).toEqual(brief([shapes[0], shapes[3]]))
    await read(root, { kind: 'coco', path: keypoints })
    const r = await save('add')
    expect([r.shapes, r.repeated]).toEqual([2, 1])
    const out = doc(dir, 'p.json').shapes
    expect(brief(out)).toEqual(brief([shapes[0], shapes[3], shapes[1], shapes[2]]))
    expect(out[3].flags).toEqual({ difficult: true })
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
    expect(group(p, 'import-keypoint-names')!.rows).toEqual(["data.yaml has no kpt_names, so the keypoints are called kp0 to kp2: pick a keypoint file to name them"])
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

  test('a thousand label files are read a few at a time, and messages come a page at a time', async () => {
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
    const started = Date.now()
    const p = await read(root, { kind: 'yolo', path: ds })
    expect(Date.now() - started).toBeLessThan(20000)
    expect([p.matched, p.fresh]).toEqual([1000, 1000])
    // Each file's repeated row is a note: 1000 of them, 20 sent, the rest a page at a time
    const repeated = group(p, 'yolo-repeated')!
    expect([repeated.count, repeated.rows.length]).toEqual([1000, 20])
    expect(importRows('issues', 'yolo-repeated', 20)).toHaveLength(200)
    expect(importRows('issues', 'yolo-repeated', 990)).toHaveLength(10)
  })
})
