import { mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, statSync, utimesSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import { afterAll, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest'
import {
  findDuplicateImages, forgetDataset, labelFileClashes, orphanLabels, readDataset, rewriteLabels, setFoldsCase
} from '../../src/main/dataset'
import { setCacheDir } from '../../src/main/images'
import { encodeMask } from '../../src/main/masks'
import { openProject, setTrash } from '../../src/main/project'
import { checkDataset, fixIssue, renameLabel, renaming } from '../../src/shared/checks'
import type { DatasetImage } from '../../src/shared/dataset'
import { readCreateML, readVoc } from '../../src/shared/formats'
import { newShape, parseLabelFile } from '../../src/shared/labelfile'
import { datasetStats } from '../../src/shared/stats'

// Whole folders read with real images, made by sharp as e2e/helpers.ts makes them, and label
// files in all four formats. The trash is a folder of the test's own, as in project.test.ts.
const bin = mkdtempSync(path.join(tmpdir(), 'seamtag-trash-'))
let trashed: string[] = []
// Where each file went last, by where it was
const binned = new Map<string, string>()
const toBin = async (file: string) => {
  const to = path.join(bin, `${binned.size}-${trashed.length}-${path.basename(file)}`)
  renameSync(file, to)
  trashed.push(file)
  binned.set(file, to)
}
setTrash(toBin)
// An image kept inside its label file is written out once, as the window opens it
const cache = mkdtempSync(path.join(tmpdir(), 'seamtag-cache-'))
setCacheDir(cache)

const made: string[] = [bin, cache]
const folder = () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-dataset-'))
  made.push(dir)
  return dir
}
afterAll(() => {
  // Windows can hold a file a moment after sharp lets go of it: what's left is the system's to clear
  for (const dir of made) {
    try {
      rmSync(dir, { recursive: true, force: true, maxRetries: 5 })
    } catch {}
  }
})

let shade = 0
/** A PNG of a colour of its own, so no two are one file by chance */
async function png(file: string, width = 40, height = 30) {
  mkdirSync(path.dirname(file), { recursive: true })
  shade += 9
  await sharp({ create: { width, height, channels: 3, background: { r: shade % 256, g: (shade * 7) % 256, b: 90 } } }).png().toFile(file)
}
function write(dir: string, files: Record<string, string | Buffer>) {
  for (const [name, content] of Object.entries(files)) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true })
    writeFileSync(path.join(dir, name), content)
  }
}
const read = (dir: string, name: string) => readFileSync(path.join(dir, name), 'utf8')
const shape = (label: string, shape_type: string, points: number[][], extra: Record<string, unknown> = {}) =>
  ({ label, points, group_id: null, description: '', shape_type, flags: {}, mask: null, ...extra })
const labelme = (image: string, shapes: unknown[], extra: Record<string, unknown> = {}) =>
  JSON.stringify({ version: '5.5.0', flags: {}, shapes, imagePath: image, imageData: null, imageHeight: 30, imageWidth: 40, ...extra })
const voc = (objects: [string, number[]][], { size = [40, 30], depth = 3, verified = false } = {}) =>
  `<annotation${verified ? ' verified="yes"' : ''}><folder>x</folder><size><width>${size[0]}</width><height>${size[1]}</height><depth>${depth}</depth></size>` +
  objects.map(([name, [x0, y0, x1, y1]]) =>
    `<object><name>${name}</name><difficult>0</difficult><bndbox><xmin>${x0}</xmin><ymin>${y0}</ymin><xmax>${x1}</xmax><ymax>${y1}</ymax></bndbox></object>`).join('') +
  '</annotation>'
const entry = (image: string, labels: string[], verified = false) =>
  ({ image, verified, annotations: labels.map((label) => ({ label, coordinates: { x: 20, y: 15, width: 10, height: 6 } })) })
const createml = (image: string, labels: string[], verified = false) => JSON.stringify([entry(image, labels, verified)])
const of = (images: DatasetImage[], name: string) => images.find((i) => i.name === name)!
/** A 24-bit BMP in one colour: every one of a size has the same byte size */
function bmp(width: number, height: number, [r, g, b]: number[]) {
  const row = Math.ceil((width * 3) / 4) * 4
  const data = Buffer.alloc(54 + row * height)
  data.write('BM', 0)
  data.writeUInt32LE(data.length, 2)
  data.writeUInt32LE(54, 10)
  data.writeUInt32LE(40, 14)
  data.writeInt32LE(width, 18)
  data.writeInt32LE(height, 22)
  data.writeUInt16LE(1, 26)
  data.writeUInt16LE(24, 28)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) data.set([b, g, r], 54 + y * row + x * 3)
  }
  return data
}

describe('a folder with every format', () => {
  let root = ''
  let names: string[] = []
  let images: DatasetImage[] = []
  beforeAll(async () => {
    root = folder()
    for (const name of ['a.png', 'b.png', 'd.png', 'e.png', 'h.png', 'sub/g.png', 'yolo/c.png']) await png(path.join(root, name))
    // Stored 40 x 20, shown upright as 20 x 40
    await sharp({ create: { width: 40, height: 20, channels: 3, background: '#808080' } }).jpeg().withMetadata({ orientation: 6 }).toFile(path.join(root, 'f.jpg'))
    const embedded = await sharp({ create: { width: 16, height: 12, channels: 3, background: '#203040' } }).png().toBuffer()
    write(root, {
      'a.json': labelme('a.png', [
        shape('car', 'polygon', [[1, 1], [20, 1], [10, 20]]),
        shape('road', 'mask', [[5, 6], [8, 9]], { mask: await encodeMask(Uint8Array.from([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]), 4, 4) })
      ], { custom: { kept: true } }),
      'b.xml': voc([['dog', [2, 3, 20, 25]]], { verified: true }),
      'bad.png': 'not an image',
      'd.json': createml('d.png', ['bird'], true),
      'e.json': '{"shapes": [',
      'f.json': labelme('f.jpg', [shape('car', 'rectangle', [[1, 1], [5, 5]])], { imageWidth: 40, imageHeight: 20 }),
      'sub/g.json': labelme('../a.png', []),
      'yolo/c.txt': '1 0.5 0.5 0.5 0.5\n',
      'yolo/classes.txt': 'cat\ndog\n',
      'emb.json': labelme('gone.png', [shape('cat', 'rectangle', [[2, 2], [9, 9]])], { imageData: embedded.toString('base64'), imageWidth: 16, imageHeight: 12 }),
      '.myseg/stray.json': labelme('nothing.png', [])
    })
    // Opening the file that holds its own image opens its folder with it
    const opened = await openProject(path.join(root, 'emb.json'), null)
    names = opened.images.map((i) => i.name)
    images = await readDataset(root)
  })

  test("every image comes in the folder's order, with its label file and the format it's in", () => {
    expect(images.map((i) => i.name)).toEqual(names)
    expect(names).toContain('emb.json')
    expect(Object.fromEntries(images.map((i) => [i.name, i.format]))).toEqual({
      'a.png': 'labelme', 'b.png': 'voc', 'bad.png': null, 'd.png': 'createml', 'e.png': 'labelme', 'f.jpg': 'labelme', 'h.png': null,
      'sub/g.png': 'labelme', 'yolo/c.png': 'yolo', 'emb.json': 'labelme'
    })
    expect(of(images, 'b.png').file).toBe(path.join(root, 'b.xml'))
    expect(of(images, 'h.png').file).toBeNull()
  })

  test("sizes are the images' own, upright after EXIF orientation, with the size each label file gives", () => {
    expect(images.map((i) => [i.name, i.width, i.height, i.labelSize])).toEqual([
      ['a.png', 40, 30, [40, 30]], ['b.png', 40, 30, [40, 30]], ['bad.png', 0, 0, null], ['d.png', 40, 30, null], ['e.png', 40, 30, null],
      ['f.jpg', 20, 40, [40, 20]], ['h.png', 40, 30, null], ['sub/g.png', 40, 30, [40, 30]], ['yolo/c.png', 40, 30, null], ['emb.json', 16, 12, [16, 12]]
    ])
    expect(of(images, 'a.png').bytes).toBe(statSync(path.join(root, 'a.png')).size)
    expect(of(images, 'emb.json').bytes).toBeUndefined()
  })

  test("YOLO's shares are read against the real size, and CreateML's and VOC's boxes, verified marks and all", () => {
    expect(of(images, 'yolo/c.png').shapes.map((s) => [s.label, s.points])).toEqual([['dog', [[10, 8], [30, 22]]]])
    expect(of(images, 'd.png').shapes.map((s) => [s.label, s.points])).toEqual([['bird', [[15, 12], [25, 18]]]])
    expect(images.filter((i) => i.verified).map((i) => i.name)).toEqual(['b.png', 'd.png'])
  })

  test("what can't be read says why: a broken label file, an image that isn't one", () => {
    expect(of(images, 'e.png')).toMatchObject({ broken: expect.stringMatching(/isn't valid JSON/), file: path.join(root, 'e.json'), shapes: [] })
    expect(of(images, 'bad.png').unreadable).toMatch(/unsupported image format/)
    expect(images.filter((i) => i.broken || i.unreadable).map((i) => i.name)).toEqual(['bad.png', 'e.png'])
  })

  test('each read says what it was read from; a file that cannot be read, or none, gives no stamp', () => {
    expect(images.filter((i) => i.stamp).map((i) => i.name)).toEqual(['a.png', 'b.png', 'd.png', 'f.jpg', 'sub/g.png', 'yolo/c.png', 'emb.json'])
  })

  test('masks are decoded where they sit; imagePath is a path in the folder, and says nothing for a file holding its image', () => {
    const [polygon, mask] = of(images, 'a.png').masks
    expect(polygon).toBeNull()
    expect({ ...mask!, data: [...mask!.data] }).toEqual({ x: 5, y: 6, width: 4, height: 4, data: [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1] })
    expect(images.map((i) => i.labelImage)).toEqual(['a.png', undefined, undefined, undefined, undefined, 'f.jpg', undefined, 'a.png', undefined, undefined])
  })

  test('the checks find what is wrong in it, and only that', async () => {
    const issues = checkDataset(images, {
      labels: null, clashes: labelFileClashes(images), duplicates: await findDuplicateImages(root, images), orphans: await orphanLabels(root, images)
    })
    const found = (name: string | null) => issues.filter((i) => i.image === name).map((i) => i.code)
    expect(found(null)).toEqual([])
    expect(found('e.png')).toEqual(['label-file-unreadable'])
    expect(found('bad.png')).toEqual(['image-unreadable', 'never-labelled'])
    expect(found('f.jpg')).toEqual(['size-mismatch', 'not-verified'])
    expect(found('sub/g.png')).toEqual(['image-path-mismatch', 'labelled-empty', 'not-verified'])
    expect(found('h.png')).toEqual(['never-labelled'])
    expect(found('b.png')).toEqual([])
    expect(found('yolo/c.png')).toEqual([])
  })

  test('statistics count it as it is', () => {
    const stats = datasetStats(images)
    expect(stats.images).toEqual({ total: 10, labelled: 6, empty: 1, never: 2, verified: 2, broken: 1, unreadable: 1, aiPending: 0 })
    expect(stats.labels.map((l) => [l.label, l.objects, l.images, l.types])).toEqual([
      ['car', 2, 2, { polygon: 1, rectangle: 1 }], ['dog', 2, 2, { rectangle: 2 }], ['bird', 1, 1, { rectangle: 1 }], ['cat', 1, 1, { rectangle: 1 }],
      ['road', 1, 1, { mask: 1 }]
    ])
    expect(stats.imageSizes).toEqual([{ width: 40, height: 30, images: 7 }, { width: 16, height: 12, images: 1 }, { width: 20, height: 40, images: 1 }])
  })
})

describe('mask shapes', () => {
  test('are placed where labelme 7.9.1 pastes them: whole-pixel corners crop or pad to the box between them', async () => {
    const root = folder()
    await png(path.join(root, 'm.png'))
    const four = await encodeMask(Uint8Array.from({ length: 16 }, (_, k) => (k % 3 ? 1 : 0)), 4, 4)
    write(root, {
      'm.json': labelme('m.png', [
        shape('a', 'mask', [[5, 6], [8, 9]], { mask: four }),
        shape('b', 'mask', [[5.5, 6.5], [8.5, 9.5]], { mask: four }),
        shape('c', 'mask', [[1, 1], [2, 2]], { mask: four }),
        shape('d', 'mask', [[1, 1], [5, 5]], { mask: four }),
        shape('e', 'mask', [[1, 1], [2, 2]], { mask: 'bm90IGEgcG5n' })
      ])
    })
    await openProject(root, null)
    const [m] = await readDataset(root)
    expect(m.masks.map((b) => b && [b.x, b.y, b.width, b.height, [...b.data].join('')])).toEqual([
      [5, 6, 4, 4, '0110110110110110'],
      // Python's round sends halves to the even number
      [6, 6, 4, 4, '0110110110110110'],
      [1, 1, 2, 2, '0111'],
      [1, 1, 5, 5, '0110011010101100110000000'],
      null
    ])
    expect(checkDataset([m], { labels: null }).filter((i) => i.code === 'mask-undecodable').map((i) => i.shape)).toEqual([4])
  })

  test('are clipped to the image as labelme pastes them, however far their corners reach, and one wholly outside keeps no pixels', async () => {
    const root = folder()
    await png(path.join(root, 'm.png'))
    const two = await encodeMask(Uint8Array.from([1, 1, 1, 1]), 2, 2)
    const four = await encodeMask(Uint8Array.from({ length: 16 }, (_, k) => (k % 3 ? 1 : 0)), 4, 4)
    write(root, {
      'm.json': labelme('m.png', [
        // Whole-pixel corners 100000 px apart: a 10 GB box, of which the 40 x 30 image holds the first rows and columns
        shape('far', 'mask', [[0, 0], [100000, 100000]], { mask: two }),
        // Past the left and top: rows 1 to 3 and columns 2 and 3 of the mask land in the image
        shape('corner', 'mask', [[-2, -1], [1, 2]], { mask: four }),
        shape('gone', 'mask', [[500, 500], [501, 501]], { mask: two }),
        shape('edge', 'mask', [[38.5, 28.5], [40.5, 30.5]], { mask: two })
      ])
    })
    await openProject(root, null)
    const [m] = await readDataset(root)
    expect(m.masks.map((b) => b && [b.x, b.y, b.width, b.height])).toEqual([[0, 0, 40, 30], [0, 0, 2, 3], [40, 30, 0, 0], [38, 28, 2, 2]])
    const [far, corner] = m.masks
    expect([...far!.data].reduce((n, v) => n + v, 0)).toBe(4)
    expect([...corner!.data].join('')).toBe('011110')
    const issues = checkDataset([m], { labels: null })
    expect(issues.filter((i) => i.code === 'mask-undecodable')).toEqual([])
    expect(issues.filter((i) => i.code === 'zero-area').map((i) => i.message)).toEqual(['Shape 3 (gone) covers no area: its mask has no pixels set inside the image'])
    // The statistics count what the image holds of them
    expect(datasetStats([m]).sizes.full).toEqual({ small: 4, medium: 0, large: 0 })
  })
})

describe('output folders and files', () => {
  test('an -O folder is read for labels, and its files without an image are found; files that are not labels are not', async () => {
    const root = folder(), out = folder()
    for (const name of ['x.png', 'sub/y.png', 'z.png']) await png(path.join(root, name))
    write(out, {
      'x.json': labelme(path.relative(out, path.join(root, 'x.png')), [shape('car', 'rectangle', [[1, 1], [9, 9]])]),
      'sub/y.xml': voc([['dog', [2, 2, 8, 8]]]),
      'w.json': labelme(path.relative(out, path.join(root, 'w.png')), []),
      'notes.json': JSON.stringify({ title: 'not labels' })
    })
    write(root, {
      'old/q.json': labelme('../gone/q.png', []),
      'caption.txt': 'a caption, not labels',
      'settings.xml': '<config><a>1</a></config>',
      'coco.json': JSON.stringify({ images: [], annotations: [], categories: [] }),
      'boxes/t.txt': '0 0.5 0.5 0.2 0.2\n',
      'boxes/classes.txt': 'car\n'
    })
    await openProject(root, out)
    const images = await readDataset(root)
    expect(images.map((i) => [i.name, i.format, i.file && path.relative(out, i.file)])).toEqual([
      ['sub/y.png', 'voc', path.join('sub', 'y.xml')], ['x.png', 'labelme', 'x.json'], ['z.png', null, null]
    ])
    expect(of(images, 'x.png').labelImage).toBe('x.png')
    expect(await orphanLabels(root, images)).toEqual([
      { file: 'boxes/t.txt', names: null },
      { file: 'old/q.json', names: 'gone/q.png' },
      { file: path.join(out, 'w.json'), names: 'w.png' }
    ])
  })

  test("a LabelMe file holding its own image is told when no image of the folder has it, as labelme's default files are", async () => {
    const root = folder()
    await png(path.join(root, 'a.png'))
    const own = (await sharp({ create: { width: 4, height: 4, channels: 3, background: '#000' } }).png().toBuffer()).toString('base64')
    write(root, {
      'a.json': labelme('a.png', [], { imageData: own }),
      // Its image renamed away: only opening it on its own shows it
      'old_name.json': labelme('old_name.png', [shape('dog', 'rectangle', [[1, 1], [3, 3]])], { imageData: own }),
      'selfie.json': labelme('', [], { imageData: own })
    })
    await openProject(root, null)
    const images = await readDataset(root)
    const orphans = await orphanLabels(root, images)
    expect(orphans).toEqual([{ file: 'old_name.json', names: 'old_name.png', holdsImage: true }, { file: 'selfie.json', names: null, holdsImage: true }])
    expect(checkDataset(images, { labels: null, orphans }).filter((i) => i.code === 'orphan-label-file').map((i) => i.message)).toEqual([
      'old_name.json has no image in the folder, so its labels are never shown or exported; its imagePath names old_name.png, and it holds its own image, so it still opens on its own',
      'selfie.json has no image in the folder, so its labels are never shown or exported; it holds its own image, so it still opens on its own'
    ])
    // The file the folder was opened from is its own image, and no orphan
    await openProject(path.join(root, 'old_name.json'), null)
    expect((await orphanLabels(root, await readDataset(root))).map((o) => o.file)).toEqual(['selfie.json'])
  })

  test("a CreateML list's entries that no image reads are told: images gone from an -O list, another image's in one's own file", async () => {
    const root = folder()
    for (const name of ['a.png', 'b.png', 'c.png']) await png(path.join(root, name))
    const one = path.join(folder(), 'labels.json')
    writeFileSync(one, JSON.stringify([entry('a.png', ['cat']), entry('gone.png', ['dog']), entry('b.png', ['dog']), entry('sub/old.png', ['cat'])]))
    await openProject(root, one)
    let images = await readDataset(root)
    expect(await orphanLabels(root, images)).toEqual([{ file: one, names: null, entries: ['gone.png', 'sub/old.png'] }])
    // An -O list that no image of the folder is in is still looked inside
    writeFileSync(one, JSON.stringify([entry('gone.png', ['dog'])]))
    images = await readDataset(root)
    expect(images.map((i) => i.format)).toEqual([null, null, null])
    expect(await orphanLabels(root, images)).toEqual([{ file: one, names: null, entries: ['gone.png'] }])

    // Beside the images, each reads its own file: b.png never reads a.json
    write(root, { 'a.json': JSON.stringify([entry('a.png', ['cat']), entry('b.png', ['dog'])]), 'c.json': createml('x.png', ['dog']) })
    await openProject(root, null)
    images = await readDataset(root)
    const orphans = await orphanLabels(root, images)
    expect(orphans).toEqual([{ file: 'a.json', names: null, entries: ['b.png'] }, { file: 'c.json', names: null, entries: ['x.png'] }])
    expect(checkDataset(images, { labels: null, orphans }).filter((i) => i.code === 'orphan-label-file').map((i) => i.message)).toEqual([
      "a.json holds labels of b.png, but no image in the folder reads them from it, so they're never shown or exported",
      "c.json holds labels of x.png, but no image in the folder reads them from it, so they're never shown or exported"
    ])
  })

  test('two images beside one label file, or every image in one -O LabelMe file, share it; one CreateML file keeps them apart', async () => {
    const root = folder()
    await png(path.join(root, 'a.png'))
    await sharp({ create: { width: 40, height: 30, channels: 3, background: '#123456' } }).jpeg().toFile(path.join(root, 'a.jpg'))
    await png(path.join(root, 'b.png'))
    write(root, { 'a.json': labelme('a.png', []) })
    await openProject(root, null)
    expect(labelFileClashes(await readDataset(root))).toEqual([{ file: path.join(root, 'a.json'), images: ['a.jpg', 'a.png'] }])

    const one = path.join(folder(), 'labels.json')
    writeFileSync(one, labelme('b.png', []))
    await openProject(root, one)
    expect(labelFileClashes(await readDataset(root))).toEqual([{ file: one, images: ['a.jpg', 'a.png', 'b.png'] }])
    writeFileSync(one, JSON.stringify([...JSON.parse(createml('a.png', ['cat'])), ...JSON.parse(createml('b.png', ['dog']))]))
    const listed = await readDataset(root)
    expect(labelFileClashes(listed)).toEqual([])
    expect(listed.map((i) => [i.name, i.format, i.shapes.map((s) => s.label)])).toEqual([['a.jpg', null, []], ['a.png', 'createml', ['cat']], ['b.png', 'createml', ['dog']]])
  })
})

describe('names that differ only in case', () => {
  beforeAll(() => setFoldsCase(true))
  afterAll(() => setFoldsCase())

  test('are one file on Windows and macOS: label files are matched to images, and shared, by name in any case', async () => {
    const root = folder()
    await png(path.join(root, 'a.jpg'))
    await png(path.join(root, 'A.png'))
    await png(path.join(root, 'img_1.jpg'))
    write(root, { 'a.json': labelme('a.jpg', [shape('car', 'rectangle', [[1, 1], [9, 9]])]), 'IMG_1.json': labelme('img_1.jpg', []) })
    await openProject(root, null)
    const images = await readDataset(root)
    // As each finds its label file there, by its own spelling of the name
    const found = images.map((i) => (i.name === 'A.png' ? { ...i, file: path.join(root, 'A.json'), format: 'labelme' as const } : i))
    expect(labelFileClashes(found)).toEqual([{ file: path.join(root, 'a.json'), images: ['a.jpg', 'A.png'] }])
    expect(await orphanLabels(root, images)).toEqual([])
    await expect(rewriteLabels(root, 'a.jpg', renaming('car', 'bus'), null, of(images, 'a.jpg'))).rejects.toThrow("a.json holds the labels of other images too, so it's left unchanged")
  })
})

describe('reading again', () => {
  let root = ''
  beforeAll(async () => {
    root = folder()
    for (const name of ['m.png', 'n.png', 'yolo/o.png']) await png(path.join(root, name))
    write(root, {
      'm.json': labelme('m.png', [shape('car', 'rectangle', [[1, 1], [9, 9]])]),
      'n.xml': voc([['dog', [1, 1, 9, 9]]]),
      'yolo/o.txt': '1 0.5 0.5 0.2 0.2\n',
      'yolo/classes.txt': 'cat\ndog\n'
    })
    await openProject(root, null)
  })
  const same = (a: DatasetImage[], b: DatasetImage[]) => a.map((image, k) => image === b[k])

  test('only label files, images and classes.txt that changed are read again', async () => {
    const first = await readDataset(root)
    const second = await readDataset(root)
    expect(same(second, first)).toEqual([true, true, true])
    write(root, { 'n.xml': voc([['dog', [1, 1, 9, 9]], ['cat', [2, 2, 8, 8]]]) })
    const third = await readDataset(root)
    expect(same(third, second)).toEqual([true, false, true])
    expect(third[1].shapes.map((s) => s.label)).toEqual(['dog', 'cat'])
    await png(path.join(root, 'm.png'), 50, 30)
    const fourth = await readDataset(root)
    expect(same(fourth, third)).toEqual([false, true, true])
    expect(fourth[0].width).toBe(50)
    write(root, { 'yolo/classes.txt': 'cat\npuppy\n' })
    const fifth = await readDataset(root)
    expect(same(fifth, fourth)).toEqual([true, true, false])
    expect(fifth[2].shapes[0].label).toBe('puppy')
    forgetDataset(root)
    expect(same(await readDataset(root), fifth)).toEqual([false, false, false])
  })

  test('a label file rewritten to the same length within one tick of a coarse clock is still read again', async () => {
    // HFS+ keeps times to the second and FAT to two: size and modified time alone can't tell these apart
    const file = path.join(root, 'm.json')
    utimesSync(file, 1700000000, 1700000000)
    const before = await readDataset(root)
    writeFileSync(`${file}.new`, labelme('m.png', [shape('bus', 'rectangle', [[1, 1], [9, 9]])]))
    renameSync(`${file}.new`, file)
    utimesSync(file, 1700000000, 1700000000)
    expect(statSync(file).mtimeMs).toBe(1700000000000)
    expect((await readDataset(root))[0].shapes.map((s) => s.label)).toEqual(['bus'])
    expect(before[0].shapes.map((s) => s.label)).toEqual(['car'])
  })

  test('the preferred format is read first when an image has files in two', async () => {
    write(root, { 'n.json': labelme('n.png', [shape('bird', 'rectangle', [[1, 1], [9, 9]])]) })
    expect((await readDataset(root))[1].format).toBe('labelme')
    expect((await readDataset(root, { prefer: 'voc' }))[1].format).toBe('voc')
    rmSync(path.join(root, 'n.json'))
  })

  test('progress is told, and a stop rejects with what was read kept for next time', async () => {
    const told: number[][] = []
    await readDataset(root, { onProgress: (done, total) => told.push([done, total]) })
    expect(told[told.length - 1]).toEqual([3, 3])
    const stop = new AbortController()
    stop.abort()
    await expect(readDataset(root, { signal: stop.signal })).rejects.toThrow(/abort/i)
    expect(await readDataset(root)).toHaveLength(3)
  })
})

describe('one CreateML list for every image', () => {
  const N = 300
  let root = '', list = ''
  beforeAll(async () => {
    root = folder()
    const tiny = await sharp({ create: { width: 40, height: 30, channels: 3, background: '#406080' } }).png().toBuffer()
    for (let k = 0; k < N; k++) writeFileSync(path.join(root, `i${k}.png`), tiny)
    list = path.join(folder(), 'labels.json')
    writeFileSync(list, JSON.stringify(Array.from({ length: N }, (_, k) => entry(`i${k}.png`, ['dog', 'cat']))))
    await openProject(root, list)
  })

  test('is read once a run, not once an image, and a re-run reads only the images whose own entries changed', async () => {
    const parse = vi.spyOn(JSON, 'parse')
    // Parses of the whole list, as long as the file is now
    const whole = () => {
      const size = readFileSync(list, 'utf8').length
      return parse.mock.calls.filter(([text]) => typeof text === 'string' && text.length === size).length
    }
    try {
      const first = await readDataset(root)
      expect(first.every((i) => i.shapes.length === 2)).toBe(true)
      expect(whole()).toBeLessThanOrEqual(2)
      parse.mockClear()
      expect((await readDataset(root)).every((image, k) => image === first[k])).toBe(true)
      expect(whole()).toBe(0)
      await rewriteLabels(root, 'i7.png', renaming('dog', 'puppy'), null, of(first, 'i7.png'))
      parse.mockClear()
      const third = await readDataset(root)
      expect(whole()).toBeLessThanOrEqual(2)
      expect(third.filter((image, k) => image !== first[k]).map((i) => [i.name, i.shapes.map((s) => s.label)])).toEqual([['i7.png', ['puppy', 'cat']]])
    } finally {
      parse.mockRestore()
    }
  })

  test('one -O LabelMe file, which every image shows, is read once a run too', async () => {
    const one = path.join(folder(), 'labels.json')
    const text = labelme('i0.png', [shape('car', 'rectangle', [[1, 1], [9, 9]])])
    writeFileSync(one, text)
    await openProject(root, one)
    const parse = vi.spyOn(JSON, 'parse')
    try {
      const images = await readDataset(root)
      expect(images.every((i) => i.file === one && i.shapes.length === 1)).toBe(true)
      expect(parse.mock.calls.filter(([read]) => read === text).length).toBe(1)
    } finally {
      parse.mockRestore()
      await openProject(root, list)
    }
  })

  test("a fix made from a read is refused once the image's own entries changed, but not when only another image's did", async () => {
    const images = await readDataset(root)
    const [a, b] = [of(images, 'i1.png'), of(images, 'i2.png')]
    await rewriteLabels(root, 'i2.png', renaming('cat', 'lynx'), null, b)
    await rewriteLabels(root, 'i1.png', a.shapes.slice(1), null, a)
    expect(readCreateML(readFileSync(list, 'utf8'), 'i1.png').shapes.map((s) => s.label)).toEqual(['cat'])
    await expect(rewriteLabels(root, 'i1.png', a.shapes.slice(0, 1), null, a)).rejects.toThrow(
      "labels.json has changed since i1.png's labels were read, so it's left unchanged: check the folder again")
  })
})

test('identical image files are found; files of one size with other bytes are not', async () => {
  const root = folder()
  write(root, {
    'p.bmp': bmp(8, 6, [200, 10, 10]), 'q.bmp': bmp(8, 6, [200, 10, 10]), 'sub/r.bmp': bmp(8, 6, [10, 200, 10]), 's.bmp': bmp(9, 6, [200, 10, 10]),
    't.bmp': bmp(9, 6, [200, 10, 10])
  })
  await openProject(root, null)
  const images = await readDataset(root)
  expect(images.map((i) => [i.name, i.width, i.height])).toEqual([['p.bmp', 8, 6], ['q.bmp', 8, 6], ['s.bmp', 9, 6], ['sub/r.bmp', 8, 6], ['t.bmp', 9, 6]])
  expect(await findDuplicateImages(root, images)).toEqual([['p.bmp', 'q.bmp'], ['s.bmp', 't.bmp']])
})

describe('rewriting labels', () => {
  let root = ''
  beforeAll(async () => {
    root = folder()
    for (const name of ['a.png', 'b.png', 'd.png', 'e.png', 'f.png', 'yolo/c.png']) await png(path.join(root, name))
    await sharp({ create: { width: 40, height: 30, channels: 3, background: '#654321' } }).jpeg().toFile(path.join(root, 'f.jpg'))
    write(root, {
      'a.json': labelme('a.png', [shape('dog', 'rectangle', [[1, 1], [9, 9]]), shape('cat', 'polygon', [[1, 1], [9, 1], [5, 9]])],
        { flags: { night: true }, custom: { kept: true }, verified: true }),
      'b.xml': voc([['dog', [2, 3, 20, 25]], ['cat', [1, 1, 5, 5]]], { depth: 1, verified: true }),
      'd.json': createml('d.png', ['dog'], true),
      'e.json': '{"shapes": [',
      'f.json': labelme('f.png', [shape('dog', 'rectangle', [[1, 1], [9, 9]])]),
      'yolo/c.txt': '1 0.512345 0.5 0.25 0.25\n0 0.2 0.2 0.1 0.1\n',
      'yolo/classes.txt': 'cat\ndog\n'
    })
    await openProject(root, null)
  })

  test("a label renamed across the folder keeps each file's format and all it holds, with the old files in the trash", async () => {
    const before = Object.fromEntries(['a.json', 'b.xml', 'd.json', 'yolo/c.txt'].map((f) => [f, read(root, f)]))
    const images = await readDataset(root)
    const plan = renameLabel(images, 'dog', 'puppy', labelFileClashes(images))
    expect(plan.images.map((i) => i.name)).toEqual(['a.png', 'b.png', 'd.png', 'yolo/c.png'])
    expect(plan.skipped.map((s) => s.name)).toEqual(['e.png', 'f.jpg', 'f.png'])
    trashed = []
    for (const image of plan.images) {
      const flags: Record<string, boolean> = image.name === 'a.png' ? { night: false } : {}
      expect((await rewriteLabels(root, image.name, image.shapes, flags, of(images, image.name))).item.classes).toMatchObject({ puppy: 1 })
    }
    expect(trashed).toEqual(['a.json', 'b.xml', 'd.json', 'yolo/c.txt'].map((f) => path.join(root, f)))
    // The trash holds the files as they were
    expect(trashed.map((f) => readFileSync(binned.get(f)!, 'utf8'))).toEqual(Object.values(before))

    const a = parseLabelFile(read(root, 'a.json'))
    expect(a.shapes.map((s) => s.label)).toEqual(['puppy', 'cat'])
    expect(a.flags).toEqual({ night: false })
    expect(a.other).toEqual({ custom: { kept: true }, verified: true })
    // Pascal VOC is changed where the name is, and nowhere else
    expect(read(root, 'b.xml')).toBe(before['b.xml'].replace('<name>dog</name>', '<name>puppy</name>'))
    expect(readVoc(read(root, 'b.xml')).verified).toBe(true)
    const d = readCreateML(read(root, 'd.json'), 'd.png')
    expect([d.verified, d.shapes.map((s) => s.label)]).toEqual([true, ['puppy']])
    // A box that didn't move keeps its numbers: only its class does, and the new name goes last in classes.txt
    expect(read(root, 'yolo/c.txt')).toBe('2 0.512345 0.5 0.25 0.25\n0 0.2 0.2 0.1 0.1\n')
    expect(read(root, 'yolo/classes.txt')).toBe('cat\ndog\npuppy\n')
    expect(readdirSync(root).filter((f) => f.endsWith('.tmp'))).toEqual([])
  })

  test('a rename given as a change is made to what the file holds as it is written, so edits since the folder was read stay', async () => {
    const read1 = of(await readDataset(root), 'a.png')
    write(root, { 'a.json': labelme('a.png', [shape('puppy', 'rectangle', [[1, 1], [9, 9]]), shape('owl', 'point', [[3, 3]])], { flags: { night: true } }) })
    const done = await rewriteLabels(root, 'a.png', renaming('puppy', 'dog'), null, read1)
    const a = parseLabelFile(read(root, 'a.json'))
    expect(read1.shapes.map((s) => s.label)).toEqual(['puppy', 'cat'])
    expect([a.shapes.map((s) => s.label), a.flags]).toEqual([['dog', 'owl'], { night: true }])
    expect(done.item.classes).toEqual({ dog: 1, owl: 1 })
  })

  test('a list of shapes made from a read is refused once the file has changed since, as it would undo the change', async () => {
    const images = await readDataset(root)
    write(root, { 'a.json': labelme('a.png', [shape('dog', 'rectangle', [[1, 1], [9, 9]]), shape('owl', 'point', [[3, 3]]), shape('emu', 'point', [[4, 4]])]) })
    const text = read(root, 'a.json')
    await expect(rewriteLabels(root, 'a.png', of(images, 'a.png').shapes.slice(1), null, of(images, 'a.png'))).rejects.toThrow(
      "a.json has changed since a.png's labels were read, so it's left unchanged: check the folder again")
    expect(read(root, 'a.json')).toBe(text)
  })

  test("a label file that can't be read, or that two images share, is never written over", async () => {
    const images = await readDataset(root)
    trashed = []
    await expect(rewriteLabels(root, 'e.png', [], {}, of(images, 'e.png'))).rejects.toThrow("e.json can't be read, so it's left unchanged: It isn't valid JSON")
    expect(read(root, 'e.json')).toBe('{"shapes": [')
    await expect(rewriteLabels(root, 'f.png', [], {}, of(images, 'f.png'))).rejects.toThrow("f.json holds the labels of other images too, so it's left unchanged")
    await expect(rewriteLabels(root, 'h.png', [], {}, { file: null, format: null })).rejects.toThrow("h.png had no label file when its labels were read, so there's nothing to rewrite")
    await expect(rewriteLabels(root, 'yolo/c.png', [], {}, of(images, 'yolo/c.png'))).resolves.toMatchObject({ format: 'yolo' })
    expect(trashed).toEqual([path.join(root, 'yolo', 'c.txt')])
  })

  test('when the old file can not go to the trash, it stays as it was and nothing is left behind', async () => {
    const text = read(root, 'a.json')
    const images = await readDataset(root)
    setTrash(async () => {
      throw new Error('no trash here')
    })
    try {
      await expect(rewriteLabels(root, 'a.png', [], {}, of(images, 'a.png'))).rejects.toThrow("a.json couldn't go to the trash, so it's left unchanged: no trash here")
    } finally {
      setTrash(toBin)
    }
    expect(read(root, 'a.json')).toBe(text)
    expect(readdirSync(root).filter((f) => f.endsWith('.tmp'))).toEqual([])
  })
})

test('an image with label files in two formats has only the one that was read rewritten, and nothing once its labels are elsewhere', async () => {
  const root = folder()
  await png(path.join(root, 'w.png'))
  const json = labelme('w.png', [shape('car', 'polygon', [[1, 1], [30, 2], [12, 25]])])
  write(root, { 'w.json': json, 'w.xml': voc([['car', [2, 3, 60, 25]]]) })
  await openProject(root, null)
  // The window's format picker on Pascal VOC: the fix is to the VOC file's box
  const [w] = await readDataset(root, { prefer: 'voc' })
  const issue = checkDataset([w], { labels: null }).find((i) => i.code === 'outside-image')!
  await rewriteLabels(root, 'w.png', fixIssue(w, issue)!, null, w)
  expect(readVoc(read(root, 'w.xml')).shapes.map((s) => s.points)).toEqual([[[2, 3], [40, 25]]])
  expect(read(root, 'w.json')).toBe(json)
  // Read as LabelMe, then that file's gone: the VOC file isn't written in its place
  const [asJson] = await readDataset(root)
  rmSync(path.join(root, 'w.json'))
  const xml = read(root, 'w.xml')
  await expect(rewriteLabels(root, 'w.png', renaming('car', 'bus'), null, asJson)).rejects.toThrow(
    "w.json was read as w.png's labels, but they're in w.xml now, so nothing is rewritten: check the folder again")
  expect(read(root, 'w.xml')).toBe(xml)
})

describe('Pascal VOC files from other tools', () => {
  // As CVAT and Datumaro write them, with VOC2012's parts and actions: fractions, a box at 0,
  // and what each object holds beyond its name and box
  const ORIGINAL = `<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <folder></folder>
  <filename>v.png</filename>
  <source>
    <database>Unknown</database>
    <annotation>Unknown</annotation>
    <image>Unknown</image>
  </source>
  <size>
    <width>100</width>
    <height>80</height>
    <depth></depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>dog</name>
    <occluded>1</occluded>
    <bndbox>
      <xmin>0.0</xmin>
      <ymin>0.0</ymin>
      <xmax>50.7</xmax>
      <ymax>40.2</ymax>
    </bndbox>
    <attributes>
      <attribute>
        <name>colour</name>
        <value>brown</value>
      </attribute>
    </attributes>
  </object>
  <object>
    <name>person</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <difficult>1</difficult>
    <bndbox>
      <xmin>10.5</xmin>
      <ymin>10.5</ymin>
      <xmax>120.5</xmax>
      <ymax>60.5</ymax>
    </bndbox>
    <part>
      <name>head</name>
      <bndbox><xmin>12</xmin><ymin>11</ymin><xmax>20</xmax><ymax>19</ymax></bndbox>
    </part>
    <actions><jumping>0</jumping></actions>
  </object>
  <object>
    <name>dog</name>
    <pose>Left</pose>
    <bndbox>
      <xmin>0.0</xmin>
      <ymin>0.0</ymin>
      <xmax>50.7</xmax>
      <ymax>40.2</ymax>
    </bndbox>
  </object>
</annotation>
`
  let root = ''
  let v: DatasetImage
  beforeAll(async () => {
    root = folder()
    await png(path.join(root, 'v.png'), 100, 80)
    await openProject(root, null)
  })
  beforeEach(async () => {
    write(root, { 'v.xml': ORIGINAL })
    ;[v] = await readDataset(root)
  })

  test('a rename changes the names of the objects renamed, and not a character more', async () => {
    expect(v.shapes.map((s) => [s.label, s.points])).toEqual([['dog', [[0, 0], [50.7, 40.2]]], ['person', [[10.5, 10.5], [120.5, 60.5]]], ['dog', [[0, 0], [50.7, 40.2]]]])
    await rewriteLabels(root, 'v.png', renaming('dog', 'puppy'), null, v)
    expect(read(root, 'v.xml')).toBe(ORIGINAL.replaceAll('<name>dog</name>', '<name>puppy</name>'))
  })

  test('a box clipped to the image changes only the number that moved', async () => {
    const issue = checkDataset([v], { labels: null }).find((i) => i.code === 'outside-image')!
    expect(issue.shape).toBe(1)
    await rewriteLabels(root, 'v.png', fixIssue(v, issue)!, null, v)
    expect(read(root, 'v.xml')).toBe(ORIGINAL.replace('<xmax>120.5</xmax>', '<xmax>100</xmax>'))
  })

  test('a copy deleted takes its object out with its line, and the copy kept keeps all it holds', async () => {
    const issue = checkDataset([v], { labels: null }).find((i) => i.code === 'duplicate-shape')!
    expect(issue.shape).toBe(2)
    await rewriteLabels(root, 'v.png', fixIssue(v, issue)!, null, v)
    const copy = ORIGINAL.indexOf('  <object>\n    <name>dog</name>\n    <pose>Left</pose>')
    expect(read(root, 'v.xml')).toBe(ORIGINAL.slice(0, copy - 1) + ORIGINAL.slice(ORIGINAL.indexOf('</object>', copy) + '</object>'.length))
  })

  test('a difficult mark is set where the object keeps it, or added before its box, indented as the box is', async () => {
    const flip = (shapes: ReturnType<typeof newShape>[]) => shapes.map((s, k) => (k < 2 ? { ...s, flags: { difficult: k === 0 } } : s))
    await rewriteLabels(root, 'v.png', flip, null, v)
    expect(read(root, 'v.xml')).toBe(ORIGINAL
      .replace('<occluded>1</occluded>\n    <bndbox>', '<occluded>1</occluded>\n    <difficult>1</difficult>\n    <bndbox>')
      .replace('<difficult>1</difficult>\n    <bndbox>\n      <xmin>10.5', '<difficult>0</difficult>\n    <bndbox>\n      <xmin>10.5'))
    expect(readVoc(read(root, 'v.xml')).shapes.map((s) => s.flags)).toEqual([{ difficult: true }, {}, {}])
  })

  test('a new box is written as LabelImg writes it, after the shape before it', async () => {
    await rewriteLabels(root, 'v.png', (shapes) => [shapes[0], newShape('cat', 'rectangle', [[5.5, 0.2], [30.9, 20.9]]), ...shapes.slice(1)], null, v)
    const first = ORIGINAL.indexOf('</object>') + '</object>'.length
    expect(read(root, 'v.xml')).toBe(ORIGINAL.slice(0, first) + '\n' + [
      '\t<object>', '\t\t<name>cat</name>', '\t\t<pose>Unspecified</pose>', '\t\t<truncated>1</truncated>', '\t\t<difficult>0</difficult>',
      '\t\t<bndbox>', '\t\t\t<xmin>5</xmin>', '\t\t\t<ymin>1</ymin>', '\t\t\t<xmax>30</xmax>', '\t\t\t<ymax>20</ymax>', '\t\t</bndbox>', '\t</object>'
    ].join('\n') + ORIGINAL.slice(first))
  })
})

test('2000 images read in a few seconds, and again from the cache faster still', async () => {
  const root = folder()
  const tiny = await sharp({ create: { width: 8, height: 6, channels: 3, background: '#406080' } }).png().toBuffer()
  for (let k = 0; k < 2000; k++) {
    writeFileSync(path.join(root, `${k}.png`), tiny)
    writeFileSync(path.join(root, `${k}.json`), labelme(`${k}.png`, [shape('car', 'rectangle', [[1, 1], [5, 4]])], { imageWidth: 8, imageHeight: 6 }))
  }
  await openProject(root, null)
  let started = performance.now()
  const images = await readDataset(root)
  const first = performance.now() - started
  started = performance.now()
  await readDataset(root)
  const again = performance.now() - started
  console.info(`readDataset: 2000 images in ${Math.round(first)} ms, ${Math.round(again)} ms from the cache`)
  expect(images.every((i) => i.width === 8 && i.shapes.length === 1)).toBe(true)
  expect(first).toBeLessThan(20000)
  expect(again).toBeLessThan(first)
})
