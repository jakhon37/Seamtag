import { existsSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, renameSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { afterEach, describe, expect, test, vi } from 'vitest'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { setFolding } from '../../src/main/names'
import {
  findLabels, labelPath, loadLabels, openProject, renameRetrying, saveLabels, type SaveRequest, setTrash, sharesStem, trashImage, writeAtomic
} from '../../src/main/project'

// Saving, switching formats and deleting, on folders of their own. The trash is a folder too.
const bin = mkdtempSync(path.join(tmpdir(), 'seamtag-trash-'))
let trashed: string[] = []
setTrash(async (file) => {
  renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
  trashed.push(file)
})

/** A folder of images, which only need their names here, and files beside them */
function folder(files: Record<string, string>, images = ['a.png']) {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-project-'))
  for (const [name, text] of [...images.map((i) => [i, ''] as const), ...Object.entries(files)]) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true })
    writeFileSync(path.join(dir, name), text)
  }
  trashed = []
  return dir
}
const text = (dir: string, name: string) => readFileSync(path.join(dir, name), 'utf8')
const box = (label: string, x0 = 10, y0 = 10, x1 = 50, y1 = 40) => newShape(label, 'rectangle', [[x0, y0], [x1, y1]])
const request = (shapes: Shape[], extra: Partial<SaveRequest> = {}): SaveRequest =>
  ({ shapes, flags: {}, size: [100, 80], imageData: null, ...extra })
const LABELME = JSON.stringify({ version: '5.5.0', flags: {}, shapes: [
  { label: 'cat', points: [[1, 1], [9, 1], [5, 9]], group_id: null, description: '', shape_type: 'polygon', flags: {}, mask: null }
], imagePath: 'a.png', imageData: null, imageHeight: 80, imageWidth: 100 })
const VOC = '<annotation><object><name>dog</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>'

describe('switching formats', () => {
  test('saves the new format, then moves the old file to the trash', async () => {
    const dir = folder({ 'a.json': LABELME })
    await openProject(dir, null)
    const saved = await saveLabels(dir, 'a.png', request([box('cat')], { format: 'voc', from: 'labelme' }))
    expect(saved.replaced).toBe(path.join(dir, 'a.json'))
    expect(text(dir, 'a.xml')).toContain('<name>cat</name>')
    expect(trashed).toEqual([path.join(dir, 'a.json')])
  })

  test('never writes over labels the image already has in the new format', async () => {
    const dir = folder({ 'a.json': LABELME, 'a.xml': VOC })
    await openProject(dir, null)
    await expect(saveLabels(dir, 'a.png', request([box('dog')], { format: 'labelme', from: 'voc' }))).rejects.toThrow(/already holds labels/)
    expect(text(dir, 'a.json')).toBe(LABELME)
    expect(text(dir, 'a.xml')).toBe(VOC)
    expect(trashed).toEqual([])
  })

  test('the preferred format opens first, then LabelMe JSON, Pascal VOC and YOLO', async () => {
    const dir = folder({ 'a.json': LABELME, 'a.xml': VOC, 'b.txt': 'a caption, not labels' }, ['a.png', 'b.png'])
    await openProject(dir, null)
    expect((await loadLabels(dir, 'a.png', 'voc')).format).toBe('voc')
    expect((await loadLabels(dir, 'a.png', 'yolo')).format).toBe('labelme')
    // A .txt with no classes.txt beside it isn't YOLO's
    expect(await findLabels(dir, 'b.png', 'yolo')).toBeNull()
  })
})

describe('YOLO', () => {
  test("classes.txt keeps its numbers, new labels go at its end, and a caption .txt is never written over", async () => {
    const dir = folder({ 'b.txt': 'a caption' }, ['a.png', 'b.png'])
    await openProject(dir, null)
    await expect(saveLabels(dir, 'b.png', request([box('cat')], { format: 'yolo' }))).rejects.toThrow(/isn't a YOLO file/)
    expect(text(dir, 'b.txt')).toBe('a caption')
    writeFileSync(path.join(dir, 'classes.txt'), 'dog\ncat\n')
    await saveLabels(dir, 'a.png', request([box('cat'), box('bird')], { format: 'yolo', classes: ['bird', 'cat', 'dog'] }))
    expect(text(dir, 'classes.txt')).toBe('dog\ncat\nbird\n')
    expect(text(dir, 'a.txt').split('\n').map((l) => l.split(' ')[0])).toEqual(['1', '2', ''])
  })
})

describe('output folders and files', () => {
  test('images of the same name in different folders keep their own label files', async () => {
    const out = mkdtempSync(path.join(tmpdir(), 'seamtag-out-'))
    writeFileSync(path.join(out, 'a.xml'), VOC)
    writeFileSync(path.join(out, 'c.json'), LABELME)
    const dir = folder({}, ['a.png', 'sub/a.png', 'deep/c.png'])
    await openProject(dir, out)
    expect(labelPath(dir, 'a.png', 'voc')).toBe(path.join(out, 'a.xml'))
    expect(labelPath(dir, 'sub/a.png', 'voc')).toBe(path.join(out, 'sub', 'a.xml'))
    expect((await loadLabels(dir, 'sub/a.png', 'voc')).doc).toBeNull()
    // A name only one image has still finds labelme's flat file
    expect(labelPath(dir, 'deep/c.png')).toBe(path.join(out, 'c.json'))
  })

  test('one CreateML file lists every image by its path; one that can not be read is never trashed whole', async () => {
    const dir = folder({}, ['a.png', 'sub/a.png', 'b.png'])
    const file = path.join(mkdtempSync(path.join(tmpdir(), 'seamtag-out-')), 'labels.json')
    await openProject(dir, file)
    await saveLabels(dir, 'a.png', request([box('cat')], { format: 'createml' }))
    await saveLabels(dir, 'sub/a.png', request([box('dog'), box('dog', 60, 10, 90, 40)], { format: 'createml' }))
    expect(JSON.parse(readFileSync(file, 'utf8')).map((e: any) => e.image)).toEqual(['a.png', 'sub/a.png'])
    expect((await loadLabels(dir, 'sub/a.png', 'createml')).doc!.shapes.map((s) => s.label)).toEqual(['dog', 'dog'])
    expect((await loadLabels(dir, 'b.png', 'createml')).doc).toBeNull()
    writeFileSync(file, '[{"image": "a.png"')
    const gone = await trashImage(dir, 'a.png')
    expect(gone.kept).toMatch(/can't be read/)
    expect(existsSync(file)).toBe(true)
    expect(trashed).toEqual([path.join(dir, 'a.png')])
  })
})

test('Delete image: the image goes first, then its label files in every format; classes.txt stays', async () => {
  const dir = folder({ 'a.json': LABELME, 'a.xml': VOC, 'a.txt': '0 0.5 0.5 0.2 0.2\n', 'classes.txt': 'dog\n' })
  await openProject(dir, null)
  expect((await trashImage(dir, 'a.png')).kept).toBeUndefined()
  expect(trashed).toEqual(['a.png', 'a.json', 'a.xml', 'a.txt'].map((f) => path.join(dir, f)))
  expect(existsSync(path.join(dir, 'classes.txt'))).toBe(true)
})

test('images whose names differ only in their extension are known, as the system compares names, until one is deleted', async () => {
  const dir = folder({}, ['a.png', 'a.jpg', 'B.png', 'sub/b.jpg', 'c.png'])
  await openProject(dir, null)
  expect(['a.png', 'B.png', 'sub/b.jpg', 'c.png'].map((name) => sharesStem(dir, name))).toEqual([true, false, false, false])
  await trashImage(dir, 'a.jpg')
  expect(sharesStem(dir, 'a.png')).toBe(false)
  setFolding('case')
  try {
    const other = folder({}, ['a.png', 'A.jpg'])
    await openProject(other, null)
    expect(sharesStem(other, 'a.png')).toBe(true)
  } finally {
    setFolding()
  }
})

describe('renames a file someone holds for a moment', () => {
  const platform = process.platform
  const as = (name: string) => Object.defineProperty(process, 'platform', { value: name })
  const busy = (code: string, times: number) => {
    const rename = fsp.rename.bind(fsp)
    let left = times
    return vi.spyOn(fsp, 'rename').mockImplementation(async (from, to) => {
      if (left-- > 0) throw Object.assign(new Error(`${code}: held`), { code })
      return rename(from, to)
    })
  }
  afterEach(() => {
    as(platform)
    vi.restoreAllMocks()
  })

  test('on Windows, EPERM, EACCES and EBUSY are tried again until the file is let go', async () => {
    as('win32')
    const dir = folder({})
    for (const code of ['EPERM', 'EACCES', 'EBUSY']) {
      const spy = busy(code, 3)
      await writeAtomic(path.join(dir, `${code}.json`), code)
      expect(spy).toHaveBeenCalledTimes(4)
      expect(text(dir, `${code}.json`)).toBe(code)
      spy.mockRestore()
    }
  })

  test('it gives up after its patience, and other errors and other systems fail at once', async () => {
    const dir = folder({ 'a.txt': 'a' })
    as('win32')
    let spy = busy('EPERM', Infinity)
    await expect(renameRetrying(path.join(dir, 'a.txt'), path.join(dir, 'b.txt'), 100)).rejects.toThrow('EPERM')
    expect(spy.mock.calls.length).toBeGreaterThan(2)
    spy.mockRestore()
    spy = busy('ENOENT', 1)
    await expect(renameRetrying(path.join(dir, 'a.txt'), path.join(dir, 'b.txt'))).rejects.toThrow('ENOENT')
    expect(spy).toHaveBeenCalledTimes(1)
    spy.mockRestore()
    as('linux')
    spy = busy('EPERM', 1)
    await expect(renameRetrying(path.join(dir, 'a.txt'), path.join(dir, 'b.txt'))).rejects.toThrow('EPERM')
    expect(spy).toHaveBeenCalledTimes(1)
    spy.mockRestore()
    // A write whose rename fails leaves the old file as it was and no temporary file behind
    busy('EPERM', 1)
    await expect(writeAtomic(path.join(dir, 'a.txt'), 'new')).rejects.toThrow('EPERM')
    expect(text(dir, 'a.txt')).toBe('a')
    expect(readdirSync(dir).sort()).toEqual(['a.png', 'a.txt'])
  })
})
