import { copyFileSync, mkdirSync, mkdtempSync, promises as fsp, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import { afterAll, beforeAll, describe, expect, test, vi } from 'vitest'
import {
  checkFolder, type CheckSummary, fixShapes, folderStats, forgetChecks, issuesOf, problemPage, type ProblemQuery, type Progress, renamePlan, renameRun,
  saveStats, statsFileName, statsText, stopJob, strayFile, together
} from '../../src/main/checking'
import { forgetDataset, readDataset } from '../../src/main/dataset'
import { Stale } from '../../src/renderer/src/stale'
import { encodeMask } from '../../src/main/masks'
import { closeProject, openProject, setTrash } from '../../src/main/project'
import { CHECKS, type CheckOptions, checkDataset } from '../../src/shared/checks'
import type { DatasetImage, Issue, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { builtInKeypoints } from '../../src/shared/skeleton'
import { datasetStats } from '../../src/shared/stats'

// The main process's side of Dataset > Check labels, Statistics and renames, on folders of
// real images made by sharp, as tests/unit/dataset.test.ts makes them. The trash is a folder
// of the test's own.
const bin = mkdtempSync(path.join(tmpdir(), 'seamtag-trash-'))
let trashed: string[] = []
setTrash(async (file) => {
  renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
  trashed.push(file)
})
const made: string[] = [bin]
const folder = () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-checking-'))
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

let shade = 0
async function png(file: string, width = 40, height = 30) {
  mkdirSync(path.dirname(file), { recursive: true })
  shade += 9
  await sharp({ create: { width, height, channels: 3, background: { r: shade % 256, g: (shade * 7) % 256, b: 90 } } }).png().toFile(file)
}
const write = (dir: string, files: Record<string, string>) => {
  for (const [name, text] of Object.entries(files)) writeFileSync(path.join(dir, name), text)
}
const read = (dir: string, name: string) => readFileSync(path.join(dir, name), 'utf8')
const shape = (label: string, shape_type: string, points: number[][], extra: Record<string, unknown> = {}) =>
  ({ label, points, group_id: null, description: '', shape_type, flags: {}, mask: null, ...extra })
const labelme = (image: string, shapes: unknown[], extra: Record<string, unknown> = {}) =>
  JSON.stringify({ version: '5.5.0', flags: {}, shapes, imagePath: image, imageData: null, imageHeight: 30, imageWidth: 40, ...extra })
const box = (label: string, x0: number, y0: number, x1: number, y1: number, extra: Record<string, unknown> = {}) =>
  shape(label, 'rectangle', [[x0, y0], [x1, y1]], extra)
const asShapes = (raw: unknown[]) => (raw as Partial<Shape>[]).map((s) => newShape(s.label!, s.shape_type!, s.points!, { ...s, other: {} }))
/** Problems in an order of their own, so lists sorted another way compare */
const canon = (issues: Issue[]) => issues.map((i) => JSON.stringify([i.code, i.image, i.shape ?? null, i.message, i.severity])).sort()
const codes = (issues: Issue[]) => [...new Set(issues.map((i) => i.code))]
const LABELS = ['cat', 'dog']
const ALL: ProblemQuery = { severities: ['error', 'warning', 'info'], image: null, limits: {} }

// A folder with one of each problem the window plants for its own tests
async function planted() {
  const root = folder()
  for (const name of ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) await png(path.join(root, `${name}.png`))
  // An identical copy of h, also never labelled
  copyFileSync(path.join(root, 'h.png'), path.join(root, 'i.png'))
  write(root, {
    'a.json': labelme('a.png', [box('cat', 2, 2, 12, 12), box('cat', 2, 2, 12, 12)]),
    'b.json': labelme('b.png', [shape('cat', 'polygon', [[1, 1], [20, 1], [20, 1], [20, 20]])]),
    'c.json': labelme('c.png', [box('dog', 30, 20, 45, 34)]),
    'd.json': labelme('d.png', [box('Cat', 5, 5, 15, 15)]),
    'e.json': labelme('e.png', [box('zebra', 5, 5, 15, 15)]),
    'f.json': '{"shapes": [',
    'g.json': labelme('g.png', [box('dog', 5, 5, 15, 15, { flags: { ai: true }, score: 0.4 })]),
    'gone.json': labelme('gone.png', [box('cat', 1, 1, 5, 5)])
  })
  await openProject(root, null)
  return root
}

describe('two passes find what one checkDataset finds', () => {
  const W = 400, H = 300
  const image = (name: string, shapes: Shape[] = [], extra: Partial<DatasetImage> = {}): DatasetImage => ({
    name, width: W, height: H, file: `/data/${name.replace(/\.\w+$/, '.json')}`, format: 'labelme', shapes, masks: shapes.map(() => null),
    flags: {}, verified: false, labelSize: [W, H], labelImage: name, bytes: 1000 + name.length, ...extra
  })
  const rect = (label: string, x0: number, y0: number, x1: number, y1: number, extra: Partial<Shape> = {}) =>
    newShape(label, 'rectangle', [[x0, y0], [x1, y1]], extra)
  const poly = (label: string, points: number[][]) => newShape(label, 'polygon', points)
  const kp = (label: string, x: number, y: number, group: number | null) => newShape(label, 'point', [[x, y]], { group_id: group })
  const bits: MaskBits = { x: 10, y: 10, width: 2, height: 2, data: Uint8Array.from([1, 1, 1, 1]) }
  const mask = newShape('road', 'mask', [[10, 10], [11, 11]], { mask: 'png' })
  // Every check fires somewhere here
  const folderOfAll = () => [
    image('broken.png', [], { broken: "It isn't valid JSON" }),
    image('unreadable.png', [rect('car', 1, 1, 9, 9)], { width: 0, height: 0, unreadable: 'Input file is missing' }),
    image('size.png', [rect('car', 1, 1, 9, 9)], { labelSize: [10, 10] }),
    image('shapes.png', [
      rect(' ', 1, 1, 9, 9), poly('car', [[1, 1], [5, 1]]), rect('car', 10, 10, 10, 50), poly('car', [[0, 0], [0, 0], [10, 0], [10, 10]]),
      poly('car', [[0, 0], [10, 10], [10, 0], [0, 10]]), rect('car', 390, 290, 420, 330), rect('car', 100, 100, 150, 150), rect('car', 100, 100, 150, 150),
      rect('car', 200, 200, 250, 250), rect('car', 201, 200, 250, 250), { ...mask }
    ], { masks: [null, null, null, null, null, null, null, null, null, null, null] }),
    image('labels.png', [rect('Car', 1, 1, 9, 9), rect('zebra', 1, 1, 9, 9), rect('car', 20, 20, 29, 29, { flags: { ai: true }, other: { score: 0.3 } })]),
    image('a.jpg', [rect('car', 1, 1, 9, 9)], { file: '/data/a.json' }),
    image('a.png', [rect('car', 1, 1, 9, 9)], { file: '/data/a.json' }),
    image('elsewhere.png', [rect('car', 1, 1, 9, 9)], { labelImage: 'other.png' }),
    image('copy1.png', [rect('car', 1, 1, 9, 9)], { bytes: 77 }),
    image('copy2.png', [rect('car', 1, 1, 9, 9)], { bytes: 77 }),
    image('never.png', [], { file: null, format: null, labelSize: null }),
    image('empty.png'),
    image('verified.png', [rect('car', 1, 1, 9, 9)], { verified: true }),
    image('masked.png', [{ ...mask }], { masks: [bits] }),
    // A person with its hips the wrong way round and its nose far above its box, a point it has no place for and a
    // second nose; a keypoint of no group; and a group of one keypoint, with no box
    image('poses.png', [
      rect('person', 100, 50, 200, 290, { group_id: 1 }), kp('nose', 150, 20, 1), kp('left_eye', 157, 65, 1), kp('right_eye', 143, 65, 1),
      kp('left_shoulder', 175, 100, 1), kp('right_shoulder', 125, 100, 1), kp('left_hip', 135, 180, 1), kp('right_hip', 165, 180, 1),
      kp('tail', 150, 200, 1), kp('nose', 151, 71, 1), kp('nose', 5, 5, null), kp('nose', 300, 50, 9)
    ])
  ]
  const options = (): CheckOptions => ({
    labels: ['car', 'bus'], keypoints: builtInKeypoints(), clashes: [{ file: '/data/a.json', images: ['a.jpg', 'a.png'] }],
    duplicates: [['copy1.png', 'copy2.png']], orphans: [{ file: 'stray.json', names: null }]
  })

  test('on a folder where every check fires', async () => {
    const images = folderOfAll()
    const whole = checkDataset(images, options())
    expect(codes(whole).sort()).toEqual(Object.keys(CHECKS).sort())
    const { issues } = await issuesOf(images, options())
    expect(canon(issues)).toEqual(canon(whole))
    // Worst first, then the folder's own, then in the folder's order
    const order = issues.map((i) => [i.severity, i.image === null ? -1 : images.findIndex((x) => x.name === i.image)])
    expect(order).toEqual([...order].sort((a, b) => ['error', 'warning', 'info'].indexOf(a[0] as string) - ['error', 'warning', 'info'].indexOf(b[0] as string) || (a[1] as number) - (b[1] as number)))
  })

  test('after one image changes, measuring it alone again: what hangs on other images follows it', async () => {
    const images = folderOfAll()
    const first = await issuesOf(images, options())
    // The verified image now spells the label another way, four times
    const changed = images.map((i) => i.name === 'verified.png'
      ? { ...i, shapes: [rect('Car', 1, 1, 9, 9), rect('Car', 30, 30, 39, 39), rect('Car', 60, 60, 69, 69), rect('Car', 90, 90, 99, 99)] }
      : i)
    const next = await issuesOf(changed, options(), first)
    expect(canon(next.issues)).toEqual(canon(checkDataset(changed, options())))
    // Images read as before keep what was found in them; only the changed one was measured again
    for (const name of ['shapes.png', 'labels.png', 'masked.png']) expect(next.own.get(name)).toBe(first.own.get(name))
    expect(next.own.get('verified.png')).not.toBe(first.own.get('verified.png'))
    expect(next.across).not.toBe(first.across)
    expect(next.issues.filter((i) => i.code === 'label-variants').map((i) => i.image)).toContain('verified.png')
  })

  test("once the folder's one verified image isn't, no image is told it isn't verified: every image is measured again", async () => {
    const images = folderOfAll()
    const first = await issuesOf(images, options())
    expect(first.issues.filter((i) => i.code === 'not-verified').length).toBeGreaterThan(5)
    const changed = images.map((i) => (i.name === 'verified.png' ? { ...i, verified: false } : i))
    const next = await issuesOf(changed, options(), first)
    expect(canon(next.issues)).toEqual(canon(checkDataset(changed, options())))
    expect(next.issues.filter((i) => i.code === 'not-verified')).toEqual([])
    expect(next.own.get('shapes.png')).not.toBe(first.own.get('shapes.png'))
    // And back: every image but the verified one says so again
    const again = await issuesOf(images, options(), next)
    expect(canon(again.issues)).toEqual(canon(first.issues))
  })

  test("a shape that moved changes nothing across the folder, so that's kept; a new label or label list doesn't", async () => {
    const images = folderOfAll()
    const first = await issuesOf(images, options())
    const moved = images.map((i) => (i.name === 'shapes.png' ? { ...i, shapes: i.shapes.map((s, k) => (k === 7 ? { ...s, points: [[300, 100], [350, 150]] } : s)) } : i))
    const next = await issuesOf(moved, options(), first)
    expect(next.across).toBe(first.across)
    expect(canon(next.issues)).toEqual(canon(checkDataset(moved, options())))
    expect(next.issues.filter((i) => i.code === 'duplicate-shape')).toEqual([])
    const relabelled = moved.map((i) => (i.name === 'shapes.png' ? { ...i, shapes: i.shapes.map((s, k) => (k === 7 ? { ...s, label: 'CAR' } : s)) } : i))
    const after = await issuesOf(relabelled, options(), next)
    expect(after.across).not.toBe(next.across)
    expect(canon(after.issues)).toEqual(canon(checkDataset(relabelled, options())))
    const listed = await issuesOf(relabelled, { ...options(), labels: ['car'] }, after)
    expect(listed.across).not.toBe(after.across)
    expect(canon(listed.issues)).toEqual(canon(checkDataset(relabelled, { ...options(), labels: ['car'] })))
  })

  test("other keypoints check every image's poses again, and what hangs on them across the folder; the same keypoints keep what was found", async () => {
    const images = folderOfAll()
    const first = await issuesOf(images, options())
    const again = await issuesOf(images, options(), first)
    expect(again.own.get('poses.png')).toBe(first.own.get('poses.png'))
    expect(again.across).toBe(first.across)
    // A keypoints file whose person has a tail, and no flip pairs: the tail is no unknown label or point, and nothing looks swapped
    const tailed: KeypointSchema = { classes: { person: ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'tail'] } }
    const next = await issuesOf(images, { ...options(), keypoints: tailed }, first)
    expect(canon(next.issues)).toEqual(canon(checkDataset(images, { ...options(), keypoints: tailed })))
    expect(next.own.get('poses.png')).not.toBe(first.own.get('poses.png'))
    expect(next.own.get('shapes.png')).not.toBe(first.own.get('shapes.png'))
    expect(next.across).not.toBe(first.across)
    const at = (found: typeof first) => codes(found.issues.filter((i) => i.image === 'poses.png')).sort()
    expect(at(first)).toEqual(expect.arrayContaining(['keypoint-swapped', 'keypoint-unknown', 'unknown-label']))
    expect(at(next)).not.toContain('keypoint-swapped')
    expect(at(next)).not.toContain('keypoint-unknown')
    expect(next.issues.find((i) => i.code === 'unknown-label' && i.image === 'poses.png')!.message).toBe('Label “person” isn\'t in the label list (shape 1)')
  })

  test('progress is told slice by slice, and a stop ends it', async () => {
    const images = Array.from({ length: 450 }, (_, k) => image(`${k}.png`, [rect('car', 1, 1, 9, 9)]))
    const told: number[] = []
    await issuesOf(images, { labels: null }, undefined, (done) => told.push(done))
    expect(told).toEqual([200, 400, 450])
    const stop = new AbortController()
    stop.abort()
    await expect(issuesOf(images, { labels: null }, undefined, undefined, stop.signal)).rejects.toThrow()
  })

  test("the pass over the whole folder's labels goes a slice at a time too, so the window gets its turns and a stop ends it part way", async () => {
    let looked = 0
    // Each image's labels are counted as they're read
    const images = Array.from({ length: 4000 }, (_, k) => {
      const { shapes, ...rest } = image(`${k}.png`, [rect(k % 2 ? 'car' : 'Car', 1, 1, 9, 9)])
      return Object.defineProperty(rest, 'shapes', { enumerable: true, get: () => (looked++, shapes) }) as DatasetImage
    })
    const stop = new AbortController()
    setImmediate(() => stop.abort())
    await expect(issuesOf(images, { labels: ['car'] }, undefined, undefined, stop.signal)).rejects.toThrow()
    expect(looked).toBeLessThan(images.length)
  })
})

describe('a folder with planted problems', () => {
  let root = ''
  let summary: CheckSummary
  const phases: Progress['phase'][] = []
  beforeAll(async () => {
    root = await planted()
    const result = await checkFolder(root, { prefer: 'labelme', labels: LABELS }, (p) => phases.push(p.phase))
    if ('stopped' in result) throw new Error('stopped')
    summary = result
  })

  test('is read, compared, searched for stray label files and checked, and each problem counted', () => {
    expect([...new Set(phases)]).toEqual(['read', 'compare', 'orphans', 'check'])
    expect(summary.images).toBe(9)
    expect(summary.counts).toEqual({ error: 1, warning: 9, info: 2 })
    // Errors and warnings only: never-labelled images are information
    expect([...summary.problemImages].sort()).toEqual(['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'])
  })

  test('the dock gets groups of the severities it shows, worst first, each with its count and rows', () => {
    const page = problemPage(root, { severities: ['error', 'warning'], image: null, limits: {} })!
    expect(page.at).toBe(summary.at)
    expect(page.counts).toEqual({ error: 1, warning: 9, info: 2 })
    expect(page.groups.map((g) => [g.code, g.title, g.count])).toEqual([
      ['label-file-unreadable', CHECKS['label-file-unreadable'].title, 1],
      ['repeated-points', CHECKS['repeated-points'].title, 1],
      ['outside-image', CHECKS['outside-image'].title, 1],
      ['duplicate-shape', CHECKS['duplicate-shape'].title, 1],
      ['unknown-label', CHECKS['unknown-label'].title, 1],
      ['label-variants', CHECKS['label-variants'].title, 1],
      ['ai-pending', CHECKS['ai-pending'].title, 1],
      ['orphan-label-file', CHECKS['orphan-label-file'].title, 1],
      ['duplicate-image', CHECKS['duplicate-image'].title, 2]
    ])
    const row = (code: string) => page.groups.find((g) => g.code === code)!.rows[0]
    // A row says which shape and label it's about, and what can be done about it
    expect(row('duplicate-shape')).toMatchObject({ image: 'a.png', shape: 1, label: 'cat', fix: true })
    expect(row('repeated-points')).toMatchObject({ image: 'b.png', shape: 0, label: 'cat', fix: true })
    expect(row('outside-image')).toMatchObject({ image: 'c.png', shape: 0, label: 'dog', fix: true })
    expect(row('label-variants')).toMatchObject({ image: 'd.png', shape: 0, label: 'Cat', to: 'cat' })
    expect(row('label-variants').fix).toBeUndefined()
    expect(row('unknown-label')).toMatchObject({ image: 'e.png', label: 'zebra' })
    expect(row('orphan-label-file')).toMatchObject({ image: null, file: 'gone.json' })
  })

  test('one image alone, information shown, and groups cut to the rows asked for', () => {
    const one = problemPage(root, { ...ALL, severities: ['error', 'warning', 'info'], image: 'h.png' })!
    expect(one.counts).toEqual({ error: 0, warning: 1, info: 1 })
    expect(one.groups.map((g) => g.code)).toEqual(['duplicate-image', 'never-labelled'])
    const cut = problemPage(root, { severities: ['warning'], image: null, limits: { 'duplicate-image': 1, 'ai-pending': 0 } })!
    const group = (code: string) => cut.groups.find((g) => g.code === code)!
    expect([group('duplicate-image').count, group('duplicate-image').rows.length]).toEqual([2, 1])
    expect([group('ai-pending').count, group('ai-pending').rows.length]).toEqual([1, 0])
    expect(group('duplicate-shape').rows).toHaveLength(1)
  })

  test("a label file without its image can be shown in its folder; no other file can", () => {
    expect(strayFile(root, 'gone.json')).toBe(path.join(root, 'gone.json'))
    expect(strayFile(root, 'a.json')).toBeNull()
    expect(strayFile(root, '../../etc/passwd')).toBeNull()
  })

  test('fixes are made to the shapes the window has, and only while the problem is still there', async () => {
    const a = asShapes(JSON.parse(read(root, 'a.json')).shapes)
    expect(await fixShapes(root, { code: 'duplicate-shape', image: 'a.png', shape: 1 }, a)).toHaveLength(1)
    // Moved in the window since the check: no longer a copy, so nothing is fixed
    const moved = [a[0], { ...a[1], points: [[3, 3], [13, 13]] }]
    expect(await fixShapes(root, { code: 'duplicate-shape', image: 'a.png', shape: 1 }, moved)).toBeNull()
    const b = asShapes(JSON.parse(read(root, 'b.json')).shapes)
    expect((await fixShapes(root, { code: 'repeated-points', image: 'b.png', shape: 0 }, b))![0].points).toEqual([[1, 1], [20, 1], [20, 20]])
    const c = asShapes(JSON.parse(read(root, 'c.json')).shapes)
    expect((await fixShapes(root, { code: 'outside-image', image: 'c.png', shape: 0 }, c))![0].points).toEqual([[30, 20], [40, 30]])
    // A problem with no safe fix, or an image that isn't in the folder
    expect(await fixShapes(root, { code: 'unknown-label', image: 'e.png', shape: 0 }, [])).toBeNull()
    await expect(fixShapes(root, { code: 'duplicate-shape', image: '../x.png', shape: 1 }, a)).rejects.toThrow("isn't in the folder")
  })

  test("masks drawn since the check are decoded to tell copies apart", async () => {
    const two = await encodeMask(Uint8Array.from([1, 1, 0, 1]), 2, 2)
    const other = await encodeMask(Uint8Array.from([1, 0, 0, 1]), 2, 2)
    const masks = (second: string) => [newShape('road', 'mask', [[4, 4], [5, 5]], { mask: two }), newShape('road', 'mask', [[4, 4], [5, 5]], { mask: second })]
    expect(await fixShapes(root, { code: 'duplicate-shape', image: 'h.png', shape: 1 }, masks(two))).toHaveLength(1)
    expect(await fixShapes(root, { code: 'duplicate-shape', image: 'h.png', shape: 1 }, masks(other))).toBeNull()
  })
})

describe('checking again', () => {
  test('after a save, only what changed is read again; identical images and stray files are kept until a full check', async () => {
    const root = await planted()
    const phases: string[] = []
    const send = (p: Progress) => phases.push(p.phase)
    await checkFolder(root, { prefer: 'labelme', labels: LABELS }, send)
    const page = () => problemPage(root, { ...ALL, severities: ['error', 'warning', 'info'] })!.groups.map((g) => [g.code, g.count])
    // b's repeated point is taken out, and the stray label file goes, outside the check
    write(root, { 'b.json': labelme('b.png', [shape('cat', 'polygon', [[1, 1], [20, 1], [20, 20]])]) })
    rmSync(path.join(root, 'gone.json'))
    phases.length = 0
    await checkFolder(root, { prefer: 'labelme', labels: LABELS, saved: ['b.png'] }, send)
    expect([...new Set(phases)]).toEqual(['read', 'check'])
    expect(page()).not.toContainEqual(['repeated-points', 1])
    expect(page()).toContainEqual(['orphan-label-file', 1])
    await checkFolder(root, { prefer: 'labelme', labels: LABELS }, send)
    expect(page()).not.toContainEqual(['orphan-label-file', 1])
    // A new image is a change to the folder's images: everything is looked at again
    await png(path.join(root, 'j.png'))
    await openProject(root, null)
    phases.length = 0
    await checkFolder(root, { prefer: 'labelme', labels: LABELS, saved: [] }, send)
    expect([...new Set(phases)]).toEqual(['read', 'compare', 'orphans', 'check'])
  })

  test("after saves only the images saved are read again: a file changed outside the window waits for a full check", async () => {
    const root = await planted()
    await checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    const codes = () => problemPage(root, ALL)!.groups.map((g) => g.code)
    const before = await readDataset(root, { prefer: 'labelme' })
    // The window saved b without its repeated point; c's box was moved inside the image by another program
    write(root, {
      'b.json': labelme('b.png', [shape('cat', 'polygon', [[1, 1], [20, 1], [20, 20]])]),
      'c.json': labelme('c.png', [box('dog', 30, 20, 40, 30)])
    })
    const summary = await checkFolder(root, { prefer: 'labelme', labels: LABELS, saved: ['b.png'] }, () => {})
    expect(summary).toHaveProperty('counts')
    expect(codes()).not.toContain('repeated-points')
    expect(codes()).toContain('outside-image')
    // Read as another format, any image could have other files: the whole folder is read
    await checkFolder(root, { prefer: 'voc', labels: LABELS, saved: ['b.png'] }, () => {})
    expect(codes()).not.toContain('outside-image')
    await checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    expect(codes()).not.toContain('outside-image')
    // readDataset's own: the images not named are the objects read before, their files not looked at
    write(root, { 'a.json': labelme('a.png', []), 'd.json': labelme('d.png', []) })
    const after = await readDataset(root, { prefer: 'labelme', only: new Set(['d.png']) })
    const named = (images: typeof before, name: string) => images.find((i) => i.name === name)!
    expect(named(after, 'd.png').shapes).toEqual([])
    expect(named(after, 'a.png').shapes).toHaveLength(2)
    expect(named(after, 'e.png')).toBe(named(before, 'e.png'))
    expect(named(await readDataset(root, { prefer: 'labelme' }), 'a.png').shapes).toEqual([])
  })

  test('a stop ends the check, and the last result stays', async () => {
    const root = await planted()
    await checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    const before = problemPage(root, ALL)
    const running = checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    stopJob(root, 'check')
    expect(await running).toEqual({ stopped: true })
    expect(problemPage(root, ALL)).toEqual(before)
    // A second check stops the first
    const first = checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    const second = checkFolder(root, { prefer: 'labelme', labels: LABELS }, () => {})
    expect(await first).toEqual({ stopped: true })
    expect(await second).toHaveProperty('counts')
    forgetChecks(root)
    expect(problemPage(root, ALL)).toBeNull()
  })

  test('a folder closed as its check ends, as another opens, keeps no check', async () => {
    const root = await planted()
    let closed = false
    const result = await checkFolder(root, { prefer: 'labelme', labels: LABELS }, (p) => {
      if (closed || p.phase !== 'check' || p.done < p.total) return
      closed = true
      // As project:close does, between the check's last slice and its end
      setImmediate(() => {
        forgetDataset(root)
        forgetChecks(root)
        closeProject(root)
      })
    })
    expect(closed).toBe(true)
    expect(result).toEqual({ stopped: true })
    expect(problemPage(root, ALL)).toBeNull()
  })
})

describe("the folder's keypoints", () => {
  /** A folder of two images: a person with a nose and a tail, the nose given twice; and a nose of no group */
  async function posedFolder() {
    const root = folder()
    for (const name of ['a', 'b']) await png(path.join(root, `${name}.png`))
    write(root, {
      'a.json': labelme('a.png', [box('person', 10, 5, 30, 28, { group_id: 1 }), shape('nose', 'point', [[20, 10]], { group_id: 1 }),
        shape('tail', 'point', [[20, 26]], { group_id: 1 }), shape('nose', 'point', [[21, 11]], { group_id: 1 })]),
      'b.json': labelme('b.png', [shape('nose', 'point', [[5, 5]])])
    })
    await openProject(root, null)
    return root
  }
  const groups = (root: string) => problemPage(root, ALL)!.groups.map((g) => [g.code, g.count])

  test("are read once a check, COCO's person until the folder has a file, and a new file checks every image's poses again", async () => {
    const root = await posedFolder()
    const reads = vi.spyOn(fsp, 'readFile')
    const keypointReads = () => reads.mock.calls.filter(([file]) => String(file).endsWith('keypoints.yaml')).length
    try {
      await checkFolder(root, { prefer: 'labelme', labels: null }, () => {})
      expect(keypointReads()).toBe(1)
      expect(groups(root)).toEqual([['keypoint-no-group', 1], ['keypoint-unknown', 1], ['keypoint-repeated', 1]])
      // The folder's own file, where a person has a tail: a check after saves, with none saved, reads no label file but checks every pose with it
      mkdirSync(path.join(root, '.myseg'), { recursive: true })
      writeFileSync(path.join(root, '.myseg', 'keypoints.yaml'), 'classes:\n  person: [nose, tail]\n')
      await checkFolder(root, { prefer: 'labelme', labels: null, saved: [] }, () => {})
      expect(keypointReads()).toBe(2)
      expect(groups(root)).toEqual([['keypoint-no-group', 1], ['keypoint-repeated', 1]])
    } finally {
      vi.restoreAllMocks()
    }
  })

  test('a keypoint given twice is fixed in the shapes the window has, by deleting the second, while it is still the second', async () => {
    const root = await posedFolder()
    await checkFolder(root, { prefer: 'labelme', labels: null }, () => {})
    const row = problemPage(root, ALL)!.groups.find((g) => g.code === 'keypoint-repeated')!.rows[0]
    expect(row).toMatchObject({ image: 'a.png', shape: 3, label: 'nose', fix: true })
    expect(problemPage(root, ALL)!.groups.find((g) => g.code === 'keypoint-unknown')!.rows[0].fix).toBeUndefined()
    const shapes = asShapes(JSON.parse(read(root, 'a.json')).shapes)
    expect(await fixShapes(root, { code: 'keypoint-repeated', image: 'a.png', shape: 3 }, shapes)).toEqual(shapes.slice(0, 3))
    // Given a group of its own in the window since the check, it's no second nose
    expect(await fixShapes(root, { code: 'keypoint-repeated', image: 'a.png', shape: 3 }, [...shapes.slice(0, 3), { ...shapes[3], group_id: 2 }])).toBeNull()
  })

  test("with only listed labels allowed, a rename may still give a keypoint's name", async () => {
    const root = await posedFolder()
    write(root, { 'b.json': labelme('b.png', [shape('Nose', 'point', [[5, 5]])]) })
    await expect(renameRun(root, 'Nose', 'snout', 'labelme', ['person'], () => {})).rejects.toThrow("“snout” isn't in the label list")
    expect((await renameRun(root, 'Nose', 'nose', 'labelme', ['person'], () => {})).changed).toEqual(['b.png'])
  })
})

describe('renaming a label across the folder', () => {
  test('is planned before anything is written, then made in each file, with the old files in the trash', async () => {
    const root = await planted()
    await png(path.join(root, 'v.png'))
    write(root, {
      'v.xml': '<annotation><size><width>40</width><height>30</height><depth>3</depth></size><object><name>Cat</name><pose>Left</pose>' +
        '<difficult>0</difficult><bndbox><xmin>1.5</xmin><ymin>0</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object></annotation>'
    })
    await openProject(root, null)
    const plan = await renamePlan(root, 'Cat', 'labelme', () => {})
    expect(plan).toEqual({
      images: 2, shapes: 2, names: ['d.png', 'v.png'], skippedCount: 1,
      skipped: [{ name: 'f.png', reason: "f.json can't be read, so it's left unchanged" }]
    })
    expect(read(root, 'd.json')).toContain('"Cat"')
    trashed = []
    const done = await renameRun(root, 'Cat', 'cat', 'labelme', null, () => {})
    // f.json is left alone whatever it holds: the window names it, and says why
    expect(done).toEqual({ stopped: false, shapes: 2, changed: ['d.png', 'v.png'], skipped: { unreadable: 1, shared: 0, file: 'f.json' }, failed: [], left: 0 })
    expect(JSON.parse(read(root, 'd.json')).shapes[0].label).toBe('cat')
    // VOC is changed where its objects are: the pose and the exact numbers stay
    expect(read(root, 'v.xml')).toContain('<name>cat</name><pose>Left</pose>')
    expect(read(root, 'v.xml')).toContain('<xmin>1.5</xmin><ymin>0</ymin>')
    expect(trashed.map((f) => path.basename(f))).toEqual(['d.json', 'v.xml'])
    expect(await renamePlan(root, 'Cat', 'labelme', () => {})).toMatchObject({ images: 0, shapes: 0 })
  })

  test("refuses a name the config doesn't allow, or none; a file that can't be written is told and the rest go on", async () => {
    const root = await planted()
    await expect(renameRun(root, 'Cat', 'kitten', 'labelme', ['cat', 'dog'], () => {})).rejects.toThrow("“kitten” isn't in the label list")
    await expect(renameRun(root, 'Cat', '  ', 'labelme', null, () => {})).rejects.toThrow('A label needs a name')
    write(root, { 'h.json': labelme('h.png', [box('Cat', 1, 1, 5, 5)]) })
    setTrash(async (file) => {
      if (path.basename(file) === 'd.json') throw new Error('The trash is full')
      renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
      trashed.push(file)
    })
    try {
      const done = await renameRun(root, 'Cat', 'cat', 'labelme', null, () => {})
      expect(done.changed).toEqual(['h.png'])
      expect(done.failed).toEqual([{ name: 'd.png', reason: "d.json couldn't go to the trash, so it's left unchanged: The trash is full" }])
      expect(read(root, 'd.json')).toContain('"Cat"')
      // d has the old name still, so the window keeps it in the label list
      expect(done.left).toBe(1)
    } finally {
      setTrash(async (file) => {
        renameSync(file, path.join(bin, `${trashed.length}-${path.basename(file)}`))
        trashed.push(file)
      })
    }
  })
})

describe('renaming: the new name and what keeps the old one', () => {
  test('the new name is trimmed, as the label dialog trims a label, before the label list is asked', async () => {
    const root = await planted()
    const done = await renameRun(root, 'zebra', ' dog ', 'labelme', ['cat', 'dog'], () => {})
    expect(done).toMatchObject({ changed: ['e.png'], shapes: 1, left: 0 })
    expect(JSON.parse(read(root, 'e.json')).shapes[0].label).toBe('dog')
    await renameRun(root, 'Cat', 'cat\t', 'labelme', null, () => {})
    expect(JSON.parse(read(root, 'd.json')).shapes[0].label).toBe('cat')
  })

  test('images that share a label file keep the old name, and say so', async () => {
    const root = await planted()
    await png(path.join(root, 'k.png'))
    await png(path.join(root, 'k.jpg'))
    write(root, { 'k.json': labelme('k.png', [box('Cat', 1, 1, 5, 5)]) })
    await openProject(root, null)
    const done = await renameRun(root, 'Cat', 'cat', 'labelme', null, () => {})
    // Both images keep the old name; their one file is left alone, and so is f.json, which can't be read
    expect(done).toMatchObject({ changed: ['d.png'], failed: [], left: 2, skipped: { unreadable: 1, shared: 1 } })
    expect(done.skipped.file).toBeUndefined()
    expect(read(root, 'k.json')).toContain('"Cat"')
    // With f.json gone, k.json is the one file left alone, and the window can name it
    rmSync(path.join(root, 'f.json'))
    write(root, { 'm.json': labelme('m.png', [box('Cat', 1, 1, 5, 5)]) })
    await png(path.join(root, 'm.png'))
    await openProject(root, null)
    expect((await renameRun(root, 'Cat', 'cat', 'labelme', null, () => {})).skipped).toEqual({ unreadable: 0, shared: 1, file: 'k.json' })
  })

  test("the images of one CreateML list, as Roboflow's _annotations.createml.json holds them, are renamed in one rewrite of it", async () => {
    const root = folder()
    const names = ['a.png', 'b.png', 'c.png']
    for (const name of names) await png(path.join(root, name))
    const list = path.join(root, '_annotations.createml.json')
    writeFileSync(list, JSON.stringify(names.map((image) => ({ image, verified: false, annotations: [{ label: 'Cat', coordinates: { x: 9, y: 9, width: 6, height: 6 } }] }))))
    await openProject(root, list)
    trashed = []
    const progress: Progress[] = []
    const done = await renameRun(root, 'Cat', 'cat', 'createml', null, (p) => progress.push(p))
    expect(done).toEqual({ stopped: false, shapes: 3, changed: names, skipped: { unreadable: 0, shared: 0 }, failed: [], left: 0 })
    // One copy of the list in the trash, as it was, not one an image
    expect(trashed).toEqual([list])
    expect(JSON.parse(readFileSync(list, 'utf8')).map((e: { annotations: { label: string }[] }) => e.annotations[0].label)).toEqual(['cat', 'cat', 'cat'])
    expect(progress.filter((p) => p.phase === 'write')).toEqual([{ phase: 'write', done: 3, total: 3 }])
  })

  test("a stop while a CreateML list's images are looked up ends the rename before the list is written", async () => {
    const root = folder()
    const names = ['a.png', 'b.png', 'c.png']
    for (const name of names) await png(path.join(root, name))
    const list = path.join(root, '_annotations.createml.json')
    const text = JSON.stringify(names.map((image) => ({ image, verified: false, annotations: [{ label: 'Cat', coordinates: { x: 9, y: 9, width: 6, height: 6 } }] })))
    writeFileSync(list, text)
    await openProject(root, list)
    trashed = []
    // Once the folder is read, the list is opened again for each image, to see it still has its labels: Stop comes at the second
    let folderRead = false, opened = 0
    const open = fsp.open.bind(fsp)
    vi.spyOn(fsp, 'open').mockImplementation(async (file, ...rest) => {
      if (folderRead && path.basename(String(file)) === path.basename(list) && ++opened === 2) stopJob(root, 'rename')
      return open(file, ...rest)
    })
    try {
      const done = await renameRun(root, 'Cat', 'cat', 'createml', null, (p) => {
        if (p.phase === 'read' && p.done === p.total) folderRead = true
      })
      expect(opened).toBe(2)
      expect(done).toEqual({ stopped: true, shapes: 0, changed: [], skipped: { unreadable: 0, shared: 0 }, failed: [], left: 3 })
    } finally {
      vi.restoreAllMocks()
    }
    expect(trashed).toEqual([])
    expect(readFileSync(list, 'utf8')).toBe(text)
  })
})

describe('out of date in the window', () => {
  test('a check takes in the saves made before it began; one made while it ran waits for the next', () => {
    const stale = new Stale()
    stale.mark(['a.png', 'b.png'])
    const first = stale.begin()
    expect(first.names).toEqual(['a.png', 'b.png'])
    // b is saved again while the check runs, which may have read it before
    stale.mark(['b.png'])
    expect(stale.settle(first.upTo)).toBe(true)
    expect([stale.has('a.png'), stale.has('b.png'), stale.size]).toEqual([false, true, 1])
    const second = stale.begin()
    expect(second.names).toEqual(['b.png'])
    expect(stale.settle(second.upTo)).toBe(false)
    expect(stale.size).toBe(0)
  })

  test("saves made while the folder's first check runs wait for the next", () => {
    const stale = new Stale()
    const first = stale.begin()
    expect(first.names).toEqual([])
    stale.mark(['c.png'])
    expect(stale.settle(first.upTo)).toBe(true)
    expect(stale.begin().names).toEqual(['c.png'])
  })
})

describe('statistics', () => {
  test('count the folder as datasetStats does, in slices, with the image sizes cut to ten', async () => {
    const root = await planted()
    for (let k = 0; k < 12; k++) await png(path.join(root, `sizes/${k}.png`), 10 + k, 10)
    await openProject(root, null)
    const phases: string[] = []
    const view = await folderStats(root, { prefer: 'labelme', labels: ['cat', 'dog', 'bus'], trainSize: 320, again: true }, (p) => phases.push(p.phase))
    if ('stopped' in view) throw new Error('stopped')
    const whole = datasetStats(await readDataset(root), { trainSize: 320, labels: ['cat', 'dog', 'bus'] })
    expect({ ...view, moreSizes: undefined }).toEqual({ ...whole, imageSizes: whole.imageSizes.slice(0, 10), moreSizes: undefined })
    expect(view.moreSizes).toBe(whole.imageSizes.length - 10)
    expect([...new Set(phases)]).toEqual(['read', 'count'])
    expect(statsText(root)!.split('\r\n')[0]).toBe('﻿label,objects,images,polygon,rectangle')
    // Another training size counts what was read last, without reading again
    write(root, { 'a.json': labelme('a.png', []) })
    phases.length = 0
    const again = await folderStats(root, { prefer: 'labelme', labels: null, trainSize: 6400, again: false }, (p) => phases.push(p.phase))
    expect(phases).not.toContain('read')
    if ('stopped' in again) throw new Error('stopped')
    expect(again.sizes.trainSize).toBe(6400)
    expect(again.labels[0]).toEqual({ label: 'cat', objects: 3, images: 2, types: { rectangle: 2, polygon: 1 } })
  })

  test("Save as CSV suggests a file beside the folder, named after it, or in it when the folder it's in can't be written to", async () => {
    const parent = folder()
    const root = path.join(parent, 'birds')
    mkdirSync(root)
    expect(await statsFileName(root)).toBe(path.join(parent, 'birds-statistics.csv'))
    // As at the top of a removable drive, whose parent (/media/<user>, /Volumes) is the system's
    const access = fsp.access.bind(fsp)
    vi.spyOn(fsp, 'access').mockImplementation(async (file, mode) => {
      if (file === parent) throw Object.assign(new Error(`EACCES: permission denied, access '${parent}'`), { code: 'EACCES' })
      return access(file, mode)
    })
    try {
      expect(await statsFileName(root)).toBe(path.join(root, 'birds-statistics.csv'))
    } finally {
      vi.restoreAllMocks()
    }
  })

  test("a CSV that can't be saved is told by the name it was given, not the file its text goes to first, and is left as it was", async () => {
    const dir = folder()
    const file = path.join(dir, 'birds-statistics.csv')
    writeFileSync(file, 'old')
    // As in a folder the user can't write to: Node's message names the file beside the one chosen
    const write = fsp.writeFile.bind(fsp)
    vi.spyOn(fsp, 'writeFile').mockImplementation(async (to, data, options) => {
      if (String(to).endsWith('.tmp')) throw Object.assign(new Error(`EACCES: permission denied, open '${String(to)}'`), { code: 'EACCES' })
      return write(to, data, options)
    })
    try {
      await expect(saveStats(file, 'label,objects\r\n')).rejects.toThrow(new Error(`birds-statistics.csv can't be written in ${dir} (EACCES)`))
    } finally {
      vi.restoreAllMocks()
    }
    expect(readdirSync(dir)).toEqual(['birds-statistics.csv'])
    expect(readFileSync(file, 'utf8')).toBe('old')
    await saveStats(file, 'label,objects\r\n')
    expect(readFileSync(file, 'utf8')).toBe('label,objects\r\n')
  })

  test("slices' statistics put together are the whole folder's", () => {
    const image = (name: string, shapes: Shape[] | null, extra: Partial<DatasetImage> = {}): DatasetImage => ({
      name, width: 400, height: 300, file: shapes ? `/d/${name}.json` : null, format: shapes ? 'labelme' : null, shapes: shapes ?? [],
      masks: (shapes ?? []).map(() => null), flags: {}, verified: false, ...extra
    })
    const rect = (label: string, x1: number, extra: Partial<Shape> = {}) => newShape(label, 'rectangle', [[0, 0], [x1, x1]], extra)
    const images = Array.from({ length: 23 }, (_, k) => image(`${k}`, k % 7 === 0 ? null : k % 5 === 0 ? [] : [
      rect(['car', 'bus', 'person'][k % 3], 5 + k * 4), newShape('nose', 'point', [[1, 1]]), rect('car', 90, k % 4 ? {} : { flags: { ai: true } })
    ], { width: 400 + (k % 3) * 100, verified: k % 6 === 0, ...(k === 9 ? { broken: 'bad' } : {}), ...(k === 11 ? { unreadable: 'gone', width: 0, height: 0 } : {}) }))
    const labels = ['car', 'van']
    const whole = datasetStats(images, { trainSize: 512, labels })
    for (const size of [1, 4, 10, 23]) {
      const parts = Array.from({ length: Math.ceil(images.length / size) }, (_, k) => datasetStats(images.slice(k * size, (k + 1) * size), { trainSize: 512 }))
      expect(together(parts, 512, labels)).toEqual(whole)
    }
  })
})
