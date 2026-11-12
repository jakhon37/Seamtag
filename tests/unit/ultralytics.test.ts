import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import yaml from 'js-yaml'
import sharp from 'sharp'
import { describe, expect, test } from 'vitest'
import { readYoloDataset } from '../../src/main/ultralytics'
import type { DatasetImage, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { circlePolygon } from '../../src/shared/outline'
import { ULTRALYTICS_YAML, yamlString } from '../../src/shared/pose'
import {
  countRows, guessTask, isYoloImage, labelPathFor, labelRows, labelText, newRowStats, poseYamlKeys, quadKind, readDataYaml, readRows, verifyRows,
  writeDataYaml, YOLO_TASKS, type YoloTask
} from '../../src/shared/ultralytics'

// What Ultralytics 8.4.211's own code makes of label files and data.yaml: tests/fixtures/ultralytics/make_verify_cases.py
const FIXTURES = path.join(__dirname, '../fixtures/ultralytics')
const ref = JSON.parse(readFileSync(path.join(FIXTURES, 'verify_cases.json'), 'utf8'))

type LabelCase = {
  name: string; task: YoloTask; nc: number; kpt_shape: [number, number] | null; text?: string | null; bytes?: string
  status: string; duplicates: number; boxes?: number[][]; segments?: number[][][]; keypoints?: number[][][]
}
const textOf = (c: LabelCase) => (c.bytes !== undefined ? labelText(Buffer.from(c.bytes, 'base64')) : c.text ?? null)
const sorted = (rows: unknown[]) => rows.map((r) => JSON.stringify(r)).sort()
// Line and paragraph separators and byte order marks, which Seamtag escapes and Ultralytics doesn't
const SEPARATORS = new RegExp('[\\u2028\\u2029\\ufeff]')

describe("label files, as Ultralytics 8.4.211's verify_image_label reads them", () => {
  for (const c of ref.labels as LabelCase[]) {
    test(`${c.task}: ${c.name}`, () => {
      const text = textOf(c)
      // Text that isn't UTF-8 is corrupt before any row is read
      const got = text === null && c.bytes !== undefined ? { status: 'corrupt', repeated: [], boxes: [], outlines: [], keypoints: null } : verifyRows(text, c.task, c.nc, c.kpt_shape)
      expect(got.status).toBe(c.status)
      expect(got.repeated.length).toBe(c.duplicates)
      if (c.status !== 'ok') return
      // Ultralytics sorts the rows it keeps when it drops repeats
      const order = c.duplicates ? sorted : (rows: unknown[]) => rows
      expect(order(got.boxes)).toEqual(order(c.boxes!))
      expect(order(got.outlines)).toEqual(order(c.segments!))
      if (c.keypoints) expect(order(got.keypoints!)).toEqual(order(c.keypoints))
      else expect(got.keypoints).toBeNull()
    })
  }
})

describe('names in data.yaml', () => {
  type Read = { type: string; str: string }
  const plain = ref.yaml_plain as { text: string; ultralytics: Read; pyyaml: Read; written: string }[]
  test('a name is written plain only when Ultralytics and PyYAML both read it back as it is', () => {
    for (const p of plain) {
      if (yamlString(p.text) !== p.text) continue
      expect(p.ultralytics, p.text).toEqual({ type: 'str', str: p.text })
      expect(p.pyyaml, p.text).toEqual({ type: 'str', str: p.text })
    }
    expect(plain.filter((p) => yamlString(p.text) === p.text).length).toBeGreaterThan(10)
  })
  test("quoted names are written as Ultralytics' YAML._repr writes them, and read back as they are", () => {
    for (const p of plain) {
      const written = yamlString(p.text)
      expect(yaml.load(`k: ${written}`, { schema: ULTRALYTICS_YAML }), p.text).toEqual({ k: p.text })
      if (written === p.text || p.written[0] !== '"' || SEPARATORS.test(p.text)) continue
      expect(written, p.text).toBe(p.written)
    }
  })
  test('names read as check_class_names gives them, or are refused where it raises', () => {
    for (const c of ref.class_names as { yaml: string; names?: Record<string, string>; error?: string }[]) {
      const { data, issues } = readDataYaml(`train: a\nval: a\n${c.yaml}`)
      if (c.error) expect(issues.some((i) => i.severity === 'error'), c.yaml).toBe(true)
      else {
        expect(issues.filter((i) => i.severity === 'error'), c.yaml).toEqual([])
        expect(data.names, c.yaml).toEqual(Object.values(c.names!))
      }
    }
  })
  test("whole data.yaml files read as Ultralytics' YAML reader gives them", () => {
    for (const c of ref.data_yaml as { name: string; yaml: string; data?: unknown; error?: string }[]) {
      // Ultralytics refuses anchors and tab indents; js-yaml takes anchors
      if (c.error) continue
      const got = yaml.load(c.yaml.charCodeAt(0) === 0xfeff ? c.yaml.slice(1) : c.yaml, { schema: ULTRALYTICS_YAML }) ?? null
      expect(JSON.parse(JSON.stringify(got)), c.name).toEqual(c.data)
    }
  })
})

// Images and their shapes, as src/main/dataset.ts reads them
const image = (shapes: Shape[], masks: (MaskBits | null)[] = [], size: [number, number] = [100, 50], name = 'a.png'): DatasetImage => ({
  name, width: size[0], height: size[1], file: 'a.json', format: 'labelme', shapes, masks: shapes.map((_, i) => masks[i] ?? null), flags: {}, verified: false
})
const bitmap = (x: number, y: number, rows: string[]): MaskBits =>
  ({ x, y, width: rows[0].length, height: rows.length, data: Uint8Array.from(rows.join('').split('').map(Number)) })
const rect = (label: string, points: number[][], extra: Partial<Shape> = {}) => newShape(label, 'rectangle', points, extra)
const poly = (label: string, points: number[][]) => newShape(label, 'polygon', points)
const lines = (text: string) => text.split('\n').filter(Boolean)
const codes = (issues: { code: string }[]) => issues.map((i) => i.code)
const CLASSES = ['cat', 'dog', 'bird']

describe('label rows written for each task', () => {
  const shapes = [
    rect('cat', [[30, 25], [10, 5]]),
    poly('cat', [[10, 10], [30, 10], [30, 10], [20, 30], [10, 10]]),
    newShape('dog', 'oriented_rectangle', [[60, 10], [70, 20], [65, 25], [55, 15]]),
    newShape('dog', 'circle', [[50, 25], [55, 25]]),
    newShape('cat', 'mask', [[1, 1], [3, 3]])
  ]
  const masks = [null, null, null, null, bitmap(1, 1, ['111', '101', '111'])]

  test("detect: each closed shape is the box around it, as shares of the image to 6 places; a mask's box runs along its pixels' edges", () => {
    const out = labelRows(image(shapes, masks), 'detect', CLASSES)
    expect(lines(out.text)).toEqual([
      '0 0.200000 0.300000 0.200000 0.400000',
      '0 0.200000 0.400000 0.200000 0.400000',
      '1 0.625000 0.350000 0.150000 0.300000',
      '1 0.500000 0.500000 0.100000 0.200000',
      '0 0.025000 0.050000 0.030000 0.060000'
    ])
    expect(out.rows).toBe(5)
    expect(out.issues).toEqual([])
  })
  test("Python's '%.6f' rounds halves to even", () => {
    // Centre 26.5 / 64 = 0.4140625 and width 28.5 / 64 = 0.4453125: halves that stay down
    expect(labelRows(image([rect('cat', [[12.25, 8.5], [40.75, 30]])], [], [64, 48]), 'detect', CLASSES).text).toBe('0 0.414062 0.401042 0.445312 0.447917\n')
  })
  test('segment: a box is its 4 corners from the top left, so box and outline rows never share a file; repeated points are dropped', () => {
    const out = labelRows(image(shapes.slice(0, 3)), 'segment', CLASSES)
    expect(lines(out.text)).toEqual([
      '0 0.100000 0.100000 0.300000 0.100000 0.300000 0.500000 0.100000 0.500000',
      '0 0.100000 0.200000 0.300000 0.200000 0.200000 0.600000',
      '1 0.600000 0.200000 0.700000 0.400000 0.650000 0.500000 0.550000 0.300000'
    ])
    expect(verifyRows(out.text, 'segment', 3)).toMatchObject({ status: 'ok', outline: true })
  })
  test("segment: a circle is labelme2coco's polygon", () => {
    const out = labelRows(image([shapes[3]]), 'segment', CLASSES)
    const want = circlePolygon([50, 25], [55, 25])!
    const got = lines(out.text)[0].split(' ').slice(1).map(Number)
    expect(got.length).toBe(want.length * 2)
    want.forEach(([px, py], i) => {
      expect(Math.abs(got[2 * i] * 100 - px)).toBeLessThan(1e-4)
      expect(Math.abs(got[2 * i + 1] * 50 - py)).toBeLessThan(1e-4)
    })
  })
  test("segment: a mask is the outline of each part along its pixels' edges; holes are filled and parts become objects, each with an issue", () => {
    const ring = labelRows(image([shapes[4]], [masks[4]], [10, 10]), 'segment', CLASSES)
    // The 3 x 3 ring at (1, 1): its outer edge, corners cut, and the hole left out
    expect(ring.text).toBe('0 0.350000 0.400000 0.150000 0.400000 0.100000 0.350000 0.100000 0.150000 0.150000 0.100000 0.350000 0.100000 0.400000 0.150000 0.400000 0.350000\n')
    expect(ring.issues).toEqual([expect.objectContaining({ severity: 'warning', code: 'yolo-mask-holes', shape: 0 })])
    const parts = labelRows(image([newShape('cat', 'mask', [[0, 0], [3, 2]])], [bitmap(0, 0, ['1100', '1100', '0001'])], [10, 10]), 'segment', CLASSES)
    expect(lines(parts.text)).toEqual([
      '0 0.150000 0.200000 0.050000 0.200000 0.000000 0.150000 0.000000 0.050000 0.050000 0.000000 0.150000 0.000000 0.200000 0.050000 0.200000 0.150000',
      '0 0.350000 0.300000 0.300000 0.250000 0.350000 0.200000 0.400000 0.250000'
    ])
    expect(parts.issues).toEqual([expect.objectContaining({ severity: 'info', code: 'yolo-mask-parts' })])
  })
  test('obb: a box from its top left round, a rotated rectangle as stored, a rectangle drawn as a polygon as it is, other shapes as the smallest rotated box around them', () => {
    const square = poly('cat', [[20, 10], [30, 20], [20, 30], [10, 20]])
    const out = labelRows(image([shapes[0], shapes[2], square, shapes[1], shapes[3], shapes[4]], [null, null, null, null, null, masks[4]]), 'obb', CLASSES)
    expect(lines(out.text)).toEqual([
      '0 0.100000 0.100000 0.300000 0.100000 0.300000 0.500000 0.100000 0.500000',
      '1 0.600000 0.200000 0.700000 0.400000 0.650000 0.500000 0.550000 0.300000',
      '0 0.200000 0.200000 0.300000 0.400000 0.200000 0.600000 0.100000 0.400000',
      '0 0.040000 0.440000 0.140000 0.040000 0.300000 0.200000 0.200000 0.600000',
      '1 0.450000 0.400000 0.550000 0.400000 0.550000 0.600000 0.450000 0.600000',
      '0 0.010000 0.020000 0.040000 0.020000 0.040000 0.080000 0.010000 0.080000'
    ])
    expect(out.issues.map((i) => [i.code, i.shape])).toEqual([['yolo-obb-fitted', 3], ['yolo-obb-fitted', 4], ['yolo-obb-fitted', 5]])
    expect(verifyRows(out.text, 'obb', 3).status).toBe('ok')
  })
  test("shapes past the image's edge are cut there, with a warning beyond a pixel; ones outside it or with no area are left out", () => {
    const out = labelRows(image([
      rect('bird', [[-5, -5], [120, 10]]), rect('bird', [[-0.5, 0], [10, 10]]), rect('bird', [[110, 10], [120, 20]]), rect('bird', [[10, 10], [10, 20]])
    ]), 'detect', CLASSES)
    expect(lines(out.text)).toEqual(['2 0.500000 0.100000 1.000000 0.200000', '2 0.050000 0.100000 0.100000 0.200000'])
    expect(out.issues.map((i) => [i.severity, i.code, i.shape])).toEqual([
      ['warning', 'yolo-clamped', 0], ['info', 'yolo-clamped', 1], ['warning', 'yolo-outside', 2], ['warning', 'yolo-no-area', 3]
    ])
  })
  test("lines, line strips, points and sets of points are left out, and so are labels not in the class list, each with a warning", () => {
    const open = ['line', 'linestrip', 'point', 'points'] as const
    for (const task of ['detect', 'segment', 'obb'] as const) {
      const out = labelRows(image([...open.map((t) => newShape('cat', t, [[1, 1], [5, 5]])), rect('fish', [[1, 1], [5, 5]])]), task, CLASSES)
      expect(out.text).toBe('')
      expect(codes(out.issues)).toEqual(['yolo-open-shape', 'yolo-open-shape', 'yolo-open-shape', 'yolo-open-shape', 'yolo-unknown-label'])
      expect(out.issues[1].message).toBe(`a.png: cat is a line strip, which YOLO ${task === 'obb' ? 'OBB' : task} files can't hold, so it's left out`)
    }
  })
  test('outlines of fewer than 3 points apart are left out; a row that repeats another is written once', () => {
    const out = labelRows(image([poly('cat', [[10, 10], [20, 10], [10, 10.0000001]]), rect('cat', [[1, 1], [5, 5]]), rect('cat', [[1, 1], [5, 5]])]), 'segment', CLASSES)
    expect(lines(out.text).length).toBe(1)
    expect(out.issues.map((i) => [i.code, i.shape])).toEqual([['yolo-few-points', 0], ['yolo-repeated', 2]])
  })
  test('a labelled image with no shapes gets an empty file, which Ultralytics reads as a background', () => {
    for (const task of YOLO_TASKS) {
      const out = labelRows(image([]), task, CLASSES, { classes: { cat: ['a'] } })
      expect(out).toEqual({ text: '', rows: 0, issues: [] })
      expect(verifyRows(out.text, task, 3, [1, 3]).status).toBe('empty')
    }
  })
  test("an image whose labels or size can't be read gets no rows, and an error", () => {
    expect(labelRows({ ...image([rect('cat', [[1, 1], [5, 5]])]), broken: "It isn't valid JSON" }, 'detect', CLASSES).issues[0]).toMatchObject({ severity: 'error', code: 'yolo-broken' })
    expect(labelRows(image([rect('cat', [[1, 1], [5, 5]])], [], [0, 0]), 'detect', CLASSES).issues[0]).toMatchObject({ severity: 'error', code: 'yolo-no-size' })
    expect(labelRows(image([newShape('cat', 'mask', [[1, 1], [3, 3]])]), 'segment', CLASSES).issues[0]).toMatchObject({ severity: 'error', code: 'yolo-mask-unreadable' })
  })

  const SCHEMA: KeypointSchema = { classes: { person: ['nose', 'left_eye', 'right_eye'], hand: ['wrist', 'thumb'] } }
  test('pose: a box and every keypoint, x, y and visibility (2 seen, 1 hidden); missing ones, ones outside the image and padding as 0 0 0', () => {
    const out = labelRows(image([
      rect('person', [[10, 5], [30, 25]], { group_id: 1 }), newShape('nose', 'point', [[20, 10]], { group_id: 1 }),
      newShape('right_eye', 'point', [[25, 15]], { group_id: 1, flags: { difficult: true } }),
      rect('hand', [[60, 10], [80, 30]], { group_id: 2 }), newShape('wrist', 'point', [[70, 20]], { group_id: 2 }), newShape('thumb', 'point', [[110, 20]], { group_id: 2 }),
      poly('car', [[1, 1], [5, 1], [5, 5]])
    ]), 'pose', ['person', 'hand'], SCHEMA)
    expect(lines(out.text)).toEqual([
      '0 0.200000 0.300000 0.200000 0.400000 0.200000 0.200000 2 0 0 0 0.250000 0.300000 1',
      '1 0.700000 0.400000 0.200000 0.400000 0.700000 0.400000 2 0 0 0 0 0 0'
    ])
    expect(out.issues.map((i) => [i.severity, i.code, i.shape])).toEqual([['warning', 'yolo-not-pose', 6], ['warning', 'yolo-keypoint-outside', 5]])
    expect(verifyRows(out.text, 'pose', 2, [3, 3]).status).toBe('ok')
    expect(() => labelRows(image([]), 'pose', ['person'])).toThrow(/keypoint file/)
    // An object wholly outside the image is left out; one cut at the edge keeps its size within the image
    const edge = labelRows(image([rect('hand', [[-30, -20], [-10, -5]]), rect('person', [[90, 40], [130, 70]])]), 'pose', ['person', 'hand'], SCHEMA)
    expect(edge.text).toBe('0 0.950000 0.900000 0.100000 0.200000 0 0 0 0 0 0 0 0 0\n')
    expect(edge.issues.map((i) => [i.code, i.shape])).toEqual([['pose-no-keypoints', 0], ['pose-no-keypoints', 1], ['yolo-outside', 0], ['yolo-clamped', 1]])
  })

  test("every file Seamtag writes passes Ultralytics' checks for its task, with nothing dropped as a repeat", () => {
    // Shapes of every kind, some reaching past the edges, at a size whose shares don't come out even
    let seed = 7
    const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const size: [number, number] = [97, 61]
    const many: Shape[] = [], maskList: (MaskBits | null)[] = []
    for (let i = 0; i < 40; i++) {
      const [cx, cy] = [random() * 110 - 5, random() * 70 - 5]
      const label = CLASSES[i % 3]
      const k = i % 5
      if (k === 0) many.push(rect(label, [[cx, cy], [cx + random() * 30, cy + random() * 20]]))
      else if (k === 1) many.push(poly(label, Array.from({ length: 3 + (i % 6) }, () => [cx + random() * 20, cy + random() * 20])))
      else if (k === 2) many.push(newShape(label, 'circle', [[cx, cy], [cx + 1 + random() * 15, cy]]))
      else if (k === 3) {
        const [a, w, h] = [random() * Math.PI, 5 + random() * 20, 3 + random() * 10]
        const [ux, uy, vx, vy] = [Math.cos(a) * w, Math.sin(a) * w, -Math.sin(a) * h, Math.cos(a) * h]
        many.push(newShape(label, 'oriented_rectangle', [[cx, cy], [cx + ux, cy + uy], [cx + ux + vx, cy + uy + vy], [cx + vx, cy + vy]]))
      } else {
        const rows = Array.from({ length: 6 }, () => Array.from({ length: 7 }, () => (random() < 0.55 ? '1' : '0')).join(''))
        many.push(newShape(label, 'mask', [[Math.round(cx), Math.round(cy)], [Math.round(cx) + 6, Math.round(cy) + 5]]))
        maskList[many.length - 1] = bitmap(Math.round(cx), Math.round(cy), rows)
      }
    }
    many.push(rect('cat', [[0, 0], [97, 61]]), rect('dog', [[0, 0], [97, 61]]))
    for (const task of ['detect', 'segment', 'obb'] as const) {
      const out = labelRows(image(many, maskList, size), task, CLASSES)
      expect(out.rows).toBeGreaterThan(30)
      const verdict = verifyRows(out.text, task, 3)
      expect(verdict, task).toMatchObject({ status: 'ok', repeated: [], outline: task !== 'detect' })
      expect(verdict.boxes.length).toBe(out.rows)
    }
  })
})

describe('label rows read back into shapes', () => {
  const near = (a: number[][], b: number[][]) => {
    expect(a.length).toBe(b.length)
    a.forEach((p, i) => p.forEach((v, j) => expect(Math.abs(v - b[i][j]), `point ${i}`).toBeLessThan(1e-4)))
  }
  const read = (text: string, task: YoloTask, extra: object = {}) => readRows(text, { task, names: CLASSES, width: 100, height: 50, ...extra })

  test('round trips: boxes, outlines and rotated boxes come back where they were, to the precision of 6 places', () => {
    const shapes = [rect('cat', [[12.3, 7.1], [40.9, 33.3]]), poly('dog', [[10.5, 10.25], [33.3, 11], [20, 31.7], [12, 25]]), newShape('bird', 'oriented_rectangle', [[60, 10], [70, 20], [65, 25], [55, 15]])]
    const detect = read(labelRows(image(shapes.slice(0, 1)), 'detect', CLASSES).text, 'detect')
    expect(detect.issues).toEqual([])
    expect(detect.shapes.map((s) => [s.label, s.shape_type])).toEqual([['cat', 'rectangle']])
    near(detect.shapes[0].points, shapes[0].points)
    const segment = read(labelRows(image(shapes.slice(1, 2)), 'segment', CLASSES).text, 'segment')
    expect(segment.shapes.map((s) => [s.label, s.shape_type])).toEqual([['dog', 'polygon']])
    near(segment.shapes[0].points, shapes[1].points)
    const obb = read(labelRows(image(shapes.slice(2)), 'obb', CLASSES).text, 'obb')
    expect(obb.shapes.map((s) => [s.label, s.shape_type])).toEqual([['bird', 'oriented_rectangle']])
    near(obb.shapes[0].points, shapes[2].points)
    // Points as read, not cut or rounded to pixels: a share of 0.123457 of 100 is 12.3457
    expect(read('0 0.123457 0.5 0.1 0.1', 'detect').shapes[0].points).toEqual([[7.3457, 22.5], [17.3457, 27.5]])
  })
  test('pose round trip: the box and named keypoints in one new group; missing ones stay missing, hidden ones difficult', () => {
    const schema: KeypointSchema = { classes: { person: ['nose', 'left_eye', 'right_eye'] } }
    const text = labelRows(image([
      rect('person', [[10, 5], [30, 25]], { group_id: 4 }), newShape('nose', 'point', [[20.5, 10.25]], { group_id: 4 }),
      newShape('right_eye', 'point', [[25, 15]], { group_id: 4, flags: { occluded: true } })
    ]), 'pose', ['person'], schema).text
    const got = readRows(text, { task: 'pose', names: ['person'], width: 100, height: 50, kptShape: [3, 3], kptNames: { 0: schema.classes.person }, firstGroup: 9 })
    expect(got.issues).toEqual([])
    expect(got.shapes.map((s) => [s.label, s.shape_type, s.points, s.group_id, s.flags])).toEqual([
      ['person', 'rectangle', [[10, 5], [30, 25]], 9, {}], ['nose', 'point', [[20.5, 10.25]], 9, {}], ['right_eye', 'point', [[25, 15]], 9, { difficult: true }]
    ])
    // Unnamed keypoints, and rows without visibility, where 0 0 or below 0 is missing
    const plain = readRows('0 0.5 0.5 0.2 0.2 0.1 0.1 0 0 -0.1 0.2\n', { task: 'pose', names: ['person'], width: 100, height: 50, kptShape: [3, 2] })
    expect(plain.shapes.map((s) => [s.label, s.points])).toEqual([['person', [[40, 20], [60, 30]]], ['kp0', [[10, 5]]]])
    expect(readRows('0 0.5 0.5 0.2 0.2', { task: 'pose', names: ['person'], width: 10, height: 10 }).issues[0].code).toBe('yolo-no-kpt-shape')
  })
  test('a rotated box whose corners make no rectangle in pixels is read as an outline, with a note', () => {
    // A rectangle in shares of a square, but not of a 100 x 50 image
    const got = read('0 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5', 'obb')
    expect(got.shapes[0].shape_type).toBe('polygon')
    expect(codes(got.issues)).toEqual(['yolo-obb-outline'])
    expect(readRows('0 0.5 0.1 0.9 0.5 0.5 0.9 0.1 0.5', { task: 'obb', names: CLASSES, width: 100, height: 100 }).shapes[0].shape_type).toBe('oriented_rectangle')
  })
  test("rows of another task's shape are read as Ultralytics would train them, or as they are, with a note", () => {
    const outlineInDetect = read('0 0.1 0.1 0.3 0.1 0.3 0.5\n', 'detect')
    expect(outlineInDetect.shapes.map((s) => [s.shape_type, s.points])).toEqual([['rectangle', [[10, 5], [30, 25]]]])
    expect(codes(outlineInDetect.issues)).toEqual(['yolo-outline-as-box'])
    const boxInSegment = read('0 0.2 0.3 0.2 0.4\n', 'segment')
    expect(boxInSegment.shapes.map((s) => s.shape_type)).toEqual(['rectangle'])
    expect(codes(boxInSegment.issues)).toEqual(['yolo-box-row'])
    const keypointsInDetect = readRows('0 0.5 0.5 0.2 0.2 0.45 0.45 2\n', { task: 'detect', names: CLASSES, width: 100, height: 50, kptShape: [1, 3] })
    expect(keypointsInDetect.shapes.map((s) => s.shape_type)).toEqual(['rectangle'])
    expect(codes(keypointsInDetect.issues)).toEqual(['yolo-keypoints-dropped'])
  })
  test('rows Ultralytics would refuse are read where they can be, and the file and each row left out get an issue', () => {
    const got = read('0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2\n1 0.5 x 0.2 0.2\n2 1.5 0.5 0.2 0.2\n', 'detect')
    expect(got.verdict.status).toBe('corrupt')
    expect(got.shapes.map((s) => s.label)).toEqual(['cat', 'bird'])
    expect(got.issues.map((i) => [i.severity, i.code])).toEqual([['warning', 'yolo-corrupt'], ['warning', 'yolo-row'], ['warning', 'yolo-row']])
    expect(got.issues[1].message).toBe("The label file: line 2's class, 7, isn't one of the 3 classes (0 to 2), so it's left out")
    const repeated = read('0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n', 'detect')
    expect(repeated.shapes.length).toBe(1)
    expect(codes(repeated.issues)).toEqual(['yolo-repeated'])
  })
})

describe('data.yaml', () => {
  test('written with folders relative to it and no path key, names as a map, and read back the same, names that need quotes too', () => {
    const names = ['person', 'yes', '1', 'a: b', 'Caf\u00e9 1', "it's", 'null', '#hash', ' lead']
    const text = writeDataYaml({ names, train: 'images/train', val: 'images/val', test: 'images/test' })
    expect(text).not.toMatch(/^path:/m)
    expect(text).toContain('nc: 9\nnames:\n  0: person\n  1: "yes"\n  2: "1"\n  3: "a: b"\n')
    const { data, issues } = readDataYaml(text)
    expect(issues).toEqual([])
    expect(data).toEqual({ path: null, train: 'images/train', val: 'images/val', test: 'images/test', names, kptShape: null, kptNames: null, flipIdx: null })
    // Read by YAML 1.2 readers too
    expect((yaml.load(text) as { names: Record<number, string> }).names).toEqual(Object.fromEntries(names.map((n, i) => [i, n])))
  })
  test("pose keys come from the keypoint file as coco8-pose.yaml has them: kpt_shape [K, 3], flip_idx when the classes agree, kpt_names by class", () => {
    const schema: KeypointSchema = { classes: { person: ['nose', 'left_eye', 'right_eye'], hand: ['wrist', 'thumb'] }, flip: { person: [['left_eye', 'right_eye']], hand: [] } }
    const keys = poseYamlKeys(schema)
    expect(keys).toEqual({ names: ['person', 'hand'], kptShape: [3, 3], kptNames: [['nose', 'left_eye', 'right_eye'], ['wrist', 'thumb']], flipIdx: null, issues: [expect.objectContaining({ code: 'pose-flip' })] })
    const one = poseYamlKeys({ classes: { person: schema.classes.person }, flip: { person: schema.flip!.person } })
    const text = writeDataYaml({ ...one, train: 'images/train', val: 'images/val' })
    expect(text).toBe(`# Folders are relative to this file: with no path key, Ultralytics takes its folder as the root
train: images/train
val: images/val

kpt_shape: [3, 3]
flip_idx: [0, 2, 1]

nc: 1
names:
  0: person

kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
`)
    expect(readDataYaml(text).data).toMatchObject({ kptShape: [3, 3], flipIdx: [0, 2, 1], kptNames: { 0: ['nose', 'left_eye', 'right_eye'] }, names: ['person'] })
  })
  test("Roboflow's, CVAT's and coco8-pose's files read as Ultralytics takes them", () => {
    const files = Object.fromEntries((ref.data_yaml as { name: string; yaml: string }[]).map((c) => [c.name, c.yaml]))
    expect(readDataYaml(files.roboflow)).toEqual({
      data: { path: null, train: '../train/images', val: '../valid/images', test: '../test/images', names: ['head', 'helmet', 'person'], kptShape: null, kptNames: null, flipIdx: null },
      issues: []
    })
    expect(readDataYaml(files.cvat).data).toMatchObject({ path: '.', train: 'train.txt', val: 'val.txt', test: null, names: ['car', 'person'] })
    expect(readDataYaml(files['coco8-pose']).data).toMatchObject({ path: 'coco8-pose', test: null, names: ['person'], kptShape: [17, 3], flipIdx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], kptNames: { 0: ['nose', 'left_eye', 'right_eye'] } })
    expect(readDataYaml(files.lists).data).toMatchObject({ train: ['images/a', 'images/b'], val: ['images/c', 'images/d'] })
    expect(readDataYaml(files['nc only']).data.names).toEqual(['class_0', 'class_1'])
    expect(readDataYaml(files['nc as text']).issues).toEqual([])
    expect(readDataYaml(files['validation key'])).toMatchObject({ data: { val: 'b' }, issues: [{ severity: 'info' }] })
    expect(readDataYaml(files['byte order mark and CRLF'])).toMatchObject({ data: { train: 'a', val: 'b', names: ['x'] }, issues: [] })
    expect(readDataYaml(files['quoted names and escapes']).data.names).toEqual(['say "hi"', "it's", 'tab\there'])
  })
  test('what check_det_dataset refuses comes back as an error that says why', () => {
    const errors = (text: string) => readDataYaml(text, 'data.yaml').issues.filter((i) => i.severity === 'error').map((i) => i.message)
    expect(errors('train: a\nnames: [x]')).toEqual(['data.yaml has no val, which Ultralytics needs'])
    expect(errors('train: a\nval: a')).toEqual(['data.yaml has neither names nor nc, so the classes are unknown'])
    expect(errors('train: a\nval: a\nnc: 3\nnames: [x, y]')).toEqual(['data.yaml has 2 names but nc: 3, which must agree'])
    expect(errors('train: a\nval: a\nnc: 1.5\nnames: [x]')).toEqual(["data.yaml has nc: 1.5, which isn't a whole number"])
    expect(errors('train: a\nval: a\nnames: {0: x, 2: y}')).toEqual(['data.yaml numbers its 2 classes 0 to 2, where they must run from 0 to 1'])
    expect(errors('train: a\nval: a\nnames: x')).toEqual(['data.yaml has text for names, where Ultralytics needs a list or a map', 'data.yaml has neither names nor nc, so the classes are unknown'])
    expect(errors('train: a\nval: a\nnames: [x]\nkpt_shape: [17]')).toEqual(["data.yaml has a kpt_shape that isn't [keypoints, 2 or 3], such as [17, 3]"])
    expect(errors('train: a\nval: a\nnames: [x]\nkpt_shape: [3, 3]\nflip_idx: [1, 0]')).toEqual(['data.yaml has a flip_idx of 2 keypoints where kpt_shape has 3, which stops Ultralytics when it mirrors images'])
    expect(errors('train: [a\n')[0]).toMatch(/^data.yaml isn't valid YAML/)
    expect(errors('- a\n- b\n')).toEqual(["data.yaml isn't a map of keys and values, so it can't be read"])
  })
})

describe('which task a dataset is for', () => {
  const guess = (texts: string[], kptShape: [number, number] | null = null, size?: [number, number]) => {
    const stats = newRowStats()
    for (const t of texts) countRows(stats, t, size)
    return guessTask(stats, kptShape)
  }
  test('box rows are detect, outline rows segment, and both together segment with a note that Ultralytics skips files mixing them', () => {
    expect(guess(['0 0.5 0.5 0.2 0.2\n', '1 0.1 0.1 0.1 0.1\n'])).toMatchObject({ task: 'detect', ask: false })
    expect(guess(['0 0.1 0.1 0.3 0.1 0.3 0.4\n', '1 0.1 0.1 0.3 0.1 0.3 0.4 0.2 0.5 0.1 0.5\n'])).toMatchObject({ task: 'segment', ask: false })
    const mixed = guess(['0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.3 0.4\n', '1 0.1 0.1 0.3 0.1 0.3 0.4 0.2 0.5\n'])
    expect(mixed).toMatchObject({ task: 'segment', others: ['detect'], ask: true })
    expect(mixed.stats.mixed).toBe(1)
    expect(mixed.reason).toMatch(/1 files? mix both, which Ultralytics skips/)
  })
  test('rows of 4 corners are rotated boxes only when every one is a rectangle and some are turned; the window asks either way', () => {
    const turned = '0 0.2 0.1 0.5 0.3 0.4 0.45 0.1 0.25\n'
    const upright = '0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n'
    const other = '0 0.1 0.1 0.3 0.1 0.35 0.4 0.1 0.4\n'
    expect(guess([turned, upright])).toMatchObject({ task: 'obb', others: ['segment'], ask: true })
    expect(guess([upright])).toMatchObject({ task: 'segment', others: ['obb'], ask: true })
    expect(guess([turned, other])).toMatchObject({ task: 'segment', others: ['obb'] })
    // In pixels, a skewed rectangle of shares is no rectangle
    expect(quadKind([[0.2, 0.1], [0.5, 0.3], [0.4, 0.45], [0.1, 0.25]])).toBe('rotated')
    expect(quadKind([[0.2, 0.1], [0.5, 0.3], [0.4, 0.45], [0.1, 0.25]], [100, 50])).toBe('other')
  })
  test("kpt_shape in data.yaml means pose; without it, rows whose every third value is 0, 1 or 2 look like keypoints", () => {
    expect(guess(['0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n'], [2, 3])).toMatchObject({ task: 'pose', kptShape: [2, 3], ask: false })
    expect(guess(['0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n'], [17, 3]).reason).toMatch(/1 of the 1 rows don't have the 56 values/)
    expect(guess(['0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n', '0 0.3 0.3 0.1 0.1 0.3 0.3 1 0.32 0.31 2\n'])).toMatchObject({ task: 'pose', kptShape: [2, 3], others: ['segment'], ask: true })
    // One length of odd values: outlines, or keypoints without visibility
    expect(guess(['0 0.1 0.1 0.3 0.1 0.3 0.4\n', '1 0.2 0.2 0.4 0.2 0.4 0.5\n'])).toMatchObject({ task: 'segment', others: ['pose'], kptShape: [1, 2], ask: true })
  })
  test('label files with no rows are all backgrounds, and there is nothing to ask', () => {
    expect(guess(['', '\n'])).toMatchObject({ task: 'detect', ask: false, stats: { files: 2, empty: 2, rows: 0 } })
  })
})

describe("Ultralytics' file rules", () => {
  test('img2label_paths: the last images folder becomes labels, and the extension txt; without one, the label is beside the image', () => {
    expect(labelPathFor('/ds/images/train/a/b.jpg')).toBe('/ds/labels/train/a/b.txt')
    expect(labelPathFor('/x/images/y/images/z.PNG')).toBe('/x/images/y/labels/z.txt')
    expect(labelPathFor('/flat/x.y.jpeg')).toBe('/flat/x.y.txt')
    expect(labelPathFor('/my_images/a.jpg')).toBe('/my_images/a.txt')
    expect(labelPathFor('images/train/a.jpg')).toBe('labels/train/a.txt')
    expect(labelPathFor('a.png')).toBe('a.txt')
    expect(labelPathFor('C:\\ds\\images\\a.jpg', '\\')).toBe('C:\\ds\\labels\\a.txt')
  })
  test('IMG_FORMATS decide what an image is, by what follows the last dot, in any case', () => {
    expect(['a.JPG', 'b.jpeg', 'c.png', 'd.webp', 'e.tif', 'f.bmp', 'g.avif', 'h.dng'].every(isYoloImage)).toBe(true)
    expect(['a.gif', 'b.jfif', 'c.txt', 'd.jpg.txt', 'noext'].some(isYoloImage)).toBe(false)
  })
})

describe('datasets on disk, laid out as tools write them (tests/fixtures/ultralytics/make_layouts.py)', () => {
  const LAYOUTS = path.join(FIXTURES, 'layouts')
  const items = (d: Awaited<ReturnType<typeof readYoloDataset>>) =>
    d.splits.map((s) => [s.name, s.items.map((i) => [path.relative(d.root, i.image), i.label && path.relative(d.root, i.label)])])

  test("a Roboflow download: ../train/images is read as train/images, its outlines are segment's", async () => {
    for (const target of ['roboflow', 'roboflow/data.yaml']) {
      const d = await readYoloDataset(path.join(LAYOUTS, target))
      expect(d.yaml).toBe(path.join(LAYOUTS, 'roboflow', 'data.yaml'))
      expect(d.root).toBe(path.join(LAYOUTS, 'roboflow'))
      expect(d.names).toEqual(['head', 'helmet', 'person'])
      expect(items(d)).toEqual([
        ['train', [['train/images/a.jpg', 'train/labels/a.txt'], ['train/images/b.jpg', 'train/labels/b.txt']]],
        ['val', [['valid/images/c.jpg', 'valid/labels/c.txt']]],
        ['test', [['test/images/d.jpg', null]]]
      ])
      expect(d.guess).toMatchObject({ task: 'segment', ask: false })
      expect(d.issues).toEqual([])
    }
  })
  test("a CVAT export: its list files' data/ paths are found in the dataset, and path: . is its folder", async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'cvat'))
    expect(d.names).toEqual(['car', 'person'])
    expect(items(d)).toEqual([
      ['train', [['images/train/frame_000000.png', 'labels/train/frame_000000.txt'], ['images/train/frame_000001.png', 'labels/train/frame_000001.txt']]],
      ['val', [['images/val/frame_000002.png', null]]]
    ])
    expect(d.guess.task).toBe('detect')
    expect(d.issues).toEqual([])
  })
  test("Ultralytics' own layout: images in nested folders keep their labels, and kpt_shape makes it pose", async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'ultralytics'))
    expect(d).toMatchObject({ names: ['person'], kptShape: [3, 3], flipIdx: [0, 2, 1], kptNames: { 0: ['nose', 'left_eye', 'right_eye'] } })
    expect(items(d)).toEqual([
      ['train', [['images/train/a.png', 'labels/train/a.txt'], ['images/train/sub/b.png', 'labels/train/sub/b.txt']]],
      ['val', [['images/val/c.png', 'labels/val/c.txt']]]
    ])
    expect(d.guess).toMatchObject({ task: 'pose', kptShape: [3, 3] })
    expect(d.guess.stats).toMatchObject({ files: 3, empty: 1, rows: 2 })
  })
  test('a folder without data.yaml: one split of the images under images/, names from labels/classes.txt, rotated boxes found', async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'bare'))
    expect(d.yaml).toBeNull()
    expect(d.names).toEqual(['plane', 'ship'])
    expect(items(d)).toEqual([['', [['images/a.jpg', 'labels/a.txt'], ['images/b.jpg', 'labels/b.txt']]]])
    expect(d.guess).toMatchObject({ task: 'obb', ask: true })
    expect(d.issues.map((i) => i.code)).toEqual(['yolo-no-yaml'])
  })
  test("X-AnyLabeling's export: each label beside its image, names from classes.txt", async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'flat'))
    expect(d.names).toEqual(['cat', 'dog'])
    expect(items(d)).toEqual([['', [['x.jpg', 'x.txt'], ['y.jpg', 'y.txt']]]])
    expect(d.guess.task).toBe('detect')
  })
  test("lists of images that aren't there fall back to images/<split>; a path from another computer to the YAML's folder; several YAMLs and none at all are errors", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-yolo-'))
    try {
      const put = (p: string, text = '') => { mkdirSync(path.dirname(path.join(dir, p)), { recursive: true }); writeFileSync(path.join(dir, p), text) }
      put('ds/images/train/a.jpg')
      put('ds/labels/train/a.txt', '0 0.5 0.5 0.2 0.2\n')
      put('ds/train.txt', '/content/datasets/ds/images/train/a.jpg\n')
      put('ds/data.yaml', 'path: /content/datasets/ds\ntrain: train.txt\nval: images/train\nnames: [x]\n')
      const d = await readYoloDataset(path.join(dir, 'ds'))
      expect(d.root).toBe(path.join(dir, 'ds'))
      expect(items(d)).toEqual([['train', [['images/train/a.jpg', 'labels/train/a.txt']]], ['val', [['images/train/a.jpg', 'labels/train/a.txt']]]])
      expect(d.issues.map((i) => [i.severity, i.code])).toEqual([['info', 'yolo-root'], ['info', 'yolo-list-fallback'], ['info', 'yolo-two-splits']])
      expect(d.issues[2].message).toBe('1 image is in both train and val')
      put('ds/other.yaml', 'train: a\n')
      expect((await readYoloDataset(path.join(dir, 'ds'))).issues[0]).toMatchObject({ severity: 'error', code: 'yolo-many-yaml' })
      put('empty/notes.txt', 'hello')
      expect((await readYoloDataset(path.join(dir, 'empty'))).issues.map((i) => i.code)).toEqual(['yolo-no-images'])
      expect((await readYoloDataset(path.join(dir, 'nowhere'))).issues.map((i) => i.code)).toEqual(['yolo-missing'])
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

// With YOLO_PARITY_DIR set, a dataset of each task is written there for tests/parity/ultralytics_reads.py,
// which runs Ultralytics' own checks on it, with what each label file holds in <task>.expected.json
const PARITY = process.env.YOLO_PARITY_DIR
describe.skipIf(!PARITY)("datasets for Ultralytics' own checks", () => {
  const NAMES = ['person', 'yes', '1', 'a: b', 'Caf\u00e9 1', "it's"]
  const SCHEMA: KeypointSchema = {
    classes: { person: ['nose', 'left_eye', 'right_eye'], yes: ['x', 'left', 'right'] },
    flip: { person: [['left_eye', 'right_eye']], yes: [['left', 'right']] }
  }
  // Every kind of shape, some past the edges, a mask with a hole and two parts
  const sample = (size: [number, number], name: string, labels: string[]): DatasetImage => {
    const [w, h] = size
    const shapes = [
      rect(labels[0], [[w * 0.1, h * 0.1], [w * 0.4, h * 0.5]]),
      poly(labels[1 % labels.length], [[w * 0.5, h * 0.1], [w * 0.9, h * 0.15], [w * 0.95, h * 0.6], [w * 0.6, h * 0.5]]),
      newShape(labels[2 % labels.length], 'circle', [[w * 0.8, h * 0.8], [w * 0.95, h * 0.8]]),
      newShape(labels[3 % labels.length], 'oriented_rectangle', [[w * 0.2, h * 0.6], [w * 0.4, h * 0.7], [w * 0.35, h * 0.8], [w * 0.15, h * 0.7]]),
      newShape(labels[4 % labels.length], 'mask', [[2, 2], [8, 7]]),
      rect(labels[5 % labels.length], [[-3, h * 0.8], [w * 0.1, h + 4]]),
      newShape(labels[0], 'line', [[1, 1], [5, 5]])
    ]
    return image(shapes, [null, null, null, null, bitmap(2, 2, ['1111100', '1001100', '1111100', '0000000', '0000011', '0000011'])], size, name)
  }
  const poseSample = (size: [number, number], name: string): DatasetImage => {
    const [w, h] = size
    return image([
      rect('person', [[w * 0.1, h * 0.1], [w * 0.5, h * 0.9]], { group_id: 1 }),
      newShape('nose', 'point', [[w * 0.3, h * 0.2]], { group_id: 1 }), newShape('right_eye', 'point', [[w * 0.35, h * 0.15]], { group_id: 1, flags: { difficult: true } }),
      rect('yes', [[w * 0.6, h * 0.2], [w * 0.9, h * 0.6]], { group_id: 2 }), newShape('left', 'point', [[w * 0.7, h * 0.3]], { group_id: 2 }),
      newShape('right', 'point', [[w * 1.2, h * 0.3]], { group_id: 2 })
    ], [], size, name)
  }

  test.each(YOLO_TASKS)('%s', async (task) => {
    const root = path.join(PARITY!, task)
    rmSync(root, { recursive: true, force: true })
    const pose = task === 'pose' ? poseYamlKeys(SCHEMA) : null
    const names = pose ? pose.names : NAMES
    const images: [string, DatasetImage | null][] = [
      ['images/train/a.png', task === 'pose' ? poseSample([64, 48], 'a.png') : sample([64, 48], 'a.png', names)],
      ['images/train/sub/b.jpg', task === 'pose' ? poseSample([80, 61], 'b.jpg') : sample([80, 61], 'b.jpg', [...names].reverse())],
      ['images/val/c.png', image([], [], [40, 30], 'c.png')],
      // Never labelled: no label file, a background
      ['images/val/d.jpg', null]
    ]
    const expected: Record<string, { status: string; rows: number[][] }> = {}
    for (const [rel, img] of images) {
      const size = img ? [img.width, img.height] : [32, 24]
      const file = path.join(root, rel)
      mkdirSync(path.dirname(file), { recursive: true })
      const pixels = sharp({ create: { width: size[0], height: size[1], channels: 3, background: { r: 90, g: 120, b: 150 } } })
      await (rel.endsWith('.jpg') ? pixels.jpeg() : pixels.png()).toFile(file)
      if (!img) {
        expected[rel] = { status: 'missing', rows: [] }
        continue
      }
      const out = labelRows(img, task, names, SCHEMA)
      expect(out.issues.filter((i) => i.severity === 'error')).toEqual([])
      const label = path.join(root, labelPathFor(rel))
      mkdirSync(path.dirname(label), { recursive: true })
      writeFileSync(label, out.text)
      expected[rel] = { status: out.rows ? 'ok' : 'empty', rows: lines(out.text).map((l) => l.split(' ').map(Number)) }
      expect(verifyRows(out.text, task, names.length, pose?.kptShape ?? null)).toMatchObject({ status: out.rows ? 'ok' : 'empty', repeated: [] })
    }
    writeFileSync(path.join(root, 'data.yaml'), writeDataYaml({ names, train: 'images/train', val: 'images/val', ...(pose ?? {}) }))
    writeFileSync(`${root}.expected.json`, JSON.stringify({ task, names, kpt_shape: pose?.kptShape ?? null, flip_idx: pose?.flipIdx ?? null, images: expected }, null, 1))
  })
})
