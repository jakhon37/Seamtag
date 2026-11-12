import { chmodSync, cpSync, existsSync, promises as fsp, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import yaml from 'js-yaml'
import sharp from 'sharp'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { readYoloDataset } from '../../src/main/ultralytics'
import type { DatasetImage, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { circlePolygon } from '../../src/shared/outline'
import { ULTRALYTICS_YAML, yamlString } from '../../src/shared/pose'
import {
  countRows, guessTask, isDatasetYaml, isYoloImage, joinOutlines, labelPathFor, labelRows, labelText, maskOutlines, newRowStats, poseYamlKeys, quadKind,
  readDataYaml, readRows, verifyRows, writeDataYaml, YOLO_TASKS, type YoloTask
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
// Paths and messages as written on Linux and macOS, so the tests read the same on Windows
const slash = (text: string) => text.split(path.sep).join('/')
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
  test("segment: a mask is its outline along its pixels' edges, holes filled; a mask in separate parts is one object, its outlines joined as Ultralytics joins them", () => {
    const ring = labelRows(image([shapes[4]], [masks[4]], [10, 10]), 'segment', CLASSES)
    // The 3 x 3 ring at (1, 1): its outer edge, corners cut, and the hole left out
    expect(ring.text).toBe('0 0.350000 0.400000 0.150000 0.400000 0.100000 0.350000 0.100000 0.150000 0.150000 0.100000 0.350000 0.100000 0.400000 0.150000 0.400000 0.350000\n')
    expect(ring.issues).toEqual([expect.objectContaining({ severity: 'warning', code: 'yolo-mask-holes', shape: 0 })])
    const parts = labelRows(image([newShape('cat', 'mask', [[0, 0], [3, 2]])], [bitmap(0, 0, ['1100', '1100', '0001'])], [10, 10]), 'segment', CLASSES)
    // The square's outline from its corner nearest the dot, round it, then over to the dot and round that: Ultralytics'
    // merge_multi_segment gives these points for the two outlines (tests/parity/ultralytics_reads.py runs it)
    expect(lines(parts.text)).toEqual([
      '0 0.200000 0.150000 0.150000 0.200000 0.050000 0.200000 0.000000 0.150000 0.000000 0.050000 0.050000 0.000000 0.150000 0.000000 0.200000 0.050000 ' +
      '0.200000 0.150000 0.300000 0.250000 0.350000 0.200000 0.400000 0.250000 0.350000 0.300000 0.300000 0.250000'
    ])
    expect(parts.issues).toEqual([expect.objectContaining({ severity: 'info', code: 'yolo-joined', shape: 0 })])
    expect(parts.issues[0].message).toBe("a.png: cat's mask is in 2 separate parts, one object, so they're joined into one outline, as Ultralytics' convert_coco joins an object's parts")
    expect(verifyRows(parts.text, 'segment', 3)).toMatchObject({ status: 'ok', outline: true })
  })
  test('shapes sharing a label and a group_id are one object: one box in detect, one joined outline in segment, the smallest rotated box around them in OBB', () => {
    const group = [
      poly('cat', [[10, 10], [20, 10], [20, 20], [10, 20]]), rect('dog', [[50, 5], [60, 15]], { group_id: 1 }),
      poly('cat', [[30, 10], [40, 10], [40, 20]]), poly('cat', [[70, 30], [80, 30], [80, 40]])
    ]
    group[0].group_id = 1
    group[2].group_id = 1
    // A shape of another label in the group, and a shape with no group, are objects of their own
    const detect = labelRows(image(group), 'detect', CLASSES)
    expect(lines(detect.text)).toEqual(['0 0.250000 0.300000 0.300000 0.200000', '1 0.550000 0.200000 0.100000 0.200000', '0 0.750000 0.700000 0.100000 0.200000'])
    expect(detect.issues.map((i) => [i.severity, i.code, i.shape])).toEqual([['info', 'yolo-joined', 0]])
    expect(detect.issues[0].message).toBe("a.png: cat is 2 shapes of group 1, one object, so it's one box around them all")
    const segment = labelRows(image(group), 'segment', CLASSES)
    expect(lines(segment.text)).toEqual([
      '0 0.200000 0.200000 0.200000 0.400000 0.100000 0.400000 0.100000 0.200000 0.200000 0.200000 0.300000 0.200000 0.400000 0.200000 0.400000 0.400000 0.300000 0.200000',
      '1 0.500000 0.100000 0.600000 0.100000 0.600000 0.300000 0.500000 0.300000',
      '0 0.700000 0.600000 0.800000 0.600000 0.800000 0.800000'
    ])
    expect(segment.issues.map((i) => [i.code, i.shape, i.message])).toEqual([
      ['yolo-joined', 0, "a.png: cat is 2 shapes of group 1, one object, so its 2 separate parts are joined into one outline, as Ultralytics' convert_coco joins an object's parts"]
    ])
    expect(verifyRows(segment.text, 'segment', 3).status).toBe('ok')
    const obb = labelRows(image(group.slice(0, 3)), 'obb', CLASSES)
    expect(lines(obb.text)).toEqual(['0 0.100000 0.200000 0.400000 0.200000 0.400000 0.400000 0.100000 0.400000', '1 0.500000 0.100000 0.600000 0.100000 0.600000 0.300000 0.500000 0.300000'])
    expect(obb.issues.map((i) => [i.code, i.shape, i.message])).toEqual([['yolo-obb-fitted', 0, 'a.png: cat is 2 shapes of group 1, one object, so the smallest rotated box around them all is written']])
  })
  test("a part of an object that can't be written is left out and the rest joined; a mask's parts and other shapes in its group are one outline", () => {
    const shapes = [
      poly('cat', [[10, 10], [20, 10], [20, 20], [10, 20]]), poly('cat', [[200, 10], [210, 10], [210, 20]]), poly('cat', [[30, 10], [40, 10], [30, 10.0000001]]),
      newShape('cat', 'mask', [[50, 30], [53, 32]])
    ].map((s) => ({ ...s, group_id: 7 }))
    const out = labelRows(image(shapes, [null, null, null, bitmap(50, 30, ['1100', '1100', '0001'])]), 'segment', CLASSES)
    expect(out.issues.map((i) => [i.severity, i.code, i.shape])).toEqual([
      ['warning', 'yolo-outside', 1], ['warning', 'yolo-few-points', 2], ['info', 'yolo-joined', 0]
    ])
    expect(out.issues[1].message).toBe('a.png: cat has a part with fewer than 3 points apart, so that part is left out')
    expect(out.issues[2].message).toBe("a.png: cat is 3 shapes of group 7, one object, so its 3 separate parts are joined into one outline, as Ultralytics' convert_coco joins an object's parts")
    expect(out.rows).toBe(1)
    // The square, then the mask's two parts, each tied to the one before at their closest points
    const want = joinOutlines([[[10, 10], [20, 10], [20, 20], [10, 20]], [[51.5, 32], [50.5, 32], [50, 31.5], [50, 30.5], [50.5, 30], [51.5, 30], [52, 30.5], [52, 31.5]], [[53.5, 33], [53, 32.5], [53.5, 32], [54, 32.5]]])
    expect(lines(out.text)[0]).toBe(`0 ${want.map(([x, y]) => `${(x / 100).toFixed(6)} ${(y / 50).toFixed(6)}`).join(' ')}`)
  })
  test("joinOutlines ties each outline to the next at their closest points, the first such pair as numpy's argmin finds it, as Ultralytics' merge_multi_segment does", () => {
    // What Ultralytics 8.4.211's own merge_multi_segment gives for these outlines
    const square = (x: number, y: number) => [[x, y], [x + 4, y], [x + 4, y + 4], [x, y + 4]]
    expect(joinOutlines([square(0, 0), square(10, 0), square(20, 0)])).toEqual([
      [4, 0], [4, 4], [0, 4], [0, 0], [4, 0], [10, 0], [14, 0], [20, 0], [24, 0], [24, 4], [20, 4], [20, 0], [14, 0], [14, 4], [10, 4], [10, 0]
    ])
    // A middle outline whose tie with the next comes first in its points is walked the other way round
    expect(joinOutlines([square(0, 0), [[14, 4], [10, 4], [10, 0], [14, 0]], square(20, 0)])).toEqual([
      [4, 0], [4, 4], [0, 4], [0, 0], [4, 0], [10, 0], [10, 4], [14, 4], [20, 4], [20, 0], [24, 0], [24, 4], [20, 4], [14, 4], [14, 0], [10, 0]
    ])
    expect(joinOutlines([square(0, 0), square(10, 10), square(0, 20), [[30, 30], [34, 30], [34, 34]]])).toEqual([
      [4, 4], [0, 4], [0, 0], [4, 0], [4, 4], [10, 10], [14, 10], [14, 14], [10, 14], [4, 20], [4, 24], [30, 30], [34, 30], [34, 34], [30, 30],
      [4, 24], [0, 24], [0, 20], [4, 20], [10, 14], [10, 10]
    ])
    expect(joinOutlines([square(0, 0)])).toEqual(square(0, 0))
    expect(joinOutlines([])).toEqual([])
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
    expect(() => labelRows(image([]), 'pose', ['person'])).toThrow(/keypoints file/)
    // An object wholly outside the image is left out; one cut at the edge keeps its size within the image
    const edge = labelRows(image([rect('hand', [[-30, -20], [-10, -5]]), rect('person', [[90, 40], [130, 70]])]), 'pose', ['person', 'hand'], SCHEMA)
    expect(edge.text).toBe('0 0.950000 0.900000 0.100000 0.200000 0 0 0 0 0 0 0 0 0\n')
    expect(edge.issues.map((i) => [i.code, i.shape])).toEqual([['pose-no-keypoints', 0], ['pose-no-keypoints', 1], ['yolo-outside', 0], ['yolo-clamped', 1]])
  })

  test('pose: shapes of a class sharing a group are one object in parts, as labelme draws a person hidden in part, its box around them all', () => {
    const out = labelRows(image([
      newShape('person', 'polygon', [[10, 5], [30, 5], [20, 20]], { group_id: 1 }), newShape('person', 'polygon', [[40, 15], [60, 15], [50, 25]], { group_id: 1 }),
      newShape('nose', 'point', [[20, 10]], { group_id: 1 })
    ]), 'pose', ['person', 'hand'], SCHEMA)
    expect(lines(out.text)).toEqual(['0 0.350000 0.300000 0.500000 0.400000 0.200000 0.200000 2 0 0 0 0 0 0'])
    expect(out.issues.map((i) => [i.severity, i.code, i.shape, i.message])).toEqual([
      ['info', 'pose-parts', 0, 'a.png: person is 2 shapes of group 1, one object, so its box is around them all']
    ])
  })

  test("pose: an object whose box has no width or height is left out with its keypoints, as Ultralytics drops it in training, and so is one only touching the image", () => {
    const out = labelRows(image([
      rect('person', [[20, 10], [20, 30]], { group_id: 1 }), newShape('nose', 'point', [[20, 20]], { group_id: 1 }),
      newShape('wrist', 'point', [[60, 20]], { group_id: 2 }),
      rect('hand', [[-10, 0], [0, 10]], { group_id: 3 }), newShape('wrist', 'point', [[0, 5]], { group_id: 3 }),
      rect('person', [[10, 5], [30, 25]], { group_id: 4 }), newShape('nose', 'point', [[20, 10]], { group_id: 4 })
    ]), 'pose', ['person', 'hand'], SCHEMA)
    expect(lines(out.text)).toEqual(['0 0.200000 0.300000 0.200000 0.400000 0.200000 0.200000 2 0 0 0 0 0 0'])
    expect(out.issues.map((i) => [i.severity, i.code, i.shape])).toEqual([
      ['warning', 'pose-no-box', 2], ['warning', 'yolo-no-area', 0], ['warning', 'yolo-no-area', 2], ['warning', 'yolo-outside', 3]
    ])
    expect(out.issues[0].message).toBe("a.png: group 2 has keypoints of hand but no box, and the box around its one point has no area: add a box with the class's label")
    expect(out.issues[1].message).toBe("a.png: person's box has no width or height, so it's left out with its keypoints, as Ultralytics would drop it in training")
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
  test('a file that starts with a byte order mark, which Ultralytics refuses, is read without it, and the warning says so', () => {
    const got = read('\ufeff0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n', 'detect')
    expect(got.verdict).toMatchObject({ status: 'corrupt', line: 1 })
    expect(got.shapes.map((s) => [s.label, s.points])).toEqual([['cat', [[40, 20], [60, 30]]], ['dog', [[25, 12.5], [35, 17.5]]]])
    expect(got.issues.map((i) => i.message)).toEqual([
      "The label file: it starts with a byte order mark, which Ultralytics reads as part of the first number, so it would leave this image out of training; it's read here without the mark"
    ])
    const more = read('\ufeff0 0.5 0.5 0.2 0.2\n7 0.5 0.5 0.2 0.2\n', 'detect')
    expect(more.issues.map((i) => i.code)).toEqual(['yolo-corrupt', 'yolo-row'])
    expect(more.issues[0].message).toMatch(/training, as it would without the mark, since line 2's class, 7, is past the 3 classes \(0 to 2\); it's read here without the mark$/)
  })
  test("values holding characters that don't show are quoted with them written out", () => {
    const got = read('0 0.5\u200b 0.5 0.2 0.2\n', 'detect')
    expect(got.issues.map((i) => i.message)).toEqual([
      "The label file: Ultralytics would leave this image out of training, as line 1 has 0.5\\u200b, which isn't a number",
      "The label file: line 1 has 0.5\\u200b, which isn't a number, so it's left out"
    ])
    expect(verifyRows('0\ufeff 0.5 0.5 0.2 0.2\n', 'detect', 3).reason).toBe("line 1 has 0\\ufeff, which isn't a number")
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
  test('a row whose numbers, in pixels, come to Infinity, which no label file can hold, is left out', () => {
    const rows = (task: YoloTask, text: string, extra: object = {}) => {
      const got = read(text, task, extra)
      return [got.shapes.map((s) => s.label), got.issues.filter((i) => i.code === 'yolo-row').map((i) => i.message)]
    }
    const tooLarge = (line: number) => `The label file: line ${line} has a value too large to place on the image, so it's left out`
    expect(rows('detect', '0 1e308 0.5 0.2 0.2\n1 0.5 0.5 1e308 0.2\n2 0.5 0.5 0.2 0.2\n')).toEqual([['bird'], [tooLarge(1), tooLarge(2)]])
    expect(rows('segment', '0 0.1 0.1 1e308 0.1 0.5 0.5\n1 0.1 0.1 0.9 0.1 0.5 0.5\n')).toEqual([['dog'], [tooLarge(1)]])
    expect(rows('detect', '0 0.1 0.1 0.9 0.1 0.5 1e308\n')).toEqual([[], [tooLarge(1)]])
    // A keypoint not labelled is left out whatever it holds
    const pose = { kptShape: [2, 3], kptNames: { 0: ['nose', 'tail'] } }
    expect(rows('pose', '0 0.5 0.5 0.2 0.2 1e308 0.5 2 0.4 0.4 2\n0 0.5 0.5 0.2 0.2 1e308 0.5 0 0.4 0.4 2\n', pose)).toEqual([['cat', 'tail'], [tooLarge(1)]])
    // Large, but a number in pixels too, is read as it is
    const far = read('0 1e290 0.5 0.2 0.2\n', 'detect').shapes[0].points[1][0]
    expect(far / 1e292).toBeCloseTo(1, 9)
  })
  test('a value from the file is quoted cut short, however long', () => {
    const got = read(`0 ${'9'.repeat(100000)}x 0.5 0.2 0.2\n`, 'detect')
    expect(got.issues.map((i) => i.message)).toEqual([
      `The label file: Ultralytics would leave this image out of training, as line 1 has ${'9'.repeat(29)}…${'9'.repeat(29)}x, which isn't a number`,
      `The label file: line 1 has ${'9'.repeat(29)}…${'9'.repeat(29)}x, which isn't a number, so it's left out`
    ])
  })
  test('an outline of more points than a call takes arguments is read as its box for detect', () => {
    const points = Array.from({ length: 200001 }, (_, i) => `${(i % 100) / 100} ${(i % 7) / 10}`).join(' ')
    expect(read(`0 ${points}\n`, 'detect').shapes.map((s) => s.points)).toEqual([[[0, 0], [99, 30]]])
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
  test("a YAML is a dataset's when it holds train, val or validation as a folder, a file or a list", () => {
    expect(['train: images/train\nval: images/val\n', 'val: [a, b]\n', 'validation: v.txt\n', '\ufefftrain: a\n'].map(isDatasetYaml)).toEqual([true, true, true, true])
    // X-AnyLabeling's keypoint file, a model's settings, a training run's args.yaml, and what isn't a map of keys
    const others = ['classes:\n  person: [nose]\nhas_visible: true\n', 'nc: 80\nbackbone: []\n', 'task: detect\nval: true\n', 'train: [a\n', '- train\n', '']
    expect(others.map(isDatasetYaml)).toEqual(others.map(() => false))
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
  test('nc without names is class_0 on, for at least one class and at most 100,000, which a long list of names is not held to', () => {
    const read = (nc: string, names = '') => {
      const t = performance.now()
      const got = readDataYaml(`train: a\nval: a\nnc: ${nc}\n${names}`, 'data.yaml')
      // Each class's name was made, so nc: 4294967295 took 45 s and nc: 100000000 ran out of memory
      expect(performance.now() - t).toBeLessThan(500)
      return { names: got.data.names, errors: got.issues.filter((i) => i.severity === 'error').map((i) => i.message) }
    }
    expect(read('3')).toEqual({ names: ['class_0', 'class_1', 'class_2'], errors: [] })
    expect(read('100000').names).toHaveLength(100000)
    expect(read('0')).toEqual({ names: [], errors: ['data.yaml has nc: 0, and Ultralytics needs at least one class'] })
    expect(read('-1')).toEqual({ names: [], errors: ['data.yaml has nc: -1, and Ultralytics needs at least one class'] })
    for (const [nc, shown] of [['100001', '100001'], ['4294967296', '4294967296'], ['1e10', '10000000000']]) {
      expect(read(nc)).toEqual({ names: [], errors: [`data.yaml has nc: ${shown} but no names, and Seamtag names at most 100,000 classes itself, so the classes are unknown`] })
    }
    // Names written out are as many as the file holds
    const many = Array.from({ length: 100001 }, (_, i) => `n${i}`)
    expect(read('100001', `names: [${many.join(', ')}]\n`)).toEqual({ names: many, errors: [] })
  })
  test("names that are what every object has, such as constructor, are names; a long value is read in a time its length gives, not its square", () => {
    const { data, issues } = readDataYaml('train: a\nval: a\nnames: [car, constructor, toString, __proto__, valueOf]\nkpt_shape: [2, 3]\nkpt_names: {0: [constructor, hasOwnProperty]}\n')
    expect([data.names, data.kptNames, issues]).toEqual([['car', 'constructor', 'toString', '__proto__', 'valueOf'], { 0: ['constructor', 'hasOwnProperty'] }, []])
    const map = readDataYaml('train: a\nval: a\nnames: {0: car, 1: constructor, 2: isPrototypeOf}\n')
    expect(map.data.names).toEqual(['car', 'constructor', 'isPrototypeOf'])
    // Stripping both ends at once tried every run of spaces inside a value: 20,000 took 2 s
    for (const name of [`1${' '.repeat(100000)}1`, `${'1'.repeat(100000)}x1`]) {
      const t = performance.now()
      const yamlText = `train: a\nval: a\nnames: {0: ${name}}\n`
      expect(readDataYaml(yamlText).data.names).toEqual([name])
      expect(isDatasetYaml(yamlText)).toBe(true)
      expect(performance.now() - t).toBeLessThan(500)
    }
  })
  test('a key or value from the file is quoted cut short, however long', () => {
    const long = 'k'.repeat(100000)
    expect(readDataYaml(`train: a\nval: a\nnames: {${long}: x}\nnc: ${long}\n`).issues.map((i) => i.message)).toEqual([
      `data.yaml has nc: ${'k'.repeat(29)}…${'k'.repeat(30)}, which isn't a whole number`,
      `data.yaml has ${'k'.repeat(29)}…${'k'.repeat(30)} as a class number in names`
    ])
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
    expect(mixed.reason).toMatch(/1 file mixes both, which Ultralytics skips/)
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
    expect(guess(['0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n'], [17, 3]).reason).toMatch(/1 of the 1 rows doesn't have the 56 values/)
    expect(guess(['0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n', '0 0.3 0.3 0.1 0.1 0.3 0.3 1 0.32 0.31 2\n'])).toMatchObject({ task: 'pose', kptShape: [2, 3], others: ['segment'], ask: true })
    // One length of odd values: outlines, or keypoints without visibility
    expect(guess(['0 0.1 0.1 0.3 0.1 0.3 0.4\n', '1 0.2 0.2 0.4 0.2 0.4 0.5\n'])).toMatchObject({ task: 'segment', others: ['pose'], kptShape: [1, 2], ask: true })
  })
  test('label files with no rows, or no label files at all, make every image a background, and there is nothing to ask', () => {
    expect(guess(['', '\n'])).toMatchObject({ task: 'detect', ask: false, reason: 'The label files have no rows, so every image is a background', stats: { files: 2, empty: 2, rows: 0 } })
    expect(guess([])).toMatchObject({ task: 'detect', ask: false, reason: 'No label files were found, so every image is a background' })
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
    d.splits.map((s) => [s.name, s.items.map((i) => [slash(path.relative(d.root, i.image)), i.label && slash(path.relative(d.root, i.label))])])

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
  test("a CVAT YOLO 1.1 export: no data.yaml, the class names in obj.names, which obj.data names as data/obj.names", async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'cvat11'))
    expect(d.yaml).toBeNull()
    expect(d.names).toEqual(['car', 'tree'])
    expect(items(d)).toEqual([['', [['obj_train_data/frame_000000.png', 'obj_train_data/frame_000000.txt'], ['obj_train_data/frame_000001.png', 'obj_train_data/frame_000001.txt']]]])
    expect(d.issues.map((i) => [i.code, i.message])).toEqual([['yolo-no-yaml', 'cvat11 has no data.yaml, so the class names come from obj.names']])
    expect(d.guess.task).toBe('detect')
  })
  test("a Roboflow YOLO Darknet download: no data.yaml, the class names in each split's _darknet.labels", async () => {
    const d = await readYoloDataset(path.join(LAYOUTS, 'darknet'))
    expect(d.names).toEqual(['head', 'helmet'])
    expect(items(d)).toEqual([['', [['train/e.jpg', 'train/e.txt'], ['valid/f.jpg', 'valid/f.txt']]]])
    expect(d.issues.map((i) => [i.code, slash(i.message)])).toEqual([['yolo-no-yaml', 'darknet has no data.yaml, so the class names come from train/_darknet.labels']])
  })
  test("lists of images that aren't there fall back to images/<split>; a path from another computer to the YAML's folder; two datasets' YAMLs and no images are errors", async () => {
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
      // Of several datasets' YAML files data.yaml is taken, and two others are asked about
      put('ds/other.yaml', 'train: a\nval: a\n')
      expect((await readYoloDataset(path.join(dir, 'ds'))).yaml).toBe(path.join(dir, 'ds', 'data.yaml'))
      put('two/a.yaml', 'train: a\nval: a\n')
      put('two/b.yaml', 'train: b\nval: b\n')
      expect((await readYoloDataset(path.join(dir, 'two'))).issues[0]).toMatchObject({ severity: 'error', code: 'yolo-many-yaml' })
      put('empty/notes.txt', 'hello')
      expect((await readYoloDataset(path.join(dir, 'empty'))).issues.map((i) => i.code)).toEqual(['yolo-no-images'])
      expect((await readYoloDataset(path.join(dir, 'nowhere'))).issues.map((i) => i.code)).toEqual(['yolo-missing'])
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

describe('datasets on disk, wherever they sit', () => {
  const LAYOUTS = path.join(FIXTURES, 'layouts')
  const items = (d: Awaited<ReturnType<typeof readYoloDataset>>) =>
    d.splits.map((s) => [s.name, s.items.map((i) => [slash(path.relative(d.root, i.image)), i.label && slash(path.relative(d.root, i.label))])])
  const codes = (d: Awaited<ReturnType<typeof readYoloDataset>>) => d.issues.map((i) => [i.severity, i.code])
  let dir = ''
  const put = (p: string, text = '') => {
    mkdirSync(path.dirname(path.join(dir, p)), { recursive: true })
    writeFileSync(path.join(dir, p), text)
  }
  beforeEach(() => {
    dir = mkdtempSync(path.join(tmpdir(), 'seamtag-yolo-'))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    rmSync(dir, { recursive: true, force: true })
  })

  test("a folder without data.yaml is read in place, so a folder above it named images doesn't hide the labels beside its images", async () => {
    const pets = path.join(dir, 'photos', 'images', 'pets')
    cpSync(path.join(LAYOUTS, 'flat'), pets, { recursive: true })
    const d = await readYoloDataset(pets)
    expect(d.names).toEqual(['cat', 'dog'])
    expect(items(d)).toEqual([['', [['x.jpg', 'x.txt'], ['y.jpg', 'y.txt']]]])
    expect(d.guess).toMatchObject({ task: 'detect', ask: false, stats: { files: 2, rows: 2 } })
    expect(codes(d)).toEqual([['info', 'yolo-no-yaml']])
  })
  test("labels beside their images in an images folder are read from there, and a dataset's images folder picked alone finds its labels in ../labels, each with a note", async () => {
    put('tree/classes.txt', 'x\n')
    put('tree/images/a.jpg')
    put('tree/images/a.txt', '0 0.5 0.5 0.2 0.2\n')
    const tree = await readYoloDataset(path.join(dir, 'tree'))
    expect(items(tree)).toEqual([['', [['images/a.jpg', 'images/a.txt']]]])
    expect(codes(tree)).toEqual([['info', 'yolo-no-yaml'], ['info', 'yolo-label-beside']])
    expect(slash(tree.issues[1].message)).toBe("images/a.txt is beside its image in an images folder, so it's read from there, though Ultralytics would look in labels/")
    put('ds/images/b.jpg')
    put('ds/images/c.jpg')
    put('ds/labels/b.txt', '0 0.5 0.5 0.2 0.2\n')
    const alone = await readYoloDataset(path.join(dir, 'ds', 'images'))
    expect(items(alone)).toEqual([['', [['b.jpg', '../labels/b.txt'], ['c.jpg', null]]]])
    expect(codes(alone)).toEqual([['info', 'yolo-label-outside'], ['info', 'yolo-no-yaml']])
    expect(slash(alone.issues[0].message)).toBe("../labels/b.txt is outside images, where Ultralytics' rule puts it (the last images folder in an image's path becomes labels), so it's read from there")
  })
  test("a data.yaml's images have their labels only where Ultralytics looks; ones beside them it wouldn't reach are a warning, and no label files at all another", async () => {
    put('images/ds/data.yaml', 'train: train\nval: train\nnames: [x]\n')
    put('images/ds/train/a.jpg')
    put('images/ds/train/a.txt', '0 0.5 0.5 0.2 0.2\n')
    put('images/ds/train/b.jpg')
    const d = await readYoloDataset(path.join(dir, 'images', 'ds'))
    expect(items(d)).toEqual([['train', [['train/a.jpg', null], ['train/b.jpg', null]]], ['val', [['train/a.jpg', null], ['train/b.jpg', null]]]])
    expect(codes(d)).toEqual([['warning', 'yolo-label-unreached'], ['info', 'yolo-two-splits']])
    expect(d.issues[0].image).toBe(path.join(dir, 'images', 'ds', 'train', 'a.jpg'))
    expect(slash(d.issues[0].message)).toBe("train/a.txt is beside its image, but Ultralytics looks for its labels in ../../labels/ds/train (the last images folder in the path becomes labels), so it would train that image as a background, and it's read as such here: move the label files there")
    expect(d.guess.reason).toBe('No label files were found, so every image is a background')
    rmSync(path.join(dir, 'images', 'ds', 'train', 'a.txt'))
    const none = await readYoloDataset(path.join(dir, 'images', 'ds'))
    expect(codes(none)).toEqual([['info', 'yolo-two-splits'], ['warning', 'yolo-no-labels']])
    expect(none.issues[1].message).toBe("None of the 2 images has a label file where Ultralytics looks (labels for the last images folder in its path, else beside it), so they'd train as backgrounds")
  })
  test("only a dataset's YAML is taken for its data.yaml: a keypoint file or a training run's args.yaml is passed over, and the folder read as images and labels", async () => {
    const xany = path.join(dir, 'xany')
    cpSync(path.join(LAYOUTS, 'flat'), xany, { recursive: true })
    put('xany/pose_config.yaml', 'classes:\n  person: [nose, left_eye]\nhas_visible: true\n')
    put('xany/runs/detect/train/args.yaml', 'task: detect\nmode: train\ndata: data.yaml\nval: true\nsplit: val\n')
    // A data.yaml below the top is taken only for what it holds
    put('xany/config/data.yaml', 'model: yolo11n.pt\nepochs: 100\n')
    const d = await readYoloDataset(xany)
    expect(d.yaml).toBeNull()
    expect(d.names).toEqual(['cat', 'dog'])
    expect(items(d)).toEqual([['', [['x.jpg', 'x.txt'], ['y.jpg', 'y.txt']]]])
    expect(codes(d)).toEqual([['info', 'yolo-yaml-passed'], ['info', 'yolo-no-yaml']])
    expect(d.issues[0].message).toBe("pose_config.yaml and 2 more YAML files aren't datasets' (there's no train or val in them), so xany is read as images and labels")
  })
  test("a dataset's YAML below the top is found when the top has none; the one named after the folder comes first, and two others are asked about", async () => {
    put('proj/notes.yaml', 'title: x\n')
    put('proj/sets/one/data.yaml', 'train: images\nval: images\nnames: [a]\n')
    put('proj/sets/one/images/a.jpg')
    expect((await readYoloDataset(path.join(dir, 'proj'))).yaml).toBe(path.join(dir, 'proj', 'sets', 'one', 'data.yaml'))
    put('proj/sets/two/data.yaml', 'train: images\nval: images\nnames: [a]\n')
    expect((await readYoloDataset(path.join(dir, 'proj'))).issues.map((i) => i.code)).toEqual(['yolo-many-yaml'])
    // As find_dataset_yaml prefers it
    put('proj/proj.yaml', 'train: sets/one/images\nval: sets/one/images\nnames: [a]\n')
    put('proj/data.yaml', 'train: sets/one/images\nval: sets/one/images\nnames: [a]\n')
    expect((await readYoloDataset(path.join(dir, 'proj'))).yaml).toBe(path.join(dir, 'proj', 'proj.yaml'))
  })
  test('label files are found as the file system finds them: through links, never folders, other cases only where it ignores case, and in folders that can\'t be listed', async () => {
    put('fs/data.yaml', 'train: images\nval: images\nnames: [x]\n')
    for (const name of ['a', 'b', 'c', 'd', 'e']) put(`fs/images/${name}.jpg`)
    put('fs/labels/a.txt', '0 0.5 0.5 0.2 0.2\n')
    put('fs/real/b.txt', '0 0.5 0.5 0.2 0.2\n')
    try {
      symlinkSync(path.join(dir, 'fs', 'real', 'b.txt'), path.join(dir, 'fs', 'labels', 'b.txt'))
    } catch {
      // Windows makes links only for administrators and in developer mode
      put('fs/labels/b.txt', '0 0.5 0.5 0.2 0.2\n')
    }
    mkdirSync(path.join(dir, 'fs', 'labels', 'c.txt'))
    put('fs/labels/D.TXT', '0 0.5 0.5 0.2 0.2\n')
    const read = async () => Object.fromEntries((await readYoloDataset(path.join(dir, 'fs'))).splits[0].items.map((i) => [path.basename(i.image), i.label && path.basename(i.label)]))
    // Whether D.TXT is d's is the file system's to say, as it is for Ultralytics
    const caseless = existsSync(path.join(dir, 'fs', 'labels', 'd.txt'))
    expect(await read()).toEqual({ 'a.jpg': 'a.txt', 'b.jpg': 'b.txt', 'c.jpg': null, 'd.jpg': caseless ? 'd.txt' : null, 'e.jpg': null })
    // A folder that can be entered but not listed (as root, it can be listed anyway)
    chmodSync(path.join(dir, 'fs', 'labels'), 0o311)
    try {
      expect(await read()).toEqual({ 'a.jpg': 'a.txt', 'b.jpg': 'b.txt', 'c.jpg': null, 'd.jpg': caseless ? 'd.txt' : null, 'e.jpg': null })
    } finally {
      chmodSync(path.join(dir, 'fs', 'labels'), 0o755)
    }
  })
  test("folders holding more files than a dataset may have are an error and nothing is read, wherever data.yaml's path leads", async () => {
    for (let i = 0; i < 6; i++) put(`ds/images/train/${i}.jpg`)
    put('ds/labels/train/0.txt', '0 0.5 0.5 0.2 0.2\n')
    put('ds/data.yaml', 'train: images/train\nval: images/train\nnames: [x]\n')
    const ds = path.join(dir, 'ds')
    // The splits are listed once each, so 12 files in all
    expect((await readYoloDataset(ds, { most: 12 })).splits.map((s) => s.items.length)).toEqual([6, 6])
    const yamlRead = await readYoloDataset(path.join(ds, 'data.yaml'), { most: 11 })
    expect([yamlRead.splits, codes(yamlRead)]).toEqual([[], [['error', 'yolo-too-many-files']]])
    expect(yamlRead.issues[0].message).toBe("The splits data.yaml names hold more than 11 files, more than Seamtag reads for one dataset, so none are read")
    // A folder without data.yaml is listed whole: 6 images and a label file
    rmSync(path.join(ds, 'data.yaml'))
    expect((await readYoloDataset(ds, { most: 7 })).splits.map((s) => s.items.length)).toEqual([6])
    const folder = await readYoloDataset(ds, { most: 6 })
    expect([folder.splits, folder.names, codes(folder)]).toEqual([[], [], [['error', 'yolo-too-many-files']]])
    expect(folder.issues[0].message).toBe('ds holds more than 6 files, more than Seamtag reads for one dataset, so none are read')
  })

  test('Stop ends the search for a dataset\'s files at the next folder', async () => {
    for (let i = 0; i < 30; i++) put(`ds/images/f${i}/a.jpg`)
    const listed: string[] = []
    const stop = new AbortController()
    const readdir = fsp.readdir.bind(fsp)
    vi.spyOn(fsp, 'readdir').mockImplementation(((p: string, o: object) => {
      listed.push(p)
      // Stopped while its third folder is listed
      if (listed.length === 3) stop.abort()
      return readdir(p, o as never)
    }) as never)
    await expect(readYoloDataset(path.join(dir, 'ds'), { signal: stop.signal })).rejects.toThrow(/aborted/)
    expect(listed.length).toBe(3)
    await expect(readYoloDataset(path.join(dir, 'ds'), { signal: AbortSignal.abort() })).rejects.toThrow(/aborted/)
  })

  test("darknet's names: the file obj.data names, inside the dataset, before obj.names or the one .names file; labelmaps that differ name nothing", async () => {
    put('dn/a.jpg')
    put('dn/a.txt', '1 0.5 0.5 0.2 0.2\n')
    put('dn/obj.names', 'car\ntree\n')
    put('dn/classes.names', 'cat\ndog\n')
    put('dn/obj.data', 'classes = 2\r\nnames = data\\classes.names\r\n')
    const names = async () => (await readYoloDataset(path.join(dir, 'dn'))).names
    expect(await names()).toEqual(['cat', 'dog'])
    // A file outside the dataset isn't read, as datumaro reads none
    put('elsewhere.names', 'x\ny\n')
    put('dn/obj.data', 'names = ../elsewhere.names\n')
    expect(await names()).toEqual(['car', 'tree'])
    rmSync(path.join(dir, 'dn', 'obj.names'))
    expect(await names()).toEqual(['cat', 'dog'])
    put('dn/more.names', 'x\n')
    expect(await names()).toEqual(['class_0', 'class_1'])
    // Roboflow's labelmaps, one in each split's folder, must agree
    put('rf/train/_darknet.labels', 'head\nhelmet\n')
    put('rf/valid/_darknet.labels', 'helmet\nhead\n')
    put('rf/train/a.jpg')
    put('rf/train/a.txt', '1 0.5 0.5 0.2 0.2\n')
    const rf = await readYoloDataset(path.join(dir, 'rf'))
    expect(rf.names).toEqual(['class_0', 'class_1'])
    expect(rf.issues.map((i) => [i.severity, slash(i.message)])).toEqual([
      ['warning', 'train/_darknet.labels and 1 more labelmap name the classes differently, so neither is used'],
      ['info', 'rf has neither data.yaml nor classes.txt, so the classes are named class_0 to class_1']
    ])
  })

  test("files of class names: a long run of blank lines is read in no time, and a file too big to be a list of names isn't read", async () => {
    put('dn/a.jpg')
    put('dn/a.txt', '0 0.5 0.5 0.2 0.2\n')
    // The blank lines stay, as LabelImg keeps them: 50,000 took five seconds
    put('dn/obj.names', `car${'\n'.repeat(50_000)}tree\n`)
    const t = performance.now()
    let d = await readYoloDataset(path.join(dir, 'dn'))
    expect(performance.now() - t).toBeLessThan(500)
    expect([d.names.length, d.names[0], d.names[1], d.names[50_000]]).toEqual([50_001, 'car', '', 'tree'])
    // Past 4 MB, classes.txt is passed over for the next place names can come from
    const big = 'x\n'.repeat(2.5 * 2 ** 20)
    put('dn/classes.txt', big)
    d = await readYoloDataset(path.join(dir, 'dn'))
    expect(d.names[0]).toBe('car')
    expect(d.issues.map((i) => [i.severity, i.message])).toEqual([
      ['warning', "classes.txt is 5.0 MB, more than the 4 MB a file of class names may be, so it isn't read"],
      ['info', 'dn has no data.yaml, so the class names come from obj.names']
    ])
    rmSync(path.join(dir, 'dn', 'classes.txt'))
    put('dn/obj.names', big)
    d = await readYoloDataset(path.join(dir, 'dn'))
    expect(d.names).toEqual(['class_0'])
    expect(d.issues.map((i) => [i.severity, i.message])).toEqual([
      ['warning', "obj.names is 5.0 MB, more than the 4 MB a file of class names may be, so it isn't read"],
      ['info', 'dn has neither data.yaml nor classes.txt, so the classes are named class_0 to class_0']
    ])
  })

  test('without data.yaml, classes are named up to class_99999: a row of a class past it is told and left out', async () => {
    put('ds/images/a.jpg')
    put('ds/labels/a.txt', '1 0.5 0.5 0.2 0.2\n100000000 0.5 0.5 0.2 0.2\n4294967295 0.5 0.5 0.2 0.2\n')
    const t = performance.now()
    const d = await readYoloDataset(path.join(dir, 'ds'))
    // Each class up to the highest was named: 100,000,000 of them took the app's memory
    expect(performance.now() - t).toBeLessThan(2000)
    expect(d.names).toEqual(['class_0', 'class_1'])
    expect(d.issues.map((i) => [i.code, slash(i.message)])).toEqual([
      ['yolo-no-yaml', 'ds has neither data.yaml nor classes.txt, so the classes are named class_0 to class_1'],
      ['yolo-no-yaml', 'labels/a.txt has a row of class 100000000, and without data.yaml Seamtag names at most 100,000 classes, so rows past class_99999 are left out']
    ])
  })

  test('a split of more images than a call takes arguments is listed whole, from a folder or a list file', async () => {
    // Over 125,000 paths spread into a call overflow the stack. So many files would take half a
    // minute to make, so a folder's listing is made up, and val lists the same images
    const N = 140_000
    const ds = path.join(dir, 'ds')
    const many = path.join(ds, 'images', 'train', 'many')
    mkdirSync(many, { recursive: true })
    const names = Array.from({ length: N }, (_, i) => `${i}.jpg`)
    put('ds/data.yaml', 'train: images/train\nval: val.txt\nnames: [x]\n')
    put('ds/val.txt', names.map((n) => path.join(many, n)).join('\n'))
    const entries = names.map((name) => ({ name, isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false }))
    const readdir = fsp.readdir.bind(fsp)
    vi.spyOn(fsp, 'readdir').mockImplementation(((p: string, o: object) => (p === many ? Promise.resolve(entries) : readdir(p, o as never))) as never)
    const d = await readYoloDataset(ds)
    expect(d.splits.map((s) => [s.name, s.items.length])).toEqual([['train', N], ['val', N]])
    expect(d.issues.map((i) => [i.code, i.message])).toEqual([
      ['yolo-two-splits', `${N} images are in both train and val`],
      ['yolo-no-labels', `None of the ${N} images has a label file where Ultralytics looks (labels for the last images folder in its path, else beside it), so they'd train as backgrounds`]
    ])
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
  // Every kind of shape, some past the edges, a mask with a hole and two parts, and two objects of two polygons each,
  // one with a part past the image's edge: kept inside it, that part's closest point to the other is another one
  const MASK = bitmap(2, 2, ['1111100', '1001100', '1111100', '0000000', '0000011', '0000011'])
  const sample = (size: [number, number], name: string, labels: string[]): DatasetImage => {
    const [w, h] = size
    const shapes = [
      rect(labels[0], [[w * 0.1, h * 0.1], [w * 0.4, h * 0.5]]),
      poly(labels[1 % labels.length], [[w * 0.5, h * 0.1], [w * 0.9, h * 0.15], [w * 0.95, h * 0.6], [w * 0.6, h * 0.5]]),
      newShape(labels[2 % labels.length], 'circle', [[w * 0.8, h * 0.8], [w * 0.95, h * 0.8]]),
      newShape(labels[3 % labels.length], 'oriented_rectangle', [[w * 0.2, h * 0.6], [w * 0.4, h * 0.7], [w * 0.35, h * 0.8], [w * 0.15, h * 0.7]]),
      newShape(labels[4 % labels.length], 'mask', [[2, 2], [8, 7]]),
      rect(labels[5 % labels.length], [[-3, h * 0.8], [w * 0.1, h + 4]]),
      newShape(labels[0], 'line', [[1, 1], [5, 5]]),
      newShape(labels[0], 'polygon', [[w * 0.55, h * 0.65], [w * 0.65, h * 0.66], [w * 0.6, h * 0.75]], { group_id: 3 }),
      newShape(labels[0], 'polygon', [[w * 0.7, h * 0.85], [w * 0.75, h * 0.86], [w * 0.72, h * 0.95], [w * 0.69, h * 0.9]], { group_id: 3 }),
      newShape(labels[1 % labels.length], 'polygon', [[w * 0.75, h * 0.2], [w * 0.9, h * 0.2], [w * 0.825, h * 0.35]], { group_id: 4 }),
      newShape(labels[1 % labels.length], 'polygon', [[w * 1.2, h * 0.17], [w * 0.975, h * 0.03], [w * 0.9875, h * 0.6]], { group_id: 4 })
    ]
    return image(shapes, [null, null, null, null, MASK], size, name)
  }
  // The objects written as outlines joined from several parts, with the parts in pixels, so the parity script can join them with Ultralytics' own code
  const joined = (img: DatasetImage, names: string[]) => [
    { class: names.indexOf(img.shapes[4].label), parts: maskOutlines(MASK).outer },
    { class: names.indexOf(img.shapes[7].label), parts: [img.shapes[7].points, img.shapes[8].points] },
    { class: names.indexOf(img.shapes[9].label), parts: [img.shapes[9].points, img.shapes[10].points] }
  ]
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
    const expected: Record<string, { status: string; rows: number[][]; joined?: ReturnType<typeof joined> }> = {}
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
      if (task === 'segment' && img.shapes.length > 8) expected[rel].joined = joined(img, names)
      expect(verifyRows(out.text, task, names.length, pose?.kptShape ?? null)).toMatchObject({ status: out.rows ? 'ok' : 'empty', repeated: [] })
    }
    writeFileSync(path.join(root, 'data.yaml'), writeDataYaml({ names, train: 'images/train', val: 'images/val', ...(pose ?? {}) }))
    writeFileSync(`${root}.expected.json`, JSON.stringify({ task, names, kpt_shape: pose?.kptShape ?? null, flip_idx: pose?.flipIdx ?? null, images: expected }, null, 1))
  })
})
