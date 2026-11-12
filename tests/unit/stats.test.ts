import { expect, test } from 'vitest'
import type { DatasetImage, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { datasetStats, statsCsv } from '../../src/shared/stats'

function image(name: string, shapes: Shape[] | null, extra: Partial<DatasetImage> = {}): DatasetImage {
  return {
    name, width: 400, height: 300, file: shapes ? `/data/${name}.json` : null, format: shapes ? 'labelme' : null, shapes: shapes ?? [],
    masks: (shapes ?? []).map(() => null), flags: {}, verified: false, ...extra
  }
}
const box = (label: string, x0: number, y0: number, x1: number, y1: number, extra: Partial<Shape> = {}) =>
  newShape(label, 'rectangle', [[x0, y0], [x1, y1]], extra)
const ai = { flags: { ai: true }, other: { score: 0.3 } }

// A folder whose every count is known: sizes are COCO's (small below 32², large from 96²), and
// the 6000 x 4000 image is scaled by 640 / 6000 for training
const FOLDER = [
  image('a', [box('car', 0, 0, 10, 10), box('car', 0, 0, 40, 40), newShape('person', 'polygon', [[0, 0], [100, 0], [100, 100], [0, 100]]), box('car', 0, 0, 50, 50, ai)]),
  image('b', [box('car', 0, 0, 100, 100)], { width: 6000, height: 4000 }),
  image('c', []),
  image('d', null),
  image('e', null, { file: '/data/e.json', format: 'labelme', broken: "It isn't valid JSON" }),
  image('f', [newShape('nose', 'point', [[5, 5]])], { verified: true }),
  image('g', [box('car', 0, 0, 10, 10)], { width: 0, height: 0, unreadable: 'Input file is missing' }),
  image('h', [box('car', 0, 0, 10, 10, ai)])
]

test('images are counted by status: AI proposals alone are neither labelled nor empty', () => {
  expect(datasetStats(FOLDER).images).toEqual({ total: 8, labelled: 4, empty: 1, never: 1, verified: 1, broken: 1, unreadable: 1, aiPending: 2 })
})

test('objects per label count accepted shapes, the images they are in and their shape types, most objects first', () => {
  const stats = datasetStats(FOLDER)
  expect(stats.labels).toEqual([
    { label: 'car', objects: 4, images: 3, types: { rectangle: 4 } },
    { label: 'nose', objects: 1, images: 1, types: { point: 1 } },
    { label: 'person', objects: 1, images: 1, types: { polygon: 1 } }
  ])
  expect(stats.types).toEqual(['polygon', 'rectangle', 'point'])
  // The label list's labels that nothing uses come last
  expect(datasetStats(FOLDER, { labels: ['bus', 'car'] }).labels.map((l) => [l.label, l.objects])).toEqual([['car', 4], ['nose', 1], ['person', 1], ['bus', 0]])
})

test('objects per image leave out images with no label file or one that cannot be read', () => {
  expect(datasetStats(FOLDER).objectsPerImage).toEqual([{ objects: 0, images: 2 }, { objects: 1, images: 3 }, { objects: 3, images: 1 }])
})

test("object sizes fall in COCO's ranges at full size and at the training size; points and images of unknown size have none", () => {
  expect(datasetStats(FOLDER).sizes).toEqual({
    trainSize: 640,
    // a: 100, 1600 and 10000 square pixels; b: 10000
    full: { small: 1, medium: 1, large: 2 },
    // a grows by 640 / 400: 256, 4096 and 25600; b shrinks by 640 / 6000 to 114
    train: { small: 2, medium: 1, large: 1 },
    unsized: 2
  })
  expect(datasetStats(FOLDER, { trainSize: 6000 }).sizes.train).toEqual({ small: 0, medium: 0, large: 4 })
})

test("shapes sharing a label and a group_id are one object, as the exports take them, and a pose's points are no objects", () => {
  const part = (label: string, x0: number, y0: number, x1: number, y1: number, group: number) => box(label, x0, y0, x1, y1, { group_id: group })
  const point = (label: string, x: number, y: number, group: number | null) => newShape(label, 'point', [[x, y]], { group_id: group })
  const images = [
    image('a', [
      // A cat in two parts, another in one, and a dog sharing the first cat's group: three objects
      part('cat', 0, 0, 10, 10, 1), part('cat', 20, 0, 30, 10, 1), box('cat', 50, 50, 60, 60), part('dog', 100, 100, 110, 110, 1),
      // A pose: its box, and points that are its keypoints
      part('person', 0, 100, 50, 200, 2), point('nose', 20, 120, 2), point('left_eye', 15, 115, 2),
      // A point of no group is an object of its own, and so is a line, though the COCO and YOLO exports leave both out
      point('dot', 5, 5, null), newShape('lane', 'linestrip', [[0, 250], [200, 260], [390, 250]])
    ]),
    // Boxes of two labels in one group make no pose, so its point is an object
    image('b', [part('car', 0, 0, 50, 50, 3), part('bus', 60, 0, 110, 50, 3), point('wheel', 10, 45, 3)])
  ]
  const stats = datasetStats(images)
  expect(stats.labels).toEqual([
    { label: 'cat', objects: 2, images: 1, types: { rectangle: 3 } },
    { label: 'bus', objects: 1, images: 1, types: { rectangle: 1 } },
    { label: 'car', objects: 1, images: 1, types: { rectangle: 1 } },
    { label: 'dog', objects: 1, images: 1, types: { rectangle: 1 } },
    { label: 'dot', objects: 1, images: 1, types: { point: 1 } },
    { label: 'lane', objects: 1, images: 1, types: { linestrip: 1 } },
    { label: 'person', objects: 1, images: 1, types: { rectangle: 1 } },
    { label: 'wheel', objects: 1, images: 1, types: { point: 1 } },
    // Keypoints count as shapes of their labels, with no objects
    { label: 'left_eye', objects: 0, images: 1, types: { point: 1 } },
    { label: 'nose', objects: 0, images: 1, types: { point: 1 } }
  ])
  expect(stats.objectsPerImage).toEqual([{ objects: 3, images: 1 }, { objects: 6, images: 1 }])
  // The dot, the lane and the wheel have no area
  expect(stats.sizes.unsized).toBe(3)
})

test("an object of several shapes is ranged by its area as one, where they overlap counted once, as COCO's export merges them", () => {
  const mask: MaskBits = { x: 0, y: 200, width: 40, height: 40, data: new Uint8Array(1600).fill(1) }
  const images = [image('m', [
    // Two 25 x 25 squares, one 5 px right of the other: 625 square pixels each, small; 750 as one, small, where adding them up would give 1,250, medium
    box('a', 0, 0, 25, 25, { group_id: 1 }), box('a', 5, 0, 30, 25, { group_id: 1 }),
    // Two 20 x 30 boxes apart: 600 each, small; 1,200 as one, medium
    box('b', 100, 0, 120, 30, { group_id: 1 }), box('b', 200, 0, 220, 30, { group_id: 1 }),
    // A mask of 1,600 pixels and a 20 x 20 box: 2,000 as one, medium
    newShape('road', 'mask', [[0, 200], [39, 239]], { mask: 'png', group_id: 2 }), box('road', 100, 200, 120, 220, { group_id: 2 })
  ], { masks: [null, null, null, null, mask, null] })]
  const stats = datasetStats(images)
  expect(stats.sizes.full).toEqual({ small: 1, medium: 2, large: 0 })
  expect(stats.objectsPerImage).toEqual([{ objects: 3, images: 1 }])
  expect(stats.labels.map((l) => [l.label, l.objects, l.types])).toEqual([['a', 1, { rectangle: 2 }], ['b', 1, { rectangle: 2 }], ['road', 1, { mask: 1, rectangle: 1 }]])
})

test("masks are measured by their pixels and circles by their radius", () => {
  const mask: MaskBits = { x: 0, y: 0, width: 40, height: 40, data: new Uint8Array(1600).fill(1) }
  const images = [image('m', [newShape('road', 'mask', [[0, 0], [39, 39]], { mask: 'png' }), newShape('ball', 'circle', [[100, 100], [100, 110]])], { masks: [mask, null] })]
  // 1600 pixels; π 10² = 314
  expect(datasetStats(images).sizes.full).toEqual({ small: 1, medium: 1, large: 0 })
})

test('a polygon with a hole is measured without the hole, whichever way round the hole is wound', () => {
  // 40 x 40 with a 30 x 30 hole joined by a cut: 700 square pixels, small; counting the hole in, 2500 would be medium
  const ring = (hole: number[][]) => [[0, 0], [40, 0], [40, 40], [0, 40], [0, 0], ...hole, [5, 5]]
  const images = [image('h', [
    newShape('frame', 'polygon', ring([[5, 5], [35, 5], [35, 35], [5, 35]])),
    newShape('frame', 'polygon', ring([[5, 5], [5, 35], [35, 35], [35, 5]]))
  ])]
  expect(datasetStats(images).sizes.full).toEqual({ small: 2, medium: 0, large: 0 })
})

test('image sizes are listed with how many images have each, the most common first', () => {
  expect(datasetStats(FOLDER).imageSizes).toEqual([{ width: 400, height: 300, images: 6 }, { width: 6000, height: 4000, images: 1 }])
})

test('the per-label table saves as CSV, quoting what needs it', () => {
  expect(statsCsv(datasetStats(FOLDER))).toBe('\ufefflabel,objects,images,polygon,rectangle,point\r\ncar,4,3,0,4,0\r\nnose,1,1,0,0,1\r\nperson,1,1,1,0,0\r\n')
  const odd = datasetStats([image('a', [box('a, "b"', 0, 0, 5, 5), box('a, "b"', 9, 0, 15, 5), box(' c', 0, 0, 5, 5)])])
  expect(statsCsv(odd)).toBe('\ufefflabel,objects,images,rectangle\r\n"a, ""b""",2,1,2\r\n" c",1,1,1\r\n')
})

test('the CSV starts with a byte order mark, so Excel reads labels beyond ASCII as UTF-8', () => {
  const csv = statsCsv(datasetStats([image('a', [box('고양이', 0, 0, 5, 5), box('café', 9, 0, 15, 5)])]))
  expect([...Buffer.from(csv, 'utf8').subarray(0, 3)]).toEqual([0xef, 0xbb, 0xbf])
  expect(csv.slice(1)).toBe('label,objects,images,rectangle\r\ncafé,1,1,1\r\n고양이,1,1,1\r\n')
})

test('labels a spreadsheet would run as formulas are saved as text: a quote first, and quoted, as OWASP advises', () => {
  const labels = ['=HYPERLINK("http://x","y")', '+1', '-car', '@SUM(A1)', '\tcat', '\rdog', 'a=b']
  const stats = datasetStats([image('a', labels.map((l, k) => box(l, k * 10, 0, k * 10 + 5, 5)))])
  expect(statsCsv(stats).split('\r\n').slice(1, -1).sort()).toEqual([
    '"\'\tcat",1,1,1', '"\'\rdog",1,1,1', '"\'+1",1,1,1', '"\'-car",1,1,1', '"\'=HYPERLINK(""http://x"",""y"")",1,1,1', '"\'@SUM(A1)",1,1,1', 'a=b,1,1,1'
  ].sort())
})
