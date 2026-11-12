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
  expect(statsCsv(datasetStats(FOLDER))).toBe('label,objects,images,polygon,rectangle,point\r\ncar,4,3,0,4,0\r\nnose,1,1,0,0,1\r\nperson,1,1,1,0,0\r\n')
  const odd = datasetStats([image('a', [box('a, "b"', 0, 0, 5, 5), box('a, "b"', 9, 0, 15, 5), box(' c', 0, 0, 5, 5)])])
  expect(statsCsv(odd)).toBe('label,objects,images,rectangle\r\n"a, ""b""",2,1,2\r\n" c",1,1,1\r\n')
})

test('labels a spreadsheet would run as formulas are saved as text: a quote first, and quoted, as OWASP advises', () => {
  const labels = ['=HYPERLINK("http://x","y")', '+1', '-car', '@SUM(A1)', '\tcat', '\rdog', 'a=b']
  const stats = datasetStats([image('a', labels.map((l, k) => box(l, k * 10, 0, k * 10 + 5, 5)))])
  expect(statsCsv(stats).split('\r\n').slice(1, -1).sort()).toEqual([
    '"\'\tcat",1,1,1', '"\'\rdog",1,1,1', '"\'+1",1,1,1', '"\'-car",1,1,1', '"\'=HYPERLINK(""http://x"",""y"")",1,1,1', '"\'@SUM(A1)",1,1,1', 'a=b,1,1,1'
  ].sort())
})
