import { describe, expect, test } from 'vitest'
import {
  boxesClipped, CHECKS, type CheckCode, type CheckOptions, checkDataset, crossing, fixIssue, renameLabel, withoutDuplicates, withoutRepeatedPoints
} from '../../src/shared/checks'
import type { DatasetImage, Issue, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'

// The checks run on what readDataset gives: images 400 x 300 with LabelMe files, built here by hand
const W = 400, H = 300
function image(name: string, shapes: Shape[] = [], extra: Partial<DatasetImage> = {}): DatasetImage {
  return {
    name, width: W, height: H, file: `/data/${name.replace(/\.\w+$/, '.json')}`, format: 'labelme', shapes, masks: shapes.map(() => null),
    flags: {}, verified: false, labelSize: [W, H], labelImage: name, bytes: 1000, ...extra
  }
}
const box = (label: string, x0: number, y0: number, x1: number, y1: number, extra: Partial<Shape> = {}) =>
  newShape(label, 'rectangle', [[x0, y0], [x1, y1]], extra)
const poly = (label: string, points: number[][], extra: Partial<Shape> = {}) => newShape(label, 'polygon', points, extra)
const square = (label: string, x0: number, y0: number, x1: number, y1: number) => poly(label, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
const ai = { flags: { ai: true }, other: { score: 0.4 } }
const check = (images: DatasetImage[], options: Partial<CheckOptions> = {}) => checkDataset(images, { labels: null, ...options })
const of = (issues: Issue[], code: CheckCode) => issues.filter((i) => i.code === code)
/** A mask's pixels from rows of 0 and 1, its top-left pixel at (x, y) */
function bits(x: number, y: number, rows: string[]): MaskBits {
  const width = rows[0].length, height = rows.length
  return { x, y, width, height, data: Uint8Array.from(rows.join('').split('').map(Number)) }
}
const maskShape = (label: string, m: MaskBits) => newShape(label, 'mask', [[m.x, m.y], [m.x + m.width - 1, m.y + m.height - 1]], { mask: 'png' })

describe('errors', () => {
  test("a label file that can't be read is an error, and nothing else is said of its image", () => {
    const issues = check([image('a.png', [], { broken: "It isn't valid JSON (Unexpected end of JSON input)" }), image('b.png', [box('car', 1, 1, 9, 9)])])
    expect(issues).toEqual([{
      severity: 'error', code: 'label-file-unreadable', image: 'a.png',
      message: "a.json can't be read, so a.png's labels can't be checked or exported: It isn't valid JSON (Unexpected end of JSON input)"
    }])
  })

  test("an image that can't be read is an error, and its shapes aren't measured against a size it doesn't have", () => {
    const far = box('car', 900, 900, 950, 950)
    const issues = check([image('a.png', [far], { width: 0, height: 0, unreadable: 'Input file contains unsupported image format', labelSize: [10, 10] })])
    expect(issues.map((i) => i.code)).toEqual(['image-unreadable'])
    expect(issues[0].message).toBe("a.png can't be read, so its labels can't be checked against it or exported: Input file contains unsupported image format")
    // YOLO's shares were read against no size at all: their geometry means nothing
    const yolo = check([image('b.png', [box('car', 0, 0, 0, 0)], { format: 'yolo', width: 0, height: 0, unreadable: 'gone', labelSize: null })])
    expect(yolo.map((i) => i.code)).toEqual(['image-unreadable'])
  })

  test("a label file that gives another size than the image's is an error; one that gives none, or the same, isn't", () => {
    const issues = check([image('a.png', [], { labelSize: [300, 400] }), image('b.png', [], { labelSize: null, format: 'yolo' }), image('c.png')])
    expect(of(issues, 'size-mismatch')).toEqual([{
      severity: 'error', code: 'size-mismatch', image: 'a.png',
      message: "a.json gives the image size as 300 x 400, but a.png is 400 x 300, so its shapes may be out of place, and labelme 7 won't open it"
    }])
  })

  test('a shape with no label is an error; one with a label is not', () => {
    const issues = check([image('a.png', [box('car', 1, 1, 9, 9), box('  ', 1, 1, 9, 9)])])
    expect(of(issues, 'empty-label')).toEqual([
      { severity: 'error', code: 'empty-label', image: 'a.png', shape: 1, message: 'Shape 2 has no label, so no export can name its class' }
    ])
  })

  test("a mask whose pixels can't be read, or that has none, is an error; one that decoded is not", () => {
    const good = bits(10, 10, ['11', '11'])
    const shapes = [maskShape('road', good), maskShape('road', bits(50, 50, ['1'])), { ...maskShape('road', bits(70, 70, ['1'])), mask: null }]
    const issues = check([image('a.png', shapes, { masks: [good, null, null] })])
    expect(of(issues, 'mask-undecodable').map((i) => [i.shape, i.message])).toEqual([
      [1, "Shape 2 (road) is a mask whose pixels can't be read, so it can't be exported"],
      [2, 'Shape 3 (road) is a mask, but the file holds no pixels for it']
    ])
  })

  test('images sharing one label file are each an error, and not also told their imagePath differs', () => {
    const images = [image('a.jpg', [], { labelImage: 'a.png' }), image('a.png'), image('b.png')]
    const issues = check(images, { clashes: [{ file: '/data/a.json', images: ['a.jpg', 'a.png'] }] })
    expect(of(issues, 'shared-label-file').map((i) => [i.image, i.message])).toEqual([
      ['a.jpg', "a.json is the label file of a.jpg and a.png, so saving one replaces the others' labels"],
      ['a.png', "a.json is the label file of a.jpg and a.png, so saving one replaces the others' labels"]
    ])
    expect(of(issues, 'image-path-mismatch')).toEqual([])
  })
})

describe('shapes', () => {
  test('a shape with too few points for its type, or the wrong exact count, is a warning', () => {
    const shapes = [box('car', 1, 1, 9, 9), poly('car', [[1, 1], [5, 1], [3, 4]]), poly('car', [[1, 1], [5, 1]]),
      newShape('car', 'rectangle', [[1, 1], [5, 5], [9, 9]]), newShape('road', 'linestrip', [[1, 1]])]
    expect(of(check([image('a.png', shapes)]), 'point-count').map((i) => i.message)).toEqual([
      'Shape 3 (car) has 2 points: a polygon needs at least 3',
      'Shape 4 (car) has 3 points: a rectangle needs exactly 2',
      'Shape 5 (road) has 1 point: a line strip needs at least 2'
    ])
  })

  test('a shape that covers no area, or less than a pixel, is a warning; a thin but real box is not', () => {
    const shapes = [box('car', 10, 10, 10, 50), box('car', 10, 10, 10.5, 10.5), box('car', 10, 10, 11, 60),
      poly('car', [[0, 0], [5, 5], [10, 10]]), newShape('ball', 'circle', [[50, 50], [50, 50]]), maskShape('road', bits(0, 0, ['00']))]
    const issues = check([image('a.png', shapes, { masks: [null, null, null, null, null, bits(0, 0, ['00'])] })])
    expect(of(issues, 'zero-area').map((i) => i.message)).toEqual([
      'Shape 1 (car) covers no area: it has no width',
      'Shape 2 (car) covers less than a pixel',
      'Shape 4 (car) covers no area: its points lie on one line',
      'Shape 5 (ball) covers no area: its rim is at its centre',
      'Shape 6 (road) covers no area: its mask has no pixels set'
    ])
  })

  test("a bow tie's lobes cancel in the shoelace sum, but it still covers an area", () => {
    const issues = check([image('a.png', [poly('car', [[0, 0], [10, 10], [10, 0], [0, 10]])])])
    expect(of(issues, 'zero-area')).toEqual([])
  })

  test('points that repeat the one before, or a closing point that repeats the first, are a warning; a point visited twice apart is not', () => {
    const issues = check([image('a.png', [
      poly('car', [[0, 0], [0, 0], [10, 0], [10, 10], [0, 0]]),
      poly('car', [[0, 0], [10, 0], [5, 5], [10, 10], [0, 10], [5, 5]]),
      newShape('road', 'linestrip', [[0, 0], [5, 5], [5, 5], [5, 5]])
    ])])
    expect(of(issues, 'repeated-points').map((i) => i.message)).toEqual([
      'Shape 1 (car) has 1 point that repeats the one before, and ends with its first point again: taking them out leaves its outline as it is',
      'Shape 3 (road) has 2 points that repeat the one before: taking them out leaves its outline as it is'
    ])
  })

  test('a polygon that crosses itself is a warning, with where', () => {
    const issues = check([image('a.png', [poly('car', [[0, 0], [10, 10], [10, 0], [0, 10]]), square('car', 20, 20, 30, 30)])])
    expect(of(issues, 'self-intersecting')).toEqual([{
      severity: 'warning', code: 'self-intersecting', image: 'a.png', shape: 0, message: 'Shape 1 (car) crosses itself near (5, 5), so tools may fill it differently'
    }])
  })

  test("the zero-width cut joining a hole to its outline in Seamtag's polygons isn't a crossing", () => {
    // As geometry.ts withHoles joins a 20 px hole to the outline: out along y = 40 and back
    const holed = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 40], [40, 40], [40, 60], [60, 60], [60, 40], [40, 40], [0, 40]]
    expect(crossing(holed)).toBeNull()
    expect(of(check([image('a.png', [poly('car', holed)])]), 'self-intersecting')).toEqual([])
  })

  test('long outlines are swept: 20000 points take little time, and one crossing among them is found', () => {
    const n = 20000
    const ring = Array.from({ length: n }, (_, k) => [200 + 100 * Math.cos((2 * Math.PI * k) / n), 150 + 100 * Math.sin((2 * Math.PI * k) / n)])
    const started = Date.now()
    expect(crossing(ring)).toBeNull()
    expect(Date.now() - started).toBeLessThan(2000)
    // Swap two neighbours: their edges now cross
    const twisted = ring.map((p) => [...p])
    ;[twisted[500], twisted[501]] = [twisted[501], twisted[500]]
    const at = crossing(twisted)!
    expect(Math.hypot(at[0] - ring[500][0], at[1] - ring[500][1])).toBeLessThan(0.1)
  })

  test('a shape more than a pixel outside the image is a warning, by the edge it passes most; within a pixel it is not', () => {
    const shapes = [box('car', 10, 10, W + 1.5, 50), box('car', -0.9, 10, W + 0.9, 50), newShape('ball', 'circle', [[5, 150], [5, 135]]), box('car', -30, -5, 20, 20)]
    expect(of(check([image('a.png', shapes)]), 'outside-image').map((i) => i.message)).toEqual([
      'Shape 1 (car) reaches 1.5 px past the right edge of the image',
      'Shape 3 (ball) reaches 10 px past the left edge of the image',
      'Shape 4 (car) reaches 30 px past the left edge of the image'
    ])
  })

  test('an exact copy is a warning on the copy that would go: the one without a description', () => {
    const shapes = [box('car', 10, 10, 50, 50), box('car', 50.004, 50, 10, 10.005, { description: 'front' }), box('bus', 10, 10, 50, 50), box('car', 10.02, 10, 50, 50)]
    const issues = check([image('a.png', shapes)])
    expect(of(issues, 'duplicate-shape')).toEqual([
      { severity: 'warning', code: 'duplicate-shape', image: 'a.png', shape: 0, message: 'Shape 1 (car) is an exact copy of shape 2' }
    ])
  })

  test("a person's shape is kept over a model's proposal of it", () => {
    const shapes = [box('car', 10, 10, 50, 50, ai), box('car', 10, 10, 50, 50)]
    expect(of(check([image('a.png', shapes)]), 'duplicate-shape').map((i) => i.shape)).toEqual([0])
  })

  test('shapes of one label almost on top of each other are a warning; less overlap, or another label, is not', () => {
    const shapes = [square('car', 0, 0, 100, 100), square('car', 2, 0, 102, 100), square('car', 200, 0, 300, 100), square('car', 208, 0, 308, 100),
      box('bus', 0, 0, 100, 100), box('bus', 0, 2, 100, 101), square('van', 0, 0, 100, 100)]
    expect(of(check([image('a.png', shapes)]), 'near-duplicate').map((i) => i.message)).toEqual([
      'Shape 2 (car) covers almost the same area as shape 1 (they overlap 96%)',
      'Shape 6 (bus) covers almost the same area as shape 5 (they overlap 97%)'
    ])
  })

  test('a box and a polygon of one label over the same area are near-duplicates; circles are compared by their outlines', () => {
    const shapes = [box('car', 0, 0, 100, 100), square('car', 1, 0, 100, 100), newShape('ball', 'circle', [[200, 150], [250, 150]]), newShape('ball', 'circle', [[201, 150], [251, 150]])]
    expect(of(check([image('a.png', shapes)]), 'near-duplicate').map((i) => [i.shape, i.message])).toEqual([
      [1, 'Shape 2 (car) covers almost the same area as shape 1 (they overlap 99%)'],
      [3, 'Shape 4 (ball) covers almost the same area as shape 3 (they overlap 97%)']
    ])
  })

  test('masks are compared by their pixels, and never with outlines', () => {
    const a = bits(0, 0, ['1111111111', '1111111111']), b = bits(0, 0, ['1111111111', '1111111110']), c = bits(5, 5, ['11', '11'])
    const shapes = [maskShape('road', a), maskShape('road', b), maskShape('road', c), box('road', 0, 0, 10, 2)]
    expect(of(check([image('a.png', shapes, { masks: [a, b, c, null] })]), 'near-duplicate').map((i) => i.message)).toEqual([
      'Shape 2 (road) covers almost the same area as shape 1 (they overlap 95%)'
    ])
  })
})

describe('labels', () => {
  test('labels missing from the label list are a warning, once an image; with no list set, nothing is', () => {
    const images = [image('a.png', [box('car', 1, 1, 9, 9), box('bus', 1, 1, 9, 9), box('bus', 20, 1, 29, 9)])]
    expect(of(check(images, { labels: ['car'] }), 'unknown-label')).toEqual([
      { severity: 'warning', code: 'unknown-label', image: 'a.png', shape: 1, message: 'Label "bus" isn\'t in the label list (shapes 2 and 3)' }
    ])
    expect(of(check(images), 'unknown-label')).toEqual([])
  })

  test("one label written several ways is a warning on the rarer spellings; the label list's spelling wins", () => {
    const images = [
      image('a.png', [box('car', 1, 1, 9, 9), box('car', 20, 1, 29, 9), box('traffic_light', 40, 1, 49, 9), box('traffic_light', 80, 1, 89, 9)]),
      image('b.png', [box('Car ', 1, 1, 9, 9), box('car', 20, 1, 29, 9), box('Traffic-Light', 40, 1, 49, 9), box('cart', 60, 1, 69, 9)])
    ]
    expect(of(check(images), 'label-variants').map((i) => [i.image, i.shape, i.message])).toEqual([
      ['b.png', 0, 'Label "Car " looks like "car" written another way (3 shapes use "car")'],
      ['b.png', 2, 'Label "Traffic-Light" looks like "traffic_light" written another way (2 shapes use "traffic_light")']
    ])
    const listed = check(images, { labels: ['Car ', 'traffic_light', 'cart'] })
    expect(of(listed, 'label-variants').map((i) => [i.image, i.message])).toEqual([
      ['a.png', 'Label "car" looks like "Car " from the label list, written another way'],
      ['b.png', 'Label "car" looks like "Car " from the label list, written another way'],
      ['b.png', 'Label "Traffic-Light" looks like "traffic_light" from the label list, written another way']
    ])
    // A variant of a listed label is told so, not that it's missing from the list
    expect(of(listed, 'unknown-label')).toEqual([])
  })

  test('labels in the list that no accepted shape uses are noted; a proposal alone is no use', () => {
    const images = [image('a.png', [box('car', 1, 1, 9, 9), box('bus', 1, 1, 9, 9, ai)])]
    expect(of(check(images, { labels: ['car', 'bus', 'van'] }), 'unused-label').map((i) => [i.image, i.message])).toEqual([
      [null, 'Label "bus" is in the label list, but no accepted shape uses it'],
      [null, 'Label "van" is in the label list, but no accepted shape uses it']
    ])
  })

  test('AI proposals not yet accepted are a warning on their image; accepted shapes are not', () => {
    const issues = check([image('a.png', [box('car', 1, 1, 9, 9), box('car', 20, 1, 29, 9, ai), box('car', 40, 1, 49, 9, ai)]), image('b.png', [box('car', 1, 1, 9, 9)])])
    expect(of(issues, 'ai-pending')).toEqual([
      { severity: 'warning', code: 'ai-pending', image: 'a.png', shape: 1, message: 'a.png has 2 AI proposals not yet accepted' }
    ])
  })
})

describe('files', () => {
  test('a label file naming another image is a warning; naming its own, or holding its own image, is not', () => {
    const issues = check([image('a.png', [], { labelImage: 'b.png' }), image('c.png', [], { labelImage: null }), image('d.png'), image('e.json', [], { labelImage: undefined })])
    expect(of(issues, 'image-path-mismatch').map((i) => i.message)).toEqual([
      'a.json names b.png as its image, not a.png, so labelme would show these labels on that image',
      'c.json names an image outside the folder as its image, not c.png, so labelme would show these labels on that image'
    ])
  })

  test('label files without their image are a warning for the folder', () => {
    const issues = check([image('a.png')], { orphans: [{ file: 'labels/b.json', names: 'images/b.png' }, { file: 'c.xml', names: null }] })
    expect(of(issues, 'orphan-label-file').map((i) => [i.image, i.message])).toEqual([
      [null, 'labels/b.json has no image in the folder, so its labels are never shown or exported; its imagePath names images/b.png'],
      [null, 'c.xml has no image in the folder, so its labels are never shown or exported']
    ])
  })

  test('identical image files are a warning on each of them', () => {
    const issues = check([image('a.png'), image('b.png'), image('c.png')], { duplicates: [['a.png', 'c.png']] })
    expect(of(issues, 'duplicate-image').map((i) => [i.image, i.message])).toEqual([
      ['a.png', 'a.png has the same bytes as c.png, so a split can put one copy in training and another in validation'],
      ['c.png', 'c.png has the same bytes as a.png, so a split can put one copy in training and another in validation']
    ])
  })

  test('never labelled and labelled with no shapes are told apart', () => {
    const issues = check([image('a.png', [], { file: null, format: null, labelSize: null }), image('b.png', []), image('c.png', [box('car', 1, 1, 9, 9)])])
    expect(issues.filter((i) => i.severity === 'info').map((i) => [i.code, i.message])).toEqual([
      ['never-labelled', 'a.png has no label file yet'],
      ['labelled-empty', 'b.png is labelled with no shapes, so it counts as a background']
    ])
  })

  test("unverified images are noted once any image is verified; YOLO's files can't say, so they aren't", () => {
    const images = [image('a.png', [], { verified: true }), image('b.png'), image('c.png', [], { format: 'yolo', labelSize: null }), image('d.png', [], { file: null, format: null })]
    expect(of(check(images), 'not-verified').map((i) => i.message)).toEqual(["b.png isn't verified yet"])
    expect(of(check(images.slice(1)), 'not-verified')).toEqual([])
  })
})

test('every check has a stable code, a severity and a title; issues come worst first, the folder before its images', () => {
  for (const [code, { severity, title }] of Object.entries(CHECKS)) {
    expect(code).toMatch(/^[a-z]+(-[a-z]+)*$/)
    expect(['error', 'warning', 'info']).toContain(severity)
    expect(title).toBeTruthy()
  }
  const issues = check([image('a.png', [], { file: null, format: null }), image('b.png', [box('car', 1, 1, W + 9, 9)]), image('c.png', [], { broken: 'bad' })],
    { labels: ['car', 'bus'], orphans: [{ file: 'x.json', names: null }] })
  expect(issues.map((i) => [i.code, i.image])).toEqual([
    ['label-file-unreadable', 'c.png'], ['orphan-label-file', null], ['outside-image', 'b.png'], ['unused-label', null], ['never-labelled', 'a.png']
  ])
})

describe('fixes', () => {
  test('repeated points come out, the outline stays, and a proposal stays a proposal', () => {
    const shapes = [poly('car', [[0, 0], [0, 0], [10, 0], [10, 10], [0, 0]], ai), box('car', 1, 1, 2, 2)]
    const fixed = withoutRepeatedPoints(shapes)!
    expect(fixed[0].points).toEqual([[0, 0], [10, 0], [10, 10]])
    expect(fixed[0].flags).toEqual({ ai: true })
    expect(fixed[1]).toBe(shapes[1])
    expect(withoutRepeatedPoints(fixed)).toBeNull()
  })

  test('exact copies go, keeping the copy with a description or flags', () => {
    const shapes = [box('car', 10, 10, 50, 50), box('car', 10, 10, 50, 50, { flags: { difficult: true } }), box('car', 10, 10, 50, 50), box('bus', 10, 10, 50, 50)]
    expect(withoutDuplicates(shapes)).toEqual([shapes[1], shapes[3]])
    expect(withoutDuplicates([shapes[0], shapes[3]])).toBeNull()
  })

  test('boxes are clipped to the image; one wholly outside is left, as clipping would leave nothing', () => {
    const shapes = [box('car', -20, 10, 50, W), box('car', 500, 500, 600, 600), box('car', 10, 10, 20, 20)]
    const fixed = boxesClipped(shapes, W, H)!
    expect(fixed.map((s) => s.points)).toEqual([[[0, 10], [50, H]], [[500, 500], [600, 600]], [[10, 10], [20, 20]]])
    expect(boxesClipped(fixed, W, H)).toBeNull()
  })

  test('each fixable issue, fixed, is gone when the image is checked again', () => {
    const img = image('a.png', [poly('car', [[0, 0], [10, 0], [10, 10], [10, 10]]), box('car', 50, 50, 90, 90), box('car', 50, 50, 90, 90), box('car', 100, 100, W + 30, 150)])
    for (const issue of check([img]).filter((i) => i.code !== 'labelled-empty')) {
      const shapes = fixIssue(img, issue)
      expect(shapes, issue.code).not.toBeNull()
      expect(of(check([{ ...img, shapes: shapes!, masks: shapes!.map(() => null) }]), issue.code as CheckCode), issue.code).toEqual([])
    }
    expect(fixIssue(img, { severity: 'warning', code: 'near-duplicate', image: 'a.png', shape: 1, message: '' })).toBeNull()
  })

  test("renaming a label lists the images that change; files that can't be read, or are shared, are left alone", () => {
    const images = [
      image('a.png', [box('car', 1, 1, 9, 9), box('bus', 1, 1, 9, 9), box('car', 20, 1, 29, 9, ai)]),
      image('b.png', [box('bus', 1, 1, 9, 9)]),
      image('c.png', [], { broken: 'bad' }),
      image('d.jpg', [box('car', 1, 1, 9, 9)]), image('d.png', [box('car', 1, 1, 9, 9)])
    ]
    const plan = renameLabel(images, 'car', 'auto', [{ file: '/data/d.json', images: ['d.jpg', 'd.png'] }])
    expect(plan.images.map((i) => [i.name, i.renamed, i.shapes.map((s) => s.label)])).toEqual([['a.png', 2, ['auto', 'bus', 'auto']]])
    expect(plan.images[0].shapes[2].flags).toEqual({ ai: true })
    expect(plan.shapes).toBe(2)
    expect(plan.skipped).toEqual([
      { name: 'c.png', reason: "c.json can't be read, so it's left unchanged" },
      { name: 'd.jpg', reason: "d.json holds the labels of d.jpg and d.png, so it's left unchanged" },
      { name: 'd.png', reason: "d.json holds the labels of d.jpg and d.png, so it's left unchanged" }
    ])
    expect(() => renameLabel(images, 'car', ' ')).toThrow(/needs a name/)
  })
})
