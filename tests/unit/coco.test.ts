import { mkdirSync, readFileSync, writeFileSync } from 'fs'
import path from 'path'
import { describe, expect, test, vi } from 'vitest'
import { shapesOf } from '../../src/main/importWorker'
import {
  type CocoAnnotation, type CocoEntry, type CocoImage, type CocoRle, cocoLabels, MASK_PIXELS, matchCocoImages, readCoco, writeCoco
} from '../../src/shared/coco'
import type { DatasetImage, KeypointSchema, MaskBits, PoseInstance } from '../../src/shared/dataset'
import { newShape, parseLabelFile, type Shape } from '../../src/shared/labelfile'
import { circlePolygon } from '../../src/shared/outline'
import { poseInstances } from '../../src/shared/pose'
import { rleArea, rleDecode, rleEncode, rleFrMaskBits, rleFrPoly, rleFrString, rleMerge, rleToBbox, rleToString } from '../../src/shared/rle'

const INFO = { description: 'Seamtag test', version: '3.3.0', year: 2026, date_created: '2026-11-12 00:00:00' }

const image = (name: string, width: number, height: number, shapes: Shape[], masks: (MaskBits | null)[] = []): DatasetImage => ({
  name, width, height, file: name.replace(/\.\w+$/, '.json'), format: 'labelme', shapes, masks: shapes.map((_, i) => masks[i] ?? null), flags: {}, verified: false
})

// Rows of 0 and 1 to a mask shape's pixels with its top-left pixel at x, y
function bits(x: number, y: number, rows: string[]): MaskBits {
  const height = rows.length, width = rows[0].length
  const data = new Uint8Array(width * height)
  rows.forEach((row, j) => { for (let i = 0; i < width; i++) data[j * width + i] = row[i] === '1' ? 1 : 0 })
  return { x, y, width, height, data }
}

const decode = (seg: CocoRle) => rleDecode(rleFrString(seg.counts as string, seg.size[0], seg.size[1]))
const instancesOf = (entries: CocoEntry[], labels: string[] = []) => writeCoco(entries, { task: 'instances', labels, info: INFO })

// The four corners of a box of half-size hw x hh at cx, cy, turned by some degrees
function rotated(cx: number, cy: number, hw: number, hh: number, degrees: number) {
  const t = (degrees * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t)
  return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([dx, dy]) => [cx + dx * c - dy * s, cy + dx * s + dy * c])
}

describe("as labelme 7.9.1's labelme2coco writes the same LabelMe files", () => {
  // tests/fixtures/coco/make_labelme2coco.py runs labelme2coco itself
  const fx = JSON.parse(readFileSync(path.join(__dirname, '../fixtures/coco/labelme2coco.json'), 'utf8'))
  const files = Object.keys(fx.files).sort()
  const entries = files.map((file): CocoEntry => {
    const doc = parseLabelFile(JSON.stringify(fx.files[file]))
    // labelme's rule for a mask with a whole-pixel box: its top-left corner, both corners in
    const masks = doc.shapes.map((s, i) => (fx.masks[file][i] ? bits(s.points[0][0], s.points[0][1], fx.masks[file][i]) : null))
    return { image: image(file.replace('.json', '.png'), doc.imageWidth!, doc.imageHeight!, doc.shapes, masks), fileName: `images/${file.replace('.json', '.png')}` }
  })
  const { dataset: ours, issues } = instancesOf(entries, fx.labels)
  const theirs = fx.coco
  const nameOf = (cats: { id: number; name: string }[], id: number) => cats.find((c) => c.id === id)!.name
  const pairsOf = () => files.map((_, k) => [
    ours.annotations.filter((a) => a.image_id === k + 1),
    theirs.annotations.filter((a: CocoAnnotation) => a.image_id === k),
    entries[k].image
  ] as [CocoAnnotation[], CocoAnnotation[], DatasetImage])
  // The shape type behind each polygon of one of our annotations
  const typesOf = (a: CocoAnnotation): string[] => {
    const r = a.attributes ?? {}
    if (Array.isArray(r.shapes)) return r.shapes.map((s: { shape_type?: string }) => s.shape_type ?? 'polygon')
    return [(r.shape_type as string) ?? 'polygon']
  }

  test('the fixture comes from labelme 7.9.1 and pycocotools 2.0.11', () => {
    expect([fx.labelme, fx.pycocotools]).toEqual(['7.9.1', '2.0.11'])
    expect(issues).toEqual([])
  })

  test('shapes sharing a label and a group_id are one object, in the same order, of the same categories', () => {
    for (const [mine, its] of pairsOf()) {
      expect(mine.map((a) => nameOf(ours.categories, a.category_id))).toEqual(its.map((a) => nameOf(theirs.categories, a.category_id)))
      expect(mine.map((a) => (Array.isArray(a.segmentation) ? a.segmentation.length : 'mask'))).toEqual(
        its.map((a) => (Array.isArray(a.segmentation) ? a.segmentation.length : 'mask')))
    }
  })

  test('polygons, rectangles and rotated boxes are the same numbers; circles the same vertices to 1e-9', () => {
    let circles = 0
    for (const [mine, its] of pairsOf()) {
      mine.forEach((a, k) => {
        if (!Array.isArray(a.segmentation)) return
        const types = typesOf(a)
        a.segmentation.forEach((poly, p) => {
          const want = (its[k].segmentation as number[][])[p]
          expect(poly.length).toBe(want.length)
          if (types[p] !== 'circle') return expect(poly).toEqual(want)
          circles++
          poly.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThan(1e-9))
        })
      })
    }
    expect(circles).toBe(3)
  })

  test('a mask is the same pixels; a mask with a polygon holds both', () => {
    let masks = 0
    for (const [mine, its, im] of pairsOf()) {
      mine.forEach((a, k) => {
        if (Array.isArray(a.segmentation)) return
        masks++
        const ourPixels = decode(a.segmentation), theirPixels = decode(its[k].segmentation as CocoRle)
        if (typesOf(a).length === 1) return expect(ourPixels).toEqual(theirPixels)
        // labelme2coco draws the polygon with Pillow, which takes in its edge pixels too
        expect(ourPixels.every((v, i) => !v || theirPixels[i])).toBe(true)
        const own = decode({ size: [im.height, im.width], counts: rleToString(rleFrMaskBits(im.masks.find((m) => m)!, im.height, im.width)) })
        expect(own.every((v, i) => !v || ourPixels[i])).toBe(true)
        // and the polygon's pixels besides
        expect(ourPixels.reduce((n, v) => n + v, 0)).toBeGreaterThan(own.reduce((n, v) => n + v, 0))
      })
    }
    expect(masks).toBe(2)
  })
})

describe('writing COCO', () => {
  test('ids count from 1, and categories follow the label list, then labels found past it', () => {
    const entries = [
      { image: image('a.png', 40, 30, [newShape('dog', 'polygon', [[1, 1], [9, 1], [5, 8]]), newShape('emu', 'rectangle', [[2, 2], [8, 9]])]), fileName: 'images/a.png' },
      { image: image('b.png', 40, 30, [newShape('cat', 'rectangle', [[2, 2], [8, 9]]), newShape('fox', 'circle', [[20, 15], [24, 15]])]), fileName: 'images/b.png' }
    ]
    const { dataset, issues } = instancesOf(entries, ['cat', 'dog'])
    expect(dataset.images.map((i) => [i.id, i.file_name])).toEqual([[1, 'images/a.png'], [2, 'images/b.png']])
    expect(dataset.annotations.map((a) => [a.id, a.image_id, a.category_id])).toEqual([[1, 1, 2], [2, 1, 3], [3, 2, 1], [4, 2, 4]])
    expect(dataset.categories).toEqual([
      { id: 1, name: 'cat', supercategory: '' }, { id: 2, name: 'dog', supercategory: '' },
      { id: 3, name: 'emu', supercategory: '' }, { id: 4, name: 'fox', supercategory: '' }
    ])
    expect(issues.map((i) => i.message)).toEqual([
      "“emu” isn't in the label list, so it's category 3, after the list",
      "“fox” isn't in the label list, so it's category 4, after the list"
    ])
    expect(cocoLabels(['cat', 'dog'], entries)).toEqual(['cat', 'dog', 'emu', 'fox'])
  })

  test('info and licences are written as given, with no clock read, so the same labels give the same file', () => {
    const entries = [{ image: image('a.png', 40, 30, [newShape('dog', 'polygon', [[1, 1], [9, 1], [5, 8]])]), fileName: 'a.png' }]
    const one = instancesOf(entries).json, two = instancesOf(entries).json
    expect(one).toBe(two)
    const data = JSON.parse(one)
    expect(Object.keys(data)).toEqual(['info', 'licenses', 'images', 'annotations', 'categories'])
    expect(data.info).toEqual({ description: 'Seamtag test', url: '', version: '3.3.0', year: 2026, contributor: '', date_created: '2026-11-12 00:00:00' })
    expect(data.licenses).toEqual([])
  })

  test("areas are pycocotools' pixel counts and polygon boxes their corners'", () => {
    const shapes = [
      newShape('a', 'rectangle', [[10, 10], [20, 20]]),
      newShape('a', 'rectangle', [[10.2, 10.2], [19.7, 19.7]]),
      newShape('a', 'polygon', [[0, 0], [10, 0], [0, 10]])
    ]
    const { dataset } = instancesOf([{ image: image('a.png', 40, 40, shapes), fileName: 'a.png' }])
    expect(dataset.annotations.map((a) => [a.area, a.bbox])).toEqual([
      [100, [10, 10, 10, 10]], [100, [10.2, 10.2, 19.7 - 10.2, 19.7 - 10.2]], [45, [0, 0, 10, 10]]
    ])
    expect(dataset.annotations[0].segmentation).toEqual([[10, 10, 20, 10, 20, 20, 10, 20]])
  })

  test('a mask is compressed runs of the whole image, with its box and area from them, not a crowd', () => {
    const m = bits(5, 3, ['0110', '1111', '0100'])
    const { dataset } = instancesOf([{ image: image('a.png', 12, 8, [newShape('blob', 'mask', [[5, 3], [8, 5]])], [m]), fileName: 'a.png' }])
    const [a] = dataset.annotations
    const want = rleFrMaskBits(m, 8, 12)
    expect(a.segmentation).toEqual({ size: [8, 12], counts: rleToString(want) })
    expect([a.area, a.bbox, a.iscrowd]).toEqual([7, [5, 3, 4, 3], 0])
    expect(a.attributes).toEqual({ shape_type: 'mask', points: [[5, 3], [8, 5]] })
  })

  test('an object with a mask in it is one mask: the mask and its other shapes filled as pycocotools fills polygons', () => {
    const m = bits(1, 1, ['11', '11'])
    const shapes = [newShape('blob', 'mask', [[1, 1], [2, 2]], { group_id: 4 }), newShape('blob', 'polygon', [[5, 5], [9, 5], [9, 9]], { group_id: 4 })]
    const { dataset } = instancesOf([{ image: image('a.png', 12, 12, shapes, [m]), fileName: 'a.png' }])
    const [a] = dataset.annotations
    const union = rleMerge([rleFrMaskBits(m, 12, 12), rleFrPoly([5, 5, 9, 5, 9, 9], 12, 12)])
    expect((a.segmentation as CocoRle).counts).toBe(rleToString(union))
    expect([a.area, a.bbox]).toEqual([rleArea(union), rleToBbox(union)])
    expect((a.attributes!.shapes as unknown[]).length).toBe(2)
  })

  test('shapes with no inside, too few points, no radius or an unreadable mask are left out, each with a reason', () => {
    const shapes = [
      newShape('rope', 'line', [[1, 1], [9, 9]]), newShape('path', 'linestrip', [[1, 1], [9, 9], [3, 7]]), newShape('nose', 'point', [[4, 4]]),
      newShape('dots', 'points', [[1, 1], [2, 2], [3, 3]]), newShape('bad', 'polygon', [[1, 1], [9, 9]]), newShape('ball', 'circle', [[5, 5], [5, 5]]),
      newShape('blob', 'mask', [[1, 1], [4, 4]]), newShape('box', 'rectangle', [[1, 1]])
    ]
    const { dataset, issues } = instancesOf([{ image: image('sub/a.png', 20, 20, shapes), fileName: 'a.png' }])
    expect(dataset.annotations).toEqual([])
    expect(issues.map((i) => [i.severity, i.code, i.image, i.shape])).toEqual([
      ['warning', 'coco-no-inside', 'sub/a.png', 0], ['warning', 'coco-no-inside', 'sub/a.png', 1], ['warning', 'coco-no-inside', 'sub/a.png', 2],
      ['warning', 'coco-no-inside', 'sub/a.png', 3], ['warning', 'coco-too-few-points', 'sub/a.png', 4], ['warning', 'coco-no-radius', 'sub/a.png', 5],
      ['warning', 'coco-mask-unreadable', 'sub/a.png', 6], ['warning', 'coco-bad-shape', 'sub/a.png', 7]
    ])
    expect(issues[0].message).toBe("sub/a.png: shape 1, a line labelled “rope”, has no inside for COCO to hold, so it's left out")
    expect(issues[4].message).toBe("sub/a.png: shape 5, a polygon labelled “bad”, has 2 points, and COCO needs 3, so it's left out")
    expect(issues[7].message).toBe("sub/a.png: shape 8, a rectangle labelled “box”, has 1 point, not 2, so it's left out")
  })

  test("a mask without exactly two corners is left out with a reason, as labelme2coco can't place it either", () => {
    // labelme's label reader takes a mask of one point
    const [one] = parseLabelFile(JSON.stringify({ shapes: [{ label: 'blob', points: [[2, 2]], shape_type: 'mask', mask: 'iVBORw0KGgo=' }] })).shapes
    const m = bits(2, 2, ['11', '11'])
    const shapes = [one, newShape('blob', 'mask', [[2, 2], [3, 3], [4, 4]]), newShape('blob', 'mask', [[2, 2], [3, 3]])]
    const { dataset, issues } = instancesOf([{ image: image('a.png', 10, 8, shapes, [m, m, m]), fileName: 'a.png' }], ['blob'])
    expect(dataset.annotations.map((a) => [a.area, a.attributes])).toEqual([[4, { shape_type: 'mask', points: [[2, 2], [3, 3]] }]])
    expect(issues.map((i) => [i.severity, i.code, i.shape, i.message])).toEqual([
      ['warning', 'coco-bad-shape', 0, "a.png: shape 1, a mask labelled “blob”, has 1 point, not 2, so it's left out"],
      ['warning', 'coco-bad-shape', 1, "a.png: shape 2, a mask labelled “blob”, has 3 points, not 2, so it's left out"]
    ])
  })

  test('points past the image are moved onto its edges, and mask pixels past it left out, each noted', () => {
    const m = bits(17, 18, ['1111', '1111'])
    const shapes = [newShape('cat', 'polygon', [[-5, 10], [12, -3], [25, 15]]), newShape('blob', 'mask', [[17, 18], [20, 19]], { group_id: 2 })]
    const { dataset, issues } = instancesOf([{ image: image('a.png', 20, 19, shapes, [null, m]), fileName: 'a.png' }], ['cat', 'blob'])
    expect(dataset.annotations[0].segmentation).toEqual([[0, 10, 12, 0, 20, 15]])
    expect(dataset.annotations[0].bbox).toEqual([0, 0, 20, 15])
    expect(dataset.annotations[1].area).toBe(3)
    expect(issues.map((i) => [i.severity, i.code, i.shape])).toEqual([['info', 'coco-outside', 0], ['info', 'coco-outside', 1]])
    expect(issues[1].message).toBe("a.png: shape 2, a mask labelled “blob”, has 5 pixels past the image, so they're left out")
  })

  test('an object that covers no whole pixel is left out', () => {
    const shapes = [newShape('dust', 'polygon', [[3.1, 3.1], [3.9, 3.2], [3.5, 3.4]]), newShape('cat', 'polygon', [[-9, 2], [-3, 2], [-6, 8]])]
    const { dataset, issues } = instancesOf([{ image: image('a.png', 20, 20, shapes), fileName: 'a.png' }])
    expect(dataset.annotations).toEqual([])
    expect(issues.filter((i) => i.code === 'coco-no-pixels').map((i) => i.message)).toEqual([
      "a.png: “dust” (shape 1) covers no whole pixel, so it's left out", "a.png: “cat” (shape 2) covers no whole pixel, so it's left out"
    ])
  })

  test('images whose label file or pixels can\'t be read are left out with an error', () => {
    const broken = { ...image('a.png', 20, 20, []), broken: "It isn't valid JSON" }
    const unread = { ...image('b.png', 0, 0, []), unreadable: 'Input file is missing' }
    const { dataset, issues } = instancesOf([{ image: broken, fileName: 'a.png' }, { image: unread, fileName: 'b.png' }, { image: image('c.png', 5, 5, []), fileName: 'c.png' }])
    expect(dataset.images).toEqual([{ id: 1, width: 5, height: 5, file_name: 'c.png' }])
    expect(issues.map((i) => [i.severity, i.message])).toEqual([
      ['error', "a.png: its label file can't be read (It isn't valid JSON), so the image is left out"],
      ['error', "b.png can't be read (Input file is missing), so it's left out"]
    ])
  })
})

// A posed person: a box, a polygon of another, and one made of its points alone
const SCHEMA: KeypointSchema = {
  classes: { person: ['nose', 'left_eye', 'right_eye', 'left_ear'] },
  skeleton: { person: [['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'left_ear']] }
}
function posed() {
  const shapes = [
    newShape('person', 'rectangle', [[10, 10], [40, 60]], { group_id: 7 }),
    newShape('nose', 'point', [[20, 15]], { group_id: 7 }),
    newShape('left_eye', 'point', [[18.5, 13.25]], { group_id: 7, flags: { difficult: true } }),
    newShape('left_ear', 'point', [[16, 14]], { group_id: 7 }),
    newShape('person', 'polygon', [[50, 10], [70, 12], [66, 40], [52, 38]], { group_id: 8 }),
    newShape('nose', 'point', [[60, 20]], { group_id: 8 }),
    newShape('right_eye', 'point', [[95, 20]], { group_id: 8 }),
    newShape('nose', 'point', [[30, 70]], { group_id: 9 }),
    newShape('left_eye', 'point', [[38, 77]], { group_id: 9 })
  ]
  const poses: PoseInstance[] = [
    { label: 'person', group: 7, box: [10, 10, 40, 60], keypoints: [[20, 15, 2], [18.5, 13.25, 1], null, [16, 14, 2]], shapes: [0, 1, 2, 3] },
    { label: 'person', group: 8, box: [50, 10, 70, 40], keypoints: [[60, 20, 2], null, [95, 20, 2], null], shapes: [4, 5, 6] },
    { label: 'person', group: 9, box: [30, 70, 38, 77], keypoints: [[30, 70, 2], [38, 77, 2], null, null], shapes: [7, 8] }
  ]
  return { image: image('people.png', 90, 80, shapes), fileName: 'images/people.png', poses }
}

describe('writing COCO keypoints', () => {
  const entry = posed()
  const { dataset, issues } = writeCoco([entry], { task: 'keypoints', labels: ['car', 'person'], info: INFO, schema: SCHEMA })

  test("categories keep the instances file's ids, with keypoint names and a skeleton counted from 1", () => {
    expect(dataset.categories).toEqual([{
      id: 2, name: 'person', supercategory: '', keypoints: ['nose', 'left_eye', 'right_eye', 'left_ear'], skeleton: [[1, 2], [1, 3], [2, 4]]
    }])
  })

  test('keypoints are triplets with 0, 0, 0 for those not labelled, and num_keypoints on every annotation', () => {
    expect(dataset.annotations.map((a) => [a.keypoints, a.num_keypoints])).toEqual([
      [[20, 15, 2, 18.5, 13.25, 1, 0, 0, 0, 16, 14, 2], 3],
      [[60, 20, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1],
      [[30, 70, 2, 38, 77, 2, 0, 0, 0, 0, 0, 0], 2]
    ])
    expect(issues.map((i) => i.message)).toEqual(["people.png: keypoint “right_eye” of “person” (group 8) lies past the image, so it's written as not labelled"])
  })

  test('the area is the outline\'s pixel count when the object has one, else its box\'s width x height', () => {
    const [box, polygon, points] = dataset.annotations
    expect([box.area, box.bbox, box.segmentation]).toEqual([1500, [10, 10, 30, 50], [[10, 10, 40, 10, 40, 60, 10, 60]]])
    expect(polygon.area).toBe(rleArea(rleFrPoly([50, 10, 70, 12, 66, 40, 52, 38], 80, 90)))
    expect(polygon.bbox).toEqual([50, 10, 20, 30])
    expect([points.area, points.bbox, points.segmentation]).toEqual([56, [30, 70, 8, 7], [[30, 70, 38, 70, 38, 77, 30, 77]]])
  })

  test("attributes keep the group, and the outline's shapes as the instances file keeps them", () => {
    // A rectangle needs only its type; a polygon nothing; a posed object of points alone has its box, marked as one
    expect(dataset.annotations.map((a) => a.attributes)).toEqual([{ group_id: 7, shape_type: 'rectangle' }, { group_id: 8 }, { group_id: 9, shape_type: 'rectangle' }])
  })

  test("the instances file leaves a posed object's points to the keypoints file", () => {
    const { dataset: objects, issues: notes } = instancesOf([entry], ['car', 'person'])
    expect(objects.annotations.map((a) => a.category_id)).toEqual([2, 2])
    expect(notes).toEqual([])
  })

  test('posed objects of a class with no keypoint names, or the wrong number of keypoints, are left out', () => {
    const bad = { ...entry, poses: [{ ...entry.poses[0], label: 'dog' }, { ...entry.poses[1], keypoints: [null] }] }
    const out = writeCoco([bad], { task: 'keypoints', labels: ['person'], info: INFO, schema: SCHEMA })
    expect(out.dataset.annotations).toEqual([])
    expect(out.issues.map((i) => i.message)).toEqual([
      "people.png: “dog” (group 7) has no keypoint names in the keypoints file, so it's left out",
      "people.png: “person” (group 8) has 1 keypoint where the keypoints file names 4, so it's left out"
    ])
  })
})

// Every kind of shape Seamtag keeps, and what makes them hard to give back
function everything() {
  const m1 = bits(44, 2, ['1101100000', '1111111111', '0011111100', '0001111000', '0000110000', '1000000001', '1100000011', '1111111111'])
  const m2 = bits(30, 31, ['01110', '11111', '01010', '00100', '11011', '10001'])
  const m3 = bits(58, 40, Array.from({ length: 11 }, (_, j) => (j < 8 ? '1111100000' : '0000000000')))
  const m4 = bits(2, 30, ['111', '101', '111'])
  const m5 = bits(5, 5, ['1111', '1111', '1111'])
  const shapes: Shape[] = [
    newShape('cat', 'polygon', [[5.5, 4.25], [25, 6], [20.75, 24.5], [7, 20]], { description: 'tabby', flags: { difficult: true, truncated: false } }),
    newShape('dog', 'rectangle', [[40, 30], [30.5, 10.25]]),
    newShape('dog', 'rectangle', [[2, 2], [9.5, 8]]),
    newShape('ball', 'circle', [[50, 38], [54, 41]]),
    newShape('plate', 'oriented_rectangle', rotated(18, 33, 6.5, 4, 20)),
    newShape('dog', 'polygon', [[2, 40], [12, 40], [8, 46]], { group_id: 1 }),
    newShape('cat', 'polygon', [[-5, 10], [12, -3], [15, 15]]),
    newShape('dog', 'rectangle', [[14, 41], [20, 46]], { group_id: 1 }),
    newShape('rope', 'line', [[1, 1], [9, 9]]),
    newShape('blob', 'mask', [[44, 2], [53, 9]]),
    newShape('dog', 'circle', [[25, 44], [27, 44]], { group_id: 1 }),
    newShape('blob', 'mask', [[30.4, 30.6], [35.2, 34.9]]),
    newShape('blob', 'mask', [[58, 40], [67, 50]], { group_id: 6 }),
    newShape('blob', 'mask', [[2, 30], [4, 32]], { group_id: 3 }),
    newShape('nose', 'point', [[4, 4]]),
    newShape('blob', 'polygon', [[6, 30], [12, 30], [9, 36]], { group_id: 3 }),
    newShape('crowd', 'mask', [[5, 5], [8, 7]], { other: { iscrowd: 1 } }),
    newShape('cat', 'polygon', [[30, 2], [38, 2], [34, 8]], { locked: true, flags: { ai: true }, other: { score: 0.93, source: 'sam' } }),
    newShape('plate', 'oriented_rectangle', rotated(60, 20, 8, 3, -35).map(([x, y]) => [x + 3, y]), { group_id: 9 })
  ]
  const masks = [null, null, null, null, null, null, null, null, null, m1, null, m2, m3, m4, null, null, m5, null, null]
  return { image: image('all.png', 64, 48, shapes, masks) }
}

describe('reading back what Seamtag wrote', () => {
  test('gives back the same shapes in the same order, mask pixels and all', () => {
    const { image: all } = everything()
    const { json } = instancesOf([{ image: all, fileName: 'images/all.png' }], ['cat', 'dog'])
    const read = readCoco(json, 'instances_default.json')
    expect(read.issues.map((i) => i.code)).toEqual(['coco-crowd'])
    const got = read.results.get(1)!
    // All but the line and the point, which COCO can't hold
    const kept = all.shapes.map((_, i) => i).filter((i) => all.shapes[i].shape_type !== 'line' && all.shapes[i].shape_type !== 'point')
    expect(got.shapes).toEqual(kept.map((i) => all.shapes[i]))
    expect(got.masks).toEqual(kept.map((i) => all.masks[i]))
    expect(got.poses).toEqual([])
  })

  test('posed objects come back as their outlines, boxes and named keypoints, hidden ones marked 1', () => {
    const entry = posed()
    const { json } = writeCoco([entry], { task: 'keypoints', labels: ['person'], info: INFO, schema: SCHEMA })
    const read = readCoco(json, 'person_keypoints_default.json')
    expect(read.results.get(1)!.poses).toEqual([
      { label: 'person', group: 7, box: [10, 10, 40, 60], names: SCHEMA.classes.person, keypoints: [[20, 15, 2], [18.5, 13.25, 1], null, [16, 14, 2]], outlined: true },
      { label: 'person', group: 8, box: [50, 10, 70, 40], names: SCHEMA.classes.person, keypoints: [[60, 20, 2], null, null, null], outlined: true },
      // Made of its points alone, its outline is the box Seamtag wrote, a rectangle
      { label: 'person', group: 9, box: [30, 70, 38, 77], names: SCHEMA.classes.person, keypoints: [[30, 70, 2], [38, 77, 2], null, null], outlined: true }
    ])
    // The rectangle of group 7 and the polygon of group 8, as they were, and group 9's box
    expect(read.results.get(1)!.shapes).toEqual([entry.image.shapes[0], entry.image.shapes[4], newShape('person', 'rectangle', [[30, 70], [38, 77]], { group_id: 9 })])
  })

  test("a posed object's polygon lying on its box comes back as that polygon; another tool's outline that's only the box is the box", () => {
    const shapes = [newShape('person', 'polygon', [[100, 10], [140, 10], [140, 40], [100, 40]], { group_id: 5 }), newShape('nose', 'point', [[110, 20]], { group_id: 5 })]
    const poses: PoseInstance[] = [{ label: 'person', group: 5, box: [100, 10, 140, 40], keypoints: [[110, 20, 2], null, null, null], shapes: [0, 1] }]
    const { json } = writeCoco([{ image: image('d.png', 160, 60, shapes), fileName: 'd.png', poses }], { task: 'keypoints', labels: ['person'], info: INFO, schema: SCHEMA })
    const ours = readCoco(json, 'person_keypoints.json').results.get(1)!
    expect(ours.shapes).toEqual([shapes[0]])
    expect(ours.poses.map((p) => [p.group, p.outlined])).toEqual([[5, true]])
    // With CVAT's attributes in place of Seamtag's, the same segmentation is only a box
    const theirs = JSON.parse(json)
    theirs.annotations[0].attributes = { occluded: false }
    const got = readCoco(JSON.stringify(theirs), 'person_keypoints.json').results.get(1)!
    expect(got.shapes).toEqual([])
    expect(got.poses.map((p) => [p.box, p.outlined])).toEqual([[[100, 10, 140, 40], false]])
  })

  test("a keypoints file alone gives back its posed objects' outlines exactly: circles, rotated boxes past the image, masks", () => {
    const m = bits(60, 50, ['0110', '1111', '0110'])
    const shapes = [
      newShape('person', 'circle', [[20, 20], [28, 20]], { group_id: 1 }),
      newShape('nose', 'point', [[20, 16]], { group_id: 1 }),
      newShape('person', 'oriented_rectangle', rotated(84, 30, 10, 4, 30), { group_id: 2 }),
      newShape('nose', 'point', [[80, 30]], { group_id: 2 }),
      newShape('person', 'mask', [[60.4, 49.8], [63.2, 52.1]], { group_id: 3, description: 'small' }),
      newShape('left_eye', 'point', [[61, 50]], { group_id: 3, flags: { difficult: true } }),
      newShape('person', 'rectangle', [[40, 70], [30, 55]], { group_id: 4, flags: { truncated: true } }),
      newShape('nose', 'point', [[35, 60]], { group_id: 4 }),
      // People with no keypoints, in a group and not
      newShape('person', 'polygon', [[2, 70], [12, 70], [7, 78]], { group_id: 5 }),
      newShape('person', 'rectangle', [[50, 70], [58, 79]])
    ]
    const none = [null, null, null, null]
    const poses: PoseInstance[] = [
      { label: 'person', group: 1, box: [12, 12, 28, 28], keypoints: [[20, 16, 2], null, null, null], shapes: [0, 1] },
      { label: 'person', group: 2, box: [74, 20, 90, 40], keypoints: [[80, 30, 2], null, null, null], shapes: [2, 3] },
      { label: 'person', group: 3, box: [60, 50, 64, 53], keypoints: [null, [61, 50, 1], null, null], shapes: [4, 5] },
      { label: 'person', group: 4, box: [30, 55, 40, 70], keypoints: [[35, 60, 2], null, null, null], shapes: [6, 7] },
      { label: 'person', group: 5, box: [2, 70, 12, 78], keypoints: none, shapes: [8] },
      { label: 'person', group: null, box: [50, 70, 58, 79], keypoints: none, shapes: [9] }
    ]
    const entry = { image: image('p.png', 90, 80, shapes, [null, null, null, null, m]), fileName: 'p.png', poses }
    const { json, issues } = writeCoco([entry], { task: 'keypoints', labels: ['person'], info: INFO, schema: SCHEMA })
    expect(issues).toEqual([])
    const read = readCoco(json, 'person_keypoints.json'), got = read.results.get(1)!
    expect(got.shapes).toEqual([shapes[0], shapes[2], shapes[4], shapes[6], shapes[8], shapes[9]])
    expect(got.masks).toEqual([null, null, m, null, null, null])
    expect(got.poses.map((p) => [p.group, p.outlined, p.keypoints])).toEqual(poses.slice(0, 4).map((p) => [p.group, true, p.keypoints]))
    expect(read.issues.map((i) => i.code)).toEqual(['coco-unlabelled-keypoints', 'coco-no-keypoints'])
  })
})

describe('the order of shapes', () => {
  // A dog in two parts with a cat between them
  const shapes = [
    newShape('dog', 'polygon', [[1, 1], [9, 1], [5, 8]], { group_id: 1 }), newShape('cat', 'polygon', [[20, 1], [29, 1], [25, 8]]),
    newShape('dog', 'rectangle', [[1, 20], [9, 28]], { group_id: 1 })
  ]
  const { dataset, json } = instancesOf([{ image: image('a.png', 40, 40, shapes), fileName: 'a.png' }])

  test("an object is written whole where its first shape was, and each shape's place comes back with it", () => {
    expect(dataset.annotations.map((a) => a.attributes)).toEqual([{ group_id: 1, shapes: [{ index: 0 }, { shape_type: 'rectangle', index: 2 }] }, { index: 1 }])
    expect(readCoco(json, 'a.json').results.get(1)!.shapes).toEqual(shapes)
  })

  test('shapes already in the order of their objects keep no places', () => {
    const { dataset: kept } = instancesOf([{ image: image('a.png', 40, 40, [shapes[0], shapes[2], shapes[1]]), fileName: 'a.png' }])
    expect(kept.annotations.map((a) => a.attributes)).toEqual([{ group_id: 1, shapes: [{}, { shape_type: 'rectangle' }] }, undefined])
  })

  test("other tools' annotations of the image go after Seamtag's own", () => {
    const data = JSON.parse(json)
    data.annotations.unshift({ id: 9, image_id: 1, category_id: 2, segmentation: [[30, 30, 35, 30, 33, 34]], area: 9, bbox: [30, 30, 5, 4], iscrowd: 0 })
    expect(readCoco(JSON.stringify(data), 'a.json').results.get(1)!.shapes).toEqual([...shapes, newShape('cat', 'polygon', [[30, 30], [35, 30], [33, 34]])])
  })
})

// A seeded random image of shapes of every kind with an inside: some grouped, some partly past
// the image, masks with boxes past it or at fractions of a pixel. Mask pixels stay inside, so
// nothing is cut and every shape can come back as it was.
function randomImage(seed: number) {
  let state = seed
  const random = () => (state = (state * 16807) % 2147483647) / 2147483647
  const int = (a: number, b: number) => a + Math.floor(random() * (b - a + 1))
  const w = int(20, 80), h = int(20, 80)
  const shapes: Shape[] = [], masks: (MaskBits | null)[] = []
  for (let k = int(1, 9); k > 0; k--) {
    const label = random() < 0.5 ? 'a' : 'b'
    const extra: Partial<Shape> = { group_id: [null, null, 1, 2][int(0, 3)] }
    if (random() < 0.2) extra.description = `note ${k}`
    if (random() < 0.2) extra.flags = { difficult: random() < 0.5 }
    if (random() < 0.1) extra.locked = true
    if (random() < 0.1) extra.other = { score: Math.round(random() * 100) / 100 }
    const cx = random() * w, cy = random() * h
    const kind = int(0, 4)
    let bitsOf: MaskBits | null = null
    if (kind === 0) {
      const n = int(3, 8), r = 2 + random() * Math.min(w, h) / 2
      const angles = Array.from({ length: n }, () => random() * 2 * Math.PI).sort((p, q) => p - q)
      shapes.push(newShape(label, 'polygon', angles.map((t) => [cx + r * (0.4 + random() * 0.6) * Math.cos(t), cy + r * (0.4 + random() * 0.6) * Math.sin(t)]), extra))
    } else if (kind === 1) {
      const hw = 1 + random() * w / 3, hh = 1 + random() * h / 3
      const corners = [[cx - hw, cy - hh], [cx + hw, cy + hh]]
      shapes.push(newShape(label, 'rectangle', random() < 0.5 ? corners : [corners[1], corners[0]], extra))
    } else if (kind === 2) {
      const r = 2 + random() * Math.min(w, h) / 2, t = random() * 2 * Math.PI
      shapes.push(newShape(label, 'circle', [[cx, cy], [cx + r * Math.cos(t), cy + r * Math.sin(t)]], extra))
    } else if (kind === 3) {
      shapes.push(newShape(label, 'oriented_rectangle', rotated(cx, cy, 1.5 + random() * w / 4, 1.5 + random() * h / 4, random() * 180), extra))
    } else {
      const bw = int(2, 12), bh = int(2, 12), x = int(-3, w - 1), y = int(-3, h - 1)
      const data = new Uint8Array(bw * bh)
      for (let j = 0; j < bh; j++) for (let i = 0; i < bw; i++) data[j * bw + i] = x + i >= 0 && y + j >= 0 && x + i < w && y + j < h && random() < 0.6 ? 1 : 0
      // One pixel at least, inside the image
      data[(Math.max(0, -y)) * bw + Math.max(0, -x)] = 1
      bitsOf = { x, y, width: bw, height: bh, data }
      const fraction = random() < 0.3
      const points = fraction ? [[x + 0.3, y - 0.2], [x + bw - 1.4, y + bh - 0.6]] : [[x, y], [x + bw - 1, y + bh - 1]]
      if (extra.group_id === null && random() < 0.2) extra.other = { iscrowd: 1 }
      shapes.push(newShape(label, 'mask', points, extra))
    }
    masks.push(bitsOf)
  }
  return image(`random/${seed}.png`, w, h, shapes, masks)
}

test('random images of every closed shape come back from COCO as they went in, in their order', () => {
  let checked = 0
  for (let seed = 1; seed <= 300; seed++) {
    const im = randomImage(seed)
    const { json, issues } = instancesOf([{ image: im, fileName: im.name }], ['a', 'b'])
    // A thin sliver can cover no whole pixel and is left out; such images prove nothing here
    if (issues.some((i) => i.code === 'coco-no-pixels')) continue
    const read = readCoco(json, 'random.json'), got = read.results.get(1)!
    expect(got.shapes, `seed ${seed}`).toEqual(im.shapes)
    expect(got.masks, `seed ${seed}`).toEqual(im.masks)
    // Read without making masks' pixels, all is the same but the pixels, which are counted
    const counted = readCoco(json, 'random.json', { pixels: false })
    expect(counted.issues, `seed ${seed}`).toEqual(read.issues)
    expect(counted.results.get(1)!.shapes, `seed ${seed}`).toEqual(im.shapes)
    expect(counted.results.get(1)!.masks, `seed ${seed}`).toEqual(im.masks.map(() => null))
    expect(counted.results.get(1)!.pixels, `seed ${seed}`).toBe(im.masks.reduce((n, m) => n + (m ? m.width * m.height : 0), 0))
    expect(got.pixels).toBe(counted.results.get(1)!.pixels)
    checked++
  }
  expect(checked).toBeGreaterThan(250)
})

describe("Seamtag's attributes", () => {
  test("a segmentation another tool changed counts over them, and a rounding of it doesn't", () => {
    const m = bits(2, 2, ['1111', '1111', '1111'])
    const shapes = [newShape('ball', 'circle', [[20, 20], [25, 20]]), newShape('blob', 'mask', [[2, 2], [5, 4]])]
    const { json } = instancesOf([{ image: image('a.png', 100, 100, shapes, [null, m]), fileName: 'a.png' }])
    const data = JSON.parse(json)
    // The circle is moved, and the mask gains a pixel past its box at x 9, y 9
    data.annotations[0].segmentation = [[70, 70, 90, 70, 90, 90, 70, 90]]
    const pixels = rleDecode(rleFrString(data.annotations[1].segmentation.counts, 100, 100))
    pixels[9 * 100 + 9] = 1
    data.annotations[1].segmentation.counts = rleToString(rleEncode(pixels, 100, 100))
    const read = readCoco(JSON.stringify(data), 'a.json')
    expect(read.results.get(1)!.shapes).toEqual([newShape('ball', 'polygon', [[70, 70], [90, 70], [90, 90], [70, 90]]), newShape('blob', 'mask', [[2, 2], [9, 9]])])
    expect(read.issues.map((i) => i.message)).toEqual([
      "a.png: annotation 1 (“ball”) carries Seamtag's attributes, but its polygon differs from the shape they give, so it's imported from its segmentation",
      "a.png: annotation 2 (“blob”) carries Seamtag's attributes, but its mask differs from the shapes they give, so it's imported from its segmentation"
    ])
    // Rounded to two decimals, as some tools write them, the circle's points are still the circle's
    const rounded = JSON.parse(json)
    rounded.annotations[0].segmentation = rounded.annotations[0].segmentation.map((p: number[]) => p.map((v) => Math.round(v * 100) / 100))
    expect(rounded.annotations[0].segmentation).not.toEqual(JSON.parse(json).annotations[0].segmentation)
    expect(readCoco(JSON.stringify(rounded), 'a.json').results.get(1)!.shapes).toEqual(shapes)
  })

  test('a mask far bigger than the image is read from the segmentation, not held in memory', () => {
    const mask = { size: [8, 10], counts: rleToString({ h: 8, w: 10, counts: [11, 3, 66] }) }
    // 300000 x 300000 pixels in runs of at most 2^32 - 1, as COCO's counts are
    const huge = rleToString({ h: 3e5, w: 3e5, counts: [...Array(22).fill(4e9), 2e9] })
    const text = JSON.stringify({
      images: [{ id: 1, width: 10, height: 8, file_name: 'a.jpg' }], categories: [{ id: 1, name: 'cat' }],
      annotations: [
        { id: 1, image_id: 1, category_id: 1, segmentation: mask, bbox: [1, 3, 1, 3], iscrowd: 0, attributes: { shape_type: 'mask', points: [[0, 0], [3e5, 3e5]] } },
        { id: 2, image_id: 1, category_id: 1, segmentation: mask, bbox: [1, 3, 1, 3], iscrowd: 0, attributes: { shape_type: 'mask', points: [[1, 3], [1, 5]], mask: { size: [3e5, 3e5], counts: huge } } },
        { id: 3, image_id: 1, category_id: 1, segmentation: mask, bbox: [1, 3, 1, 3], iscrowd: 0, attributes: { shape_type: 'mask', points: [[0, 0], [20, 3]] } },
        { id: 4, image_id: 1, category_id: 1, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3], iscrowd: 0 }
      ]
    })
    const read = readCoco(text, 'x.json')
    const fromMask = newShape('cat', 'mask', [[1, 3], [1, 5]])
    expect(read.results.get(1)!.shapes).toEqual([fromMask, fromMask, fromMask, newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]])])
    expect(read.results.get(1)!.masks.slice(0, 3)).toEqual(Array(3).fill({ x: 1, y: 3, width: 1, height: 3, data: Uint8Array.from([1, 1, 1]) }))
    expect(read.issues.map((i) => i.message)).toEqual([
      "a.jpg: annotation 1 (“cat”) carries Seamtag's attributes, but shape 1's mask is 300001 x 300001 pixels, too big for the 10 x 8 image, so it's imported from its segmentation",
      "a.jpg: annotation 2 (“cat”) carries Seamtag's attributes, but shape 1's mask is 300000 x 300000 pixels, too big for the 10 x 8 image, so it's imported from its segmentation",
      "a.jpg: annotation 3 (“cat”) carries Seamtag's attributes, but shape 1's mask is 21 x 4 pixels, too big for the 10 x 8 image, so it's imported from its segmentation"
    ])
  })

  test("a crowd mark comes back on the shapes that carried it, or on all of an object's shapes when another tool set it", () => {
    const shapes = [
      newShape('cat', 'polygon', [[1, 1], [6, 1], [3, 5]], { group_id: 2, other: { iscrowd: 1 } }), newShape('cat', 'polygon', [[5, 5], [9, 5], [7, 7]], { group_id: 2 }),
      newShape('dog', 'polygon', [[1, 1], [8, 1], [4, 7]], { description: 'brown' })
    ]
    const { json } = instancesOf([{ image: image('a.png', 10, 8, shapes), fileName: 'a.png' }])
    expect(readCoco(json, 'a.json').results.get(1)!.shapes).toEqual(shapes)
    const data = JSON.parse(json)
    data.annotations[1].iscrowd = 1
    expect(readCoco(JSON.stringify(data), 'a.json').results.get(1)!.shapes[2]).toEqual({ ...shapes[2], other: { iscrowd: 1 } })
  })
})

// A COCO file as other tools write them
function coco(annotations: Record<string, unknown>[], extra: Record<string, unknown> = {}) {
  return JSON.stringify({
    images: [{ id: 0, width: 10, height: 8, file_name: 'train/a.jpg' }, { id: 7, width: 20, height: 20, file_name: 'b.jpg' }],
    categories: [{ id: 0, name: 'background', supercategory: '' }, { id: 3, name: 'cat', supercategory: 'animal' }, { id: 9, name: 'person', keypoints: ['nose', 'eye'], skeleton: [[1, 2]] }],
    annotations, ...extra
  })
}

describe('reading COCO from other tools', () => {
  test("pycocotools' compressed masks and CVAT's lists of runs become mask shapes with their pixels", () => {
    const m = new Uint8Array(80)
    for (const i of [11, 12, 13, 19, 20, 21, 30]) m[i] = 1
    const runs = [11, 3, 5, 3, 8, 1, 49]
    const text = coco([
      { id: 1, image_id: 0, category_id: 3, segmentation: { size: [8, 10], counts: rleToString({ h: 8, w: 10, counts: runs }) }, area: 7, bbox: [1, 3, 3, 4], iscrowd: 0 },
      { id: 2, image_id: 0, category_id: 3, segmentation: { size: [8, 10], counts: runs }, area: 7, bbox: [1, 3, 3, 4], iscrowd: 1, attributes: { occluded: false, rotation: 0 } }
    ])
    const read = readCoco(text, 'instances.json')
    const got = read.results.get(0)!
    expect(got.shapes).toEqual([
      newShape('cat', 'mask', [[1, 3], [3, 6]]),
      newShape('cat', 'mask', [[1, 3], [3, 6]], { other: { iscrowd: 1 } })
    ])
    // Columns of 8 from the left: pixels 11-13 are x 1, y 3-5; 19-21 x 2, y 3-5; 30 x 3, y 6
    const want = { x: 1, y: 3, width: 3, height: 4, data: Uint8Array.from([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]) }
    expect(got.masks).toEqual([want, want])
    expect(read.issues.map((i) => [i.severity, i.code])).toEqual([['info', 'coco-crowd']])
  })

  test('polygons become polygons, an object of several sharing a new group_id; a box alone becomes a rectangle', () => {
    const text = coco([
      { id: 1, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 4, 3], iscrowd: 0 },
      { id: 2, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4], [10, 10, 15, 10, 12, 18]], area: 9, bbox: [1, 1, 14, 17], iscrowd: 0 },
      { id: 3, image_id: 7, category_id: 3, segmentation: [], area: 12, bbox: [2, 3, 4, 3], iscrowd: 0 },
      { id: 4, image_id: 7, category_id: 3, segmentation: [[]], area: 12, bbox: [2.5, 3, 4, 3], iscrowd: 0 },
      { id: 5, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 4, 3], iscrowd: 0, attributes: { group_id: 4 } }
    ])
    const read = readCoco(text, 'instances.json')
    expect(read.results.get(7)!.shapes).toEqual([
      newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]]),
      newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]], { group_id: 5 }),
      newShape('cat', 'polygon', [[10, 10], [15, 10], [12, 18]], { group_id: 5 }),
      newShape('cat', 'rectangle', [[2, 3], [6, 6]]),
      newShape('cat', 'rectangle', [[2.5, 3], [6.5, 6]]),
      newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]], { group_id: 4 })
    ])
    expect(read.issues.map((i) => i.message)).toEqual(['instances.json: 2 annotations hold only a box, so they become rectangles'])
  })

  test("CVAT's rotated boxes, an upright box and how far it's turned in attributes.rotation, become the rotated boxes CVAT shows", () => {
    const box = (id: number, rotation: unknown) => ({ id, image_id: 7, category_id: 3, segmentation: [], area: 32, bbox: [2, 3, 8, 4], iscrowd: 0, attributes: { occluded: false, rotation } })
    const read = readCoco(coco([box(1, 30), box(2, 90), box(3, -90), box(4, 0), box(5, 180), box(6, 360.0001), box(7, '30')]), 'instances_default.json')
    const [turned, quarter, back, ...upright] = read.results.get(7)!.shapes
    // Turned clockwise about the box's centre, as CVAT's RotatedBoxesToPolygons turns its corners
    expect(turned.shape_type).toBe('oriented_rectangle')
    turned.points.forEach(([x, y], k) => {
      expect(x).toBeCloseTo(rotated(6, 5, 4, 2, 30)[k][0], 9)
      expect(y).toBeCloseTo(rotated(6, 5, 4, 2, 30)[k][1], 9)
    })
    // A quarter turn exactly, its sides swapped
    expect([quarter.shape_type, quarter.points, back.points]).toEqual(['oriented_rectangle', [[8, 1], [8, 9], [4, 9], [4, 1]], [[4, 9], [4, 1], [8, 1], [8, 9]]])
    // With no turn, half a turn, a turn that moves no corner a hundredth of a pixel or a turn that isn't a number, it stays upright
    expect(upright).toEqual(Array(4).fill(newShape('cat', 'rectangle', [[2, 3], [10, 7]])))
    expect(read.issues.map((i) => i.message)).toEqual([
      'instances_default.json: 4 annotations hold only a box, so they become rectangles',
      'instances_default.json: 3 annotations are boxes CVAT rotated (attributes.rotation), so they become rotated boxes, as CVAT shows them'
    ])
  })

  test('keypoints not labelled are left out, hidden ones kept as 1, and a crowd is read as its outline', () => {
    const text = coco([
      { id: 1, image_id: 7, category_id: 9, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 8, 9], iscrowd: 0, keypoints: [2, 3, 2, 0, 0, 0], num_keypoints: 1 },
      { id: 2, image_id: 7, category_id: 9, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 8, 9], iscrowd: 0, keypoints: [2, 3, 1, 4, 5, 2], num_keypoints: 2 },
      { id: 3, image_id: 7, category_id: 9, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [0, 0, 0, 0], iscrowd: 0, keypoints: [2, 3, 2, 6, 9, 2], num_keypoints: 2 },
      { id: 4, image_id: 7, category_id: 9, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 4, 3], iscrowd: 1, keypoints: [0, 0, 0, 0, 0, 0], num_keypoints: 0 },
      { id: 5, image_id: 7, category_id: 9, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 4, 3], iscrowd: 0, keypoints: [2, 3] }
    ])
    const read = readCoco(text, 'person_keypoints.json')
    const got = read.results.get(7)!
    expect(got.poses).toEqual([
      { label: 'person', group: 1, box: [1, 1, 9, 10], names: ['nose', 'eye'], keypoints: [[2, 3, 2], null], outlined: true },
      { label: 'person', group: 2, box: [1, 1, 9, 10], names: ['nose', 'eye'], keypoints: [[2, 3, 1], [4, 5, 2]], outlined: true },
      // With no box of its own, it takes its keypoints'
      { label: 'person', group: 3, box: [2, 3, 6, 9], names: ['nose', 'eye'], keypoints: [[2, 3, 2], [6, 9, 2]], outlined: true }
    ])
    const triangle = [[1, 1], [5, 1], [3, 4]]
    expect(got.shapes).toEqual([
      newShape('person', 'polygon', triangle, { group_id: 1 }), newShape('person', 'polygon', triangle, { group_id: 2 }),
      newShape('person', 'polygon', triangle, { group_id: 3 }), newShape('person', 'polygon', triangle, { other: { iscrowd: 1 } }),
      newShape('person', 'polygon', triangle)
    ])
    expect(read.issues.map((i) => i.message)).toEqual([
      "b.jpg: annotation 5 (“person”) has 2 keypoint numbers where its 2 keypoints need 6, so it's imported without them",
      'person_keypoints.json: 1 keypoint is marked not labelled (visibility 0), so it\'s left out',
      'person_keypoints.json: 1 annotation is a crowd region (iscrowd 1): the shapes made from it keep iscrowd, so an export writes it as a crowd again'
    ])
  })

  test("COCO's own keypoints files keep each person's outline in its group, and read people with no keypoints as plain objects", () => {
    // As person_keypoints_val2017.json has them: outlines, boxes and 17 keypoints, two here
    const text = JSON.stringify({
      images: [{ id: 1, width: 100, height: 100, file_name: 'a.jpg' }],
      categories: [{ id: 1, name: 'person', supercategory: 'person', keypoints: ['nose', 'eye'], skeleton: [[1, 2]] }],
      annotations: [
        { id: 1, image_id: 1, category_id: 1, segmentation: [[10, 10, 50, 12, 48, 60, 12, 58]], area: 1800, bbox: [10, 10, 40, 50], iscrowd: 0, keypoints: [20, 20, 2, 30, 30, 2], num_keypoints: 2 },
        { id: 2, image_id: 1, category_id: 1, segmentation: [[60, 60, 70, 60, 70, 70]], area: 50, bbox: [60, 60, 10, 10], iscrowd: 0, keypoints: [0, 0, 0, 0, 0, 0], num_keypoints: 0 },
        { id: 3, image_id: 1, category_id: 1, segmentation: [[1, 1, 9, 1, 5, 8], [80, 80, 90, 80, 85, 95]], area: 90, bbox: [1, 1, 89, 94], iscrowd: 0, keypoints: [3, 3, 1, 0, 0, 0], num_keypoints: 1 },
        // An outline that's only the box, in any vertex order, is the box
        { id: 4, image_id: 1, category_id: 1, segmentation: [[90, 40, 90, 10, 70, 10, 70, 40]], area: 600, bbox: [70, 10, 20, 30], iscrowd: 0, keypoints: [80, 20, 2, 0, 0, 0], num_keypoints: 1 },
        // An outline that can't be read leaves the box and the keypoints
        { id: 5, image_id: 1, category_id: 1, segmentation: { size: [5, 5], counts: [25] }, area: 1, bbox: [1, 90, 5, 5], iscrowd: 0, keypoints: [3, 92, 2, 0, 0, 0], num_keypoints: 1 }
      ]
    })
    const read = readCoco(text, 'person_keypoints_val2017.json')
    const got = read.results.get(1)!
    expect(got.shapes).toEqual([
      newShape('person', 'polygon', [[10, 10], [50, 12], [48, 60], [12, 58]], { group_id: 1 }),
      newShape('person', 'polygon', [[60, 60], [70, 60], [70, 70]]),
      newShape('person', 'polygon', [[1, 1], [9, 1], [5, 8]], { group_id: 2 }), newShape('person', 'polygon', [[80, 80], [90, 80], [85, 95]], { group_id: 2 })
    ])
    expect(got.poses.map((p) => [p.group, p.box, p.keypoints, p.outlined])).toEqual([
      [1, [10, 10, 50, 60], [[20, 20, 2], [30, 30, 2]], true],
      [2, [1, 1, 90, 95], [[3, 3, 1], null], true],
      [3, [70, 10, 90, 40], [[80, 20, 2], null], false],
      [4, [1, 90, 6, 95], [[3, 92, 2], null], false]
    ])
    expect(read.issues.map((i) => i.message)).toEqual([
      "a.jpg: annotation 5 (“person”) has a 5 x 5 mask on a 100 x 100 image, so only its box and keypoints are imported",
      'person_keypoints_val2017.json: 3 keypoints are marked not labelled (visibility 0), so they\'re left out',
      "person_keypoints_val2017.json: 1 annotation has no keypoints labelled, so it's read as a plain object"
    ])
  })

  test("a person in two polygons, as COCO's keypoints files have one, comes in in one group and goes out again as one person, its box around both", () => {
    const schema = { classes: { person: ['nose', 'eye'] } }
    const person = { id: 1, image_id: 1, category_id: 1, segmentation: [[10, 10, 40, 10, 25, 30], [60, 50, 90, 50, 75, 80]], area: 750, bbox: [10, 10, 80, 70], iscrowd: 0, keypoints: [20, 15, 2, 0, 0, 0], num_keypoints: 1 }
    const text = JSON.stringify({ images: [{ id: 1, width: 100, height: 100, file_name: 'a.jpg' }], categories: [{ id: 1, name: 'person', supercategory: 'person', keypoints: schema.classes.person }], annotations: [person] })
    const { shapes, masks } = shapesOf(readCoco(text, 'person_keypoints.json').results.get(1)!)
    const im = image('a.jpg', 100, 100, shapes, masks)
    const { instances, issues } = poseInstances(im, schema)
    expect(issues.map((i) => i.code)).toEqual(['pose-parts'])
    const [out] = writeCoco([{ image: im, fileName: 'a.jpg', poses: instances }], { task: 'keypoints', labels: ['person'], info: INFO, schema }).dataset.annotations
    expect([out.segmentation, out.bbox, out.keypoints, out.num_keypoints]).toEqual([person.segmentation, person.bbox, person.keypoints, 1])
  })

  test("an annotation that fails in a way nothing here foresaw is left out with a reason, and the file's others are read", async () => {
    vi.resetModules()
    vi.doMock('../../src/shared/rle', async (importOriginal) => ({
      ...(await importOriginal<typeof import('../../src/shared/rle')>()),
      rleToMaskBits: () => { throw new RangeError('Array buffer allocation failed') }
    }))
    try {
      const { readCoco: read } = await import('../../src/shared/coco')
      const got = read(coco([
        { id: 1, image_id: 0, category_id: 3, segmentation: { size: [8, 10], counts: [11, 3, 66] }, area: 3, bbox: [1, 3, 1, 3], iscrowd: 1 },
        { id: 2, image_id: 0, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], area: 6, bbox: [1, 1, 4, 3], iscrowd: 0 }
      ]), 'instances.json')
      expect(got.results.get(0)!.shapes).toEqual([newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]])])
      // The crowd that couldn't be read isn't counted as one
      expect(got.issues.map((i) => i.message)).toEqual(["train/a.jpg: annotation 1 (“cat”) can't be read (Array buffer allocation failed), so it's left out"])
    } finally {
      vi.doUnmock('../../src/shared/rle')
      vi.resetModules()
    }
  })

  test('what can\'t be read is left out with a reason: unknown ids, short or odd polygons, masks of the wrong size', () => {
    const text = coco([
      { id: 1, image_id: 3, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3] },
      { id: 2, image_id: 7, category_id: 4, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3] },
      { id: 3, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1], [1, 1, 5, 1, 3, 4, 9]], bbox: [1, 1, 4, 3] },
      { id: 4, image_id: 7, category_id: 3, segmentation: { size: [10, 20], counts: [200] }, bbox: [1, 1, 4, 3] },
      { id: 5, image_id: 7, category_id: 3, segmentation: { size: [20, 20], counts: '3!1' }, bbox: [1, 1, 4, 3] },
      { id: 6, image_id: 7, category_id: 3, segmentation: { size: [20, 20], counts: [100, 5] }, bbox: [1, 1, 4, 3] },
      { id: 7, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3], attributes: { shape_type: 'circle' } },
      { id: 8, image_id: 7, category_id: 3, bbox: [1, 1, 0, 3] }
    ])
    const read = readCoco(text, 'instances.json')
    expect(read.issues.map((i) => i.message)).toEqual([
      "instances.json: annotation 1 is of image 3, which isn't in its images list, so it's left out",
      "b.jpg: annotation 2 is of category 4, which isn't in the categories list, so it's left out",
      "b.jpg: annotation 3 (“cat”) has polygon 1 of 2 points, and a polygon needs 3, so that polygon is left out",
      "b.jpg: annotation 3 (“cat”) has polygon 2 of an odd count of numbers (7), so that polygon is left out",
      "b.jpg: annotation 3 (“cat”) keeps only its box, so it becomes a rectangle",
      "b.jpg: annotation 4 (“cat”) has a 10 x 20 mask on a 20 x 20 image, so it's left out",
      "b.jpg: annotation 5 (“cat”) has a mask that can't be read (Its counts hold \"!\", which isn't one of COCO's characters), so it's left out",
      "b.jpg: annotation 6 (“cat”) has a mask that can't be read (Its runs cover 105 pixels, not 400), so it's left out",
      "b.jpg: annotation 7 (“cat”) carries Seamtag's attributes, but shape 1 has no points, so it's imported from its segmentation",
      "b.jpg: annotation 8 (“cat”) has neither a shape nor a box with an area, so it's left out"
    ])
    expect(read.results.get(7)!.shapes).toEqual([newShape('cat', 'rectangle', [[1, 1], [5, 4]]), newShape('cat', 'polygon', [[1, 1], [5, 1], [3, 4]])])
  })

  test("a box whose far corner comes to Infinity, which no label file can hold, is no box", () => {
    const text = coco([
      { id: 1, image_id: 7, category_id: 3, bbox: [1e308, 2, 1e308, 10] },
      { id: 2, image_id: 7, category_id: 3, bbox: [2, -1e308, 10, -1e308] },
      { id: 3, image_id: 7, category_id: 9, bbox: [1e308, 2, 1e308, 10], keypoints: [5, 5, 2, 0, 0, 0] },
      { id: 4, image_id: 7, category_id: 9, bbox: [1e308, 2, 1e308, 10], keypoints: [5, 5, 2, 8, 9, 2] },
      { id: 5, image_id: 7, category_id: 3, bbox: [8.9e307, 2, 8.9e307, 10] }
    ])
    const read = readCoco(text, 'instances.json')
    const got = read.results.get(7)!
    expect(got.shapes).toEqual([newShape('cat', 'rectangle', [[8.9e307, 2], [1.78e308, 12]])])
    // A posed object's box then comes from its keypoints, when they span one
    expect(got.poses.map((p) => p.box)).toEqual([[5, 5, 8, 9]])
    expect(read.issues.filter((i) => i.code === 'coco-no-pixels').map((i) => i.message)).toEqual([
      "b.jpg: annotation 1 (“cat”) has neither a shape nor a box with an area, so it's left out",
      "b.jpg: annotation 2 (“cat”) has neither a shape nor a box with an area, so it's left out",
      "b.jpg: annotation 3 (“person”) has neither a box nor keypoints that span one, so it's left out"
    ])
  })

  test('group ids past 2^31 are kept, and new ones count on from those below, so separate objects stay apart', () => {
    const two = (x: number) => [[x, 1, x + 2, 1, x + 2, 3], [x + 4, 1, x + 6, 1, x + 6, 3]]
    const text = coco([
      { id: 1, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3], attributes: { group_id: Number.MAX_SAFE_INTEGER } },
      { id: 2, image_id: 7, category_id: 3, segmentation: [[1, 1, 5, 1, 3, 4]], bbox: [1, 1, 4, 3], attributes: { group_id: 2 } },
      { id: 3, image_id: 7, category_id: 3, segmentation: two(1), bbox: [1, 1, 6, 2] },
      { id: 4, image_id: 7, category_id: 3, segmentation: two(10), bbox: [10, 1, 6, 2] }
    ])
    // 2^53 + 1 is 2^53: counted on from it, every new object was one group
    expect(readCoco(text, 'a.json').results.get(7)!.shapes.map((s) => s.group_id)).toEqual([Number.MAX_SAFE_INTEGER, 2, 3, 3, 4, 4])
  })

  test('a mask shape of more pixels than sharp takes for its PNG is left out, its pixels never made', () => {
    expect(MASK_PIXELS).toBe(16383 * 16383)
    // 16384 x 16384, every pixel set: a mask whose box is the whole image
    const text = JSON.stringify({
      images: [{ id: 1, width: 16384, height: 16384, file_name: 'a.jpg' }], categories: [{ id: 1, name: 'sky' }],
      annotations: [
        { id: 1, image_id: 1, category_id: 1, segmentation: { size: [16384, 16384], counts: [0, 16384 * 16384] }, bbox: [0, 0, 16384, 16384] },
        // Seamtag's own mask of the shape, which may reach past the image, is held to it too
        {
          id: 2, image_id: 1, category_id: 1, segmentation: { size: [16384, 16384], counts: [5, 1, 16384 * 16384 - 6] }, bbox: [0, 5, 1, 1],
          attributes: { shape_type: 'mask', points: [[0, 5], [0, 5]], mask: { size: [16384, 16384], counts: rleToString({ h: 16384, w: 16384, counts: [0, 16384 * 16384] }) } }
        }
      ]
    })
    const t = performance.now()
    const read = readCoco(text, 'sky.json')
    expect(performance.now() - t).toBeLessThan(1000)
    expect(read.results.get(1)!.shapes).toEqual([newShape('sky', 'mask', [[0, 5], [0, 5]])])
    expect(read.results.get(1)!.masks).toEqual([{ x: 0, y: 5, width: 1, height: 1, data: Uint8Array.from([1]) }])
    expect(read.issues.map((i) => i.message)).toEqual([
      "a.jpg: annotation 1 (“sky”) has a mask of 16384 x 16384 pixels, more than the 268,402,689 a mask shape can hold, so it's left out",
      "a.jpg: annotation 2 (“sky”) carries Seamtag's attributes, but shape 1's mask is 16384 x 16384 pixels, more than the 268,402,689 a mask shape can hold, so it's imported from its segmentation"
    ])
    // 16383 x 16383 is a mask shape, counted without making it
    const most = JSON.stringify({
      images: [{ id: 1, width: 16383, height: 16384, file_name: 'b.jpg' }], categories: [{ id: 1, name: 'sky' }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, segmentation: { size: [16384, 16383], counts: [0, ...Array(16383).fill([16383, 1]).flat()] }, bbox: [0, 0, 16383, 16383] }]
    })
    const counted = readCoco(most, 'sky.json', { pixels: false }).results.get(1)!
    expect([counted.shapes, counted.masks, counted.pixels]).toEqual([[newShape('sky', 'mask', [[0, 0], [16382, 16382]])], [null], MASK_PIXELS])
  })

  test("an outline that zigzags across the image is too long to fill and compare with the object's mask, so the mask is read", () => {
    // Corner to corner of a 100,000-pixel square 20 times: 9.5 million points to trace, 76 MB, at 5 times the size
    const zigzag = Array.from({ length: 20 }, (_, i) => (i % 2 ? [100000, 100000] : [0, 0]))
    const text = JSON.stringify({
      images: [{ id: 1, width: 100000, height: 100000, file_name: 'a.jpg' }], categories: [{ id: 1, name: 'cat' }],
      annotations: [{
        id: 1, image_id: 1, category_id: 1, segmentation: { size: [100000, 100000], counts: [5, 1, 1e10 - 6] }, bbox: [0, 5, 1, 1],
        attributes: { group_id: 1, shapes: [{ shape_type: 'mask', points: [[0, 5], [0, 5]] }, { points: zigzag }] }
      }]
    })
    const read = readCoco(text, 'a.json')
    expect(read.results.get(1)!.shapes).toEqual([newShape('cat', 'mask', [[0, 5], [0, 5]])])
    expect(read.issues.map((i) => i.message)).toEqual([
      "a.jpg: annotation 1 (“cat”) carries Seamtag's attributes, but shape 2's outline is too long to compare with its mask, so it's imported from its segmentation"
    ])
  })

  test('values, names and file names from the file are quoted cut short, however long', () => {
    const name = `${'deep/'.repeat(20000)}a.jpg`
    const text = JSON.stringify({
      images: [{ id: 1, width: 10, height: 8, file_name: name }, { id: 2, file_name: name }],
      categories: [{ id: 1, name: 'c'.repeat(100000) }],
      annotations: [
        { id: 1, image_id: Array(100000).fill(123456789), category_id: 1, bbox: [1, 1, 2, 2] },
        { id: 2, image_id: 1, category_id: 'x'.repeat(100000), bbox: [1, 1, 2, 2] },
        { id: 3, image_id: 1, category_id: 1, segmentation: [[1, 2]], bbox: [1, 1, 2, 2] }
      ]
    })
    const shown = (text: string, most: number) => `${text.slice(0, Math.floor((most - 1) / 2))}…${text.slice(text.length - (most - 1 - Math.floor((most - 1) / 2)))}`
    const file = shown(name, 120), category = shown('c'.repeat(100000), 60)
    expect(readCoco(text, 'x.json').issues.map((i) => i.message)).toEqual([
      `x.json: image 2 in its images list (${file}) has no width and height, so it's left out`,
      `x.json: annotation 1 is of image ${shown(JSON.stringify(Array(100000).fill(123456789)), 60)}, which isn't in its images list, so it's left out`,
      `${file}: annotation 2 is of category ${shown(JSON.stringify('x'.repeat(100000)), 60)}, which isn't in the categories list, so it's left out`,
      `${file}: annotation 3 (“${category}”) has its polygon of 1 point, and a polygon needs 3, so that polygon is left out`,
      `${file}: annotation 3 (“${category}”) keeps only its box, so it becomes a rectangle`
    ])
  })

  test('a file that isn\'t COCO is refused whole', () => {
    expect(() => readCoco('{', 'a.json')).toThrow("a.json isn't valid JSON")
    expect(() => readCoco('[]', 'a.json')).toThrow("a.json isn't a COCO file: the top level must be an object")
    expect(() => readCoco('{"annotations": []}', 'a.json')).toThrow("a.json isn't a COCO file: it has no images list")
  })

  test('images without an id, a name or a size, and repeated ids, are left out', () => {
    const text = JSON.stringify({
      images: [{ id: 1, width: 5, height: 5, file_name: 'a.jpg' }, { width: 5, height: 5, file_name: 'b.jpg' }, { id: 2, width: 5, file_name: 'c.jpg' }, { id: 1, width: 5, height: 5, file_name: 'd.jpg' }],
      categories: [], annotations: []
    })
    const read = readCoco(text, 'x.json')
    expect(read.images.map((i) => i.file_name)).toEqual(['a.jpg'])
    expect(read.issues.map((i) => i.message)).toEqual([
      "x.json: image 2 in its images list has no whole-number id, so it's left out",
      "x.json: image 3 in its images list (c.jpg) has no width and height, so it's left out",
      "x.json: image 4 in its images list (d.jpg) has the id 1, which an earlier image has, so it's left out"
    ])
  })
})

describe('matching COCO images to the folder', () => {
  const im = (id: number, file_name: string): CocoImage => ({ id, width: 10, height: 10, file_name })

  test('by path under the base folder first, then by a name only one image has on each side', () => {
    const m = matchCocoImages(
      [im(1, 'train/a.jpg'), im(2, 'b.jpg'), im(3, 'other/c.jpg'), im(4, '.\\train\\d.jpg')],
      ['images/train/a.jpg', 'images/b.jpg', 'elsewhere/c.jpg', 'images/train/d.jpg', 'images/val/a.jpg'],
      'images'
    )
    expect([...m.names]).toEqual([[1, 'images/train/a.jpg'], [2, 'images/b.jpg'], [3, 'elsewhere/c.jpg'], [4, 'images/train/d.jpg']])
    expect(m.issues).toEqual([])
  })

  test('never guesses between images of one name, on either side', () => {
    const m = matchCocoImages([im(1, 'x/a.jpg'), im(2, 'p.jpg'), im(3, 'q/p.jpg'), im(4, 'z.jpg')], ['one/a.jpg', 'two/a.jpg', 'sub/p.jpg'])
    expect([...m.names]).toEqual([])
    expect(m.ambiguous.map((a) => [a.image.id, a.candidates])).toEqual([[1, ['one/a.jpg', 'two/a.jpg']], [2, ['sub/p.jpg']], [3, ['sub/p.jpg']]])
    expect(m.missing.map((i) => i.id)).toEqual([4])
    expect(m.issues.map((i) => i.message)).toEqual([
      "x/a.jpg isn't at that path in the folder and its name matches one/a.jpg, two/a.jpg, so its annotations are left out rather than guessed",
      "p.jpg isn't at that path in the folder and other images in the COCO file are named p.jpg too, so its annotations are left out rather than guessed",
      "q/p.jpg isn't at that path in the folder and other images in the COCO file are named p.jpg too, so its annotations are left out rather than guessed",
      "z.jpg isn't in the folder, so its annotations are left out"
    ])
  })

  test('joins a file name to the base before resolving .., and counts every image of the COCO file when matching by name', () => {
    // LabelMe's imagePath, ../images/a.jpg, copied into file_name by a converter
    const up = matchCocoImages([im(1, '../images/a.jpg'), im(2, '..\\images\\sub\\b.jpg')], ['images/a.jpg', 'images/sub/b.jpg'], 'annotations')
    expect([...up.names]).toEqual([[1, 'images/a.jpg'], [2, 'images/sub/b.jpg']])
    // A path that climbs out of the folder is still found by its name
    expect([...matchCocoImages([im(1, '../images/a.jpg')], ['images/a.jpg']).names]).toEqual([[1, 'images/a.jpg']])
    const two = matchCocoImages([im(1, '../x/a.jpg'), im(2, 'y/a.jpg')], ['z/a.jpg'])
    expect([...two.names]).toEqual([])
    expect(two.ambiguous.map((a) => [a.image.id, a.candidates])).toEqual([[1, ['z/a.jpg']], [2, ['z/a.jpg']]])
  })

  test("an image whose name many of the folder's images have names a few of them, and long names are cut short", () => {
    // A video's frames, named alike in each of its folders
    const folder = Array.from({ length: 1000 }, (_, i) => `video${i}/frame_0001.jpg`)
    const m = matchCocoImages([im(1, 'frame_0001.jpg'), im(2, `${'x'.repeat(100000)}.jpg`)], folder)
    expect(m.issues.map((i) => i.message)).toEqual([
      "frame_0001.jpg isn't at that path in the folder and its name matches video0/frame_0001.jpg, video1/frame_0001.jpg, video2/frame_0001.jpg and 997 more, so its annotations are left out rather than guessed",
      `${'x'.repeat(59)}…${'x'.repeat(56)}.jpg isn't in the folder, so its annotations are left out`
    ])
    expect(m.ambiguous[0].candidates).toHaveLength(1000)
  })

  test('a path or a name in another case or Unicode form is matched by the key given, when exactly one image on each side has it', () => {
    // As macOS compares names; Linux compares them as they are
    const key = (p: string) => p.normalize('NFC').toLowerCase()
    const nfc = (n: string) => n.normalize('NFC'), nfd = (n: string) => n.normalize('NFD')
    const folder = [nfd('사과.png'), 'images/photo.png', nfd('images/café.png'), 'x/A.png', 'x/a.png', 'b.png', 'sub/leaf.jpg', 'images/plain.png']
    const images = [
      im(1, nfc('사과.png')), im(2, 'images/PHOTO.PNG'), im(3, nfc('images/café.png')), im(4, 'x/a.PNG'), im(5, 'B.png'), im(6, 'b.PNG'), im(7, 'other/Leaf.JPG'), im(8, 'images/plain.png')
    ]
    const m = matchCocoImages(images, folder, '', key)
    expect([...m.names]).toEqual([[1, nfd('사과.png')], [2, 'images/photo.png'], [3, nfd('images/café.png')], [7, 'sub/leaf.jpg'], [8, 'images/plain.png']])
    expect([...m.byName]).toEqual([7])
    // Two images of the folder have x/a.PNG's key, and two of the COCO file b.png's: never a guess
    expect(m.ambiguous.map((a) => [a.image.id, a.candidates])).toEqual([[4, ['x/A.png', 'x/a.png']], [5, ['b.png']], [6, ['b.png']]])
    const exact = matchCocoImages(images, folder)
    expect([...exact.names]).toEqual([[8, 'images/plain.png']])
    expect(exact.missing.map((i) => i.id)).toEqual([1, 2, 3, 4, 5, 6, 7])
  })

  test("a folder of 50,000 images of one name, one to a folder, is matched in a time their number gives, not its square", () => {
    const folder = Array.from({ length: 50_000 }, (_, i) => `sample_${i}/rgb.png`)
    const t = performance.now()
    const m = matchCocoImages([im(1, 'sample_7/rgb.png'), im(2, 'rgb.png')], folder)
    expect(performance.now() - t).toBeLessThan(2000)
    expect([...m.names]).toEqual([[1, 'sample_7/rgb.png']])
    expect(m.ambiguous.map((a) => a.candidates.length)).toEqual([50_000])
  })

  test('an image listed twice gets the annotations of both, with a warning', () => {
    const m = matchCocoImages([im(1, 'a.jpg'), im(2, './a.jpg')], ['a.jpg'])
    expect([...m.names]).toEqual([[1, 'a.jpg'], [2, 'a.jpg']])
    expect(m.issues.map((i) => i.message)).toEqual(['a.jpg is listed twice in the COCO file (images 1 and 2), so the annotations of both go to it'])
  })
})

// COCO_PARITY_DIR=dir npx vitest run tests/unit/coco.test.ts, then tests/parity/coco_reads.py dir
test.runIf(!!process.env.COCO_PARITY_DIR)('writes a sample for pycocotools to read', () => {
  const dir = path.join(process.env.COCO_PARITY_DIR!, 'annotations')
  mkdirSync(dir, { recursive: true })
  const fx = JSON.parse(readFileSync(path.join(__dirname, '../fixtures/coco/labelme2coco.json'), 'utf8'))
  const entries: CocoEntry[] = Object.keys(fx.files).sort().map((file) => {
    const doc = parseLabelFile(JSON.stringify(fx.files[file]))
    const masks = doc.shapes.map((s, i) => (fx.masks[file][i] ? bits(s.points[0][0], s.points[0][1], fx.masks[file][i]) : null))
    return { image: image(file.replace('.json', '.png'), doc.imageWidth!, doc.imageHeight!, doc.shapes, masks), fileName: `images/${file.replace('.json', '.png')}` }
  })
  entries.push({ image: everything().image, fileName: 'images/all.png' }, posed())
  for (const seed of [3, 5, 8, 13]) entries.push({ image: randomImage(seed), fileName: `images/random/${seed}.png` })
  const labels = cocoLabels(['person', 'cat', 'dog'], entries)
  writeFileSync(path.join(dir, 'instances_sample.json'), writeCoco(entries, { task: 'instances', labels, info: INFO }).json)
  writeFileSync(path.join(dir, 'person_keypoints_sample.json'), writeCoco(entries, { task: 'keypoints', labels, info: INFO, schema: SCHEMA }).json)
})

test('circles are labelme2coco\'s polygons', () => {
  const { dataset } = instancesOf([{ image: image('a.png', 100, 100, [newShape('ball', 'circle', [[50, 50], [50, 10]])]), fileName: 'a.png' }])
  expect(dataset.annotations[0].segmentation).toEqual([circlePolygon([50, 50], [50, 10])!.flat()])
  expect(dataset.annotations[0].attributes).toEqual({ shape_type: 'circle', points: [[50, 50], [50, 10]] })
})
