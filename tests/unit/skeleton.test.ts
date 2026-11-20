import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { describe, expect, test } from 'vitest'
import { folderKeypoints, keypointsFile, saveFolderKeypoints } from '../../src/main/keypoints'
import type { KeypointSchema } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { readPoseYaml, writePoseYaml } from '../../src/shared/pose'
import { builtInKeypoints, COCO_PERSON, isHidden, keypointNames, keypointOnlyNames, nextGroupId, posePoints, poseShapes, posesIn, sidesOf, withKnownLines } from '../../src/shared/skeleton'

const box = (label: string, group: number | null, type: Shape['shape_type'] = 'rectangle') =>
  newShape(label, type, type === 'rectangle' ? [[10, 10], [50, 60]] : [[10, 10], [50, 10], [50, 60]], { group_id: group })
const point = (label: string, group: number | null, flags: Record<string, boolean> = {}) => newShape(label, 'point', [[20, 20]], { group_id: group, flags })

describe('the built-in keypoints: COCO person', () => {
  const coco = builtInKeypoints()
  test('17 keypoints, 19 lines and 8 flip pairs, all by COCO names', () => {
    expect(coco.classes.person).toEqual(COCO_PERSON)
    expect(COCO_PERSON).toHaveLength(17)
    expect(coco.skeleton!.person).toHaveLength(19)
    expect(coco.flip!.person).toHaveLength(8)
    for (const [a, b] of [...coco.skeleton!.person, ...coco.flip!.person]) {
      expect(COCO_PERSON).toContain(a)
      expect(COCO_PERSON).toContain(b)
    }
    // A few of COCO's lines, from person_keypoints_2017.json's 1-based skeleton
    expect(coco.skeleton!.person).toContainEqual(['left_ankle', 'left_knee'])
    expect(coco.skeleton!.person).toContainEqual(['left_shoulder', 'right_shoulder'])
    expect(coco.skeleton!.person).toContainEqual(['right_ear', 'right_shoulder'])
  })
  test('reads back as written, as the folder file holds it', () => {
    expect(readPoseYaml(writePoseYaml(coco)).schema).toEqual(coco)
  })
  test('each call gives its own copy', () => {
    builtInKeypoints().classes.person.push('tail')
    expect(builtInKeypoints().classes.person).toHaveLength(17)
  })
})

describe('withKnownLines', () => {
  test("a class with COCO's person keypoints, in any order, gets COCO's lines and flip pairs", () => {
    const made: KeypointSchema = { classes: { walker: [...COCO_PERSON].reverse(), cat: ['nose', 'tail'] } }
    const out = withKnownLines(made)
    expect(out.skeleton!.walker).toHaveLength(19)
    expect(out.flip!.walker).toHaveLength(8)
    expect(out.skeleton!.cat).toBeUndefined()
  })
  test('lines a class already has stay', () => {
    const made: KeypointSchema = { classes: { person: COCO_PERSON }, skeleton: { person: [['nose', 'left_eye']] } }
    expect(withKnownLines(made).skeleton!.person).toEqual([['nose', 'left_eye']])
  })
  test('no lines or pairs at all leaves the keys out', () => {
    const out = withKnownLines({ classes: { cat: ['nose', 'tail'] } })
    expect(out).toEqual({ classes: { cat: ['nose', 'tail'] } })
  })
})

describe('sidesOf', () => {
  test("COCO's person: left, right and centre from the flip pairs", () => {
    const sides = sidesOf(builtInKeypoints(), 'person')
    expect(sides[0]).toBe('centre')
    expect(sides[COCO_PERSON.indexOf('left_eye')]).toBe('left')
    expect(sides[COCO_PERSON.indexOf('right_ankle')]).toBe('right')
    expect(sides.filter((s) => s === 'centre')).toHaveLength(1)
  })
  test('a pair whose names say nothing: the first is on the left', () => {
    expect(sidesOf({ classes: { fish: ['a', 'b', 'c'] }, flip: { fish: [['b', 'a']] } }, 'fish')).toEqual(['right', 'left', 'centre'])
  })
  test('the name saying left wins, whichever way round the pair is', () => {
    expect(sidesOf({ classes: { h: ['RightHand', 'LeftHand'] }, flip: { h: [['RightHand', 'LeftHand']] } }, 'h')).toEqual(['right', 'left'])
  })
  test('without flip pairs, from the names', () => {
    expect(sidesOf({ classes: { h: ['head', 'Left_paw', 'right_paw'] } }, 'h')).toEqual(['centre', 'left', 'right'])
  })
  test('a class the schema lacks has none', () => {
    expect(sidesOf(builtInKeypoints(), 'cat')).toEqual([])
  })
})

describe('posesIn', () => {
  const schema = builtInKeypoints()
  test('a box and its points, in the keypoints order', () => {
    const shapes = [box('person', 1), point('left_eye', 1), point('nose', 1), point('nose', 2)]
    const [pose] = posesIn(shapes, schema)
    expect(pose).toMatchObject({ group: 1, cls: 'person', box: 0, extra: [] })
    expect(pose.points[0]).toBe(2)
    expect(pose.points[1]).toBe(1)
    expect(pose.points.filter((p) => p !== null)).toHaveLength(2)
    expect(poseShapes(pose).sort()).toEqual([0, 1, 2])
  })
  test("a group without keypoints isn't a pose: a person and their bag, as the README's photos group them", () => {
    expect(posesIn([box('person', 1, 'polygon'), box('bag', 1, 'polygon'), box('person', 2)], schema)).toEqual([])
  })
  test('points without a box take the one class they all fit', () => {
    const [pose] = posesIn([point('nose', 4), point('left_knee', 4)], schema)
    expect(pose).toMatchObject({ group: 4, cls: 'person', box: null })
  })
  test("points that fit two classes, and no box, aren't read one way", () => {
    const two: KeypointSchema = { classes: { cat: ['nose', 'tail'], dog: ['nose', 'tail', 'ear'] } }
    expect(posesIn([point('nose', 1)], two)).toEqual([])
    expect(posesIn([point('ear', 1)], two)[0].cls).toBe('dog')
  })
  test('a repeated keypoint and a name not in the class are extra', () => {
    const shapes = [box('person', 3), point('nose', 3), point('nose', 3), point('tail', 3)]
    const [pose] = posesIn(shapes, { classes: { person: COCO_PERSON, cat: ['tail'] } })
    expect(pose.points[0]).toBe(1)
    expect(pose.extra).toEqual([2, 3])
  })
  test('the rectangle is the box when the group has other shapes of the class', () => {
    const shapes = [box('person', 1, 'polygon'), box('person', 1), point('nose', 1)]
    expect(posesIn(shapes, schema)[0].box).toBe(1)
  })
  test('ungrouped shapes, lines and other classes are left alone; poses come in the order of their first shape', () => {
    const shapes = [point('nose', null), newShape('person', 'line', [[0, 0], [5, 5]], { group_id: 7 }), point('nose', 9), box('person', 2), point('nose', 2)]
    expect(posesIn(shapes, schema).map((p) => p.group)).toEqual([9, 2])
  })
})

test('nextGroupId: one more than the largest', () => {
  expect(nextGroupId([])).toBe(1)
  expect(nextGroupId([{ group_id: null }, { group_id: 5 }, { group_id: 2 }])).toBe(6)
})

test('isHidden: difficult, as X-AnyLabeling writes it, or occluded', () => {
  expect(isHidden(point('nose', 1))).toBe(false)
  expect(isHidden(point('nose', 1, { difficult: true }))).toBe(true)
  expect(isHidden(point('nose', 1, { occluded: true }))).toBe(true)
})

test('keypointNames: every class\'s', () => {
  expect(keypointNames({ classes: { a: ['x', 'y'], b: ['y', 'z'] } })).toEqual(new Set(['x', 'y', 'z']))
})

describe("a folder's keypoints file", () => {
  const folder = () => mkdtempSync(path.join(tmpdir(), 'seamtag-keypoints-'))
  test("COCO's person until there is one", async () => {
    expect(await folderKeypoints(folder())).toEqual({ schema: builtInKeypoints(), file: null, problem: null })
  })
  test('saved in .myseg, then read back', async () => {
    const root = folder()
    const schema: KeypointSchema = { classes: { cat: ['nose', 'tail'] }, skeleton: { cat: [['nose', 'tail']] } }
    const saved = await saveFolderKeypoints(root, schema)
    expect(saved).toEqual({ schema, file: keypointsFile(root), problem: null })
    expect(keypointsFile(root)).toBe(path.join(root, '.myseg', 'keypoints.yaml'))
    expect(readFileSync(keypointsFile(root), 'utf8')).toContain('classes:')
    expect(await folderKeypoints(root)).toEqual(saved)
  })
  test("a file that can't be read gives COCO's person and says why", async () => {
    const root = folder()
    mkdirSync(path.join(root, '.myseg'))
    writeFileSync(keypointsFile(root), 'classes: [nose\n')
    const k = await folderKeypoints(root)
    expect(k.schema).toEqual(builtInKeypoints())
    expect(k.file).toBeNull()
    expect(k.problem).toMatch(/^The keypoints file .*keypoints\.yaml isn't valid YAML .*, so COCO's person is used$/)
  })
  test("a schema the reader refuses isn't written", async () => {
    const root = folder()
    await expect(saveFolderKeypoints(root, { classes: { cat: ['nose', 'nose'] } })).rejects.toThrow(/names nose twice/)
    expect(await folderKeypoints(root)).toMatchObject({ file: null, problem: null })
  })
})

test("keypointOnlyNames: keypoint names that no class has as its name", () => {
  expect(keypointOnlyNames({ classes: { person: ['nose', 'hand'], hand: ['thumb'] } })).toEqual(new Set(['nose', 'thumb']))
})

describe('fixes from review', () => {
  test("a group with a closed shape of a class the keypoints don't cover isn't a pose: a dog's box and its nose", () => {
    const shapes = [box('dog', 1), point('nose', 1), point('left_eye', 1), box('person', 2), point('nose', 2)]
    expect(posesIn(shapes, builtInKeypoints()).map((p) => p.group)).toEqual([2])
  })
  test('posePoints: only the keypoints a pose has a place for', () => {
    const shapes = [box('person', 1), point('nose', 1), point('nose', 1), point('nose', null), newShape('left_eye', 'rectangle', [[0, 0], [2, 2]])]
    expect([...posePoints(posesIn(shapes, builtInKeypoints()))]).toEqual([1])
  })
  test("withKnownLines with cocoOrder puts a class of COCO's keypoints in COCO's order; without it, the order stays", () => {
    const made: KeypointSchema = { classes: { person: [...COCO_PERSON.slice(1), 'nose'] } }
    expect(withKnownLines(made, true).classes.person).toEqual(COCO_PERSON)
    expect(withKnownLines(made).classes.person).toEqual(made.classes.person)
    expect(made.classes.person[16]).toBe('nose')
  })
  test("a folder's file of COCO's person without lines, as X-AnyLabeling writes it, is read with COCO's lines and flip pairs", async () => {
    const root = mkdtempSync(path.join(tmpdir(), 'seamtag-keypoints-'))
    mkdirSync(path.join(root, '.myseg'))
    writeFileSync(keypointsFile(root), `classes:\n  person: [${COCO_PERSON.join(', ')}]\nhas_visible: true\n`)
    const k = await folderKeypoints(root)
    expect(k.schema.skeleton!.person).toHaveLength(19)
    expect(k.schema.flip!.person).toHaveLength(8)
  })
})
