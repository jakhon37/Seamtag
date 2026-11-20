import { describe, expect, test } from 'vitest'
import { isAi, newShape, type Shape } from '../../src/shared/labelfile'
import { builtInKeypoints, COCO_PERSON, type PosePoint, posesIn } from '../../src/shared/skeleton'
import {
  acceptTouched, aiCounts, aiPosePoints, askedNow, classForModel, keptAsBoxes, layoutOf, mirrorPose, placeLayout, poseIndex, type PoseSession,
  regroup, reviewItems, scoreOfPose, shapeRows, spoken, stepOf, swapSides, unposed, withHidden, withPoints, withSession
} from '../../src/renderer/src/posing'

const coco = builtInKeypoints()
const box = (group: number | null, extra: Partial<Shape> = {}) => newShape('person', 'rectangle', [[10, 20], [110, 220]], { group_id: group, ...extra })
const point = (label: string, x: number, y: number, group: number | null, extra: Partial<Shape> = {}) => newShape(label, 'point', [[x, y]], { group_id: group, ...extra })
const ai = (score: number) => ({ flags: { ai: true }, other: { score } })

describe('the pose of each shape, and a box with its points', () => {
  const shapes = [box(1), point('nose', 50, 40, 1), newShape('car', 'rectangle', [[0, 0], [5, 5]]), point('left_eye', 55, 35, 1), box(2), point('nose', 300, 40, 2)]
  const poses = posesIn(shapes, coco)
  test('each pose by its shapes', () => {
    const of = poseIndex(poses)
    expect([0, 1, 3].map((i) => of.get(i)!.group)).toEqual([1, 1, 1])
    expect(of.has(2)).toBe(false)
    expect(of.get(5)!.group).toBe(2)
  })
  test('a box chosen brings its points, after what was chosen; a point chosen comes alone', () => {
    expect(withPoints([2, 0], poses)).toEqual([2, 0, 1, 3])
    expect(withPoints([3], poses)).toEqual([3])
    expect(withPoints([0, 1], poses)).toEqual([0, 1, 3])
  })
})

test('copies of a pose take a group_id of their own; other groups keep theirs', () => {
  const existing = [box(1), point('nose', 50, 40, 1), box(4)]
  const copies = [box(1), point('nose', 50, 40, 1), point('left_eye', 55, 35, 1), newShape('bag', 'polygon', [[0, 0], [9, 0], [9, 9]], { group_id: 7 })]
  const out = regroup(copies, existing, coco)
  // After the image's 4 and the copied bag's 7
  expect(out.map((s) => s.group_id)).toEqual([8, 8, 8, 7])
  // Pasted on an image with nothing on it, a pose takes the first free group_id
  expect(regroup(copies.slice(0, 2), [], coco).map((s) => s.group_id)).toEqual([1, 1])
  // Nothing posed, nothing changes
  const plain = [newShape('car', 'rectangle', [[0, 0], [5, 5]], { group_id: 3 })]
  expect(regroup(plain, existing, coco)).toBe(plain)
})

test('hidden is marked difficult; seen takes occluded away too, keeping the flags label_flags lists', () => {
  const s = point('nose', 1, 1, 1, { flags: { truncated: true } })
  const hidden = withHidden(s, true)
  expect(hidden.flags).toEqual({ truncated: true, difficult: true })
  expect(withHidden({ ...s, flags: { occluded: true, truncated: true } }, false).flags).toEqual({ truncated: true })
  expect(withHidden(hidden, false, { difficult: false }).flags).toEqual({ truncated: true, difficult: false })
})

describe('left and right', () => {
  const pose = [box(1), point('left_shoulder', 30, 60, 1), point('right_shoulder', 90, 60, 1), point('left_wrist', 20, 150, 1), point('nose', 60, 30, 1)]
  const [p] = posesIn(pose, coco)
  test('M swaps the names across every flip pair, and the points stay', () => {
    const out = swapSides(pose, p, coco)
    expect(out.map((s) => s.label)).toEqual(['person', 'right_shoulder', 'left_shoulder', 'right_wrist', 'nose'])
    expect(out.map((s) => s.points)).toEqual(pose.map((s) => s.points))
  })
  test('with points chosen, only their pairs swap', () => {
    expect(swapSides(pose, p, coco, new Set([3])).map((s) => s.label)).toEqual(['person', 'left_shoulder', 'right_shoulder', 'right_wrist', 'nose'])
  })
  test('swapping an AI pose accepts the points it changes', () => {
    const proposed = pose.map((s) => ({ ...s, ...ai(0.8) }))
    const out = swapSides(proposed, p, coco)
    expect(out[1].flags.ai).toBeUndefined()
    expect(out[1].other.score).toBeUndefined()
    expect(out[4].flags.ai).toBe(true)
  })
  test('Shift+M mirrors the points inside the box, keeping their names', () => {
    const out = mirrorPose(pose, p)
    // The box runs from 10 to 110
    expect(out.slice(1).map((s) => s.points[0])).toEqual([[90, 60], [30, 60], [100, 150], [60, 30]])
    expect(out.map((s) => s.label)).toEqual(pose.map((s) => s.label))
    expect(out[0]).toBe(pose[0])
  })
  test('without a box, inside the points own width', () => {
    const loose = pose.slice(1)
    const [q] = posesIn(loose, coco)
    expect(mirrorPose(loose, q).map((s) => s.points[0][0])).toEqual([80, 20, 90, 50])
  })
})

describe('AI pose', () => {
  const names = [...COCO_PERSON]
  const found: PosePoint[] = names.map((_, k) => ({ x: 20 + k * 5, y: 30 + k * 10, score: 0.9 }))
  test("a COCO model's keypoints go to the class that has them", () => {
    expect(classForModel(coco, names)).toBe('person')
    const schema = { classes: { hand: ['wrist', 'thumb'], body: ['nose', 'left_eye', 'tail'], person: [...COCO_PERSON] } }
    expect(classForModel(schema, names)).toBe('person')
    // The class sharing most names, when none has them all
    expect(classForModel({ classes: { hand: ['wrist'], face: ['nose', 'left_eye'] } }, names)).toBe('face')
    expect(classForModel({ classes: { hand: ['wrist'] } }, names)).toBeNull()
  })
  test("points in the class's order, each an AI shape with its score, the box taking their mean", () => {
    const { points, score } = aiPosePoints(coco, 'person', names, found, { size: [400, 300], threshold: 0.3, group: 3 })
    expect(points.map((s) => s.label)).toEqual(COCO_PERSON)
    expect(points.every((s) => s.shape_type === 'point' && s.group_id === 3 && s.flags.ai === true && s.other.score === 0.9)).toBe(true)
    expect(points[1].points).toEqual([[25, 40]])
    expect(score).toBe(0.9)
  })
  test('points below the threshold, not placed or past the edge are left out; half a pixel past it is on the edge', () => {
    const answer = [...found]
    answer[0] = null
    answer[1] = { x: 50, y: 50, score: 0.29 }
    answer[2] = { x: 400.4, y: -0.5, score: 0.5 }
    answer[3] = { x: 420, y: 100, score: 0.95 }
    answer[4] = { x: 100, y: 300.6, score: 0.95 }
    const { points, score } = aiPosePoints(coco, 'person', names, answer, { size: [400, 300], threshold: 0.3, group: 1, flagsOf: (l) => (l === 'right_eye' ? { blurry: false } : {}) })
    expect(points.map((s) => s.label)).toEqual([COCO_PERSON[2], ...COCO_PERSON.slice(5)])
    expect(points[0].points).toEqual([[400, 0]])
    expect(points[0].flags).toEqual({ blurry: false, ai: true })
    expect(points[1].flags).toEqual({ ai: true })
    expect(score).toBe(Math.round(((0.5 + 0.9 * 12) / 13) * 1000) / 1000)
  })
  test("names in the model's own order, and names the class doesn't have, are matched by name", () => {
    const reversed = [...names].reverse()
    const answer = [...found].reverse()
    const schema = { classes: { person: ['left_eye', 'nose', 'tail'] } }
    const { points } = aiPosePoints(schema, 'person', [...reversed, 'extra'], [...answer, { x: 1, y: 1, score: 1 }], { size: [400, 300], threshold: 0, group: 1 })
    expect(points.map((s) => [s.label, s.points[0]])).toEqual([['left_eye', [25, 40]], ['nose', [20, 30]]])
  })
  test('nothing above the threshold: no points and no score', () => {
    expect(aiPosePoints(coco, 'person', names, names.map(() => null), { size: [400, 300], threshold: 0.3, group: 1 })).toEqual({ points: [], score: null })
  })
})

describe('the Pose tool, one keypoint at a time', () => {
  const session = (extra: Partial<PoseSession> = {}): PoseSession => ({ cls: 'person', group: 2, box: box(2), fresh: true, asked: [0, 1, 2], done: [], ...extra })
  test('asks for each keypoint in turn', () => {
    expect(askedNow(session())).toBe(0)
    expect(askedNow(session({ done: [null] }))).toBe(1)
    expect(askedNow(session({ done: [null, null, null] }))).toBeNull()
  })
  test('a new box, then its points in the order asked, skipped ones left out', () => {
    const s = session({ done: [point('nose', 1, 1, 2), null, point('right_eye', 3, 3, 2)] })
    const before = [newShape('car', 'rectangle', [[0, 0], [5, 5]])]
    expect(withSession(before, s).map((x) => x.label)).toEqual(['car', 'person', 'nose', 'right_eye'])
    // Stopped straight after the box, the box stays
    expect(withSession(before, session()).map((x) => x.label)).toEqual(['car', 'person'])
  })
  test("carrying on a box that had no group: its points go at the end, and it takes the pose's group", () => {
    const own = box(null)
    const shapes = [own, newShape('car', 'rectangle', [[0, 0], [5, 5]])]
    const s = session({ box: own, fresh: false, group: 1, done: [point('nose', 1, 1, 1)] })
    const out = withSession(shapes, s)
    expect(out.map((x) => [x.label, x.group_id])).toEqual([['person', 1], ['car', null], ['nose', 1]])
    // Nothing placed, nothing changes
    expect(withSession(shapes, { ...s, done: [null] })).toBe(shapes)
  })
})

describe('the Shapes list folds each pose into one row', () => {
  const shapes = [
    newShape('car', 'rectangle', [[0, 0], [5, 5]]), box(1), point('left_eye', 55, 35, 1), point('nose', 50, 40, 1), point('nose', 51, 41, 1),
    newShape('dog', 'polygon', [[0, 0], [9, 0], [9, 9]])
  ]
  const poses = posesIn(shapes, coco)
  test('folded, a pose is one row where its first shape is', () => {
    expect(shapeRows(shapes.length, poses, () => false).map((r) => (r.kind === 'shape' ? r.index : r.kind))).toEqual([0, 'pose', 5])
  })
  test("opened: its box, then each keypoint in the class's order, missing ones too, then other points", () => {
    const rows = shapeRows(shapes.length, poses, () => true)
    expect(rows.slice(0, 6).map((r) => r.kind)).toEqual(['shape', 'pose', 'shape', 'shape', 'shape', 'missing'])
    expect(rows.filter((r) => r.kind === 'missing' || (r.kind === 'shape' && r.keypoint !== undefined))).toHaveLength(17)
    expect(rows[2]).toMatchObject({ kind: 'shape', index: 1 })
    expect(rows[3]).toMatchObject({ kind: 'shape', index: 3, keypoint: 0 })
    expect(rows[4]).toMatchObject({ kind: 'shape', index: 2, keypoint: 1 })
    expect(rows[5]).toMatchObject({ kind: 'missing', keypoint: 2 })
    // The second nose is the pose's, but not one of its keypoints
    expect(rows.slice(-2).map((r) => (r.kind === 'shape' ? r.index : r.kind))).toEqual([4, 5])
    expect(rows).toHaveLength(1 + 1 + 1 + 17 + 1 + 1)
  })
})

test("review takes an AI pose as one, by its box's score, least certain first, and other AI shapes one by one", () => {
  const shapes = [
    newShape('cat', 'polygon', [[0, 0], [9, 0], [9, 9]], ai(0.5)),
    box(1, ai(0.8)), point('nose', 50, 40, 1, ai(0.4)), point('left_eye', 55, 35, 1),
    newShape('cat', 'polygon', [[0, 0], [9, 0], [9, 9]], ai(0.2)),
    newShape('cat', 'polygon', [[0, 0], [9, 0], [9, 9]], { flags: { ai: true } })
  ]
  const poses = posesIn(shapes, coco)
  const items = reviewItems(shapes, poses)
  // The pose's box holds its keypoints' mean: its weakest point, the nose, doesn't rank it
  expect(items.map((i) => [i.shapes, i.score])).toEqual([[[4], 0.2], [[0], 0.5], [[1, 2], 0.8], [[5], undefined]])
  expect(items[2].pose?.group).toBe(1)
  // Any of an AI pose's shapes stands for the pose, its accepted ones too; an AI shape stands for itself only alone
  expect(stepOf(items, [3])).toBe(items[2])
  expect(stepOf(items, [1, 2, 3])).toBe(items[2])
  expect(stepOf(items, [3, 4])).toBeNull()
  expect(stepOf(items, [4])).toBe(items[0])
  expect(stepOf(items, [0, 4])).toBeNull()
  expect(stepOf(items, [])).toBeNull()
  // The review bar counts the pose once
  expect(aiCounts(shapes, poses)).toEqual({ poses: 1, shapes: 3 })
})

test("an AI pose on the image's own box scores its AI keypoints' mean", () => {
  const shapes = [box(1), point('nose', 50, 40, 1, ai(0.4)), point('left_eye', 55, 35, 1, ai(0.9)), point('right_eye', 65, 35, 1, ai(0.65))]
  const [pose] = posesIn(shapes, coco)
  expect(scoreOfPose(shapes, pose)).toBe(0.65)
  expect(reviewItems(shapes, [pose]).map((i) => [i.shapes, i.score])).toEqual([[[1, 2, 3], 0.65]])
  // Accepted, it has no score
  const plain = [box(1), point('nose', 50, 40, 1)]
  expect(scoreOfPose(plain, posesIn(plain, coco)[0])).toBeUndefined()
})

describe('an edit to an AI pose accepts all of it', () => {
  const shapes = [
    box(1, ai(0.7)), point('nose', 50, 40, 1, ai(0.6)), point('left_eye', 55, 35, 1, ai(0.8)),
    newShape('cat', 'polygon', [[0, 0], [9, 0], [9, 9]], ai(0.5)), box(2), point('nose', 300, 40, 2, ai(0.6))
  ]
  const poses = posesIn(shapes, coco)
  test("a point moved: its pose's AI shapes accepted, others' left to review", () => {
    const moved = shapes.map((s, i) => (i === 2 ? { ...s, points: [[60, 30]] } : s))
    const out = acceptTouched(moved, poses, [2])
    expect(out.slice(0, 3).some(isAi)).toBe(false)
    expect(out.slice(0, 3).every((s) => s.other.score === undefined)).toBe(true)
    expect(out[2].points).toEqual([[60, 30]])
    expect(isAi(out[3])).toBe(true)
    expect(isAi(out[5])).toBe(true)
  })
  test('a pose on its own box: its AI points accepted, the box as it was', () => {
    const out = acceptTouched(shapes, poses, [4])
    expect(isAi(out[5])).toBe(false)
    expect(out[4]).toBe(shapes[4])
    expect(isAi(out[0])).toBe(true)
  })
  test('nothing of an AI pose touched: nothing changes', () => {
    expect(acceptTouched(shapes, poses, [3])).toBe(shapes)
    expect(acceptTouched(shapes, poses, [])).toBe(shapes)
  })
})

test("LabelImg's formats keep boxes: shapes become the box around them, and points, which have none, are left out", () => {
  const shapes = [box(1), point('nose', 50, 40, 1), newShape('car', 'polygon', [[5, 1], [9, 4], [2, 8]]), point('tree', 3, 3, null), newShape('cut', 'line', [[0, 0], [4, 6]])]
  const { shapes: kept, boxed, left } = keptAsBoxes(shapes)
  expect(kept.map((s) => [s.label, s.shape_type])).toEqual([['person', 'rectangle'], ['car', 'rectangle'], ['cut', 'rectangle']])
  expect(kept[0]).toBe(shapes[0])
  expect(kept[1].points).toEqual([[2, 1], [9, 8]])
  expect({ boxed, left }).toEqual({ boxed: 2, left: 2 })
  expect(keptAsBoxes([])).toEqual({ shapes: [], boxed: 0, left: 0 })
})

describe('Pose every person: the people still to pose', () => {
  const polygon = (label: string, x0: number, y0: number, x1: number, y1: number, group: number | null) =>
    newShape(label, 'polygon', [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], { group_id: group })
  const shapes = [
    // A person drawn as two polygons and their bag, in one group
    polygon('person', 220, 40, 280, 200, 1), newShape('person', 'rectangle', [[230, 30], [300, 220]], { group_id: 1 }), polygon('bag', 285, 100, 300, 150, 1),
    // A person posed already
    box(2), point('nose', 50, 40, 2),
    // Two people in no group, one of them hidden, and a person's line, which is no box
    polygon('person', 20, 60, 90, 280, null), polygon('person', 320, 60, 390, 280, null), newShape('person', 'line', [[0, 0], [5, 5]]),
    polygon('dog', 100, 200, 150, 280, null)
  ]
  shapes[6] = { ...shapes[6], hidden: true }
  const poses = posesIn(shapes, coco)
  test("a group's shapes of the class are one person, in the box around them; a shape in no group is one; posed groups and hidden shapes are left alone", () => {
    expect(unposed(shapes, poses, 'person')).toEqual([
      { box: [220, 30, 300, 220], shapes: [0, 1], group: 1 },
      { box: [20, 60, 90, 280], shapes: [5], group: null }
    ])
  })
  test('those a click chose: the person a shape is part of', () => {
    expect(unposed(shapes, poses, 'person', [1])).toEqual([{ box: [220, 30, 300, 220], shapes: [0, 1], group: 1 }])
    expect(unposed(shapes, poses, 'person', [3])).toEqual([])
    expect(unposed(shapes, poses, 'person', [8])).toEqual([])
  })
  test('once every person has keypoints, nobody', () => {
    const posed = [...shapes.slice(0, 5), point('nose', 250, 50, 1)]
    expect(unposed(posed, posesIn(posed, coco), 'person')).toEqual([])
  })
})

test("the last pose's layout, scaled into the next box", () => {
  const shapes = [box(1), point('nose', 60, 70, 1), point('left_eye', 35, 120, 1)]
  const layout = layoutOf(shapes, posesIn(shapes, coco)[0])!
  expect(layout.at.slice(0, 2)).toEqual([[0.5, 0.25], [0.25, 0.5]])
  expect(placeLayout(layout, [0, 0, 200, 400]).slice(0, 3)).toEqual([[100, 100], [50, 200], null])
  // One point says nothing of a pose's shape
  expect(layoutOf(shapes.slice(0, 2), posesIn(shapes.slice(0, 2), coco)[0])).toBeNull()
})

test('names are said with spaces', () => {
  expect(spoken('left_shoulder')).toBe('left shoulder')
})
