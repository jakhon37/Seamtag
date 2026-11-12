import yaml from 'js-yaml'
import { describe, expect, test } from 'vitest'
import type { DatasetImage, KeypointSchema, MaskBits } from '../../src/shared/dataset'
import { newShape, type Shape } from '../../src/shared/labelfile'
import { flipIndex, keypointCount, poseInstances, pyFloat, pyInt, readPoseYaml, shapesFromPose, short, ULTRALYTICS_YAML, writePoseYaml } from '../../src/shared/pose'

const image = (shapes: Shape[], masks: (MaskBits | null)[] = []): DatasetImage => ({
  name: 'a.jpg', width: 100, height: 80, file: 'a.json', format: 'labelme', shapes, masks: shapes.map((_, i) => masks[i] ?? null), flags: {}, verified: false
})
const box = (label: string, group: number | null, points = [[10, 10], [50, 60]]) => newShape(label, 'rectangle', points, { group_id: group })
const point = (label: string, group: number | null, x: number, y: number, flags: Record<string, boolean> = {}) => newShape(label, 'point', [[x, y]], { group_id: group, flags })

// X-AnyLabeling 4.3.2's assets/yolov8_pose.yaml, cut to two classes
const XANYLABELING = `has_visible: true
classes:
  person:
    - nose
    - left_eye
    - right_eye
  hand:
    - wrist
    - thumb
`
const SCHEMA: KeypointSchema = { classes: { person: ['nose', 'left_eye', 'right_eye'], hand: ['wrist', 'thumb'] } }

describe("X-AnyLabeling's pose file", () => {
  test('reads as each class and its keypoints in order, with has_visible true when it says so or says nothing', () => {
    expect(readPoseYaml(XANYLABELING)).toEqual({ schema: SCHEMA, hasVisible: true })
    expect(readPoseYaml('classes:\n  dog: [nose, tail]\n')).toEqual({ schema: { classes: { dog: ['nose', 'tail'] } }, hasVisible: true })
    expect(readPoseYaml('classes:\n  dog: [nose]\nhas_visible: false\n').hasVisible).toBe(false)
    // PyYAML, which X-AnyLabeling reads it with, takes YAML 1.1's yes and no
    expect(readPoseYaml('classes:\n  dog: [nose]\nhas_visible: no\n').hasVisible).toBe(false)
  })
  test("Seamtag's skeleton and flip pairs ride along, and X-AnyLabeling's keys stay as it reads them", () => {
    const schema: KeypointSchema = {
      classes: { person: ['nose', 'left_eye', 'right_eye'], 'yes': ['1', 'a: b'] },
      skeleton: { person: [['nose', 'left_eye'], ['nose', 'right_eye']] },
      flip: { person: [['left_eye', 'right_eye']], 'yes': [] }
    }
    const text = writePoseYaml(schema)
    expect(readPoseYaml(text)).toEqual({ schema, hasVisible: true })
    // As X-AnyLabeling's LabelConverter loads it: classes and has_visible
    const loaded = yaml.load(text) as Record<string, unknown>
    expect(loaded.classes).toEqual(schema.classes)
    expect(loaded.has_visible).toBe(true)
    expect(writePoseYaml({ classes: { dog: ['nose'] } }, false)).toBe('# Keypoints of each class, in order, as X-AnyLabeling lists them\nclasses:\n  dog:\n    - nose\nhas_visible: false\n')
  })
  test("what can't be used is refused with a reason", () => {
    const bad = (text: string, message: RegExp) => expect(() => readPoseYaml(text)).toThrow(message)
    bad('classes: [a, b]', /needs classes/)
    bad('names: {0: a}', /needs classes/)
    bad('classes:\n  dog: []\n', /list of keypoint names for dog/)
    bad('classes:\n  dog: [nose, nose]\n', /names nose twice in dog/)
    bad('classes:\n  dog: [nose, ~]\n', /no name for keypoint 2 of dog/)
    bad('classes:\n  dog: [nose]\nhas_visible: maybe\n', /has_visible/)
    bad('classes:\n  dog: [nose]\nflip:\n  cat: []\n', /names cat under flip/)
    bad('classes:\n  dog: [a, b, c]\nflip:\n  dog: [[a, x]]\n', /pairs x under flip of dog/)
    bad('classes:\n  dog: [a, b, c]\nflip:\n  dog: [[a, b], [b, c]]\n', /flips b of dog in two pairs/)
    bad('classes:\n  dog: [a, b]\nskeleton:\n  dog: [[a, a]]\n', /pairs a with itself/)
    bad('classes: {dog: [a]\n', /isn't valid YAML/)
  })

  test('a long list of keypoints reads quickly, and aliases repeating one for every class are stopped by a count of them all', () => {
    const names = Array.from({ length: 20000 }, (_, i) => `k${i}`)
    const read = (text: string) => {
      try {
        return readPoseYaml(text).schema.classes
      } catch (e) {
        return (e as Error).message
      }
    }
    const quick = (text: string) => {
      const t = performance.now()
      const result = read(text)
      // Checked name by name against the whole list, 20,000 names took 1.5 s, and 5 classes of them 7 s
      expect(performance.now() - t).toBeLessThan(500)
      return result
    }
    const one = `classes:\n  person: [${names.join(', ')}]\n`
    expect(quick(one)).toEqual({ person: names })
    expect(quick(`classes:\n  person: [${names.join(', ')}, k7]\n`)).toBe('The keypoint file names k7 twice in person')
    // Each alias is the whole list again at no cost in the file: 10 classes of 20,000 is past the 100,000 it may name
    const aliased = (n: number) => `classes:\n  c0: &L [${names.join(', ')}]\n${Array.from({ length: n - 1 }, (_, i) => `  c${i + 1}: *L\n`).join('')}`
    expect(Object.keys(quick(aliased(5)) as object)).toEqual(['c0', 'c1', 'c2', 'c3', 'c4'])
    expect(quick(aliased(10))).toBe('The keypoint file names more than 100,000 keypoints in all, the most Seamtag reads')
    // Pairs are counted too, and checked against a class's keypoints at once
    const pairs = (n: number) => `[${Array.from({ length: n }, (_, i) => `[k${i % 20000}, k${(i + 1) % 20000}]`).join(', ')}]`
    const skeleton = (n: number) => `${aliased(5)}skeleton:\n  c0: &P ${pairs(n)}\n${[1, 2, 3, 4].map((i) => `  c${i}: *P\n`).join('')}`
    expect(read(skeleton(20000))).toEqual(expect.objectContaining({ c4: names }))
    expect(read(skeleton(20001))).toBe('The keypoint file gives more than 100,000 pairs under skeleton, the most Seamtag reads')
  })

  test('a class named after what every object has is a class like any other', () => {
    const { schema } = readPoseYaml('classes:\n  __proto__: [nose, tail]\n  constructor: [nose]\nflip:\n  __proto__: [[nose, tail]]\n')
    expect(Object.keys(schema.classes)).toEqual(['__proto__', 'constructor'])
    expect(Object.hasOwn(schema.classes, '__proto__') && schema.classes.__proto__).toEqual(['nose', 'tail'])
    expect(Object.getPrototypeOf(schema.classes)).toBe(Object.prototype)
    expect(Object.hasOwn(schema.flip!, '__proto__') && schema.flip!.__proto__).toEqual([['nose', 'tail']])
  })

  test('names from the file are quoted cut short in its messages', () => {
    const long = 'x'.repeat(100000)
    expect(() => readPoseYaml(`classes:\n  dog: [${long}, ${long}]\n`)).toThrow(`The keypoint file names ${'x'.repeat(29)}…${'x'.repeat(30)} twice in dog`)
  })
})

describe('text from a file in a message', () => {
  test("is cut in the middle past 60 characters, or as many as asked, never between a character's two halves", () => {
    expect(short('a.png')).toBe('a.png')
    expect(short('x'.repeat(60))).toBe('x'.repeat(60))
    const path = `images/${'deep/'.repeat(30)}IMG_0001.jpg`
    expect(short(path)).toBe(`${path.slice(0, 29)}…${path.slice(-30)}`)
    expect(short(path).length).toBe(60)
    expect(short(path, 120)).toBe(`${path.slice(0, 59)}…${path.slice(-60)}`)
    // 🐱 is two halves in JavaScript's strings
    const cats = '🐱'.repeat(40)
    expect([...short(cats)].every((c) => c === '🐱' || c === '…')).toBe(true)
  })
})

describe("Ultralytics' flip_idx from flip pairs", () => {
  test('one class: each keypoint and the one it swaps with', () => {
    expect(flipIndex({ classes: { person: ['nose', 'left_eye', 'right_eye'] }, flip: { person: [['left_eye', 'right_eye']] } })).toEqual({ flipIdx: [0, 2, 1], issues: [] })
  })
  test('classes that swap the same places share one; padding keypoints stay where they are', () => {
    const schema: KeypointSchema = { classes: { a: ['l', 'r', 'm'], b: ['left', 'right'] }, flip: { a: [['l', 'r']], b: [['left', 'right']] } }
    expect(flipIndex(schema)).toEqual({ flipIdx: [1, 0, 2], issues: [] })
  })
  test("classes that swap different places, or some without pairs, give none, with a warning; no pairs at all give none quietly", () => {
    const differ = flipIndex({ classes: { a: ['l', 'r', 'm'], b: ['x', 'l', 'r'] }, flip: { a: [['l', 'r']], b: [['l', 'r']] } })
    expect(differ.flipIdx).toBeNull()
    expect(differ.issues[0]).toMatchObject({ severity: 'warning', code: 'pose-flip' })
    const some = flipIndex({ classes: { a: ['l', 'r'], b: ['x'] }, flip: { a: [['l', 'r']] } })
    expect(some.flipIdx).toBeNull()
    expect(some.issues[0].message).toMatch(/not b/)
    expect(flipIndex(SCHEMA)).toEqual({ flipIdx: null, issues: [] })
    expect(flipIndex({ classes: { a: ['l', 'r'] }, flip: { a: [['l', 'x']] } }).issues[0].message).toMatch(/names a keypoint a doesn't have/)
  })
  test('rows hold as many keypoints as the class with most', () => {
    expect(keypointCount(SCHEMA)).toBe(3)
  })
})

describe('posed objects in an image', () => {
  test("a box and the points sharing its group, in the schema's order; difficult or occluded points are hidden (1), others seen (2)", () => {
    const { instances, issues } = poseInstances(image([
      point('right_eye', 1, 30, 20, { occluded: true }), box('person', 1), point('nose', 1, 25, 30),
      box('hand', 2, [[60, 60], [70, 75]]), point('thumb', 2, 65, 70, { difficult: true })
    ]), SCHEMA)
    expect(issues).toEqual([])
    expect(instances).toEqual([
      { label: 'person', group: 1, box: [10, 10, 50, 60], keypoints: [[25, 30, 2], null, [30, 20, 1]], shapes: [0, 1, 2] },
      { label: 'hand', group: 2, box: [60, 60, 70, 75], keypoints: [null, [65, 70, 1]], shapes: [3, 4] }
    ])
  })
  test("a group without a box gets the box around its points, with a warning, when only one class has them all", () => {
    const { instances, issues } = poseInstances(image([point('nose', 4, 20, 30), point('left_eye', 4, 30, 10)]), SCHEMA)
    expect(instances).toEqual([{ label: 'person', group: 4, box: [20, 10, 30, 30], keypoints: [[20, 30, 2], [30, 10, 2], null], shapes: [0, 1] }])
    expect(issues).toEqual([expect.objectContaining({ severity: 'warning', code: 'pose-no-box', shape: 0 })])
    const two = poseInstances(image([point('k', 1, 1, 1)]), { classes: { a: ['k'], b: ['k'] } })
    expect(two.instances).toEqual([])
    expect(two.issues[0]).toMatchObject({ severity: 'error', code: 'pose-no-class' })
  })
  test("a group without a box whose points lie in a line, or that has one point, is told its box has no area", () => {
    const line = poseInstances(image([point('nose', 4, 20, 30), point('left_eye', 4, 20, 10)]), SCHEMA)
    expect(line.instances[0].box).toEqual([20, 10, 20, 30])
    expect(line.issues.map((i) => i.message)).toEqual(["a.jpg: group 4 has keypoints of person but no box, and the box around its points, which lie in a line, has no area: add a box with the class's label"])
    const one = poseInstances(image([point('thumb', 2, 5, 5)]), SCHEMA)
    expect(one.issues.map((i) => [i.severity, i.code, i.message])).toEqual([
      ['warning', 'pose-no-box', "a.jpg: group 2 has keypoints of hand but no box, and the box around its one point has no area: add a box with the class's label"]
    ])
  })
  test('shapes of two classes in a group, unknown keypoint names and repeated ones are errors, which stop a pose export', () => {
    const { instances, issues } = poseInstances(image([
      box('person', 1), box('hand', 1), point('nose', 1, 20, 20),
      box('person', 2), point('tail', 2, 20, 20), point('nose', 2, 20, 20), point('nose', 2, 25, 25)
    ]), SCHEMA)
    expect(issues.map((i) => [i.severity, i.code, i.shape])).toEqual([
      ['error', 'pose-two-classes', 1], ['error', 'pose-unknown-keypoint', 4], ['error', 'pose-repeated-keypoint', 6]
    ])
    expect(issues[0].message).toBe("a.jpg: group 1 has shapes of person and hand, so it's left out: give each object its own group")
    expect(instances).toEqual([{ label: 'person', group: 2, box: [10, 10, 50, 60], keypoints: [[20, 20, 2], null, null], shapes: [3, 5] }])
  })
  test("shapes of one class in a group are one object in parts, its box around them all, as labelme draws a person hidden in part", () => {
    const { instances, issues } = poseInstances(image([
      newShape('person', 'polygon', [[10, 20], [30, 20], [20, 40]], { group_id: 0 }), newShape('person', 'polygon', [[50, 30], [70, 30], [60, 55]], { group_id: 0 }),
      point('nose', 0, 20, 25),
      // With no keypoints, as labelme's own example 2011_000003.json has one
      newShape('person', 'polygon', [[1, 1], [5, 1], [3, 6]], { group_id: 3 }), box('person', 3, [[80, 60], [90, 70]])
    ]), SCHEMA)
    expect(instances).toEqual([
      { label: 'person', group: 0, box: [10, 20, 70, 55], keypoints: [[20, 25, 2], null, null], shapes: [0, 1, 2] },
      { label: 'person', group: 3, box: [1, 1, 90, 70], keypoints: [null, null, null], shapes: [3, 4] }
    ])
    expect(issues.map((i) => [i.severity, i.code, i.shape, i.message])).toEqual([
      ['info', 'pose-parts', 0, 'a.jpg: person is 2 shapes of group 0, one object, so its box is around them all'],
      ['info', 'pose-parts', 3, 'a.jpg: person is 2 shapes of group 3, one object, so its box is around them all']
    ])
  })
  test('keypoints without a group are left out with a warning; a box without one has no keypoints; groups of other shapes are left alone', () => {
    const { instances, issues } = poseInstances(image([
      point('nose', null, 5, 5), box('person', null), newShape('car', 'polygon', [[0, 0], [5, 0], [5, 5]], { group_id: 7 }), point('wheel', 7, 1, 1)
    ]), SCHEMA)
    expect(instances).toEqual([{ label: 'person', group: null, box: [10, 10, 50, 60], keypoints: [null, null, null], shapes: [1] }])
    expect(issues.map((i) => [i.severity, i.code, i.shape])).toEqual([['warning', 'pose-no-group', 0], ['info', 'pose-no-keypoints', 1]])
  })
  test("any closed shape of a class can be the box: a polygon's around its points, a mask's around its pixels", () => {
    const mask: MaskBits = { x: 20, y: 30, width: 3, height: 2, data: Uint8Array.from([0, 1, 0, 0, 1, 1]) }
    const { instances } = poseInstances(image([
      newShape('person', 'polygon', [[10, 20], [40, 5], [30, 50]], { group_id: 1 }), point('nose', 1, 20, 20),
      newShape('hand', 'mask', [[20, 30], [22, 31]], { group_id: 2 }), point('wrist', 2, 21, 31)
    ], [null, null, mask]), SCHEMA)
    expect(instances.map((p) => p.box)).toEqual([[10, 5, 40, 50], [21, 30, 23, 32]])
  })
})

describe('posed objects as shapes, for import', () => {
  test("a rectangle and a point for each keypoint that's there, all in one group; hidden keypoints are marked difficult", () => {
    const shapes = shapesFromPose('person', [1, 2, 30, 40], [[5, 6, 2], [7, 8, 0], null, [9, 10, 1]], ['nose', 'left_eye', 'right_eye'], 3)
    expect(shapes.map((s) => [s.label, s.shape_type, s.points, s.group_id, s.flags])).toEqual([
      ['person', 'rectangle', [[1, 2], [30, 40]], 3, {}],
      ['nose', 'point', [[5, 6]], 3, {}],
      ['kp3', 'point', [[9, 10]], 3, { difficult: true }]
    ])
  })
})

describe("Python's float() and int(), which Ultralytics reads numbers with", () => {
  test('signs, exponents, underscores between digits, inf and nan, and any script\'s digits', () => {
    const cases: [string, number | null][] = [
      ['0.5', 0.5], ['+.5', 0.5], ['5.', 5], ['-5e-1', -0.5], ['1E+2', 100], ['1_000.5', 1000.5], ['1e1_0', 1e10],
      ['inf', Infinity], ['-Infinity', -Infinity], ['NaN', NaN], [' 2 ', 2], [String.fromCharCode(0x661), 1], ['0.' + String.fromCharCode(0xff15), 0.5],
      ['1__0', null], ['_1', null], ['1_', null], ['0x1', null], ['1,5', null], ['', null], ['.', null], ['e5', null]
    ]
    for (const [s, want] of cases) expect(pyFloat(s), s).toEqual(want)
    expect([pyInt('007'), pyInt('1_000'), pyInt('-3'), pyInt('1.0'), pyInt('0x1f')]).toEqual([7, 1000, -3, null, null])
  })
  test('a long value that is no number is refused in a time that grows with its length, not its square', () => {
    // A run of digits that doesn't end as a number could be split two ways, 50,000 digits taking 6 s;
    // so could a run of spaces inside a value when stripping its ends
    for (const value of ['1'.repeat(50000) + 'x', '1' + ' '.repeat(50000) + '1', '1' + '\u3000'.repeat(50000) + '1']) {
      const t = performance.now()
      expect(pyFloat(value)).toBeNull()
      expect(pyInt(value)).toBeNull()
      expect(performance.now() - t).toBeLessThan(200)
    }
    expect(pyFloat('\u3000 7.5\u2003')).toBe(7.5)
    expect(pyFloat('1'.repeat(400))).toBe(Number('1'.repeat(400)))
  })

  test("YAML's words for true, false, infinity and not a number are its own: a name such as constructor stays a name", () => {
    const names = ['constructor', 'toString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf', '__proto__', '__defineGetter__']
    expect(yaml.load(`[${names.join(', ')}]`, { schema: ULTRALYTICS_YAML })).toEqual(names)
    expect(yaml.load('[True, false, .inf, -.Inf, .nan, 1_0]', { schema: ULTRALYTICS_YAML })).toEqual([true, false, Infinity, -Infinity, NaN, 10])
  })

  test("plain values, which label files are made of, read within a few times what Number() takes", () => {
    // Every token of every label file is read this way, 40 million of them in a COCO-sized dataset
    const values = Array.from({ length: 100000 }, (_, i) => (((i * 7919) % 1000003) / 1000003).toFixed(6))
    const fastest = (read: (s: string) => number | null) => {
      let best = Infinity, sum = 0
      for (let k = 0; k < 3; k++) {
        const t = performance.now()
        for (const v of values) sum += read(v) ?? 0
        best = Math.min(best, performance.now() - t)
      }
      // The sum is used, so no read can be left out as unused
      expect(sum).toBeGreaterThan(0)
      return best
    }
    expect(fastest(pyFloat)).toBeLessThan(10 * fastest(Number))
  })
})
