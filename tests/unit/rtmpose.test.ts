import { createHash } from 'crypto'
import { copyFileSync, existsSync, mkdtempSync, readFileSync, statSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { MODELS, release } from '../../src/main/ai'
import { openProject } from '../../src/main/project'
import { areaOf, cropOf, decode, fillInput, poseInBoxes, type Pixels } from '../../src/main/rtmpose'
import { COCO_PERSON, type PosePoint } from '../../src/shared/skeleton'
import { labelmeFile } from '../labelme-files'

const MEAN = [123.675, 116.28, 103.53], STD = [58.395, 57.12, 57.375]
const close = (a: number, b: number, within: number) => expect(Math.abs(a - b), `${a} against ${b}`).toBeLessThanOrEqual(within)

describe('the crop: the box grown by a quarter, then to the input’s proportions', () => {
  test('a tall box is widened, a wide one heightened, about its centre', () => {
    // 100 x 400 grows to 125 x 500, then to 375 x 500, as 192 / 256 is 0.75
    expect(cropOf([100, 50, 200, 450], 192, 256)).toEqual({ cx: 150, cy: 250, w: 375, h: 500 })
    const wide = cropOf([0, 0, 400, 100], 192, 256)
    expect(wide).toMatchObject({ cx: 200, cy: 50, w: 500 })
    close(wide.h, 500 / 0.75, 1e-9)
    // RTMPose-l's 288 x 384 has the same proportions; a square input makes a square crop
    expect(cropOf([100, 50, 200, 450], 288, 384)).toEqual(cropOf([100, 50, 200, 450], 192, 256))
    expect(cropOf([0, 0, 400, 100], 256, 256)).toEqual({ cx: 200, cy: 50, w: 500, h: 500 })
  })

  test('corners in either order', () => {
    expect(cropOf([200, 450, 100, 50], 192, 256)).toEqual(cropOf([100, 50, 200, 450], 192, 256))
  })

  test('the area of the image its samples read, inside the image', () => {
    // Samples at 6, 8, 10 and 12 read pixels 6 to 13
    expect(areaOf({ cx: 10, cy: 10, w: 8, h: 8 }, 4, 4, 100, 100)).toEqual({ left: 6, top: 6, width: 8, height: 8 })
    // At -3, -1, 1 and 3: pixels 0 to 4 are in the image
    expect(areaOf({ cx: 1, cy: 1, w: 8, h: 8 }, 4, 4, 100, 100)).toEqual({ left: 0, top: 0, width: 5, height: 5 })
    expect(areaOf({ cx: 98, cy: 50, w: 8, h: 8 }, 4, 4, 100, 100)).toEqual({ left: 94, top: 46, width: 6, height: 8 })
    expect(areaOf({ cx: -50, cy: 10, w: 8, h: 8 }, 4, 4, 100, 100)).toBeNull()
    expect(areaOf({ cx: 10, cy: 150, w: 8, h: 8 }, 4, 4, 100, 100)).toBeNull()
  })
})

describe('the warp to the input', () => {
  // A 4 x 3 image whose channels each change evenly across and down, so bilinear sampling inside it is
  // exact, and whole at the half pixels sampled here
  const f = (x: number, y: number) => [10 + 20 * x + 60 * y, 200 - 30 * x + 6 * y, 50 + 40 * y]
  const image = (width: number, height: number): Pixels => {
    const data = new Uint8Array(width * height * 3)
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data.set(f(x, y), (y * width + x) * 3)
    return { data, width, height, area: { left: 0, top: 0, width, height } }
  }
  /** The input's channel values, back from the normalisation, input pixel by input pixel */
  const values = (input: Float32Array, width: number, height: number) =>
    Array.from({ length: width * height }, (_, i) => [0, 1, 2].map((c) => Math.round(input[c * width * height + i] * STD[c] + MEAN[c]) + 0))

  test('each input pixel samples the image bilinearly, normalised in RGB order', () => {
    const input = new Float32Array(3 * 16)
    // Samples at x = 0.5, 1, 1.5, 2 and y = 0, 0.5, 1, 1.5
    fillInput(input, 0, image(4, 3), { cx: 1.5, cy: 1, w: 2, h: 2 }, 4, 4)
    const want = [0, 0.5, 1, 1.5].flatMap((y) => [0.5, 1, 1.5, 2].map((x) => f(x, y)))
    expect(values(input, 4, 4)).toEqual(want)
    // Red's first value as it went in: (f(0.5, 0)[0] - mean) / std
    close(input[0], (20 - MEAN[0]) / STD[0], 1e-6)
    close(input[16], (185 - MEAN[1]) / STD[1], 1e-6)
  })

  test('outside the image it reads black, and half a pixel out it mixes the edge with black', () => {
    const input = new Float32Array(3 * 16)
    // Samples at x = -1.5, -0.5, 0.5, 1.5 and y = -2, -1, 0, 1
    fillInput(input, 0, image(4, 3), { cx: 0.5, cy: 0, w: 4, h: 4 }, 4, 4)
    const got = values(input, 4, 4)
    for (let u = 0; u < 4; u++) for (const v of [0, 1]) expect(got[v * 4 + u]).toEqual([0, 0, 0])
    expect(got[2 * 4 + 0]).toEqual([0, 0, 0])
    expect(got[2 * 4 + 1]).toEqual(f(0, 0).map((c) => c / 2))
    expect(got[3 * 4 + 1]).toEqual(f(0, 1).map((c) => c / 2))
    expect(got[3 * 4 + 2]).toEqual(f(0.5, 1))
  })

  test('a box wholly outside the image is all black', () => {
    const input = new Float32Array(3 * 16).fill(9)
    fillInput(input, 0, null, { cx: -50, cy: -50, w: 4, h: 4 }, 4, 4)
    expect(values(input, 4, 4)).toEqual(Array(16).fill([0, 0, 0]))
  })

  test('pixels read shrunk are sampled in their own grid, about pixel centres', () => {
    // An area of 8 x 6 at (10, 20) read as 4 x 3: each pixel read covers 2 x 2 of the image's
    const shrunk = { ...image(4, 3), area: { left: 10, top: 20, width: 8, height: 6 } }
    const input = new Float32Array(3 * 4)
    // Samples at the image's x 9.5, 11.5, 13.5, 15.5 and y 20.5: the read pixels' -0.5, 0.5, 1.5, 2.5 and 0
    fillInput(input, 0, shrunk, { cx: 13.5, cy: 21.5, w: 8, h: 2 }, 4, 1)
    expect(values(input, 4, 1)).toEqual([f(0, 0).map((c) => c / 2), f(0.5, 0), f(1.5, 0), f(2.5, 0)])
  })

  test('each box fills its own part of a batch', () => {
    const input = new Float32Array(2 * 3 * 16)
    fillInput(input, 3 * 16, image(4, 3), { cx: 1.5, cy: 1, w: 2, h: 2 }, 4, 4)
    expect(input.subarray(0, 48).every((v) => v === 0)).toBe(true)
    expect(values(input.subarray(48), 4, 4)[0]).toEqual(f(0.5, 0))
  })
})

describe("reading the model's outputs", () => {
  // Two boxes, three keypoints, an input of 4 x 4: 8 bins across and 8 down
  const W = 4, H = 4, K = 3
  const x = new Float32Array(2 * K * 2 * W), y = new Float32Array(2 * K * 2 * H)
  const set = (out: Float32Array, box: number, k: number, bin: number, value: number) => { out[(box * K + k) * 8 + bin] = value }
  // Box 1, keypoint 0: across at bin 3, down at bin 6, so input (1.5, 3), scored by the lower maximum
  set(x, 1, 0, 3, 0.9); set(y, 1, 0, 6, 0.7)
  // Keypoint 1: two highest across, the first taken
  set(x, 1, 1, 2, 0.8); set(x, 1, 1, 5, 0.8); set(y, 1, 1, 1, 0.6)
  // Keypoint 2: the mean of its maxima is above 0, the lower is not: MMPose leaves it out
  set(x, 1, 2, 4, 0.5); y.fill(-0.1, (1 * K + 2) * 8, (1 * K + 3) * 8)
  // Box 0's keypoints peak at 0 exactly: none
  const crop = { cx: 100, cy: 200, w: 40, h: 40 }

  test('each axis at its peak, two bins to an input pixel, mapped back through the crop', () => {
    const [a, b, c] = decode(x, y, 1, K, crop, W, H)
    // 1.5 / 4 * 40 + 100 - 20 and 3 / 4 * 40 + 200 - 20
    expect(a).toEqual({ x: 95, y: 210, score: Math.fround(0.7) })
    expect(b).toEqual({ x: 90, y: 185, score: Math.fround(0.6) })
    expect(c).toBeNull()
  })

  test('a score of 0 or less is no keypoint', () => {
    expect(decode(x, y, 0, K, crop, W, H)).toEqual([null, null, null])
  })
})

// The pose path whole, with tests/fixtures/models/pose.onnx (make_pose.py): RTMPose-m's shape, its
// keypoints peaking where discs of their colour are, the head's red, the arms' green, the legs' blue
const FIXTURE = path.resolve(__dirname, '../fixtures/models/pose.onnx')
const REAL_MODELS = process.env.SEAMTAG_MODELS
type Person = Record<'head' | 'body' | 'legs', [number, number]>
const GROUP = COCO_PERSON.map((_, k): keyof Person => (k < 5 ? 'head' : k < 11 ? 'body' : 'legs'))
const COLOURS = { head: '#ff0000', body: '#00ff00', legs: '#0000ff' }

/** A black image with each person's three discs */
function picture(width: number, height: number, people: Person[], r = 18) {
  const discs = people.flatMap((p) => (['head', 'body', 'legs'] as const).map((g) => `<circle cx="${p[g][0]}" cy="${p[g][1]}" r="${r}" fill="${COLOURS[g]}"/>`))
  return sharp({ create: { width, height, channels: 3, background: '#000000' } }).composite([{ input: Buffer.from(`<svg width="${width}" height="${height}">${discs.join('')}</svg>`) }])
}

/** Each keypoint on its colour's disc, but the two ears, which the fixture leaves out */
function expectOn(pose: PosePoint[], person: Person, within: number) {
  pose.forEach((p, k) => {
    if (k === 3 || k === 4) return expect(p, COCO_PERSON[k]).toBeNull()
    expect(p, COCO_PERSON[k]).not.toBeNull()
    close(p!.x, person[GROUP[k]][0], within)
    close(p!.y, person[GROUP[k]][1], within)
    expect(p!.score).toBeGreaterThan(5)
  })
}

describe('poses with a small model shaped like RTMPose', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-pose-'))
  const models = mkdtempSync(path.join(tmpdir(), 'seamtag-pose-models-'))
  const a: Person = { head: [150, 110], body: [150, 210], legs: [150, 330] }
  const b: Person = { head: [470, 90], body: [470, 190], legs: [470, 300] }
  const boxA: [number, number, number, number] = [100, 70, 200, 400], boxB: [number, number, number, number] = [420, 50, 520, 360]
  const left: Person = { head: [150, 60], body: [150, 140], legs: [150, 230] }, right: Person = { head: [4300, 60], body: [4300, 140], legs: [4300, 230] }
  const ask = (name: string, boxes: [number, number, number, number][], model = 'test-pose') => poseInBoxes({ root: dir, name, model, boxes })

  beforeAll(async () => {
    const hash = createHash('sha256').update(readFileSync(FIXTURE)).digest('hex')
    copyFileSync(FIXTURE, path.join(models, `sha256-${hash}`))
    process.env.SEAMTAG_MODELS = models
    const blob = { url: 'https://example.invalid/pose.onnx', hash: `sha256:${hash}`, size: statSync(FIXTURE).size }
    const def = { title: 'Test pose', kind: 'rtmpose' as const, points: false, box: false, text: false, keypoints: COCO_PERSON, license: { name: 'none', url: '' } }
    MODELS.push({ ...def, name: 'test-pose', blobs: { model: blob } }, { ...def, name: 'test-pose-missing', blobs: { model: { ...blob, hash: `sha256:${'0'.repeat(64)}` } } })
    await picture(640, 480, [a, b]).png().toFile(path.join(dir, 'two.png'))
    // Stored turned a quarter anticlockwise, with EXIF saying to turn it back as it's shown
    const upright = await picture(640, 480, [a, b]).png().toBuffer()
    await sharp(upright).rotate(270).withMetadata({ orientation: 6 }).jpeg({ quality: 95 }).toFile(path.join(dir, 'turned.jpg'))
    expect((await sharp(path.join(dir, 'turned.jpg')).metadata()).orientation).toBe(6)
    // Two people farther apart than one read takes, and one lying across a picture wider than that
    await picture(4400, 300, [left, right]).png().toFile(path.join(dir, 'wide.png'))
    await picture(4400, 600, [{ head: [1000, 300], body: [2200, 300], legs: [3400, 300] }], 120).png().toFile(path.join(dir, 'lying.png'))
    await openProject(dir, null)
  })
  afterAll(async () => {
    await release()
    // Set to undefined, the variable would read "undefined"
    if (REAL_MODELS === undefined) delete process.env.SEAMTAG_MODELS
    else process.env.SEAMTAG_MODELS = REAL_MODELS
    MODELS.splice(MODELS.findIndex((m) => m.name === 'test-pose'), 2)
  })

  test("each box's keypoints, in the boxes' order, named in the model's order", async () => {
    const answer = await ask('two.png', [boxA, boxB])
    expect(answer.names).toEqual(COCO_PERSON)
    expect(answer.poses).toHaveLength(2)
    expectOn(answer.poses[0], a, 1.5)
    expectOn(answer.poses[1], b, 1.5)
    expect((await ask('two.png', [boxB, boxA])).poses).toEqual([answer.poses[1], answer.poses[0]])
  })

  test('a photo stored turned, in the pixels it is shown in', async () => {
    const answer = await ask('turned.jpg', [boxA, boxB])
    expectOn(answer.poses[0], a, 2)
    expectOn(answer.poses[1], b, 2)
  })

  test('a box partly outside the image', async () => {
    // The crop reaches past the left edge and the top, where it reads black
    const { poses: [pose] } = await ask('two.png', [[-60, -40, 200, 400]])
    expectOn(pose, a, 2.5)
  })

  test('more boxes than go in one run', async () => {
    const boxes = Array.from({ length: 21 }, (_, i) => (i % 3 ? boxA : boxB))
    const { poses } = await ask('two.png', boxes)
    expect(poses).toHaveLength(21)
    const [one, two] = (await ask('two.png', [boxA, boxB])).poses
    poses.forEach((p, i) => expect(p).toEqual(i % 3 ? one : two))
  })

  test('boxes far apart are each read on their own; a box larger than one read is read shrunk', async () => {
    // Their parts of the image span 4358 pixels across, more than the 4096 one read takes
    const { poses } = await ask('wide.png', [[100, 30, 200, 260], [4250, 30, 4350, 260]])
    expectOn(poses[0], left, 1.5)
    expectOn(poses[1], right, 1.5)
    // A crop of 4500 x 6000 in an image 4400 wide: an input pixel is 23 of the image's
    const { poses: [lying] } = await ask('lying.png', [[400, 100, 4000, 500]])
    expectOn(lying, { head: [1000, 300], body: [2200, 300], legs: [3400, 300] }, 15)
  })

  test('a box of no size has no keypoints; a box must be four numbers', async () => {
    const { poses } = await ask('two.png', [[150, 150, 150, 150], boxA])
    expect(poses[0]).toEqual(COCO_PERSON.map(() => null))
    expectOn(poses[1], a, 1.5)
    await expect(ask('two.png', [[0, 0, Number.NaN, 10]])).rejects.toThrow('four numbers')
  })

  test('a model that finds no poses, or isn’t downloaded', async () => {
    await expect(ask('two.png', [boxA], 'sam2:small')).rejects.toThrow("doesn't find poses")
    await expect(ask('two.png', [boxA], 'test-pose-missing')).rejects.toThrow("isn't downloaded yet")
    await expect(ask('two.png', [boxA], 'nothing')).rejects.toThrow('Unknown model')
  })
})

// The real models, when SEAMTAG_MODELS has them, on the photo of two men from labelme's examples:
// both face the camera, so each one's left is on the photo's right
const REAL = [
  { name: 'rtmpose-m', hash: '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c' },
  { name: 'rtmpose-l', hash: '62425affc6b9edba8f8a65f55f5e278fec5d905d0c90578e0f8503bdf3954587' }
]
for (const model of REAL) {
  const has = !!REAL_MODELS && existsSync(path.join(REAL_MODELS, `sha256-${model.hash}`))
  test.skipIf(!has)(`${model.name} finds each man's keypoints on him (needs it in SEAMTAG_MODELS)`, async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-pose-photo-'))
    copyFileSync(await labelmeFile('v5.11.4', 'examples/instance_segmentation/data_annotated/2011_000003.jpg'), path.join(dir, 'p.jpg'))
    await openProject(dir, null)
    const boxes: [number, number, number, number][] = [[192, 107, 314, 327], [366, 87, 500, 337]]
    const { names, poses } = await poseInBoxes({ root: dir, name: 'p.jpg', model: model.name, boxes })
    expect(names).toEqual(COCO_PERSON)
    for (const [i, pose] of poses.entries()) {
      const at = (n: string) => pose[COCO_PERSON.indexOf(n)]!
      for (const n of ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']) {
        expect(at(n), n).not.toBeNull()
        expect(at(n).score, n).toBeGreaterThan(0.3)
        const [x0, y0, x1, y1] = boxes[i]
        expect(at(n).x, n).toBeGreaterThan(x0 - 10)
        expect(at(n).x, n).toBeLessThan(x1 + 10)
        expect(at(n).y, n).toBeGreaterThan(y0 - 10)
        expect(at(n).y, n).toBeLessThan(y1 + 10)
      }
      // Head above shoulders above hips; each man's left on the photo's right
      expect(at('nose').y).toBeLessThan(Math.min(at('left_shoulder').y, at('right_shoulder').y))
      expect(Math.max(at('left_shoulder').y, at('right_shoulder').y)).toBeLessThan(Math.min(at('left_hip').y, at('right_hip').y))
      expect(at('left_eye').x).toBeGreaterThan(at('right_eye').x)
      expect(at('left_shoulder').x).toBeGreaterThan(at('right_shoulder').x)
      expect(at('left_hip').x).toBeGreaterThan(at('right_hip').x)
    }
    await release()
  })
}
