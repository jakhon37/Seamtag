// AI pose: RTMPose finds a person's keypoints in each box it's given, top-down, ported from rtmlib
// (github.com/Tau-J/rtmlib at 5ea82df, rtmlib/tools/pose_estimation/) and MMPose
// (github.com/open-mmlab/mmpose at 759b39c), both Apache-2.0 (see NOTICE.md). A box is grown by a
// quarter and to the model's proportions, that part of the image is warped to the model's input,
// and each keypoint is where the model's two SimCC outputs, one across and one down, peak.
import { Tensor } from 'onnxruntime-node'
import type { PoseAnswer, PosePoint, PoseRequest } from '../shared/skeleton'
import { poseModel } from './ai'
import { aiImage, rgbRegion } from './images'

// The normalisation RTMPose was trained with, in RGB order (the zip's pipeline.json: Normalize, to_rgb
// true). rtmlib normalises OpenCV's BGR pixels as they come; sharp's are RGB, as training's were.
const MEAN = [123.675, 116.28, 103.53], STD = [58.395, 57.12, 57.375]
/** How much a box grows, for the person's surroundings (rtmlib rtmpose.py L61) */
const PADDING = 1.25
/** Bins of the SimCC outputs to a pixel of the input (rtmlib rtmpose.py L79 and L97) */
const SPLIT = 2
// Boxes go through the model together, up to 16 in a run at RTMPose-m's 256 × 192 (7 at RTMPose-l's
// 384 × 288). More in one run is no quicker on a CPU, and each box adds about 4.4 MB to a run of
// RTMPose-m and 13 MB to one of RTMPose-l, so a crowd in one run could ask for more than Electron gives.
const BATCH_PIXELS = 16 * 256 * 192

/** A box as the model takes it: its centre, and the size of the part of the image warped to the model's input */
export type Crop = { cx: number; cy: number; w: number; h: number }
/** Whole pixels of the image, from `left`, `top` */
export type Area = { left: number; top: number; width: number; height: number }
/** RGB pixels read from an area of the image, shrunk to `width` × `height` when the area is bigger */
export type Pixels = { data: Uint8Array; width: number; height: number; area: Area }

/**
 * A box's crop for a model input `width` × `height`: the box's centre, and its size grown by a
 * quarter, then widened or heightened to the input's proportions so the warp doesn't stretch the
 * person (rtmlib pre_processings.py L30-32 and L150-154). The corners may come in either order.
 */
export function cropOf([x0, y0, x1, y1]: readonly number[], width: number, height: number): Crop {
  let w = Math.abs(x1 - x0) * PADDING, h = Math.abs(y1 - y0) * PADDING
  const aspect = width / height
  if (w > h * aspect) h = w / aspect
  else w = h * aspect
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w, h }
}

/**
 * The area of an image `imageWidth` × `imageHeight` a crop's input reads, null when it's all
 * outside: input pixel u samples the image at cx + (u - width / 2) · w / width, and a sample
 * reads the whole pixels either side of it
 */
export function areaOf(crop: Crop, width: number, height: number, imageWidth: number, imageHeight: number): Area | null {
  const span = (centre: number, size: number, input: number, limit: number) => {
    const first = Math.max(0, Math.floor(centre - size / 2))
    const last = Math.min(limit - 1, Math.floor(centre - size / 2 + ((input - 1) * size) / input) + 1)
    return [first, last - first + 1]
  }
  const [left, w] = span(crop.cx, crop.w, width, imageWidth)
  const [top, h] = span(crop.cy, crop.h, height, imageHeight)
  return w > 0 && h > 0 ? { left, top, width: w, height: h } : null
}

/**
 * One box's input, NCHW from `offset`: each input pixel samples the image bilinearly where the
 * crop's warp takes it from, black outside the image, as cv2.warpAffine with INTER_LINEAR and its
 * zero border does (rtmlib pre_processings.py L160-163), and is rounded as that warp's 8-bit result
 * is; then normalised by RTMPose's mean and spread (rtmlib rtmpose.py L67-70), in RGB order.
 * `pixels` is null when the crop is all outside the image.
 */
export function fillInput(input: Float32Array, offset: number, pixels: Pixels | null, crop: Crop, width: number, height: number) {
  const plane = width * height
  if (!pixels) {
    for (let c = 0; c < 3; c++) input.fill(-MEAN[c] / STD[c], offset + c * plane, offset + (c + 1) * plane)
    return
  }
  const { data, width: pw, height: ph, area } = pixels
  // A point of the image in the pixels' own grid, whose pixels may each cover several of the image's
  const sx = pw / area.width, sy = ph / area.height
  const kx = crop.w / width, ky = crop.h / height
  const at = (x: number, y: number) => (x >= 0 && y >= 0 && x < pw && y < ph ? (y * pw + x) * 3 : -1)
  for (let v = 0; v < height; v++) {
    const y = (crop.cy + (v - height / 2) * ky - area.top + 0.5) * sy - 0.5
    const y0 = Math.floor(y), fy = y - y0
    for (let u = 0; u < width; u++) {
      const x = (crop.cx + (u - width / 2) * kx - area.left + 0.5) * sx - 0.5
      const x0 = Math.floor(x), fx = x - x0
      const a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1)
      for (let k = 0; k < 3; k++) {
        const value = (1 - fy) * ((1 - fx) * (a < 0 ? 0 : data[a + k]) + fx * (b < 0 ? 0 : data[b + k])) +
          fy * ((1 - fx) * (c < 0 ? 0 : data[c + k]) + fx * (d < 0 ? 0 : data[d + k]))
        input[offset + k * plane + v * width + u] = (Math.round(value) - MEAN[k]) / STD[k]
      }
    }
  }
}

/** The first of the highest values in a row of `length` from `start`, and that value */
function peak(values: Float32Array, start: number, length: number): [number, number] {
  let at = 0, best = -Infinity
  for (let i = 0; i < length; i++) {
    if (values[start + i] > best) {
      best = values[start + i]
      at = i
    }
  }
  return [at, best]
}

/**
 * Box `box`'s keypoints from the model's outputs, which have `keypoints` rows each: each axis at
 * its highest bin, two bins to an input pixel (rtmlib rtmpose.py L96-97); scored by the lower of
 * the two highest values, as MMPose scores them (mmpose codecs/utils/post_processing.py L90-99),
 * and none where that's 0 or less; then from the input back to the image (rtmlib rtmpose.py
 * L100-101, mmpose models/pose_estimators/topdown.py L154-156). The scores aren't probabilities.
 */
export function decode(simccX: Float32Array, simccY: Float32Array, box: number, keypoints: number, crop: Crop, width: number, height: number): PosePoint[] {
  const binsX = SPLIT * width, binsY = SPLIT * height
  const out: PosePoint[] = []
  for (let k = 0; k < keypoints; k++) {
    const [bx, mx] = peak(simccX, (box * keypoints + k) * binsX, binsX)
    const [by, my] = peak(simccY, (box * keypoints + k) * binsY, binsY)
    const score = Math.min(mx, my)
    out.push(score > 0 ? { x: (bx / SPLIT / width) * crop.w + crop.cx - crop.w / 2, y: (by / SPLIT / height) * crop.h + crop.cy - crop.h / 2, score } : null)
  }
  return out
}

/** The area that holds all the others, null when there are none */
function union(areas: (Area | null)[]): Area | null {
  const found = areas.filter((a): a is Area => a !== null)
  if (!found.length) return null
  const left = Math.min(...found.map((a) => a.left)), top = Math.min(...found.map((a) => a.top))
  const right = Math.max(...found.map((a) => a.left + a.width)), bottom = Math.max(...found.map((a) => a.top + a.height))
  return { left, top, width: right - left, height: bottom - top }
}

/** An area's pixels, shrunk so its longer side is no more than `maxSide` */
async function read(file: string, area: Area, maxSide: number): Promise<Pixels> {
  const shrink = maxSide / Math.max(area.width, area.height)
  const size: [number, number] | undefined = shrink < 1 ? [Math.max(1, Math.round(area.width * shrink)), Math.max(1, Math.round(area.height * shrink))] : undefined
  return { ...(await rgbRegion(file, area, size)), area }
}

/** Each box's keypoints in the image's pixels, as the boxes are, in the request's order; a box of no size has none */
export async function poseInBoxes(req: PoseRequest): Promise<PoseAnswer> {
  if (!req.boxes.every((b) => b.length === 4 && b.every(Number.isFinite))) throw new Error('A box is four numbers: x0, y0, x1, y1')
  const { session, keypoints, maxSide } = await poseModel(req.model)
  // The input's size from the model itself: the zips' pipeline.json gets it wrong for some models
  const meta = session.inputMetadata.find((m) => m.name === 'input')
  const [, , height, width] = meta?.isTensor ? meta.shape : []
  if (typeof width !== 'number' || typeof height !== 'number') throw new Error("The pose model doesn't take an image of a fixed size")
  const crops = req.boxes.map((b) => cropOf(b, width, height))
  const image = await aiImage(req.root, req.name)
  const areas = crops.map((c) => areaOf(c, width, height, image.width, image.height))
  // One read of the part every box needs when that's small enough, as for a few people near each
  // other; otherwise each box's part on its own, so a large image is never read whole
  const all = union(areas)
  const shared = all && Math.max(all.width, all.height) <= maxSide ? await read(image.file, all, maxSide) : null
  const plane = 3 * width * height
  const batch = Math.max(1, Math.floor(BATCH_PIXELS / (width * height)))
  const poses: PosePoint[][] = []
  for (let first = 0; first < crops.length; first += batch) {
    const count = Math.min(batch, crops.length - first)
    const input = new Float32Array(count * plane)
    for (let i = 0; i < count; i++) {
      const area = areas[first + i]
      fillInput(input, i * plane, shared ?? (area && (await read(image.file, area, maxSide))), crops[first + i], width, height)
    }
    const out = await session.run({ input: new Tensor('float32', input, [count, 3, height, width]) })
    const { simcc_x: x, simcc_y: y } = out
    const fits = (t: Tensor | undefined, bins: number) => t?.type === 'float32' && t.dims.join() === [count, keypoints.length, bins].join()
    if (!fits(x, SPLIT * width) || !fits(y, SPLIT * height)) throw new Error("The pose model's results aren't shaped as RTMPose's are")
    for (let i = 0; i < count; i++) {
      const crop = crops[first + i]
      poses.push(crop.w > 0 ? decode(x.data as Float32Array, y.data as Float32Array, i, keypoints.length, crop, width, height) : keypoints.map(() => null))
    }
  }
  return { names: keypoints, poses }
}
