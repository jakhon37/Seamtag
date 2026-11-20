import type { KeypointSchema } from './dataset'
import { isAi, scoreOf, type Shape } from './labelfile'
import { posePoints, posesIn } from './skeleton'

export type Summary = {
  /** Accepted shapes. AI shapes stay out of the counts until accepted, as in training. */
  count: number
  ai: number
  /** The least certain AI shape's score */
  score: number | null
  /** Accepted shapes of each class, which the Labels dock adds up across the folder */
  classes: Record<string, number>
  /** A label file exists, so the image has been looked at: zero shapes means truly empty */
  labelled: boolean
}

/**
 * What the image list shows. With the folder's `keypoints`, its poses' keypoints (skeleton.ts'
 * posePoints) are left out of the count and of `classes`: a pose counts once, by its box, and
 * its keypoints name keypoints, not classes. Any other shape labelled like a keypoint is of a
 * class as usual.
 */
export function summarize(shapes: Shape[], labelled = true, keypoints?: KeypointSchema | null): Summary {
  const classes: Record<string, number> = {}
  let ai = 0
  let score: number | null = null
  const keypointAt = keypoints ? posePoints(posesIn(shapes, keypoints)) : null
  let count = 0
  for (const [i, s] of shapes.entries()) {
    if (keypointAt?.has(i)) continue
    if (isAi(s)) {
      ai++
      const v = scoreOf(s)
      if (v !== undefined) score = score === null ? v : Math.min(score, v)
    } else {
      count++
      classes[s.label] = (classes[s.label] ?? 0) + 1
    }
  }
  return { count, ai, score, classes, labelled }
}

export const EMPTY_SUMMARY: Summary = { count: 0, ai: 0, score: null, classes: {}, labelled: false }

/** Does the shape cover pixels, as a model learns from it? Lines and points don't, nor a shape short of points. */
export function hasArea(s: Shape) {
  const n = s.points.length
  if (s.shape_type === 'polygon' || s.shape_type === 'oriented_rectangle') return n >= 3
  if (s.shape_type === 'rectangle' || s.shape_type === 'circle') return n >= 2
  return s.shape_type === 'mask' && n >= 2 && !!s.mask
}
