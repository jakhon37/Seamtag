// Poses in the editor, kept apart from the window so they can be tested: what the Pose tool adds,
// what AI pose proposes, what the pose keys change, how the Shapes list and review take a pose as
// one, and what LabelImg's formats keep of one. A pose is a box and the points sharing its
// group_id, each named for a keypoint of the box's class (src/shared/skeleton.ts); a hidden point
// is marked difficult.
import type { KeypointSchema } from '../../shared/dataset'
import { asBox, boxOf } from '../../shared/formats'
import { accepted, isAi, newShape, scoreOf, type Shape } from '../../shared/labelfile'
import { CLOSED_TYPES } from '../../shared/outline'
import { nextGroupId, type PoseGroup, type PosePoint, poseShapes, posesIn } from '../../shared/skeleton'

/** A keypoint's name as the window says it: left shoulder for left_shoulder */
export const spoken = (name: string) => name.replace(/_/g, ' ')

/** The pose each of an image's shapes is part of, by index */
export function poseIndex(poses: PoseGroup[]): Map<number, PoseGroup> {
  const out = new Map<number, PoseGroup>()
  for (const p of poses) for (const i of poseShapes(p)) out.set(i, p)
  return out
}

/** The shapes chosen, and the points of each pose whose box is among them: a box moves, copies and goes with its points */
export function withPoints(which: number[], poses: PoseGroup[]): number[] {
  const out = [...which]
  const chosen = new Set(which)
  for (const p of poses) {
    if (p.box === null || !chosen.has(p.box)) continue
    for (const i of poseShapes(p)) {
      if (chosen.has(i)) continue
      chosen.add(i)
      out.push(i)
    }
  }
  return out
}

/**
 * Copies of shapes, each pose among them under a group_id of its own, after `existing`'s, so a
 * pasted or duplicated pose is a pose of its own. Other groups keep theirs.
 */
export function regroup(copies: Shape[], existing: Shape[], schema: KeypointSchema): Shape[] {
  const groups = new Set(posesIn(copies, schema).map((p) => p.group))
  if (!groups.size) return copies
  let next = nextGroupId([...existing, ...copies.filter((s) => s.group_id === null || !groups.has(s.group_id))])
  const fresh = new Map([...groups].map((g) => [g, next++]))
  return copies.map((s) => (s.group_id !== null && fresh.has(s.group_id) ? { ...s, group_id: fresh.get(s.group_id)! } : s))
}

/**
 * Shapes after an edit, with each AI pose it touched accepted whole: review takes a pose as one,
 * so a point corrected by hand mustn't stay behind alone when the rest is rejected. `touched` are
 * the edited shapes, by index in `shapes`, and `poses` the poses those indexes had.
 */
export function acceptTouched(shapes: Shape[], poses: PoseGroup[], touched: Iterable<number>): Shape[] {
  const of = poseIndex(poses)
  const whole = new Set<number>()
  for (const i of touched) for (const k of of.has(i) ? poseShapes(of.get(i)!) : []) whole.add(k)
  if (![...whole].some((k) => shapes[k] && isAi(shapes[k]))) return shapes
  return shapes.map((s, k) => (whole.has(k) && isAi(s) ? accepted(s) : s))
}

/**
 * What LabelImg's formats keep of shapes: they hold boxes, so each shape becomes the box around
 * it, and a point, which has no box, is left out; a pose keeps its box. With how many became
 * boxes and how many points were left out, for the window to say.
 */
export function keptAsBoxes(list: Shape[]): { shapes: Shape[]; boxed: number; left: number } {
  const shapes = list.filter((s) => s.shape_type !== 'point').map(asBox)
  return { shapes, boxed: list.filter((s) => s.shape_type !== 'point' && s.shape_type !== 'rectangle').length, left: list.length - shapes.length }
}

/** A person to pose: the box around their shapes, those shapes by index, and their group_id, null for one shape in no group */
export type PoseTarget = { box: [number, number, number, number]; shapes: number[]; group: number | null }

/**
 * The people of a class still to pose in an image, in the image's order: each group's closed
 * shapes of the class as one person, in the box around them, as the checks and the pose exports
 * box a pose, and each such shape in no group on its own. Groups with keypoints already are left
 * alone, and so are hidden shapes. With `which`, only the people those shapes are part of.
 */
export function unposed(shapes: Shape[], poses: PoseGroup[], cls: string, which?: number[]): PoseTarget[] {
  const posed = new Set(poses.filter((p) => p.cls === cls).map((p) => p.group))
  const people = new Map<string, number[]>()
  shapes.forEach((s, i) => {
    if (s.label !== cls || !CLOSED_TYPES.has(s.shape_type) || s.hidden || (s.group_id !== null && posed.has(s.group_id))) return
    const key = s.group_id === null ? `shape ${i}` : `group ${s.group_id}`
    const members = people.get(key)
    if (members) members.push(i)
    else people.set(key, [i])
  })
  const chosen = which && new Set(which)
  return [...people.values()].filter((members) => !chosen || members.some((i) => chosen.has(i))).map((members) => {
    const boxes = members.map((i) => boxOf(shapes[i]))
    const box: PoseTarget['box'] = [
      Math.min(...boxes.map((b) => b[0])), Math.min(...boxes.map((b) => b[1])), Math.max(...boxes.map((b) => b[2])), Math.max(...boxes.map((b) => b[3]))
    ]
    return { box, shapes: members, group: shapes[members[0]].group_id }
  })
}

/**
 * A keypoint hidden (marked difficult, as X-AnyLabeling writes it) or seen. Seen takes occluded
 * away too, which reads as hidden; a flag the label's label_flags list stays, unticked.
 */
export function withHidden(s: Shape, hidden: boolean, listed: Record<string, boolean> = {}): Shape {
  if (hidden) return { ...s, flags: { ...s.flags, difficult: true } }
  const { difficult: _d, occluded: _o, ...flags } = s.flags
  for (const k of ['difficult', 'occluded']) if (k in listed) flags[k] = false
  return { ...s, flags }
}

/**
 * Left and right swapped on a pose: the two points of each flip pair trade names and stay where
 * they are. With `only`, just the pairs with a point among those shapes.
 */
export function swapSides(shapes: Shape[], pose: PoseGroup, schema: KeypointSchema, only?: Set<number>): Shape[] {
  const names = schema.classes[pose.cls] ?? []
  const at = (name: string) => {
    const k = names.indexOf(name)
    return k < 0 ? null : pose.points[k]
  }
  let out = shapes
  for (const [a, b] of schema.flip?.[pose.cls] ?? []) {
    const ia = at(a), ib = at(b)
    if (ia === null && ib === null) continue
    if (only && !(ia !== null && only.has(ia)) && !(ib !== null && only.has(ib))) continue
    if (out === shapes) out = [...shapes]
    if (ia !== null) out[ia] = { ...accepted(out[ia]), label: b }
    if (ib !== null) out[ib] = { ...accepted(out[ib]), label: a }
  }
  return out
}

/** A pose's points mirrored left to right inside its box, keeping their names; without a box, inside the points' own width */
export function mirrorPose(shapes: Shape[], pose: PoseGroup): Shape[] {
  const points = [...pose.points.filter((i): i is number => i !== null), ...pose.extra]
  if (!points.length) return shapes
  const xs = pose.box !== null ? [boxOf(shapes[pose.box])[0], boxOf(shapes[pose.box])[2]] : points.map((i) => shapes[i].points[0][0])
  const axis = Math.min(...xs) + Math.max(...xs)
  const out = [...shapes]
  for (const i of points) {
    const [x, y] = out[i].points[0]
    out[i] = { ...accepted(out[i]), points: [[axis - x, y]] }
  }
  return out
}

/** The keypoints file's class for a pose model's keypoints: the one sharing most of their names, or null when none shares any */
export function classForModel(schema: KeypointSchema, names: string[]): string | null {
  let best: string | null = null
  let most = 0
  for (const [cls, list] of Object.entries(schema.classes)) {
    const shared = list.filter((k) => names.includes(k)).length
    if (shared > most) {
      best = cls
      most = shared
    }
  }
  return best
}

const rounded = (score: number) => Math.round(score * 1000) / 1000

/**
 * The fewest keypoints an AI pose keeps for it to be proposed: fewer say little of a pose, and on
 * something that isn't a person, such as a bus, they're noise
 */
export const MIN_POSE_POINTS = 5

export type PoseFit = {
  /** The image's size: a point the model placed past its edge isn't labelled, as COCO leaves a body part outside the photo */
  size: [number, number]
  /** Keypoints scored below this are left out */
  threshold: number
  group: number
  /** The flags a label gets from label_flags */
  flagsOf?: (label: string) => Record<string, boolean>
}

/**
 * A pose model's keypoints for one box as AI shapes: a point for each of the class's keypoints
 * the model placed on the image with at least the threshold's score, in the class's order, each
 * with its score; and their mean score, for the box. Keypoints the class doesn't have are left
 * out, and so are those past the image's edge: a person the photo cuts off has their feet guessed
 * below it, and COCO leaves such parts not labelled (visibility 0) rather than at the edge. Half a
 * pixel past it is still on the image, at its edge.
 */
export function aiPosePoints(schema: KeypointSchema, cls: string, names: string[], found: PosePoint[], fit: PoseFit): { points: Shape[]; score: number | null } {
  const [w, h] = fit.size
  const points: Shape[] = []
  let sum = 0
  for (const name of schema.classes[cls] ?? []) {
    const p = found[names.indexOf(name)]
    if (!p || !(p.score >= fit.threshold) || !Number.isFinite(p.x) || !Number.isFinite(p.y)) continue
    if (p.x < -0.5 || p.y < -0.5 || p.x > w + 0.5 || p.y > h + 0.5) continue
    const x = Math.max(0, Math.min(w, p.x)), y = Math.max(0, Math.min(h, p.y))
    points.push(newShape(name, 'point', [[x, y]], {
      group_id: fit.group,
      flags: { ...fit.flagsOf?.(name), ai: true },
      other: { score: rounded(p.score) }
    }))
    sum += p.score
  }
  return { points, score: points.length ? rounded(sum / points.length) : null }
}

/** The Pose tool's work on one pose: the keypoints still to place, asked for one at a time */
export type PoseSession = {
  cls: string
  group: number
  /** The pose's box: drawn now and still to be added when `fresh`, else the image's own, which the points join */
  box: Shape | null
  fresh: boolean
  /** The keypoints asked for, by index in the class's list, in order */
  asked: number[]
  /** An answer for each keypoint asked so far: its point, or null when it was skipped */
  done: (Shape | null)[]
}

/** The keypoint asked for now, by index in the class's list; null once every one is answered */
export const askedNow = (s: PoseSession) => (s.done.length < s.asked.length ? s.asked[s.done.length] : null)

/**
 * The image's shapes with a session's work added at the end: a box drawn now, then the points
 * placed, in the class's order. A box the image had takes the pose's group_id once a point joins it.
 */
export function withSession(shapes: Shape[], s: PoseSession): Shape[] {
  const points = s.done.filter((p): p is Shape => p !== null)
  if (s.fresh && s.box) return [...shapes, s.box, ...points]
  if (!points.length) return shapes
  const out = [...shapes, ...points]
  const at = s.box ? shapes.indexOf(s.box) : -1
  if (at >= 0 && s.box!.group_id !== s.group) out[at] = { ...s.box!, group_id: s.group }
  return out
}

export type ShapeRow =
  | { kind: 'shape'; index: number; pose?: PoseGroup; keypoint?: number }
  | { kind: 'pose'; pose: PoseGroup; open: boolean }
  | { kind: 'missing'; pose: PoseGroup; keypoint: number }

/**
 * The Shapes list in the image's order, each pose as one row where its first shape is. Opened,
 * it's followed by its box, its keypoints in the class's order, missing ones too, and any other
 * points of its group.
 */
export function shapeRows(count: number, poses: PoseGroup[], isOpen: (group: number) => boolean): ShapeRow[] {
  const of = poseIndex(poses)
  const out: ShapeRow[] = []
  for (let i = 0; i < count; i++) {
    const p = of.get(i)
    if (!p) {
      out.push({ kind: 'shape', index: i })
      continue
    }
    if (Math.min(...poseShapes(p)) !== i) continue
    const open = isOpen(p.group)
    out.push({ kind: 'pose', pose: p, open })
    if (!open) continue
    if (p.box !== null) out.push({ kind: 'shape', index: p.box, pose: p })
    p.points.forEach((at, k) => out.push(at === null ? { kind: 'missing', pose: p, keypoint: k } : { kind: 'shape', index: at, pose: p, keypoint: k }))
    for (const e of p.extra) out.push({ kind: 'shape', index: e, pose: p })
  }
  return out
}

/**
 * An AI pose's score: its box's, which is its keypoints' mean, or, on a box of the image's own,
 * the mean of its AI keypoints'. Not its weakest point's: a pose is accepted or rejected whole,
 * and one doubtful wrist says little of the rest.
 */
export function scoreOfPose(list: Shape[], pose: PoseGroup): number | undefined {
  const box = pose.box === null ? undefined : list[pose.box]
  if (box && isAi(box) && scoreOf(box) !== undefined) return scoreOf(box)
  const scores = poseShapes(pose).filter((k) => k !== pose.box && isAi(list[k])).map((k) => scoreOf(list[k])).filter((v): v is number => v !== undefined)
  return scores.length ? rounded(scores.reduce((a, b) => a + b, 0) / scores.length) : undefined
}

/** One step of review: an AI shape, or an AI pose's AI shapes together, with its score */
export type ReviewItem = { shapes: number[]; score: number | undefined; pose: PoseGroup | null }

/** What review steps through, least certain first; shapes without a score come last */
export function reviewItems(list: Shape[], poses: PoseGroup[]): ReviewItem[] {
  const of = poseIndex(poses)
  const seen = new Set<PoseGroup>()
  const items: (ReviewItem & { first: number })[] = []
  list.forEach((s, i) => {
    if (!isAi(s) || s.hidden) return
    const p = of.get(i)
    if (!p) return void items.push({ shapes: [i], score: scoreOf(s), pose: null, first: i })
    if (seen.has(p)) return
    seen.add(p)
    const shapes = poseShapes(p).filter((k) => isAi(list[k]) && !list[k].hidden).sort((a, b) => a - b)
    items.push({ shapes, score: scoreOfPose(list, p), pose: p, first: i })
  })
  return items
    .sort((a, b) => (a.score ?? 1) - (b.score ?? 1) || a.first - b.first)
    .map(({ shapes, score, pose }) => ({ shapes, score, pose }))
}

/** What's left to review, as the review bar counts it: AI poses, each one however many shapes it has, and other AI shapes */
export function aiCounts(list: Shape[], poses: PoseGroup[]): { poses: number; shapes: number } {
  const of = poseIndex(poses)
  const posed = new Set<PoseGroup>()
  let shapes = 0
  list.forEach((s, i) => {
    if (!isAi(s)) return
    const p = of.get(i)
    if (p) posed.add(p)
    else shapes++
  })
  return { poses: posed.size, shapes }
}

/** The step of review a selection is: an AI shape alone, or any of an AI pose's shapes and nothing else; null when it's none */
export function stepOf(steps: ReviewItem[], sel: number[]): ReviewItem | null {
  if (!sel.length) return null
  return steps.find((step) => (step.pose ? sel.every((i) => poseShapes(step.pose!).includes(i)) : sel.length === 1 && sel[0] === step.shapes[0])) ?? null
}

/** A pose's points as shares of its box's width and height, for the faint guide the next pose is placed over */
export type PoseLayout = { cls: string; at: (number[] | null)[] }

/** The layout of a pose with a box and at least two points; null otherwise */
export function layoutOf(shapes: Shape[], pose: PoseGroup): PoseLayout | null {
  if (pose.box === null) return null
  const [x0, y0, x1, y1] = boxOf(shapes[pose.box])
  if (x1 - x0 <= 0 || y1 - y0 <= 0) return null
  const at = pose.points.map((i) => (i === null ? null : [(shapes[i].points[0][0] - x0) / (x1 - x0), (shapes[i].points[0][1] - y0) / (y1 - y0)]))
  return at.filter(Boolean).length >= 2 ? { cls: pose.cls, at } : null
}

/** A layout scaled into a box, x0, y0, x1, y1 */
export const placeLayout = (layout: PoseLayout, [x0, y0, x1, y1]: number[]) =>
  layout.at.map((p) => (p ? [x0 + p[0] * (x1 - x0), y0 + p[1] * (y1 - y0)] : null))
