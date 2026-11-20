// Keypoints in the editor, and what main and the window share about them:
//   - COCO's person, which Seamtag uses until a folder has its own keypoints file;
//   - each keypoint's side, worked out from the flip pairs, which gives it its colour;
//   - the posed objects in an image's shapes, as the editor draws and folds them;
//   - the AI pose call's request and answer.
// The files keep the form export and import read (src/shared/pose.ts): a pose is a box and the
// points sharing its group_id, each named for a keypoint, a hidden one marked difficult.
import type { KeypointSchema } from './dataset'
import type { Shape } from './labelfile'
import { CLOSED_TYPES } from './outline'

/** COCO's 17 person keypoints, in order (person_keypoints_2017.json) */
export const COCO_PERSON = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
// COCO's skeleton, 1-based, as person_keypoints_2017.json lists it
const COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

/** The keypoints Seamtag starts with: COCO's person, its 19 lines and the 8 pairs a mirror swaps */
export function builtInKeypoints(): KeypointSchema {
  return {
    classes: { person: [...COCO_PERSON] },
    skeleton: { person: COCO_SKELETON.map(([a, b]) => [COCO_PERSON[a - 1], COCO_PERSON[b - 1]] as [string, string]) },
    flip: { person: COCO_PERSON.filter((n) => n.startsWith('left_')).map((n) => [n, n.replace('left_', 'right_')] as [string, string]) }
  }
}

/**
 * A keypoints file made from a folder's points, or X-AnyLabeling's, has no lines: a class whose
 * keypoints are COCO's person's, in any order, gets COCO's lines, and its flip pairs when it has
 * none. With `cocoOrder`, for a file made from the order the folder's points come in, such a
 * class's keypoints go in COCO's order too, which the pose exports and COCO's models keep.
 */
export function withKnownLines(schema: KeypointSchema, cocoOrder = false): KeypointSchema {
  const coco = builtInKeypoints()
  const out: KeypointSchema = { ...schema, classes: { ...schema.classes }, skeleton: { ...schema.skeleton }, flip: { ...schema.flip } }
  for (const [cls, names] of Object.entries(schema.classes)) {
    const same = names.length === COCO_PERSON.length && COCO_PERSON.every((n) => names.includes(n))
    if (!same) continue
    if (cocoOrder) out.classes[cls] = [...COCO_PERSON]
    if (!out.skeleton![cls]?.length) out.skeleton![cls] = coco.skeleton!.person
    if (!out.flip![cls]?.length) out.flip![cls] = coco.flip!.person
  }
  if (!Object.keys(out.skeleton!).length) delete out.skeleton
  if (!Object.keys(out.flip!).length) delete out.flip
  return out
}

/** The folder's keypoints as the window gets them: `file` is null while COCO's person is used */
export type FolderKeypoints = {
  schema: KeypointSchema
  file: string | null
  /** Why the folder's file wasn't used, when it couldn't be */
  problem: string | null
}

export type Side = 'left' | 'right' | 'centre'

/**
 * Each of a class's keypoints' side, for its colour, so a left-right mix-up shows as crossed
 * colours. In a flip pair, the keypoint whose name says left (or, when the names don't say, the
 * first) is on the left. A keypoint in no pair is in the centre. A class with no flip pairs
 * takes left and right from its names.
 */
export function sidesOf(schema: KeypointSchema, cls: string): Side[] {
  const names = schema.classes[cls] ?? []
  const pairs = schema.flip?.[cls] ?? []
  const side = new Map<string, Side>()
  const says = (name: string, word: 'left' | 'right') => new RegExp(word, 'i').test(name)
  for (const [a, b] of pairs) {
    const bIsLeft = says(b, 'left') && !says(a, 'left')
    side.set(bIsLeft ? b : a, 'left')
    side.set(bIsLeft ? a : b, 'right')
  }
  return names.map((n) => side.get(n) ?? (pairs.length ? 'centre' : says(n, 'left') ? 'left' : says(n, 'right') ? 'right' : 'centre'))
}

/** A keypoint labelled but hidden: COCO's visibility 1. X-AnyLabeling and Seamtag write it as difficult. */
export const isHidden = (s: Pick<Shape, 'flags'>) => s.flags.difficult === true || s.flags.occluded === true

/** Every keypoint name of the schema */
export const keypointNames = (schema: KeypointSchema) => new Set(Object.values(schema.classes).flat())

/** Labels that name keypoints and no class: the image and label lists don't count them as classes */
export function keypointOnlyNames(schema: KeypointSchema) {
  const names = keypointNames(schema)
  for (const cls of Object.keys(schema.classes)) names.delete(cls)
  return names
}

/** The group_id a new pose gets: one more than the image's largest */
export const nextGroupId = (shapes: Pick<Shape, 'group_id'>[]) =>
  Math.max(0, ...shapes.map((s) => (s.group_id !== null && Number.isFinite(s.group_id) ? s.group_id : 0))) + 1

/** A posed object in an image's shapes, by the shapes' indexes */
export type PoseGroup = {
  group: number
  /** The pose's class: its box's label, or the one class whose keypoints its points are named for */
  cls: string
  /** The box: the group's first closed shape labelled with the class (a rectangle first); null when it has none */
  box: number | null
  /** Each keypoint of the class, in order: the index of the point named for it, or null when it isn't labelled */
  points: (number | null)[]
  /** The group's points that aren't one of the class's keypoints, or name one a second time */
  extra: number[]
}

/**
 * The posed objects in an image's shapes: each group_id with at least one point named for a
 * keypoint of the schema, and the class of its box (a closed shape labelled with a class of the
 * schema) or, with no closed shape at all, the one class whose keypoints all its named points
 * are. Groups of shapes without keypoints, such as a person and their bag in labelme's sense,
 * aren't poses, and nor is a group of another class (a dog's box and its nose and eyes): the
 * keypoints file doesn't cover it. Export reads the same groups (poseInstances).
 */
export function posesIn(shapes: Shape[], schema: KeypointSchema): PoseGroup[] {
  const names = keypointNames(schema)
  const groups = new Map<number, number[]>()
  shapes.forEach((s, i) => {
    if (s.group_id === null || (s.shape_type !== 'point' && !CLOSED_TYPES.has(s.shape_type))) return
    const g = groups.get(s.group_id)
    if (g) g.push(i)
    else groups.set(s.group_id, [i])
  })
  const out: PoseGroup[] = []
  for (const [group, members] of groups) {
    const points = members.filter((i) => shapes[i].shape_type === 'point')
    const named = points.filter((i) => names.has(shapes[i].label))
    if (!named.length) continue
    const regions = members.filter((i) => CLOSED_TYPES.has(shapes[i].shape_type) && Object.hasOwn(schema.classes, shapes[i].label))
    if (!regions.length && members.some((i) => CLOSED_TYPES.has(shapes[i].shape_type))) continue
    const box = regions.find((i) => shapes[i].shape_type === 'rectangle') ?? regions[0] ?? null
    let cls: string | null = box !== null ? shapes[box].label : null
    if (cls === null) {
      const fits = Object.keys(schema.classes).filter((c) => named.every((i) => schema.classes[c].includes(shapes[i].label)))
      if (fits.length !== 1) continue
      cls = fits[0]
    }
    const order = schema.classes[cls]
    const slots: (number | null)[] = order.map(() => null)
    const extra: number[] = []
    for (const i of points) {
      const k = order.indexOf(shapes[i].label)
      if (k < 0 || slots[k] !== null) extra.push(i)
      else slots[k] = i
    }
    out.push({ group, cls, box, points: slots, extra })
  }
  return out.sort((a, b) => Math.min(...poseShapes(a)) - Math.min(...poseShapes(b)))
}

/**
 * The points that are poses' keypoints, by index: only these name keypoints rather than a class.
 * A box or polygon labelled like a keypoint, a point in no pose, or a pose's extra point is
 * labelled with a class as any shape is.
 */
export const posePoints = (poses: PoseGroup[]) => new Set(poses.flatMap((p) => p.points.filter((i): i is number => i !== null)))

/** Every shape of a pose, by index: its box, its keypoints and its extra points */
export const poseShapes = (p: PoseGroup) => [...(p.box === null ? [] : [p.box]), ...p.points.filter((i): i is number => i !== null), ...p.extra]

// AI pose: main's `ai:pose`, which the window calls with boxes and gets keypoints back

/** Boxes to find a pose in, each x0, y0, x1, y1 in the image's own pixels */
export type PoseRequest = { root: string; name: string; model: string; boxes: [number, number, number, number][] }
/** One keypoint as the model placed it, with its score: null when it placed none there */
export type PosePoint = { x: number; y: number; score: number } | null
/** Per box, in the request's order, the model's keypoints, named in `names` (the model's own order) */
export type PoseAnswer = { names: string[]; poses: PosePoint[][] }
