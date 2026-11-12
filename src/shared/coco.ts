// COCO datasets written from a folder's labels and read back into shapes: instances (polygons,
// boxes and masks) and keypoints, as pycocotools reads them (cocodataset.org/#format-data).
// Every area, box and mask is pycocotools 2.0.11's own, through the port in rle.ts. Shapes
// become polygons as labelme 7.9.1's labelme2coco makes them (outline.ts), and shapes sharing
// a label and a group_id are one object, as there. What COCO can't hold goes in CVAT's
// per-annotation `attributes`, so importing an export gives back the same shapes: a shape's
// shape_type, description, flags, lock and other keys, its group_id, and its points wherever the
// segmentation can't give them back (a circle's centre and rim point, a rotated box moved inside
// the image, a mask's box). An object of several shapes lists them in order under `shapes`.
// When writing objects whole puts a shape before one it came after, every shape keeps its place
// in the label file as `index`, so importing puts them back in order. A keypoints file keeps its
// posed objects' outlines the same way, so it gives them back on its own.
import type { DatasetImage, Issue, KeypointSchema, MaskBits, PoseInstance, Severity } from './dataset'
import { newShape, pointCount, readShape, type Shape, type ShapeType } from './labelfile'
import { CLOSED_TYPES, outlineOf } from './outline'
import { short } from './pose'
import { type Rle, rleArea, rleFrMaskBits, rleFrPoly, rleFrString, rleLength, rleMerge, rleMove, rleToBbox, rleToMaskBits, rleToString } from './rle'

/** COCO's info, given by the caller, the moment too, so the same labels give the same file */
export type CocoInfo = { description: string; version: string; year: number; date_created: string; contributor?: string; url?: string }
/** An image: its path in the dataset, with / between folders, and its upright size */
export type CocoImage = { id: number; width: number; height: number; file_name: string }
/** A mask: compressed counts as Seamtag and pycocotools write them, or a list of runs as CVAT does */
export type CocoRle = { size: [number, number]; counts: string | number[] }
/** One object: its outline or mask, pycocotools' area of it, and its box */
export type CocoAnnotation = {
  id: number
  image_id: number
  category_id: number
  /** Polygons as flat [x0, y0, x1, y1, ...] lists, or a mask */
  segmentation: number[][] | CocoRle
  area: number
  /** x, y, width, height */
  bbox: [number, number, number, number]
  iscrowd: 0 | 1
  /** x, y and visibility per keypoint: 0 not labelled (x and y 0 too), 1 labelled but hidden, 2 seen */
  keypoints?: number[]
  num_keypoints?: number
  attributes?: Record<string, unknown>
}
/** A label; for posed objects with its keypoint names and the pairs of them joined, counted from 1 */
export type CocoCategory = { id: number; name: string; supercategory: string; keypoints?: string[]; skeleton?: [number, number][] }
/** A whole COCO file */
export type CocoDataset = {
  info: Required<CocoInfo>
  licenses: unknown[]
  images: CocoImage[]
  annotations: CocoAnnotation[]
  categories: CocoCategory[]
}

/** COCO's two files: objects with their outlines or masks, and posed objects with their keypoints */
export type CocoTask = 'instances' | 'keypoints'

/**
 * One image to export: its labels, its path in the dataset (images/train/a.jpg), and its posed
 * objects when a keypoints file is written too, as the instances file then leaves their points to it
 */
export type CocoEntry = { image: DatasetImage; fileName: string; poses?: PoseInstance[] }

/** What writeCoco writes, and the categories and info it writes them with */
export type CocoOptions = {
  task: CocoTask
  /**
   * Category names, category 1 first: cocoLabels' list, so every split numbers them alike.
   * Labels the images have and this lacks go after it, first seen first.
   */
  labels: string[]
  info: CocoInfo
  /** Keypoint names per class, for the keypoints file */
  schema?: KeypointSchema
}

const TYPE_NAMES: Record<ShapeType, string> = {
  polygon: 'polygon', rectangle: 'rectangle', oriented_rectangle: 'rotated box', circle: 'circle', line: 'line',
  point: 'point', linestrip: 'line strip', points: 'points shape', mask: 'mask'
}

const issue = (severity: Severity, code: string, image: string | null, message: string, shape?: number): Issue =>
  shape === undefined ? { severity, code, image, message } : { severity, code, image, shape, message }

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const isCount = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 0

// Round half to even, as numpy's round, which labelme uses for a mask's place
function roundHalfEven(x: number) {
  const r = Math.round(x)
  // + 0 makes Math.round's -0 a plain 0
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r) + 0
}

const flat = (points: number[][]) => points.flatMap(([x, y]) => [x, y])
const pairs = (poly: number[]) => Array.from({ length: poly.length / 2 }, (_, i) => [poly[2 * i], poly[2 * i + 1]])
const samePoints = (a: number[][], b: number[][]) => a.length === b.length && a.every((p, i) => p[0] === b[i][0] && p[1] === b[i][1])
const list = (n: number[]) => (n.length === 1 ? `shape ${n[0]}` : `shapes ${n.slice(0, -1).join(', ')} and ${n[n.length - 1]}`)
// Points within a hundredth of a pixel of each other are the same: edits move them further, and
// tools that keep coordinates as float32 or round them to two decimals move them less
const near = (a: number, b: number) => Math.abs(a - b) <= 0.01

/**
 * The categories of an export, in order: the label list, then the labels of the objects and
 * posed objects found past it, first seen first. Give the result to writeCoco for every split
 * and for both files, so they all number the categories alike.
 */
export function cocoLabels(labels: string[], entries: CocoEntry[]): string[] {
  const out = [...new Set(labels)], seen = new Set(out)
  const see = (label: string) => { if (!seen.has(label)) { seen.add(label); out.push(label) } }
  for (const { image, poses } of entries) {
    for (const s of image.shapes) if (CLOSED_TYPES.has(s.shape_type)) see(s.label)
    for (const p of poses ?? []) see(p.label)
  }
  return out
}

// One shape of an object: its outline, kept inside the image, or its mask's pixels
type Part = { index: number; shape: Shape; points: number[][] | null; bits: MaskBits | null; outside: boolean }

// A closed shape's outline as COCO holds it, its points past the image moved onto its edges;
// null for masks and for shapes too broken to have one
function outlineInside(s: Shape, w: number, h: number): { points: number[][]; outside: boolean } | null {
  const outline = outlineOf(s)
  if (!outline) return null
  let outside = false
  const points = outline.map(([x, y]) => {
    const cx = Math.min(Math.max(x, 0), w), cy = Math.min(Math.max(y, 0), h)
    if (cx !== x || cy !== y) outside = true
    return [cx, cy]
  })
  return { points, outside }
}

// A shape as a part of an object, or null, noting why it can't be one
function partOf(image: DatasetImage, i: number, issues: Issue[] | null): Part | null {
  const s = image.shapes[i], name = image.name
  const what = `shape ${i + 1}, a ${TYPE_NAMES[s.shape_type]} labelled “${s.label}”,`
  const skip = (severity: Severity, code: string, why: string) => { issues?.push(issue(severity, code, name, `${name}: ${what} ${why}`, i)); return null }
  const count = `${s.points.length} ${s.points.length === 1 ? 'point' : 'points'}`
  if (!CLOSED_TYPES.has(s.shape_type)) return skip('warning', 'coco-no-inside', "has no inside for COCO to hold, so it's left out")
  if (s.shape_type === 'mask') {
    // labelme 7.9.1's shapes_to_label (examples/utils.py) places a mask by its two corners, and
    // fails on any other count
    if (s.points.length !== 2) return skip('warning', 'coco-bad-shape', `has ${count}, not 2, so it's left out`)
    const bits = image.masks[i] ?? null
    if (!bits) return skip('warning', 'coco-mask-unreadable', "can't be read, so it's left out")
    // How many set pixels lie past the image, which COCO's mask can't hold
    let cut = 0
    for (let y = 0; y < bits.height; y++) {
      const iy = roundHalfEven(bits.y) + y
      for (let x = 0; x < bits.width; x++) {
        const ix = roundHalfEven(bits.x) + x
        if (bits.data[y * bits.width + x] && (ix < 0 || iy < 0 || ix >= image.width || iy >= image.height)) cut++
      }
    }
    if (cut) issues?.push(issue('info', 'coco-outside', name, `${name}: ${what} has ${cut} ${cut === 1 ? 'pixel' : 'pixels'} past the image, so ${cut === 1 ? "it's" : "they're"} left out`, i))
    return { index: i, shape: s, points: null, bits, outside: cut > 0 }
  }
  const outline = outlineInside(s, image.width, image.height)
  if (!outline) {
    if (s.shape_type === 'polygon') return skip('warning', 'coco-too-few-points', `has ${count}, and COCO needs 3, so it's left out`)
    if (s.shape_type === 'circle' && s.points.length === 2) return skip('warning', 'coco-no-radius', "has no radius, so it's left out")
    return skip('warning', 'coco-bad-shape', `has ${count}, not ${pointCount(s.shape_type)}, so it's left out`)
  }
  if (outline.outside) issues?.push(issue('info', 'coco-outside', name, `${name}: ${what} reaches past the image, so its points there are moved onto its edges`, i))
  return { index: i, shape: s, points: outline.points, bits: null, outside: outline.outside }
}

// Whether importing can rebuild a part's points from its polygon, so they needn't be kept
function restorable(p: Part) {
  const s = p.shape
  if (p.outside || !p.points) return false
  if (s.shape_type === 'polygon' || s.shape_type === 'oriented_rectangle') return true
  return s.shape_type === 'rectangle' && samePoints(s.points, [p.points[0], p.points[2]])
}

// The box importing reads a mask's pixels from when the shape keeps only its points: labelme's
// whole-pixel rule (examples/utils.py, shapes_to_label), both corners rounded and inclusive
function maskBox(points: number[][]) {
  const x = roundHalfEven(points[0][0]), y = roundHalfEven(points[0][1])
  return { x, y, width: roundHalfEven(points[1][0]) - x + 1, height: roundHalfEven(points[1][1]) - y + 1 }
}

// What a part needs in `attributes` to come back as it was
function partRecord(p: Part, keepPoints: boolean, ownMask: boolean) {
  const s = p.shape, r: Record<string, unknown> = {}
  if (s.shape_type !== 'polygon') r.shape_type = s.shape_type
  if (keepPoints) r.points = s.points
  if (ownMask && p.bits) {
    const own = rleFrMaskBits({ ...p.bits, x: 0, y: 0 }, p.bits.height, p.bits.width)
    r.mask = { size: [own.h, own.w], counts: rleToString(own) }
  }
  if (s.description) r.description = s.description
  if (Object.keys(s.flags).length) r.flags = s.flags
  if (s.locked) r.locked = true
  if (Object.keys(s.other).length) r.other = s.other
  return r
}

// An object's attributes: its group_id, and what each of its parts needs to come back as it
// was, with its place in the label file when `placed`
function attributesOf(parts: Part[], group: number | null, rle: boolean, w: number, h: number, placed: boolean) {
  const records = parts.map((p) => {
    const ownMask = !!p.bits && !(parts.length === 1 && !p.outside && sameBox(p.bits, maskBox(p.shape.points), w, h))
    return { ...partRecord(p, rle || !restorable(p), ownMask), ...(placed ? { index: p.index } : {}) }
  })
  const attributes: Record<string, unknown> = group === null ? {} : { group_id: group }
  if (parts.length === 1) Object.assign(attributes, records[0])
  else attributes.shapes = records
  return attributes
}

type Made = Omit<CocoAnnotation, 'id' | 'image_id'>

// The segmentation, area and box of parts of one object: polygons while they're all outlines,
// else one mask of all of them, as labelme2coco writes an object with a mask in it
function segment(parts: Part[], w: number, h: number, asMask: boolean) {
  if (asMask || parts.some((p) => p.bits)) {
    const union = rleMerge(parts.map((p) => (p.bits ? rleFrMaskBits(p.bits, h, w) : rleFrPoly(flat(p.points!), h, w))))
    return { rle: true, segmentation: { size: [h, w], counts: rleToString(union) } as CocoRle, area: rleArea(union), bbox: rleToBbox(union) }
  }
  const polygons = parts.map((p) => flat(p.points!))
  const area = rleArea(rleMerge(polygons.map((q) => rleFrPoly(q, h, w))))
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const p of parts) {
    for (const [x, y] of p.points!) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
  }
  return { rle: false, segmentation: polygons, area, bbox: [x0, y0, x1 - x0, y1 - y0] as CocoAnnotation['bbox'] }
}

function instances(entry: CocoEntry, categoryOf: (label: string) => number, issues: Issue[]): Made[] {
  const image = entry.image, { width: w, height: h, name } = image
  // A posed object's points are its keypoints, in the keypoints file
  const keypoints = new Set<number>()
  for (const p of entry.poses ?? []) for (const i of p.shapes) if (image.shapes[i]?.shape_type === 'point') keypoints.add(i)
  const objects = new Map<string | number, Part[]>()
  image.shapes.forEach((s, i) => {
    if (keypoints.has(i)) return
    const part = partOf(image, i, issues)
    if (!part) return
    // A shape without a group_id is an object of its own, as labelme2coco has it
    const key = s.group_id === null ? i : `${s.label}\u0000${s.group_id}`
    objects.set(key, [...(objects.get(key) ?? []), part])
  })
  const kept: { parts: Part[]; crowd: boolean; category: number; made: ReturnType<typeof segment> }[] = []
  for (const parts of objects.values()) {
    const first = parts[0].shape
    // Crowds, which COCO's own files keep as masks; only importing one makes such a shape
    const crowd = parts.some((p) => p.shape.other.iscrowd === 1)
    const made = segment(parts, w, h, crowd)
    if (!made.area) {
      issues.push(issue('warning', 'coco-no-pixels', name, `${name}: “${first.label}” (${list(parts.map((p) => p.index + 1))}) covers no whole pixel, so it's left out`, parts[0].index))
      continue
    }
    kept.push({ parts, crowd, category: categoryOf(first.label), made })
  }
  // Each object is written where its first shape was, with its other shapes. When that puts a
  // shape before one it came after, every shape keeps its place, so importing puts them back.
  const order = kept.flatMap((o) => o.parts.map((p) => p.index))
  const placed = order.some((n, k) => n < order[k - 1])
  return kept.map(({ parts, crowd, category, made: { rle, segmentation, area, bbox } }) => {
    const attributes = attributesOf(parts, parts[0].shape.group_id, rle, w, h, placed)
    return { category_id: category, segmentation, area, bbox, iscrowd: crowd ? 1 : 0, ...(Object.keys(attributes).length ? { attributes } : {}) }
  })
}

// Whether a mask's pixels sit in the box its points give, wholly inside the image
function sameBox(bits: MaskBits, box: { x: number; y: number; width: number; height: number }, w: number, h: number) {
  return bits.x === box.x && bits.y === box.y && bits.width === box.width && bits.height === box.height &&
    box.x >= 0 && box.y >= 0 && box.x + box.width <= w && box.y + box.height <= h
}

function poses(entry: CocoEntry, schema: KeypointSchema | undefined, categoryOf: (label: string) => number, issues: Issue[]): Made[] {
  const image = entry.image, { width: w, height: h, name } = image
  const out: Made[] = []
  for (const pose of entry.poses ?? []) {
    const what = `“${pose.label}”${pose.group === null ? '' : ` (group ${pose.group})`}`
    const names = schema?.classes[pose.label]
    if (!names) {
      issues.push(issue('warning', 'coco-pose-class', name, `${name}: ${what} has no keypoint names in the keypoints file, so it's left out`, pose.shapes[0]))
      continue
    }
    if (pose.keypoints.length !== names.length) {
      issues.push(issue('warning', 'coco-pose-keypoints', name, `${name}: ${what} has ${pose.keypoints.length} ${pose.keypoints.length === 1 ? 'keypoint' : 'keypoints'} where the keypoints file names ${names.length}, so it's left out`, pose.shapes[0]))
      continue
    }
    const keypoints: number[] = []
    let seen = 0
    pose.keypoints.forEach((k, i) => {
      if (k && (k[0] < 0 || k[1] < 0 || k[0] > w || k[1] > h)) {
        issues.push(issue('info', 'coco-outside', name, `${name}: keypoint “${names[i]}” of ${what} lies past the image, so it's written as not labelled`, pose.shapes[0]))
        k = null
      }
      if (k) seen++
      keypoints.push(...(k ?? [0, 0, 0]))
    })
    let [x0, y0, x1, y1] = pose.box
    if (x1 < x0) [x0, x1] = [x1, x0]
    if (y1 < y0) [y0, y1] = [y1, y0]
    const box = [Math.max(x0, 0), Math.max(y0, 0), Math.min(x1, w), Math.min(y1, h)]
    if (box[0] !== x0 || box[1] !== y0 || box[2] !== x1 || box[3] !== y1) {
      issues.push(issue('info', 'coco-outside', name, `${name}: the box of ${what} reaches past the image, so it's cut back to its edges`, pose.shapes[0]))
    }
    const bw = box[2] - box[0], bh = box[3] - box[1]
    if (!(bw > 0 && bh > 0)) {
      issues.push(issue('warning', 'coco-no-pixels', name, `${name}: the box of ${what} has no area, so it's left out`, pose.shapes[0]))
      continue
    }
    // Its outline: the object's own closed shapes (whose problems the instances file reports), else its box
    const parts = pose.shapes.filter((i) => image.shapes[i] && CLOSED_TYPES.has(image.shapes[i].shape_type) && image.shapes[i].label === pose.label)
      .map((i) => partOf(image, i, null)).filter((p): p is Part => p !== null)
    const outline = parts.length ? segment(parts, w, h, false) : null
    const area = outline?.area ? outline.area : bw * bh
    const segmentation = outline?.area ? outline.segmentation : [[box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]]
    // Its outline's shapes are kept as the instances file keeps them, so this file alone gives them back
    const attributes = outline?.area ? attributesOf(parts, pose.group, outline.rle, w, h, false) : pose.group === null ? {} : { group_id: pose.group }
    out.push({
      category_id: categoryOf(pose.label), segmentation, area, bbox: [box[0], box[1], bw, bh], iscrowd: 0, keypoints, num_keypoints: seen,
      ...(Object.keys(attributes).length ? { attributes } : {})
    })
  }
  return out
}

/**
 * A COCO dataset of the images given, in their order: instances_<split>.json for the task
 * 'instances', person_keypoints_<split>.json for 'keypoints'. Image and annotation ids count
 * from 1, as COCOeval needs. Everything left out, moved or cut is an Issue. Shapes are taken as
 * they are, AI proposals too. The keypoints file reports only its posed objects' problems.
 */
export function writeCoco(entries: CocoEntry[], options: CocoOptions): { dataset: CocoDataset; json: string; issues: Issue[] } {
  const issues: Issue[] = []
  const labels = [...new Set(options.labels)]
  const ids = new Map(labels.map((l, i) => [l, i + 1]))
  const categoryOf = (label: string) => {
    let id = ids.get(label)
    if (id === undefined) {
      labels.push(label)
      ids.set(label, (id = labels.length))
      issues.push(issue('info', 'coco-unlisted-label', null, `“${label}” isn't in the label list, so it's category ${id}, after the list`))
    }
    return id
  }
  const images: CocoImage[] = [], annotations: CocoAnnotation[] = []
  for (const entry of entries) {
    const { name, broken, unreadable, width, height } = entry.image
    if (broken !== undefined) {
      issues.push(issue('error', 'coco-broken-labels', name, `${name}: its label file can't be read (${broken}), so the image is left out`))
      continue
    }
    if (unreadable !== undefined || !(width > 0 && height > 0)) {
      issues.push(issue('error', 'coco-unreadable-image', name, `${name} can't be read${unreadable ? ` (${unreadable})` : ''}, so it's left out`))
      continue
    }
    const image_id = images.length + 1
    images.push({ id: image_id, width, height, file_name: entry.fileName })
    const made = options.task === 'keypoints' ? poses(entry, options.schema, categoryOf, issues) : instances(entry, categoryOf, issues)
    for (const a of made) annotations.push({ id: annotations.length + 1, image_id, ...a })
  }
  const categories: CocoCategory[] = []
  labels.forEach((label, i) => {
    if (options.task === 'instances') return categories.push({ id: i + 1, name: label, supercategory: '' })
    const names = options.schema?.classes[label]
    if (!names) return
    const index = new Map(names.map((n, k) => [n, k + 1]))
    const skeleton: [number, number][] = []
    for (const [a, b] of options.schema?.skeleton?.[label] ?? []) {
      const ia = index.get(a), ib = index.get(b)
      if (ia && ib) skeleton.push([ia, ib])
      else issues.push(issue('warning', 'coco-skeleton', null, `The skeleton of “${label}” joins “${ia ? b : a}”, which isn't one of its keypoints, so that edge is left out`))
    }
    categories.push({ id: i + 1, name: label, supercategory: '', keypoints: [...names], skeleton })
  })
  const { description, version, year, date_created, contributor = '', url = '' } = options.info
  const dataset: CocoDataset = {
    info: { description, url, version, year: Math.trunc(year), contributor, date_created },
    licenses: [], images, annotations, categories
  }
  return { dataset, json: JSON.stringify(dataset), issues }
}

// Reading

/** A posed object read from a keypoints file: its box (x0, y0, x1, y1) and its keypoints by name, in the category's order */
export type CocoPose = {
  label: string
  /** From Seamtag's attributes, else a new one, unused in the image */
  group: number
  box: [number, number, number, number]
  names: string[]
  /** x, y and 2 seen or 1 labelled but hidden; null for keypoints not labelled */
  keypoints: ([number, number, 1 | 2] | null)[]
  /**
   * Whether its outline, the shapes its segmentation gives, is among the image's shapes in its
   * group, so only its keypoints are left to add; false when its box is all there is of it
   */
  outlined: boolean
}

/** What one COCO image holds, as Seamtag's shapes */
export type CocoImageResult = {
  image: CocoImage
  /**
   * In their order in the label file where Seamtag's attributes keep it, others after them. Mask
   * shapes have points but no PNG: their pixels are in masks, for the caller to encode, unless
   * readCoco was asked not to make them.
   */
  shapes: Shape[]
  masks: (MaskBits | null)[]
  poses: CocoPose[]
  /** How many pixels its mask shapes have, made or not: making them takes a byte each */
  pixels: number
}

/** A COCO file as read: its images and categories, and what each image holds */
export type CocoRead = {
  images: CocoImage[]
  categories: CocoCategory[]
  /** By COCO image id, in the images list's order */
  results: Map<number, CocoImageResult>
  /** Problems found, each naming the image by its COCO file_name */
  issues: Issue[]
}

/**
 * The most pixels a mask shape may have: 16383 x 16383, the most sharp, which writes and reads
 * their PNGs, takes. A COCO file can give a mask any size.
 */
export const MASK_PIXELS = 0x3fff * 0x3fff
// The most points rleFrPoly may trace along an outline it fills to compare with a mask, at 5
// times the size: about 25 times round the largest mask's edge. An outline in a file's
// attributes can zigzag across the image far more often than any shape does.
const OUTLINE_POINTS = 2 ** 23

/**
 * New group ids count on from the highest below this. Past 2^53 a number plus 1 is the same
 * number, so counting on from one near it would put every new object in one group.
 */
export const MAX_GROUP = 2 ** 31

// A file name as messages quote it: room for a long path, not for megabytes of one
const pathOf = (name: string) => short(name, 120)
// Any other value from the file as messages quote it
const quote = (v: unknown) => short(String(JSON.stringify(v)))

type Note = (severity: Severity, code: string, why: string) => void
// A mask shape's pixels before they're made: those of `rle` inside `box`, with the box's top left
// at x, y in the image. They take a byte each, so they're made only when asked for.
type Pixels = { rle: Rle; box: { x: number; y: number; width: number; height: number }; x: number; y: number }
// A shape read from an annotation, with its pixels when it's a mask, and its place in the label
// file when Seamtag's attributes keep it
type Read = { shape: Shape; pixels: Pixels | null; index?: number }
const made = (p: Pixels): MaskBits => ({ ...rleToMaskBits(p.rle, p.box), x: p.x, y: p.y })
// The shapes an annotation's segmentation gives, and whether Seamtag's attributes gave them back
type Outline = { shapes: Read[]; restored: boolean }
// What a file's annotations are summed up by, in one Issue each
type Tally = { box: number; hidden: number; plain: number; crowd: number }

/**
 * A COCO file (instances or keypoints, from Seamtag, pycocotools, CVAT, X-AnyLabeling or
 * others) as Seamtag's shapes per image. Throws when it isn't COCO at all. Annotations that
 * can't be read are left out, each with an Issue. An annotation with keypoints labelled is a
 * posed object, whose outline comes back among the shapes in its group; one with none
 * labelled is read as a plain object. With `pixels` false, masks' pixels aren't made: the
 * shapes, their count and the issues are the same.
 */
export function readCoco(text: string, file: string, { pixels = true }: { pixels?: boolean } = {}): CocoRead {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (e) {
    throw new Error(`${file} isn't valid JSON (${(e as Error).message}), so nothing is imported`)
  }
  if (!isObject(raw)) throw new Error(`${file} isn't a COCO file: the top level must be an object, so nothing is imported`)
  if (!Array.isArray(raw.images)) throw new Error(`${file} isn't a COCO file: it has no images list, so nothing is imported`)
  const issues: Issue[] = []
  const images: CocoImage[] = [], results = new Map<number, CocoImageResult>()
  raw.images.forEach((im: unknown, k: number) => {
    const where = `${file}: image ${k + 1} in its images list`
    if (!isObject(im)) return issues.push(issue('warning', 'coco-bad-image', null, `${where} isn't an object, so it's left out`))
    const { id, width, height, file_name } = im
    if (!Number.isInteger(id)) return issues.push(issue('warning', 'coco-bad-image', null, `${where} has no whole-number id, so it's left out`))
    if (typeof file_name !== 'string' || !file_name) return issues.push(issue('warning', 'coco-bad-image', null, `${where} has no file_name, so it's left out`))
    if (!(Number.isInteger(width) && Number.isInteger(height) && (width as number) > 0 && (height as number) > 0)) {
      return issues.push(issue('warning', 'coco-bad-image', file_name, `${where} (${pathOf(file_name)}) has no width and height, so it's left out`))
    }
    if (results.has(id as number)) return issues.push(issue('warning', 'coco-bad-image', file_name, `${where} (${pathOf(file_name)}) has the id ${id}, which an earlier image has, so it's left out`))
    const image = { id: id as number, width: width as number, height: height as number, file_name }
    images.push(image)
    results.set(image.id, { image, shapes: [], masks: [], poses: [], pixels: 0 })
  })
  const categories: CocoCategory[] = [], byId = new Map<number, CocoCategory>()
  for (const [k, c] of (Array.isArray(raw.categories) ? raw.categories : []).entries()) {
    const where = `${file}: category ${k + 1} in its categories list`
    if (!isObject(c) || !Number.isInteger(c.id) || typeof c.name !== 'string') {
      issues.push(issue('warning', 'coco-bad-category', null, `${where} has no whole-number id or no name, so its annotations are left out`))
      continue
    }
    if (byId.has(c.id as number)) {
      issues.push(issue('warning', 'coco-bad-category', null, `${where} has the id ${c.id}, which an earlier category has, so it's left out`))
      continue
    }
    const keypoints = Array.isArray(c.keypoints) && c.keypoints.every((n) => typeof n === 'string') ? (c.keypoints as string[]) : undefined
    const category: CocoCategory = { id: c.id as number, name: c.name, supercategory: typeof c.supercategory === 'string' ? c.supercategory : '' }
    if (keypoints) category.keypoints = keypoints
    if (Array.isArray(c.skeleton)) category.skeleton = c.skeleton.filter((e): e is [number, number] => Array.isArray(e) && e.length === 2 && e.every(Number.isInteger))
    categories.push(category)
    byId.set(category.id, category)
  }
  const annotations = Array.isArray(raw.annotations) ? raw.annotations : []
  // New group ids for objects of several parts and posed objects start past those the file's
  // own attributes use in the image, of those below MAX_GROUP: a larger one is kept as it is
  const nextGroup = new Map<number, number>()
  for (const a of annotations) {
    if (!isObject(a) || !isObject(a.attributes)) continue
    const g = a.attributes.group_id
    if (Number.isInteger(g) && (g as number) < MAX_GROUP) nextGroup.set(a.image_id as number, Math.max(nextGroup.get(a.image_id as number) ?? 1, (g as number) + 1))
  }
  const newGroup = (image: number) => {
    const g = nextGroup.get(image) ?? 1
    nextGroup.set(image, g + 1)
    return g
  }
  const counted: Tally = { box: 0, hidden: 0, plain: 0, crowd: 0 }
  // Each shape's place in its label file, by image, where Seamtag's attributes keep it
  const places = new Map<number, (number | undefined)[]>()
  annotations.forEach((a: unknown, k: number) => {
    if (!isObject(a)) return issues.push(issue('warning', 'coco-bad-annotation', null, `${file}: annotation ${k + 1} in its annotations list isn't an object, so it's left out`))
    const which = `annotation ${Number.isInteger(a.id) ? a.id : `${k + 1} in the list`}`
    const result = results.get(a.image_id as number)
    if (!result) return issues.push(issue('warning', 'coco-unknown-image', null, `${file}: ${which} is of image ${quote(a.image_id)}, which isn't in its images list, so it's left out`))
    const { image } = result, name = image.file_name
    const category = byId.get(a.category_id as number)
    if (!category) return issues.push(issue('warning', 'coco-unknown-category', name, `${pathOf(name)}: ${which} is of category ${quote(a.category_id)}, which isn't in the categories list, so it's left out`))
    const what = `${which} (“${short(category.name)}”)`
    const note: Note = (severity, code, why) => issues.push(issue(severity, code, name, `${pathOf(name)}: ${what} ${why}`))
    const tally: Tally = { box: 0, hidden: 0, plain: 0, crowd: 0 }
    let read: { shapes: Read[]; pose: CocoPose | null }
    let masks: (MaskBits | null)[]
    try {
      read = readAnnotation(a, category, image, note, tally, () => newGroup(image.id))
      masks = read.shapes.map((r) => (r.pixels && pixels ? made(r.pixels) : null))
    } catch (e) {
      // An error nothing here foresaw costs this annotation alone, not the whole file
      return note('warning', 'coco-bad-annotation', `can't be read (${(e as Error).message}), so it's left out`)
    }
    for (const key of Object.keys(tally) as (keyof Tally)[]) counted[key] += tally[key]
    if (read.pose) result.poses.push(read.pose)
    if (!places.has(image.id)) places.set(image.id, [])
    read.shapes.forEach(({ shape, pixels: p, index }, i) => {
      result.shapes.push(shape)
      result.masks.push(masks[i])
      if (p) result.pixels += p.box.width * p.box.height
      places.get(image.id)!.push(index)
    })
  })
  // Seamtag's own shapes go back to their places in the label file, any others after them
  for (const [id, place] of places) {
    if (place.every((n) => n === undefined)) continue
    const result = results.get(id)!, rank = (k: number) => place[k] ?? Infinity
    const order = place.map((_, k) => k).sort((p, q) => (rank(p) === rank(q) ? p - q : rank(p) - rank(q)))
    const { shapes, masks } = result
    result.shapes = order.map((k) => shapes[k])
    result.masks = order.map((k) => masks[k])
  }
  if (counted.box) issues.push(issue('info', 'coco-box-only', null, `${file}: ${counted.box} ${counted.box === 1 ? 'annotation holds' : 'annotations hold'} only a box, so ${counted.box === 1 ? 'it becomes a rectangle' : 'they become rectangles'}`))
  if (counted.hidden) issues.push(issue('info', 'coco-unlabelled-keypoints', null, `${file}: ${counted.hidden} ${counted.hidden === 1 ? 'keypoint is' : 'keypoints are'} marked not labelled (visibility 0), so ${counted.hidden === 1 ? "it's" : "they're"} left out`))
  if (counted.plain) {
    const one = counted.plain === 1
    issues.push(issue('info', 'coco-no-keypoints', null, `${file}: ${counted.plain} ${one ? 'annotation has' : 'annotations have'} no keypoints labelled, so ${one ? "it's read as a plain object" : "they're read as plain objects"}`))
  }
  if (counted.crowd) {
    const one = counted.crowd === 1
    issues.push(issue('info', 'coco-crowd', null, `${file}: ${counted.crowd} ${one ? 'annotation is a crowd region' : 'annotations are crowd regions'} (iscrowd 1): the shapes made from ${one ? 'it' : 'them'} keep iscrowd, so an export writes ${one ? 'it as a crowd' : 'them as crowds'} again`))
  }
  return { images, categories, results, issues }
}

// An annotation as shapes, and as a posed object when it has keypoints labelled
function readAnnotation(a: Record<string, unknown>, category: CocoCategory, image: CocoImage, note: Note, tally: Tally, newGroup: () => number): { shapes: Read[]; pose: CocoPose | null } {
  const crowd = a.iscrowd === 1 || a.iscrowd === true
  if (crowd) tally.crowd++
  // A crowd has no keypoints of its own: it's read as the object it outlines
  if (category.keypoints && a.keypoints !== undefined && !crowd) {
    const read = readKeypoints(a, category.keypoints, note)
    if (read?.keypoints.some((k) => k !== null)) {
      tally.hidden += read.unlabelled
      return readPose(a, category, image, read.keypoints, note, newGroup)
    }
    // With none of its keypoints labelled it's an object like any other, as are COCO's many people of num_keypoints 0
    if (read) tally.plain++
  }
  const { shapes, restored } = readObject(a, category.name, image, note, tally, newGroup)
  // The file's iscrowd marks the object; Seamtag's attributes say which of its shapes carried it
  if (!crowd || (restored && shapes.some((r) => r.shape.other.iscrowd === 1))) return { shapes, pose: null }
  return { shapes: shapes.map((r) => ({ ...r, shape: { ...r.shape, other: { ...r.shape.other, iscrowd: 1 } } })), pose: null }
}

// A box as COCO keeps it, [x, y, width, height], when it has an area. Its far corner must be a
// number too: x plus width can come to Infinity, which no label file can hold.
function cocoBox(v: unknown): [number, number, number, number] | null {
  if (!Array.isArray(v) || v.length !== 4 || !v.every((n) => typeof n === 'number' && Number.isFinite(n))) return null
  return v[2] > 0 && v[3] > 0 && Number.isFinite(v[0] + v[2]) && Number.isFinite(v[1] + v[3]) ? (v as [number, number, number, number]) : null
}

// The box around points, x0, y0, x1, y1, when they span one
function around(points: number[][]): [number, number, number, number] | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const [x, y] of points) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null
}

// An annotation's keypoints in its category's order, null for those not labelled, and how many
// are marked not labelled (visibility 0); null when they can't be read at all
function readKeypoints(a: Record<string, unknown>, names: string[], note: Note): { keypoints: CocoPose['keypoints']; unlabelled: number } | null {
  const kp = a.keypoints
  if (!Array.isArray(kp) || kp.length !== names.length * 3 || !kp.every((n) => typeof n === 'number' && Number.isFinite(n))) {
    const n = Array.isArray(kp) ? kp.length : 0
    note('warning', 'coco-bad-keypoints', `has ${n} keypoint ${n === 1 ? 'number' : 'numbers'} where its ${names.length} keypoints need ${names.length * 3}, so it's imported without them`)
    return null
  }
  const keypoints: CocoPose['keypoints'] = []
  let unlabelled = 0
  for (let i = 0; i < names.length; i++) {
    const [x, y, v] = kp.slice(3 * i, 3 * i + 3) as number[]
    if (v === 0) { unlabelled++; keypoints.push(null); continue }
    if (v !== 1 && v !== 2) {
      note('warning', 'coco-bad-keypoints', `gives keypoint “${short(names[i])}” the visibility ${v}, not 0, 1 or 2, so that keypoint is left out`)
      keypoints.push(null)
      continue
    }
    keypoints.push([x, y, v])
  }
  return { keypoints, unlabelled }
}

// Whether an outline is only a box: one polygon of nothing but the box's four corners, each
// next to the one before, which is what Seamtag writes for a posed object made of points alone
function boxOutline(s: Shape, box: [number, number, number, number] | null) {
  if (!box || s.shape_type !== 'polygon' || s.points.length !== 4) return false
  if (s.description || s.locked || Object.keys(s.flags).length || Object.keys(s.other).length) return false
  const [x, y, bw, bh] = box
  // 0 top left, 1 top right, 2 bottom left, 3 bottom right; -1 none
  const corners = s.points.map(([px, py]) => {
    const i = near(px, x) ? 0 : near(px, x + bw) ? 1 : -1, j = near(py, y) ? 0 : near(py, y + bh) ? 2 : -1
    return i < 0 || j < 0 ? -1 : i + j
  })
  return !corners.includes(-1) && new Set(corners).size === 4 && corners.every((c, k) => [1, 2].includes(c ^ corners[(k + 1) % 4]))
}

// A posed object, its outline among the shapes, in its group, unless that's only its box. No
// pose when nothing gives it a box.
function readPose(a: Record<string, unknown>, category: CocoCategory, image: CocoImage, keypoints: CocoPose['keypoints'], note: Note, newGroup: () => number): { shapes: Read[]; pose: CocoPose | null } {
  const outline = readSegmentation(a, category.name, image, note, () => null, 'so only its box and keypoints are imported')
  let shapes = outline && outline !== 'none' ? outline.shapes : []
  const bbox = cocoBox(a.bbox)
  let box: [number, number, number, number] | null = bbox && [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
  if (shapes.length === 1 && boxOutline(shapes[0].shape, bbox)) {
    // The box's own numbers, which its width and height may have rounded
    box = around(shapes[0].shape.points)
    shapes = []
  }
  // The box of its keypoints, as pycocotools' loadRes makes one, else of its outline
  box ??= around(keypoints.filter((k) => k !== null)) ??
    around(shapes.flatMap(({ shape, pixels: p }) => (p ? [[p.x, p.y], [p.x + p.box.width, p.y + p.box.height]] : outlineOf(shape) ?? shape.points)))
  if (!box) {
    note('warning', 'coco-no-pixels', "has neither a box nor keypoints that span one, so it's left out")
    return { shapes: [], pose: null }
  }
  const group = isObject(a.attributes) && Number.isInteger(a.attributes.group_id) ? (a.attributes.group_id as number) : newGroup()
  return {
    shapes: shapes.map((r) => ({ ...r, shape: { ...r.shape, group_id: group } })),
    pose: { label: category.name, group, box, names: [...category.keypoints!], keypoints, outlined: shapes.length > 0 }
  }
}

// An annotation as an object: the shapes its segmentation gives, else a rectangle of its box
function readObject(a: Record<string, unknown>, label: string, image: CocoImage, note: Note, tally: Tally, newGroup: () => number): Outline {
  const outline = readSegmentation(a, label, image, note, (n) => (n > 1 ? newGroup() : null), "so it's left out")
  if (outline === null) return { shapes: [], restored: false }
  if (outline !== 'none') return outline
  const box = cocoBox(a.bbox)
  if (!box) {
    note('warning', 'coco-no-pixels', "has neither a shape nor a box with an area, so it's left out")
    return { shapes: [], restored: false }
  }
  // Its polygons, if any, were all left out, each with its own note
  const seg = a.segmentation
  if (Array.isArray(seg) && seg.some((p) => !Array.isArray(p) || p.length)) note('warning', 'coco-box-only', 'keeps only its box, so it becomes a rectangle')
  else tally.box++
  const [x, y, bw, bh] = box
  return { shapes: [{ shape: newShape(label, 'rectangle', [[x, y], [x + bw, y + bh]]), pixels: null }], restored: false }
}

// An annotation's segmentation as shapes: Seamtag's own when its attributes give them back,
// else its polygons, those of one annotation sharing group(count), or its mask. 'none' when it
// has no polygon or mask to read; null when its mask can't be read, noted with `lost` as the outcome.
function readSegmentation(a: Record<string, unknown>, label: string, image: CocoImage, note: Note, group: (count: number) => number | null, lost: string): Outline | 'none' | null {
  const { width: w, height: h } = image
  const seg = a.segmentation
  let polygons: number[][] | null = null, mask: Rle | null = null
  if (Array.isArray(seg) && seg.some((p) => !Array.isArray(p) || p.length)) {
    const found: number[][] = []
    seg.forEach((p, i) => {
      const which = seg.length > 1 ? `polygon ${i + 1}` : 'its polygon'
      if (!Array.isArray(p) || !p.every((n) => typeof n === 'number' && Number.isFinite(n))) return note('warning', 'coco-bad-polygon', `has a ${which} that isn't a list of numbers, so that polygon is left out`)
      if (p.length % 2) return note('warning', 'coco-bad-polygon', `has ${which} of an odd count of numbers (${p.length}), so that polygon is left out`)
      if (p.length < 6) return note('warning', 'coco-bad-polygon', `has ${which} of ${p.length / 2} ${p.length === 2 ? 'point' : 'points'}, and a polygon needs 3, so that polygon is left out`)
      found.push(p as number[])
    })
    if (!found.length) return 'none'
    polygons = found
  } else if (isObject(seg)) {
    const size = seg.size
    if (!Array.isArray(size) || size.length !== 2 || !size.every(isCount)) {
      note('warning', 'coco-bad-mask', `has a mask without a size, ${lost}`)
      return null
    }
    if (size[0] !== h || size[1] !== w) {
      note('warning', 'coco-mask-size', `has a ${size[0]} x ${size[1]} mask on a ${h} x ${w} image, ${lost}`)
      return null
    }
    try {
      if (typeof seg.counts === 'string') mask = rleFrString(seg.counts, h, w)
      else if (Array.isArray(seg.counts) && seg.counts.every(isCount)) mask = { h, w, counts: seg.counts as number[] }
      else throw new Error('Its counts are neither a string nor a list of whole numbers')
      if (rleLength(mask) !== h * w) throw new Error(`Its runs cover ${rleLength(mask)} pixels, not ${h * w}`)
    } catch (e) {
      note('warning', 'coco-bad-mask', `has a mask that can't be read (${(e as Error).message}), ${lost}`)
      return null
    }
    if (!rleArea(mask)) {
      note('warning', 'coco-no-pixels', `has a mask with no pixels set, ${lost}`)
      return null
    }
  } else return 'none'
  // Seamtag's own export: the shapes as they were
  if (isObject(a.attributes)) {
    const restored = restore(a.attributes, label, image, polygons, mask)
    if (typeof restored !== 'string') return { shapes: restored, restored: true }
    if (restored) note('info', 'coco-bad-attributes', `carries Seamtag's attributes, but ${restored}, so it's imported from its segmentation`)
  }
  if (mask) {
    const [x, y, bw, bh] = rleToBbox(mask)
    if (bw * bh > MASK_PIXELS) {
      note('warning', 'coco-mask-too-big', `has a mask of ${bw} x ${bh} pixels, more than the ${MASK_PIXELS.toLocaleString('en-GB')} a mask shape can hold, ${lost}`)
      return null
    }
    const box = { x, y, width: bw, height: bh }
    return { shapes: [{ shape: newShape(label, 'mask', [[x, y], [x + bw - 1, y + bh - 1]]), pixels: { rle: mask, box, x, y } }], restored: false }
  }
  const g = group(polygons!.length)
  return { shapes: polygons!.map((p) => ({ shape: newShape(label, 'polygon', pairs(p), { group_id: g }), pixels: null })), restored: false }
}

// The keys of attributes that make them Seamtag's record of one shape
const RECORD_KEYS = ['shape_type', 'points', 'mask', 'description', 'flags', 'locked', 'other', 'group_id', 'index']

// Seamtag's attributes back to shapes, or why they can't be ('' when they aren't Seamtag's)
function restore(attrs: Record<string, unknown>, label: string, image: CocoImage, polygons: number[][] | null, mask: Rle | null): Read[] | string {
  const { width: w, height: h } = image
  const multi = Array.isArray(attrs.shapes)
  const records = multi ? (attrs.shapes as unknown[]) : [attrs]
  if (!multi && !RECORD_KEYS.some((k) => k in attrs)) return ''
  const group = attrs.group_id ?? null
  if (group !== null && !Number.isInteger(group)) return 'its group_id isn\'t a whole number'
  if (multi && group === null) return 'its shapes have no group_id'
  if (polygons && polygons.length !== records.length) return `it lists ${records.length} shapes for ${polygons.length} polygons`
  const out: Read[] = []
  for (const [i, r] of records.entries()) {
    if (!isObject(r)) return `shape ${i + 1} isn't an object`
    const type = r.shape_type ?? 'polygon'
    if (typeof type !== 'string' || !CLOSED_TYPES.has(type)) return `shape ${i + 1} has the shape_type ${quote(type)}`
    if (type === 'mask' && !mask) return `shape ${i + 1} is a mask in polygons`
    if (r.index !== undefined && !isCount(r.index)) return `shape ${i + 1} has an index that isn't a whole number`
    let points = r.points
    if (points === undefined) {
      if (!polygons) return `shape ${i + 1} has no points`
      const p = pairs(polygons[i])
      if (type === 'polygon' || type === 'oriented_rectangle') points = p
      else if (type === 'rectangle' && p.length === 4) points = [p[0], p[2]]
      else return `shape ${i + 1} has no points`
    }
    if (r.other !== undefined && !isObject(r.other)) return `shape ${i + 1} has other keys that aren't an object`
    const other = Object.fromEntries(Object.entries((r.other ?? {}) as Record<string, unknown>).filter(([k]) => !['label', 'points', 'group_id', 'shape_type', 'flags', 'description', 'mask', 'locked'].includes(k)))
    let shape: Shape
    try {
      shape = readShape({ ...other, label, points, shape_type: type, group_id: group, description: r.description ?? '', flags: r.flags ?? {}, mask: null, ...(r.locked === undefined ? {} : { locked: r.locked }) }, i)
    } catch (e) {
      return (e as Error).message.replace(/^Shape/, 'shape')
    }
    const need = pointCount(shape.shape_type)
    if ((need && shape.points.length !== need) || (shape.shape_type === 'polygon' && shape.points.length < 3)) return `shape ${i + 1} has ${shape.points.length} points`
    let pixels: Pixels | null = null
    if (shape.shape_type === 'mask') {
      const box = maskBox(shape.points)
      let own: Rle | null = null
      if (r.mask !== undefined) {
        const m = r.mask
        if (!isObject(m) || !Array.isArray(m.size) || m.size.length !== 2 || !m.size.every(isCount) || typeof m.counts !== 'string') return `shape ${i + 1} has a mask that can't be read`
        try {
          own = rleFrString(m.counts, m.size[0], m.size[1])
        } catch (e) {
          return `shape ${i + 1} has a mask that can't be read (${(e as Error).message})`
        }
        if (rleLength(own) !== own.h * own.w) return `shape ${i + 1} has a mask whose runs don't cover it`
      } else if (!(box.width > 0 && box.height > 0)) return `shape ${i + 1} has a mask box with no pixels`
      // Its pixels take memory: no mask of the image is more than twice as wide or tall as it
      const mw = own ? own.w : box.width, mh = own ? own.h : box.height
      if (mw > 2 * w || mh > 2 * h) return `shape ${i + 1}'s mask is ${mw} x ${mh} pixels, too big for the ${w} x ${h} image`
      if (mw * mh > MASK_PIXELS) return `shape ${i + 1}'s mask is ${mw} x ${mh} pixels, more than the ${MASK_PIXELS.toLocaleString('en-GB')} a mask shape can hold`
      pixels = own ? { rle: own, box: { x: 0, y: 0, width: own.w, height: own.h }, x: box.x, y: box.y } : { rle: mask!, box, x: box.x, y: box.y }
    }
    out.push({ shape, pixels, ...(r.index === undefined ? {} : { index: r.index as number }) })
  }
  // Another tool may have changed the segmentation and kept the attributes: the segmentation is
  // what counts then. The shapes must give it back as an export would write it.
  if (polygons) {
    for (const [i, { shape }] of out.entries()) {
      const outline = outlineInside(shape, w, h)?.points
      const same = !!outline && outline.length * 2 === polygons[i].length && outline.every(([x, y], k) => near(x, polygons[i][2 * k]) && near(y, polygons[i][2 * k + 1]))
      if (!same) return `${polygons.length > 1 ? `polygon ${i + 1}` : 'its polygon'} differs from the shape they give`
    }
    return out
  }
  // Compared as runs, so no pixels are made for shapes that may not be kept
  const runs: Rle[] = []
  for (const [i, { shape, pixels: p }] of out.entries()) {
    if (p) {
      runs.push(rleMove(p.rle, p.box, p.x, p.y, h, w))
      continue
    }
    const outline = outlineInside(shape, w, h)
    if (!outline) return `shape ${i + 1} has no outline`
    try {
      runs.push(rleFrPoly(flat(outline.points), h, w, OUTLINE_POINTS))
    } catch {
      return `shape ${i + 1}'s outline is too long to compare with its mask`
    }
  }
  // The same pixels when their union holds no more than either
  const union = rleMerge(runs), both = rleMerge([union, mask!])
  if (rleArea(both) !== rleArea(union) || rleArea(both) !== rleArea(mask!)) return 'its mask differs from the shapes they give'
  return out
}

/** COCO images matched to images in the folder */
export type CocoMatch = {
  /** COCO image id to the image's path in the folder */
  names: Map<number, string>
  missing: CocoImage[]
  ambiguous: { image: CocoImage; candidates: string[] }[]
  issues: Issue[]
}

// A path with / between folders and no . or .. left in it; null when it climbs out
function normal(p: string): string | null {
  const out: string[] = []
  for (const part of p.replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue
    if (part !== '..') out.push(part)
    else if (!out.pop()) return null
  }
  return out.join('/')
}

// A file's name: the last part of its path, wherever the path leads
const baseName = (p: string) => p.slice(p.replace(/\\/g, '/').lastIndexOf('/') + 1)

/**
 * COCO images matched to a folder's images (paths with / between folders): by file_name
 * joined to `base` (the folder the file names start from, as a path in the folder, which a ..
 * climbs out of), else by the file's name alone when exactly one image in the folder and one
 * in the COCO file have it.
 * Never a guess: the rest are missing or ambiguous, each with an Issue.
 */
export function matchCocoImages(images: CocoImage[], folder: string[], base = ''): CocoMatch {
  const names = new Map<number, string>(), missing: CocoImage[] = [], ambiguous: CocoMatch['ambiguous'] = [], issues: Issue[] = []
  const inFolder = new Map<string, string>()
  const byName = new Map<string, string[]>()
  for (const f of folder) {
    const n = normal(f)
    if (n === null) continue
    inFolder.set(n, f)
    byName.set(baseName(n), [...(byName.get(baseName(n)) ?? []), f])
  }
  // Every image of the COCO file counts, whatever its path, or a name two of them share could be matched by a guess
  const cocoNames = new Map<string, number>()
  for (const im of images) cocoNames.set(baseName(im.file_name), (cocoNames.get(baseName(im.file_name)) ?? 0) + 1)
  const prefix = normal(base) ?? ''
  const taken = new Map<string, CocoImage>()
  for (const im of images) {
    // Joined before .. is resolved, so ../images/a.jpg from the annotations folder is images/a.jpg
    const path = normal(prefix ? `${prefix}/${im.file_name}` : im.file_name)
    let found = path === null ? undefined : inFolder.get(path)
    if (found === undefined) {
      const own = baseName(im.file_name)
      const candidates = byName.get(own) ?? []
      if (candidates.length === 1 && cocoNames.get(own) === 1) found = candidates[0]
      else if (candidates.length) {
        ambiguous.push({ image: im, candidates })
        // A video's frames are often named alike in each of its folders: a few are named, not every one
        const why = candidates.length === 1 ? `other images in the COCO file are named ${pathOf(own)} too`
          : `its name matches ${candidates.length > 3 ? `${candidates.slice(0, 3).join(', ')} and ${candidates.length - 3} more` : candidates.join(', ')}`
        issues.push(issue('warning', 'coco-image-ambiguous', im.file_name, `${pathOf(im.file_name)} isn't at that path in the folder and ${why}, so its annotations are left out rather than guessed`))
        continue
      }
    }
    if (found === undefined) {
      missing.push(im)
      issues.push(issue('warning', 'coco-image-missing', im.file_name, `${pathOf(im.file_name)} isn't in the folder, so its annotations are left out`))
      continue
    }
    const before = taken.get(found)
    if (before) issues.push(issue('warning', 'coco-image-twice', found, `${found} is listed twice in the COCO file (images ${before.id} and ${im.id}), so the annotations of both go to it`))
    else taken.set(found, im)
    names.set(im.id, found)
  }
  return { names, missing, ambiguous, issues }
}
