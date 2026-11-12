// Counts over a folder's labels for the Dataset window's Statistics tab: images by status,
// objects per label, grouped as the exports group them, and shapes per label by type, objects per
// image, object sizes as COCO ranges them at full size and at the training size, and image sizes.
// AI proposals aren't objects until they're accepted, as summarize() in labels.ts counts them.
import { shapeArea } from './checks'
import type { DatasetImage } from './dataset'
import { isAi, SHAPE_TYPES, type ShapeType } from './labelfile'
import { CLOSED_TYPES, outlineOf } from './outline'
import { rleArea, rleFrMaskBits, rleFrPoly, rleMerge } from './rle'

export type SizeCounts = { small: number; medium: number; large: number }
export type LabelStats = {
  label: string
  /** Objects with the label, as objectsOf finds them */
  objects: number
  /** Images with at least one accepted shape with the label */
  images: number
  /** Accepted shapes with the label by shape type, a pose's keypoints too */
  types: Partial<Record<ShapeType, number>>
}

export type DatasetStats = {
  images: {
    total: number
    /** A label file that reads, with at least one accepted shape */
    labelled: number
    /** A label file that reads, with no shapes at all: a background */
    empty: number
    /** No label file */
    never: number
    verified: number
    /** A label file that can't be read */
    broken: number
    /** An image that can't be read */
    unreadable: number
    /** AI proposals not yet accepted; an image with nothing else is neither labelled nor empty */
    aiPending: number
  }
  /** Most objects first; labels from the list that no shape uses come last with 0 */
  labels: LabelStats[]
  /** How many images have each number of objects, fewest first; images with no label file, or one that can't be read, are left out */
  objectsPerImage: { objects: number; images: number }[]
  /**
   * Objects by COCO's area ranges, at the image's own size and scaled as training at
   * `trainSize` scales it; an object of several shapes by their area together. Lines, points,
   * masks that don't decode, shapes with the wrong number of points and objects in images of
   * unknown size have no area here: `unsized` counts them.
   */
  sizes: { trainSize: number; full: SizeCounts; train: SizeCounts; unsized: number }
  /** Each size of image there is, the most common first */
  imageSizes: { width: number; height: number; images: number }[]
  /** The shape types accepted shapes have, in labelme's order: the per-label table's columns */
  types: ShapeType[]
}

// COCO's ranges, as pycocotools 2.0.11's COCOeval sets them (cocoeval.py:509): small below 32²
// square pixels, large from 96². COCOeval counts an area of exactly 32² or 96² in the ranges
// either side of it; here it goes to the larger.
const SMALL = 32 ** 2, LARGE = 96 ** 2
const range = (area: number): keyof SizeCounts => (area < SMALL ? 'small' : area < LARGE ? 'medium' : 'large')

/**
 * An image's objects, grouped as the exports group them, each a list of its shapes, AI proposals
 * left out. Shapes sharing a label and a group_id are one object, as src/shared/coco.ts'
 * instances() and ultralytics.ts' labelRows() group them, and any other shape is one of its own:
 * lines and points too, though those exports leave them out. The points of a pose, a group whose
 * closed shapes all have one label, are its keypoints and no objects of their own, as
 * src/main/export.ts' keypointSchemaOf() finds poses without a keypoints file.
 */
function objectsOf(image: DatasetImage): number[][] {
  const groups = new Map<number, { boxes: Set<string>; points: number[] }>()
  image.shapes.forEach((s, i) => {
    if (s.group_id === null || isAi(s)) return
    const g = groups.get(s.group_id) ?? { boxes: new Set<string>(), points: [] }
    groups.set(s.group_id, g)
    if (CLOSED_TYPES.has(s.shape_type)) g.boxes.add(s.label)
    else if (s.shape_type === 'point') g.points.push(i)
  })
  const keypoints = new Set<number>()
  for (const { boxes, points } of groups.values()) if (boxes.size === 1) for (const i of points) keypoints.add(i)
  const objects = new Map<string | number, number[]>()
  image.shapes.forEach((s, i) => {
    if (isAi(s) || keypoints.has(i)) return
    const key = s.group_id === null ? i : `${s.label}\u0000${s.group_id}`
    objects.set(key, [...(objects.get(key) ?? []), i])
  })
  return [...objects.values()]
}

/**
 * An object's area in square pixels: one shape's as shapeArea measures it; the pixels of several
 * together, as COCO's export merges them (coco.ts' segment()), so where they overlap counts once:
 * masks by their pixels, other shapes filled as pycocotools fills their outlines, kept inside the
 * image as the export keeps them. Null when no shape of it has an area.
 */
function objectArea(image: DatasetImage, shapes: number[]): number | null {
  const parts = shapes.map((k) => ({ k, area: shapeArea(image.shapes[k], image.masks[k] ?? null) })).filter((p) => p.area !== null)
  if (parts.length < 2) return parts[0]?.area ?? null
  const { width: w, height: h } = image
  const rles = parts.flatMap(({ k }) => {
    const s = image.shapes[k]
    if (s.shape_type === 'mask') return [rleFrMaskBits(image.masks[k]!, h, w)]
    const outline = outlineOf(s)?.map(([x, y]) => [Math.min(Math.max(x, 0), w), Math.min(Math.max(y, 0), h)])
    return outline ? [rleFrPoly(outline.flat(), h, w)] : []
  })
  return rleArea(rleMerge(rles))
}

/**
 * A folder's statistics. Training scales an image so its longer side is `trainSize`, as
 * Ultralytics 8.4.211's BaseDataset.load_image does (data/base.py:355): a 10 px object in a
 * 6000 px image shrinks to about 1 px at 640. `labels` adds the label list's labels that no
 * shape uses, with 0.
 */
export function datasetStats(images: DatasetImage[], { trainSize = 640, labels = null }: { trainSize?: number; labels?: string[] | null } = {}): DatasetStats {
  const status = { total: images.length, labelled: 0, empty: 0, never: 0, verified: 0, broken: 0, unreadable: 0, aiPending: 0 }
  const perLabel = new Map<string, LabelStats>()
  const perImage = new Map<number, number>()
  const sizes: DatasetStats['sizes'] = { trainSize, full: { small: 0, medium: 0, large: 0 }, train: { small: 0, medium: 0, large: 0 }, unsized: 0 }
  const imageSizes = new Map<string, { width: number; height: number; images: number }>()
  const used = new Set<ShapeType>()
  const rowOf = (label: string) => {
    let row = perLabel.get(label)
    if (!row) perLabel.set(label, (row = { label, objects: 0, images: 0, types: {} }))
    return row
  }
  for (const image of images) {
    if (image.verified) status.verified++
    const sized = !image.unreadable && image.width > 0 && image.height > 0
    if (image.unreadable) status.unreadable++
    else if (sized) {
      const key = `${image.width}x${image.height}`
      const known = imageSizes.get(key)
      if (known) known.images++
      else imageSizes.set(key, { width: image.width, height: image.height, images: 1 })
    }
    if (image.broken) {
      status.broken++
      continue
    }
    if (!image.file) {
      status.never++
      continue
    }
    const accepted = image.shapes.filter((s) => !isAi(s))
    if (accepted.length < image.shapes.length) status.aiPending++
    if (accepted.length) status.labelled++
    else if (!image.shapes.length) status.empty++
    const seen = new Set<string>()
    for (const s of accepted) {
      const row = rowOf(s.label)
      row.types[s.shape_type] = (row.types[s.shape_type] ?? 0) + 1
      if (!seen.has(s.label)) row.images++
      seen.add(s.label)
      used.add(s.shape_type)
    }
    const objects = objectsOf(image)
    perImage.set(objects.length, (perImage.get(objects.length) ?? 0) + 1)
    const scale = sized ? trainSize / Math.max(image.width, image.height) : 0
    for (const shapes of objects) {
      rowOf(image.shapes[shapes[0]].label).objects++
      const area = sized ? objectArea(image, shapes) : null
      if (area === null) {
        sizes.unsized++
        continue
      }
      sizes.full[range(area)]++
      sizes.train[range(area * scale * scale)]++
    }
  }
  const rows = [...perLabel.values()].sort((a, b) => b.objects - a.objects || a.label.localeCompare(b.label))
  for (const label of labels ?? []) if (!perLabel.has(label)) rows.push({ label, objects: 0, images: 0, types: {} })
  return {
    images: status,
    labels: rows,
    objectsPerImage: [...perImage].sort((a, b) => a[0] - b[0]).map(([objects, count]) => ({ objects, images: count })),
    sizes,
    imageSizes: [...imageSizes.values()].sort((a, b) => b.images - a.images || a.width - b.width || a.height - b.height),
    types: SHAPE_TYPES.filter((t) => used.has(t))
  }
}

/**
 * A cell as RFC 4180 writes it: quoted when it holds a comma, a quote, a line break or spaces
 * at either end. Labels come from anyone's files, and a spreadsheet runs a cell starting with
 * =, +, -, @, a tab or a carriage return as a formula, so such a text starts with a quote
 * and is quoted, as OWASP's CSV injection advice has it.
 */
function cell(value: string | number) {
  if (typeof value === 'number') return String(value)
  const text = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value
  return text !== value || /[",\r\n]/.test(text) || text !== text.trim() ? `"${text.replace(/"/g, '""')}"` : text
}

/**
 * The per-label table as CSV: label, objects and images, then shapes of each shape type used.
 * It starts with a byte order mark, without which Excel reads the file in the system's own code
 * page, so a label such as 고양이 or café comes out garbled.
 */
export function statsCsv(stats: DatasetStats): string {
  const rows = [
    ['label', 'objects', 'images', ...stats.types],
    ...stats.labels.map((l) => [l.label, l.objects, l.images, ...stats.types.map((t) => l.types[t] ?? 0)])
  ]
  return '\ufeff' + rows.map((row) => row.map(cell).join(',')).join('\r\n') + '\r\n'
}
