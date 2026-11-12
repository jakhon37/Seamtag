// A folder's labels, read once for exporting, checking and counting them, and the problems
// found in them. The main process reads the files (src/main/dataset.ts); what's here is
// plain data, so the format writers, the checks and the window share it.
import type { LabelFormat } from './formats'
import type { Shape } from './labelfile'

/** A mask shape's pixels, 1 byte a pixel row by row, its top-left pixel at (x, y) in the image */
export type MaskBits = { x: number; y: number; width: number; height: number; data: Uint8Array }

export type DatasetImage = {
  /** The image's path in the folder, with / between folders */
  name: string
  /** The image's size as Seamtag shows it: upright, after EXIF orientation. 0 x 0 when it can't be read. */
  width: number
  height: number
  /** Its label file and the format it was read in; null when it has none */
  file: string | null
  format: LabelFormat | null
  shapes: Shape[]
  /** For each shape, a mask shape's pixels; null for other shapes and for masks that don't decode */
  masks: (MaskBits | null)[]
  flags: Record<string, boolean>
  verified: boolean
  /** Why the label file can't be read. Such a file is never written over. */
  broken?: string
  /** Why the image can't be read */
  unreadable?: string
  /** The size the label file gives (LabelMe's imageWidth and imageHeight), when it gives one */
  labelSize?: [number, number] | null
  /** The image a LabelMe file names in imagePath, as a path in the folder; null when it names one outside it */
  labelImage?: string | null
  /** The image file's size in bytes, for finding copies of one image */
  bytes?: number
  /**
   * What the labels were read from, when they were: the label file's size, times and inode
   * (with classes.txt's for YOLO), or a digest of the image's own entries in a CreateML list.
   * A fix made from this read is refused once it differs, as it would undo what changed.
   */
  stamp?: string
}

/**
 * Keypoint names per class, in order, as X-AnyLabeling's pose file lists them
 * (classes: {person: [nose, left_eye, ...]}), with optional skeleton edges and the pairs of
 * keypoints that swap when an image is mirrored, both by name
 */
export type KeypointSchema = {
  classes: Record<string, string[]>
  skeleton?: Record<string, [string, string][]>
  flip?: Record<string, [string, string][]>
}

/**
 * One posed object: a box and the points that share its group_id, in the schema's order.
 * A point is x, y and COCO's visibility (2 seen, 1 labelled but hidden); null when missing.
 */
export type PoseInstance = {
  label: string
  group: number | null
  /** x0, y0, x1, y1 */
  box: [number, number, number, number]
  keypoints: ([number, number, 1 | 2] | null)[]
  /** The shapes it was made from, by index in the image's list */
  shapes: number[]
}

export type Severity = 'error' | 'warning' | 'info'

/**
 * A problem in a folder's labels, or a note on what an export did with a shape. `code` names
 * the check, so the window can group them; `shape` is the shape's index in the image's list.
 */
export type Issue = { severity: Severity; code: string; image: string | null; shape?: number; message: string }

export const SEVERITIES: Severity[] = ['error', 'warning', 'info']
