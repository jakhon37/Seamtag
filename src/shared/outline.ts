// The outline of a closed shape as a list of points, for formats that hold outlines only.
// Rectangles and circles become polygons as labelme 7.9.1's labelme2coco makes them
// (examples/instance_segmentation/labelme2coco.py), so COCO and YOLO files agree with it.
import type { Shape } from './labelfile'

/** The shapes with an inside. Lines, line strips and points have none. */
export const CLOSED_TYPES = new Set(['polygon', 'rectangle', 'oriented_rectangle', 'circle', 'mask'])

/** labelme2coco's vertex count: 12, doubled until the chord strays at most 1 px from the arc */
export function circleVertexCount(radius: number) {
  let n = 12
  while (radius * (1 - Math.cos(Math.PI / n)) > 1) n *= 2
  return n
}

/** labelme2coco's circle: vertices at equal angles from the +x axis. Null for a circle with no radius. */
export function circlePolygon(center: number[], edge: number[]): number[][] | null {
  const [cx, cy] = center
  const radius = Math.hypot(edge[0] - cx, edge[1] - cy)
  if (radius === 0) return null
  const n = circleVertexCount(radius)
  // numpy's linspace(0, 2π, n, endpoint=False) steps by 2π / n
  const step = (2 * Math.PI) / n
  return Array.from({ length: n }, (_, k) => [cx + radius * Math.cos(k * step), cy + radius * Math.sin(k * step)])
}

/** A rectangle's corners, sorted as labelme2coco sorts them: top left, top right, bottom right, bottom left */
export function rectangleCorners(points: number[][]): number[][] {
  const [[x1, y1], [x2, y2]] = points
  const [left, right] = x1 < x2 ? [x1, x2] : [x2, x1]
  const [top, bottom] = y1 < y2 ? [y1, y2] : [y2, y1]
  return [[left, top], [right, top], [right, bottom], [left, bottom]]
}

/**
 * A closed shape's outline: a polygon's points, a rectangle's or rotated box's four corners,
 * a circle's polygon. Null for masks, whose outline comes from their pixels, for open shapes,
 * and for shapes too broken to have one.
 */
export function outlineOf(s: Shape): number[][] | null {
  switch (s.shape_type) {
    case 'polygon':
      return s.points.length >= 3 ? s.points.map(([x, y]) => [x, y]) : null
    case 'rectangle':
      return s.points.length === 2 ? rectangleCorners(s.points) : null
    case 'oriented_rectangle':
      return s.points.length === 4 ? s.points.map(([x, y]) => [x, y]) : null
    case 'circle':
      return s.points.length === 2 ? circlePolygon(s.points[0], s.points[1]) : null
    default:
      return null
  }
}
