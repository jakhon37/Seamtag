import type { LabelFormat } from './types'

// File > Import dataset…: a COCO or Ultralytics YOLO dataset's labels, written as each image's own label file
type Props = {
  root: string
  /** The save format: an image with label files in several formats is read in this one */
  format: LabelFormat
  /** The label list: the config's labels, then the project's classes; null when there is none */
  labels: string[] | null
  onClose: () => void
  /** The images whose label files the import wrote */
  onImported: (names: string[]) => void
}

export default function ImportDialog({ onClose }: Props) {
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-label="Import dataset" onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') onClose() }}>
        <h3>Import dataset</h3>
        <div className="actions"><button className="primary" onClick={onClose}>Close</button></div>
      </div>
    </div>
  )
}
