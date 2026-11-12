import type { LabelFormat } from './types'

// File > Export dataset…: the folder's labels as a COCO or Ultralytics YOLO dataset
type Props = {
  root: string
  /** The save format: an image with label files in several formats is read in this one */
  format: LabelFormat
  /** The label list: the config's labels, then the project's classes; null when there is none */
  labels: string[] | null
  onClose: () => void
}

export default function ExportDialog({ onClose }: Props) {
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-label="Export dataset" onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') onClose() }}>
        <h3>Export dataset</h3>
        <div className="actions"><button className="primary" onClick={onClose}>Close</button></div>
      </div>
    </div>
  )
}
