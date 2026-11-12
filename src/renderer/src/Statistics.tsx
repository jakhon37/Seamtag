import { ArrowDown, ArrowUp } from 'lucide-react'
import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import type { ShapeType } from '../../shared/labelfile'
import { counted, number, StopButton } from './Problems'
import { type LabelFormat, plainError, type Progress, type StatsView } from './types'

// Dataset > Statistics: the folder's labels counted in the main process (src/main/checking.ts
// over stats.ts) as the window's save format reads them: images by status, objects per label
// and shape type, objects per image, object sizes as COCO ranges them, and image sizes.

const TYPE_NAMES: Record<ShapeType, string> = {
  polygon: 'Polygon', rectangle: 'Rectangle', oriented_rectangle: 'Rotated box', circle: 'Circle', line: 'Line', point: 'Point',
  linestrip: 'Line strip', points: 'Points', mask: 'Mask'
}
const TILES: { key: keyof StatsView['images']; name: string; title: string }[] = [
  { key: 'total', name: 'Images', title: 'Images in the folder' },
  { key: 'labelled', name: 'Labelled', title: 'With at least one accepted shape' },
  { key: 'empty', name: 'Empty', title: 'Labelled with no shapes, so they train as backgrounds' },
  { key: 'never', name: 'Never labelled', title: 'No label file yet' },
  { key: 'verified', name: 'Verified', title: 'Marked as checked' },
  { key: 'broken', name: 'Broken', title: "Label files that can't be read" },
  { key: 'unreadable', name: 'Unreadable', title: "Image files that can't be read" },
  { key: 'aiPending', name: 'AI pending', title: 'With AI proposals not yet accepted' }
]
// Label table rows shown at first, and added by Show more
const ROWS = 200
// Objects per image past this many are counted in ranges; COCO's evaluation counts at most 100 an image
const RANGES: [number, number][] = [[0, 0], [1, 1], [2, 2], [3, 5], [6, 10], [11, 20], [21, 50], [51, 100], [101, Infinity]]

type SortKey = 'label' | 'objects' | 'images' | ShapeType
type Props = {
  root: string
  /** The window's save format: an image with label files in two formats is counted from this one */
  prefer: LabelFormat
  /** The label list, whose unused labels show with 0 */
  labels: string[] | null
  colorOf: (label: string) => string
  /** Counts up when label files changed, as after a rename: the folder is read again */
  version: number
  onRename: (label: string) => void
  onError: (message: string) => void
  onClose: () => void
}

/** Objects per image as bars: one a count while they're few, ranges once some image has many */
function histogram(rows: StatsView['objectsPerImage']) {
  const most = rows.length ? rows[rows.length - 1].objects : 0
  const sum = (lo: number, hi: number) => rows.reduce((n, r) => n + (r.objects >= lo && r.objects <= hi ? r.images : 0), 0)
  if (most <= 15) return Array.from({ length: most + 1 }, (_, k) => ({ name: String(k), images: sum(k, k) }))
  return RANGES.filter(([lo]) => lo <= most).map(([lo, hi]) => ({ name: hi === Infinity ? `${lo}+` : lo === hi ? String(lo) : `${lo}–${hi}`, images: sum(lo, hi) }))
}

export default function Statistics({ root, prefer, labels, colorOf, version, onRename, onError, onClose }: Props) {
  const [stats, setStats] = useState<StatsView | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [busy, setBusy] = useState(false)
  const [size, setSize] = useState('640')
  const [sort, setSort] = useState<{ key: SortKey; down: boolean }>({ key: 'objects', down: true })
  const [shown, setShown] = useState(ROWS)
  const [savedTo, setSavedTo] = useState('')
  const trainSize = Math.round(Number(size))
  const sizeOk = Number.isFinite(trainSize) && trainSize >= 32 && trainSize <= 16384
  const read = useRef(false)

  useEffect(() => window.api.onStats((p) => p.root === root && setProgress(p)), [root])
  const count = async (again: boolean) => {
    setBusy(true)
    setProgress(null)
    try {
      const view = await window.api.statsRead(root, { prefer, labels, trainSize: sizeOk ? trainSize : 640, again })
      if ('stopped' in view) {
        if (!read.current) onClose()
        return
      }
      read.current = true
      setStats(view)
    } catch (e) {
      onError(`The folder couldn't be counted: ${plainError(e)}`)
      if (!read.current) onClose()
    } finally {
      setBusy(false)
      setProgress(null)
    }
  }
  // The folder is read on opening, and again after its label files change
  useEffect(() => {
    count(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, version])
  // Another training size counts what was read, a moment after typing stops
  const sized = useRef(trainSize)
  useEffect(() => {
    if (!read.current || !sizeOk || sized.current === trainSize) return
    const timer = setTimeout(() => {
      sized.current = trainSize
      count(false)
    }, 300)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainSize, sizeOk])

  const rows = useMemo(() => {
    if (!stats) return []
    const value = (l: StatsView['labels'][number]) => (sort.key === 'label' ? l.label : sort.key === 'objects' ? l.objects : sort.key === 'images' ? l.images : l.types[sort.key] ?? 0)
    return [...stats.labels].sort((a, b) => {
      const x = value(a), y = value(b)
      const order = typeof x === 'string' ? x.localeCompare(y as string) : x - (y as number)
      return (sort.down ? -order : order) || a.label.localeCompare(b.label)
    })
  }, [stats, sort])
  const most = Math.max(1, ...(stats?.labels.map((l) => l.objects) ?? []))
  const bars = stats ? histogram(stats.objectsPerImage) : []
  const tallest = Math.max(1, ...bars.map((b) => b.images))

  const close = () => {
    if (busy) window.api.statsStop(root)
    onClose()
  }
  const save = async () => {
    try {
      const file = await window.api.statsSave(root)
      if (file) setSavedTo(file)
    } catch (e) {
      onError(`The statistics couldn't be saved: ${plainError(e)}`)
    }
  }
  const heading = (key: SortKey, name: string, className = '') => (
    <th className={className} aria-sort={sort.key === key ? (sort.down ? 'descending' : 'ascending') : 'none'}>
      <button onClick={() => setSort((s) => ({ key, down: s.key === key ? !s.down : key !== 'label' }))}>
        {name}{sort.key === key && (sort.down ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
      </button>
    </th>
  )
  const coco = [['small', 'Small', 'under 32 × 32 px'], ['medium', 'Medium', ''], ['large', 'Large', '96 × 96 px and over']] as const

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="dialog stats" role="dialog" aria-label="Statistics" onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') close() }}>
        <h3>Statistics</h3>
        {busy && <StopButton progress={progress} onStop={() => window.api.statsStop(root)} label="Reading the label files" />}
        {stats && (
          <>
            <div className="stats-tiles">
              {TILES.map((t) => (
                <div key={t.key} className={`stats-tile${(t.key === 'broken' || t.key === 'unreadable') && stats.images[t.key] ? ' bad-tile' : ''}`} title={t.title} data-tile={t.key}>
                  <b>{number(stats.images[t.key])}</b><span>{t.name}</span>
                </div>
              ))}
            </div>

            <h4>Labels <span className="muted">accepted shapes; AI proposals count once accepted</span></h4>
            {stats.labels.length ? (
              <div className="stats-table">
                <table className="stats-labels">
                  <thead>
                    <tr>
                      {heading('label', 'Label', 'left')}
                      {heading('objects', 'Objects', 'left wide')}
                      {heading('images', 'Images')}
                      {stats.types.map((t) => <Fragment key={t}>{heading(t, TYPE_NAMES[t])}</Fragment>)}
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, shown).map((l) => (
                      <tr key={l.label} data-label={l.label}>
                        <td className="left"><span className="stats-swatch" style={{ background: colorOf(l.label) }} /><span className="stats-name" title={l.label}>{l.label || '(no label)'}</span></td>
                        <td className="left wide">
                          <span className="stats-bar"><i style={{ width: `${(l.objects / most) * 100}%`, background: colorOf(l.label) }} /></span>
                          <b>{number(l.objects)}</b>
                        </td>
                        <td>{number(l.images)}</td>
                        {stats.types.map((t) => <td key={t} className={l.types[t] ? '' : 'muted'}>{number(l.types[t] ?? 0)}</td>)}
                        <td><button className="problem-act" onClick={() => onRename(l.label)} title={`Rename “${l.label}” in every label file of the folder`}>Rename…</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rows.length > shown && (
                  <button className="ghost problems-more" onClick={() => setShown((n) => n + ROWS)}>Show {number(Math.min(ROWS, rows.length - shown))} more of {number(rows.length - shown)}</button>
                )}
              </div>
            ) : <p className="muted">No shapes yet.</p>}

            <div className="stats-pair">
              <section aria-label="Objects per image">
                <h4>Objects per image</h4>
                {bars.length ? (
                  <div className="stats-histogram">
                    {bars.map((b) => (
                      <div key={b.name} className="stats-column" title={`${counted(b.images, 'image')} with ${b.name} ${b.name === '1' ? 'object' : 'objects'}`}>
                        <span className="muted">{b.images ? number(b.images) : ''}</span>
                        <i style={{ height: `${(b.images / tallest) * 100}%` }} />
                        <span>{b.name}</span>
                      </div>
                    ))}
                  </div>
                ) : <p className="muted">No labelled images yet.</p>}
              </section>
              <section aria-label="Object sizes">
                <h4>Object sizes</h4>
                <table className="stats-sizes">
                  <thead>
                    <tr>
                      <th className="left">COCO's range</th><th>Full size</th>
                      <th>
                        <label className="stats-train">At
                          <input type="number" min={32} max={16384} step={32} value={size} aria-label="Training size" disabled={busy}
                            className={sizeOk ? '' : 'invalid'} onChange={(e) => setSize(e.target.value)} /> px
                        </label>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {coco.map(([key, name, range]) => (
                      <tr key={key}>
                        <td className="left">{name} <span className="muted">{range}</span></td>
                        <td>{number(stats.sizes.full[key])}</td>
                        <td>{number(stats.sizes.train[key])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="muted">
                  Training scales each image so its longer side is {number(stats.sizes.trainSize)} px, as Ultralytics does.
                  {stats.sizes.unsized > 0 && ` ${counted(stats.sizes.unsized, 'object has', 'objects have')} no area here: lines, points, and shapes in images that can't be read.`}
                </p>
              </section>
            </div>

            <h4>Image sizes</h4>
            {stats.imageSizes.length ? (
              <div className="stats-image-sizes">
                {stats.imageSizes.map((s) => (
                  <span key={`${s.width}x${s.height}`}><b>{number(s.width)} × {number(s.height)}</b> <span className="muted">{counted(s.images, 'image')}</span></span>
                ))}
                {stats.moreSizes > 0 && <span className="muted">and {counted(stats.moreSizes, 'size')} more</span>}
              </div>
            ) : <p className="muted">No image could be read.</p>}
          </>
        )}
        <div className="actions">
          {savedTo && <span className="muted stats-saved" title={savedTo}>Saved as {savedTo.split(/[\\/]/).pop()}</span>}
          <button className="ghost" onClick={save} disabled={!stats || busy} title="The labels table as a CSV file">Save as CSV…</button>
          <button className="primary" onClick={close}>Close</button>
        </div>
      </div>
    </div>
  )
}
