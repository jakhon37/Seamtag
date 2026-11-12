import { ChevronDown, ChevronRight, CircleAlert, FileJson, FileText, FolderOpen, Info, TriangleAlert, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { YoloTask } from '../../shared/ultralytics'
import { useModal } from './Modal'
import { showMore } from './Problems'
import {
  type ImportPolicy, type ImportPreview, type ImportProgress, type ImportReport, type ImportRequest, type IssueGroup, type LabelFormat, plainError
} from './types'

// File > Import dataset…: a COCO or Ultralytics YOLO dataset's labels, written as each image's own label file
type Props = {
  root: string
  /** The save format: an image with label files in several formats is read in this one */
  format: LabelFormat
  /** The label list: the config's labels, then the project's classes; null when there is none */
  labels: string[] | null
  /** Labels the folder has already: the label list's, and those found in its label files */
  known: string[]
  onClose: () => void
  /** The images whose label files the import wrote */
  onImported: (names: string[]) => void
}

const TASKS: { task: YoloTask; label: string; title: string }[] = [
  { task: 'detect', label: 'Detect', title: 'Boxes: a class and 4 numbers a row' },
  { task: 'segment', label: 'Segment', title: 'Outlines: a class and the points of a polygon' },
  { task: 'pose', label: 'Pose', title: 'A box and keypoints a row' },
  { task: 'obb', label: 'OBB', title: 'Rotated boxes: a class and 4 corners' }
]
const POLICIES: { policy: ImportPolicy; label: string }[] = [
  { policy: 'skip', label: 'Skip them' }, { policy: 'add', label: 'Add shapes' }, { policy: 'replace', label: 'Replace' }
]

const count = (n: number) => n.toLocaleString('en-GB')
const some = (n: number, one: string, many = `${one}s`) => `${count(n)} ${n === 1 ? one : many}`
const listed = (words: string[]) => (words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`)
const ICONS = { error: CircleAlert, warning: TriangleAlert, info: Info }
// As the Export window names them
const SEVERITIES = { error: 'Error', warning: 'Warning', info: 'Note' }

/** Issues of one kind, folded: opened, their messages, and more of them a page at a time */
function Group({ group, list }: { group: IssueGroup; list: 'issues' | 'skipped' }) {
  const [open, setOpen] = useState(false)
  const [rows, setRows] = useState(group.rows)
  useEffect(() => setRows(group.rows), [group])
  const Icon = ICONS[group.severity]
  const more = async () => setRows([...rows, ...(await window.api.importRows(list, group.code, rows.length))])
  return (
    <div className={`import-group is-${group.severity}`} data-code={group.code}>
      <button aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Icon size={14} className="sev" aria-label={SEVERITIES[group.severity]} />
        <span className="name">{group.title}</span>
        <span className="count">{count(group.count)}</span>
      </button>
      {open && (
        <ul className="import-rows">
          {rows.map((r, i) => <li key={i}>{r}</li>)}
          {rows.length < group.count && (
            <li><button className="more" onClick={more}>{showMore(group.count - rows.length, 200)}</button></li>
          )}
        </ul>
      )}
    </div>
  )
}

function Groups({ groups, list }: { groups: IssueGroup[]; list: 'issues' | 'skipped' }) {
  return <div className="import-groups">{groups.map((g) => <Group key={`${list}:${g.code}`} group={g} list={list} />)}</div>
}

/** Labels and their shapes; a label the folder doesn't have yet, in its label list or its label files, is tagged new */
function Labels({ rows, known, listing }: { rows: { label: string; shapes: number }[]; known: ReadonlySet<string>; listing: boolean }) {
  if (!rows.length) return <p className="muted">No shapes.</p>
  const fresh = listing ? "Not in the label list or this folder's label files yet" : "Not in this folder's label files yet"
  return (
    <div className="import-labels">
      <table>
        <thead><tr><th>Label</th><th className="n">Shapes</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td>{r.label}{!known.has(r.label) && <span className="tag new" title={fresh}>new</span>}</td>
              <td className="n">{count(r.shapes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// How many images a choice writes, of those the preview found: labelled ones the dataset gives no shapes keep theirs
const writes = (p: ImportPreview, policy: ImportPolicy) => p.fresh + (policy === 'skip' ? 0 : p.labelled - p.empty)

// What a choice does to the images that have labels
function labelledText(p: ImportPreview, policy: ImportPolicy) {
  const one = p.labelled === 1
  const have = `${some(p.labelled, 'image has', 'images have')} labels already`
  const left = (n: number) => `${n === 1 ? "it's" : "they're"} left as ${n === 1 ? 'it is' : 'they are'}`
  if (policy === 'skip') return `${have}: ${left(p.labelled)}.`
  if (p.empty === p.labelled) return `${have} and ${one ? 'gets' : 'get'} no shapes from it, so ${left(p.labelled)}.`
  const what = policy === 'add'
    ? `the imported shapes go after ${one ? 'its' : 'theirs'}, and shapes ${one ? 'it has' : 'they have'} already aren't added twice`
    : `${one ? 'its label file goes' : 'their label files go'} to the trash, and the imported shapes take ${one ? 'its' : 'their'} place`
  return `${have}: ${what}.${p.empty ? ` ${some(p.empty, 'of them gets', 'of them get')} no shapes from it, so ${left(p.empty)}.` : ''}`
}

export default function ImportDialog({ root, format, labels, known, onClose, onImported }: Props) {
  const [stage, setStage] = useState<'start' | 'busy' | 'preview' | 'report'>('start')
  const [request, setRequest] = useState<ImportRequest | null>(null)
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [report, setReport] = useState<ImportReport | null>(null)
  const [progress, setProgress] = useState<ImportProgress | null>(null)
  const [error, setError] = useState('')
  const [policy, setPolicy] = useState<ImportPolicy>('skip')
  // When the rows fit more than one task, the window asks: a click on one says which
  const [confirmed, setConfirmed] = useState(false)
  // The labels the folder had when the dataset was read, so the report tags the same labels new as the preview did
  const [had, setHad] = useState<ReadonlySet<string>>(() => new Set(known))
  const busy = stage === 'busy'
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => window.api.onImportProgress(setProgress), [])
  // What was read is let go of with the window
  useEffect(() => () => void window.api.importClose(), [])

  const read = async (req: ImportRequest) => {
    setError('')
    setProgress(null)
    setStage('busy')
    try {
      const p = await window.api.importRead(root, req)
      // Stopped: what was read is gone, so it starts again
      if (!p) {
        setPreview(null)
        setStage('start')
        return
      }
      setRequest(req)
      setPreview(p)
      setHad(new Set(known))
      setStage('preview')
    } catch (e) {
      setError(plainError(e))
      setPreview(null)
      setStage('start')
    }
  }
  const pick = async (kind: ImportRequest['kind'], how: 'file' | 'folder') => {
    const file = kind === 'coco'
      ? await window.api.pickFile('Choose a COCO file', [{ name: 'COCO files', extensions: ['json'] }])
      : how === 'folder'
        ? await window.api.pickFolder('Choose a YOLO dataset folder')
        : await window.api.pickFile("Choose a YOLO dataset's data.yaml", [{ name: 'Ultralytics data.yaml', extensions: ['yaml', 'yml'] }])
    if (!file) return
    setPolicy('skip')
    setConfirmed(false)
    read({ kind, path: file, prefer: format, labels: labels ?? [] })
  }
  const chooseTask = (task: YoloTask) => {
    setConfirmed(true)
    if (request && task !== preview?.task?.task) read({ ...request, task })
  }
  const pickKeypoints = async () => {
    const file = await window.api.pickFile('Choose a keypoints file', [{ name: 'Keypoints files', extensions: ['yaml', 'yml'] }])
    if (file && request) read({ ...request, task: 'pose', keypoints: file })
  }
  const write = async () => {
    setError('')
    setProgress(null)
    setStage('busy')
    try {
      const r = await window.api.importWrite(policy)
      setReport(r)
      setStage('report')
      // The open image is read again and its undo history dropped, as after pre-labelling
      if (r.names.length) onImported(r.names)
    } catch (e) {
      setError(plainError(e))
      setStage('preview')
    }
  }
  const close = () => {
    if (busy) void window.api.importStop()
    else onClose()
  }
  // Keys stay with the window when the button that had the focus goes and the one to take it is
  // disabled, as Import is while the task is asked
  useModal(box, close)

  const p = preview
  const n = p ? writes(p, policy) : 0
  const asking = !!p?.task?.ask && !confirmed
  const labelRows = p ? p.labels.map((l) => ({ label: l.label, shapes: l.fresh + (policy === 'skip' ? 0 : l.labelled) })).filter((l) => l.shapes) : []
  // The tasks the rows fit, the guess first, when they fit more than one
  const fits = p?.task?.others.length ? [p.task.guess, ...p.task.others] : []
  const shapes = labelRows.reduce((sum, l) => sum + l.shapes, 0)
  // With no label list, every label shows under Labels as found in the label files: that's nothing to say
  const unlisted = report && labels ? report.labels.filter((l) => !l.listed).map((l) => l.label) : []

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div className="dialog import" role="dialog" aria-modal="true" aria-label="Import dataset" ref={box} tabIndex={-1} onKeyDown={(e) => e.stopPropagation()}>
        <h3>Import dataset</h3>
        {stage === 'start' && (
          <div className="import-body">
            <p>Labels from a COCO file or an Ultralytics YOLO dataset go into this folder's label files as LabelMe JSON. Nothing is written until you choose Import.</p>
            <div className="import-pick">
              <button className="ghost" onClick={() => pick('coco', 'file')} autoFocus><FileJson size={15} /> COCO file…</button>
              <button className="ghost" onClick={() => pick('yolo', 'folder')}><FolderOpen size={15} /> YOLO dataset folder…</button>
              <button className="ghost" onClick={() => pick('yolo', 'file')}><FileText size={15} /> YOLO data.yaml…</button>
            </div>
            <p className="muted">COCO: instances or person_keypoints files, as pycocotools, CVAT and Seamtag write them. YOLO: Ultralytics', Roboflow's and CVAT's layouts, with or without data.yaml.</p>
            {error && <p className="bad" role="alert">{error}</p>}
          </div>
        )}
        {busy && (
          <div className="import-body import-busy" aria-live="polite">
            <p>{progress?.phase ?? 'Starting'}…</p>
            <div className={`import-bar ${progress?.total ? '' : 'busy'}`}>
              <i style={progress?.total ? { width: `${(progress.done / progress.total) * 100}%` } : undefined} />
            </div>
            <p className="muted">{progress?.total ? `${count(progress.done)} of ${count(progress.total)}` : ' '}</p>
          </div>
        )}
        {stage === 'preview' && p && (
          <div className="import-body">
            <div className="import-source">
              {p.kind === 'coco' ? <FileJson size={16} /> : <FolderOpen size={16} />}
              <b title={p.source}>{p.name}</b>
              <span className="muted">{p.kind === 'coco' ? 'COCO' : 'Ultralytics YOLO'} · {some(p.images, 'image')}</span>
            </div>
            {p.task && (
              <section className="import-section">
                <h4>Task</h4>
                <div className={`import-seg ${asking ? 'ask' : ''}`} role="radiogroup" aria-label="Task">
                  {TASKS.map((t) => {
                    // While the window asks, no task is chosen: the guess is only marked as one, and
                    // the tasks the rows fit stand out from those they don't
                    const on = !asking && p.task!.task === t.task
                    return (
                      <button key={t.task} role="radio" aria-checked={on} className={on ? 'on' : asking && fits.includes(t.task) ? 'fits' : ''} title={t.title} onClick={() => chooseTask(t.task)}>
                        {t.label}{p.task!.guess === t.task ? ' (guess)' : ''}
                      </button>
                    )
                  })}
                </div>
                <p className="muted">{p.task.reason}.</p>
                {asking && (
                  <p className="warn ask">
                    {fits.length ? `These rows fit ${listed(fits.map((task) => TASKS.find((t) => t.task === task)!.label))}: click the one they were made for.` : 'Click the task these rows were made for.'}
                  </p>
                )}
              </section>
            )}
            {p.keypoints && (
              <section className="import-section">
                <h4>Keypoint names</h4>
                <div className="import-line">
                  <span>
                    {p.keypoints.from === 'yaml' ? "From data.yaml's kpt_names."
                      : p.keypoints.from === 'file' ? `From ${p.keypoints.file?.split(/[\\/]/).pop()}.`
                        : p.keypoints.count === 1 ? "None given, so it's called kp0."
                          : `None given, so they're called kp0 ${p.keypoints.count === 2 ? 'and' : 'to'} kp${p.keypoints.count - 1}.`}
                  </span>
                  {p.keypoints.from !== 'yaml' && <button className="ghost" onClick={pickKeypoints}>Keypoints file…</button>}
                </div>
              </section>
            )}
            <section className="import-section">
              <h4>Images</h4>
              <p>
                {!p.matched ? "None of its images are in this folder, so there's nothing to import."
                  : `${p.matched < p.images ? `${count(p.matched)} of its ${count(p.images)} images ${p.matched === 1 ? 'is' : 'are'}` : p.images === 1 ? 'Its one image is' : p.images === 2 ? 'Both its images are' : `All ${count(p.images)} of its images are`} in this folder: ` +
                    `${p.matched - p.byName ? `${count(p.matched - p.byName)} by ${p.matched - p.byName === 1 ? 'its' : 'their'} path${p.base ? ` under ${p.base}` : ''}` : ''}` +
                    `${p.matched - p.byName && p.byName ? ', ' : ''}${p.byName ? `${count(p.byName)} by file name alone` : ''}.`}
              </p>
              {(p.missing > 0 || p.ambiguous > 0) && (
                <p className="muted">
                  {[p.missing && `${some(p.missing, "isn't", "aren't")} here`, p.ambiguous && (p.ambiguous === 1 ? '1 has a name that matches more than one image' : `${count(p.ambiguous)} have names that match more than one image`)].filter(Boolean).join(', and ')}: nothing is imported for {p.missing + p.ambiguous === 1 ? 'it' : 'them'}, as listed below.
                </p>
              )}
              {p.blocked > 0 && <p className="muted">{some(p.blocked, "image can't", "images can't")} take its labels, as listed below.</p>}
              {p.nothing > 0 && <p className="muted">{some(p.nothing, 'image gets', 'images get')} nothing from it, so {p.nothing === 1 ? "it's" : "they're"} left as {p.nothing === 1 ? 'it is' : 'they are'}.</p>}
              {p.blank > 0 && <p className="muted">{some(p.blank, 'image without labels has an empty label file', 'images without labels have empty label files')} in the dataset, so {p.blank === 1 ? 'it gets' : 'they get'} an empty one here and {p.blank === 1 ? 'counts' : 'count'} as labelled.</p>}
            </section>
            {(p.labelled > 0 || p.unreadable > 0) && (
              <section className="import-section">
                <h4>Images that have labels</h4>
                {p.labelled > 0 && (
                  <>
                    <div className="import-seg" role="radiogroup" aria-label="Images that have labels">
                      {POLICIES.map((o) => (
                        <button key={o.policy} role="radio" aria-checked={policy === o.policy} className={policy === o.policy ? 'on' : ''} onClick={() => setPolicy(o.policy)}>{o.label}</button>
                      ))}
                    </div>
                    <p>{labelledText(p, policy)}</p>
                    {policy !== 'skip' && p.labelImg > 0 && (
                      <p className="muted">{some(p.labelImg, 'of them has', 'of them have')} Pascal VOC, YOLO or CreateML labels: {p.labelImg === 1 ? 'it gets' : 'they get'} LabelMe JSON, which holds every shape type, with {p.labelImg === 1 ? 'its' : 'their'} boxes{policy === 'replace' ? ' replaced' : ' kept'}, and the old {p.labelImg === 1 ? 'file goes' : 'files go'} to the trash.</p>
                    )}
                  </>
                )}
                {p.unreadable > 0 && <p className="muted">{some(p.unreadable, "label file can't", "label files can't")} be read, so {p.unreadable === 1 ? "it's" : "they're"} never touched.</p>}
              </section>
            )}
            <section className="import-section">
              <h4>Shapes to import{shapes ? ` · ${count(shapes)}` : ''}</h4>
              <Labels rows={labelRows} known={had} listing={!!labels} />
            </section>
            <section className="import-section">
              <h4>Problems</h4>
              {p.groups.length ? <Groups groups={p.groups} list="issues" /> : <p className="muted">None.</p>}
            </section>
            {error && <p className="bad" role="alert">{error}</p>}
          </div>
        )}
        {stage === 'report' && report && (
          <div className="import-body">
            <p className="import-result">
              {report.stopped ? `Stopped after ${count(report.done)} of ${some(report.total, 'image')}. ` : ''}
              {report.written ? `Wrote ${some(report.written, 'label file')} with ${some(report.shapes, 'shape')}.` : 'No label file was written.'}
            </p>
            {report.trashed > 0 && <p>{some(report.trashed, 'old label file', 'old label files')} went to the trash.</p>}
            {report.repeated > 0 && <p>{some(report.repeated, 'shape was', 'shapes were')} there already, so {report.repeated === 1 ? "it wasn't" : "they weren't"} added again.</p>}
            {unlisted.length > 0 && (
              <p>{unlisted.length === 1 ? `“${unlisted[0]}” isn't` : `${unlisted.slice(0, -1).map((l) => `“${l}”`).join(', ')} and “${unlisted[unlisted.length - 1]}” aren't`} in the label list, so {unlisted.length === 1 ? 'it shows' : 'they show'} under Labels as found in the label files.</p>
            )}
            {report.labels.length > 0 && (
              <section className="import-section">
                <h4>Shapes written</h4>
                <Labels rows={report.labels} known={had} listing={!!labels} />
              </section>
            )}
            {report.skipped.length > 0 && (
              <section className="import-section">
                <h4>Left as they were</h4>
                <Groups groups={report.skipped} list="skipped" />
              </section>
            )}
          </div>
        )}
        <div className="actions">
          {busy ? (
            <button className="ghost" onClick={() => window.api.importStop()} autoFocus><X size={15} /> Stop</button>
          ) : stage === 'preview' && p ? (
            <>
              <span className="import-sum">
                {asking ? 'Pick the task first.' : n ? `${some(n, 'image gets', 'images get')} a LabelMe JSON label file.` : 'Nothing to import with this choice.'}
              </span>
              <button className="ghost" onClick={() => { setPreview(null); setStage('start') }}>Back</button>
              <button className="ghost" onClick={onClose}>Cancel</button>
              <button className="primary" onClick={write} disabled={!n || asking} autoFocus>{n ? `Import ${some(n, 'image')}` : 'Import'}</button>
            </>
          ) : stage === 'report' ? (
            <>
              <button className="ghost" onClick={() => { setReport(null); setPreview(null); setStage('start') }}>Import another…</button>
              <button className="primary" onClick={onClose} autoFocus>Close</button>
            </>
          ) : (
            <button className="ghost" onClick={onClose}>Cancel</button>
          )}
        </div>
      </div>
    </div>
  )
}
