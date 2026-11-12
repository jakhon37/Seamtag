import { ChevronDown, ChevronRight, CircleAlert, CircleCheck, FolderOpen, Info, TriangleAlert, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ExportFormat, ExportOptions, ExportProgress, ExportResult, IssueGroup, KeypointsSummary, Preflight, SplitCount } from '../../main/export'
import type { Issue, Severity } from '../../shared/dataset'
import { useModal } from './Modal'
import { showMore } from './Problems'
import { type LabelFormat, plainError } from './types'

// File > Export dataset…: the folder's labels as a COCO or Ultralytics YOLO dataset. A check runs
// whenever an option changes and says, before anything is written, what goes in and what the format
// can't hold; errors stop the export. Each image's own label file stays the working copy.
type Props = {
  root: string
  /** The save format: an image with label files in several formats is read in this one */
  format: LabelFormat
  /** The label list: the config's labels, then the project's classes; null when there is none */
  labels: string[] | null
  onClose: () => void
}

const FORMATS: { id: ExportFormat; name: string; holds: string }[] = [
  { id: 'coco', name: 'COCO', holds: "Outlines and masks, with pycocotools' areas" },
  { id: 'coco-keypoints', name: 'COCO with keypoints', holds: "The objects, and posed objects' keypoints" },
  { id: 'yolo-detect', name: 'YOLO detect', holds: 'A box around each object' },
  { id: 'yolo-segment', name: 'YOLO segment', holds: "Each object's outline" },
  { id: 'yolo-pose', name: 'YOLO pose', holds: 'A box and its keypoints for each posed object' },
  { id: 'yolo-obb', name: 'YOLO OBB', holds: 'The rotated box around each object' }
]
type Mode = ExportOptions['split']['mode']
const SPLITS: { id: Mode; name: string }[] = [{ id: 'none', name: 'None' }, { id: 'ratio', name: 'By ratio' }, { id: 'folders', name: 'Keep the folders' }]
const IMAGES: { id: ExportOptions['images']; name: string; hint: string }[] = [
  { id: 'copy', name: 'Copy', hint: 'Copied into images/, keeping their subfolders.' },
  { id: 'link', name: 'Hard link', hint: "Hard links share each file with the open folder, so they take no space; an image is copied where a link can't be made, as across drives." },
  { id: 'leave', name: 'Leave in place', hint: "No images are written: each file_name is the image's path in the open folder." }
]
const PHASES: Record<ExportProgress['phase'], string> = { read: 'Reading label files', check: 'Checking images', write: 'Writing images', finish: 'Finishing' }
const ICONS: Record<Severity, React.ReactNode> = {
  error: <CircleAlert size={15} className="export-sev is-error" aria-label="Error" />,
  warning: <TriangleAlert size={15} className="export-sev is-warning" aria-label="Warning" />,
  info: <Info size={15} className="export-sev is-info" aria-label="Note" />
}
// A group's rows grow a page at a time, up to this many in the window
const MOST = 2000
/** A button in an error's group that fixes it, where an option of the window can */
type Fix = { label: string; run: () => void }

const n = (v: number) => v.toLocaleString('en-GB')
const count = (v: number, one: string, many = `${one}s`) => `${n(v)} ${v === 1 ? one : many}`
const baseName = (file: string) => file.split(/[\\/]/).filter(Boolean).pop() ?? file
const parentOf = (file: string) => file.replace(/[\\/]+[^\\/]*[\\/]*$/, '') || '/'

// Choices the window keeps for next time, in this window's profile only
const KEPT = 'seamtag:export'
type Kept = { format: ExportFormat; mode: Mode; train: number; val: number; test: number; seed: number; images: ExportOptions['images']; backgrounds: boolean; leaveAi: boolean }
const DEFAULTS: Kept = { format: 'coco', mode: 'none', train: 80, val: 10, test: 10, seed: 0, images: 'copy', backgrounds: false, leaveAi: false }
function readKept(): Kept {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEPT) ?? '{}') }
  } catch {
    return DEFAULTS
  }
}

export default function ExportDialog({ root, format: prefer, labels, onClose }: Props) {
  const [kept, setKept] = useState<Kept>(readKept)
  const [out, setOut] = useState<{ path: string; problem: string | null } | null>(null)
  const [keypoints, setKeypoints] = useState<KeypointsSummary | null>(null)
  const [check, setCheck] = useState<Preflight | null>(null)
  const [checking, setChecking] = useState(false)
  // Checks again with the same read of the folder (a keypoints file chosen again), or with a new one (after an export fails)
  const [again, setAgain] = useState(0)
  const [reread, setReread] = useState(0)
  const [progress, setProgress] = useState<ExportProgress | null>(null)
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState<ExportResult | null>(null)
  // What went wrong with something the user did, kept until they do something else; and with the last check
  const [error, setError] = useState('')
  const [checkError, setCheckError] = useState('')
  const [note, setNote] = useState('')
  const seq = useRef(0)
  // The window's own id for the folder's read: main reads it once, and again only under a new one
  const session = useRef(`${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`)
  const box = useRef<HTMLDivElement>(null)
  const errorsBox = useRef<HTMLDivElement>(null)
  const set = (change: Partial<Kept>) => setKept((k) => ({ ...k, ...change }))

  const coco = kept.format.startsWith('coco')
  const posed = kept.format === 'coco-keypoints' || kept.format === 'yolo-pose'
  const images = !coco && kept.images === 'leave' ? 'copy' : kept.images
  const shares = [kept.train, kept.val, kept.test]
  const sharesBad = kept.mode === 'ratio' && (shares.some((p) => !Number.isInteger(p) || p < 0) || shares.reduce((a, b) => a + b, 0) !== 100 || kept.train === 0)
    ? 'Train, val and test need whole percentages adding up to 100, with some for train'
    : kept.mode === 'ratio' && (!Number.isInteger(kept.seed) || kept.seed < 0) ? 'The seed needs to be a whole number, 0 or more' : ''
  const options: ExportOptions = {
    format: kept.format,
    split: kept.mode === 'ratio' ? { mode: 'ratio', train: kept.train, val: kept.val, test: kept.test, seed: kept.seed } : { mode: kept.mode },
    images, backgrounds: kept.backgrounds, leaveAi: kept.leaveAi, keypoints: posed ? keypoints?.file ?? null : null, prefer, labels: labels ?? []
  }
  // Copying or linking reads the same: only leaving images in place changes what the check finds
  const checkKey = sharesBad ? '' : JSON.stringify({ ...options, images: images === 'leave' ? 'leave' : 'copy', again, reread })

  useEffect(() => {
    try {
      localStorage.setItem(KEPT, JSON.stringify(kept))
    } catch {
      // Not kept, then
    }
  }, [kept])
  useEffect(() => window.api.onExportProgress(setProgress), [])
  // A check still running when the window closes is stopped, and what the checks kept is let go
  useEffect(() => () => void window.api.exportClose(), [])
  useEffect(() => {
    if (!checkKey || running || done) return
    const at = ++seq.current
    setChecking(true)
    setProgress(null)
    // Typing a share or a seed checks once, when it settles
    const timer = setTimeout(() => {
      window.api.exportCheck(root, options, `${session.current}:${reread}`).then((p) => {
        if (at !== seq.current) return
        setCheck(p)
        setCheckError('')
      }).catch((e) => {
        if (at !== seq.current) return
        // What an earlier check found isn't what these options would export
        setCheck(null)
        setCheckError(plainError(e))
      }).finally(() => {
        if (at === seq.current) setChecking(false)
      })
    }, 200)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkKey])

  const chooseFolder = async () => {
    try {
      const picked = await window.api.exportFolder(root)
      if (!picked) return
      setOut(picked)
      setError('')
    } catch (e) {
      setError(plainError(e))
    }
  }
  const pickKeypoints = async (make: boolean) => {
    try {
      const picked = make ? await window.api.exportMakeKeypoints(root, prefer) : await window.api.exportKeypoints()
      if (!picked) return
      setKeypoints(picked)
      setAgain((a) => a + 1)
      setError('')
      setNote(make ? `${baseName(picked.file)} is made from the folder's posed objects: edit it to name keypoints that are missing, then choose it again.` : '')
    } catch (e) {
      setError(plainError(e))
    }
  }
  // Errors an option of the window fixes get a button for it in their group, as the option itself is
  // often out of view in a short window
  const fixes: Record<string, Fix[]> = {
    'export-ai-pending': [{ label: 'Leave them out', run: () => set({ leaveAi: true }) }],
    'export-no-keypoints': [{ label: 'Choose one…', run: () => pickKeypoints(false) }, { label: 'Make one…', run: () => pickKeypoints(true) }],
    'export-keypoints-unreadable': [{ label: 'Choose another…', run: () => pickKeypoints(false) }],
    'export-in-label-file': [{ label: 'Copy the images instead', run: () => set({ images: 'copy' }) }]
  }
  // The footer's word on what blocks the export goes to the first error, open, wherever the column is scrolled
  const showErrors = () => {
    const head = errorsBox.current?.querySelector<HTMLButtonElement>('.export-group-head')
    if (!head) return
    if (head.getAttribute('aria-expanded') === 'false') head.click()
    head.focus({ preventScroll: true })
    head.scrollIntoView({ block: 'nearest' })
  }
  const run = async () => {
    if (!out || out.problem || !check || check.blocked || checking) return
    setRunning(true)
    setProgress(null)
    setError('')
    setNote('')
    try {
      const result = await window.api.exportRun(root, options, out.path)
      if (result.stopped) setNote('Stopped: nothing was written.')
      else setDone(result)
    } catch (e) {
      setError(plainError(e))
      // The folder may have taken files meanwhile, or the labels changed: the window reads them and checks again
      setOut(null)
      setReread((r) => r + 1)
    } finally {
      setRunning(false)
    }
  }
  const close = () => {
    if (!running) onClose()
  }
  useModal(box, close)

  const ready = !!check && !check.blocked && !checking && !!out && !out.problem && !sharesBad
  const total = check ? check.splits.reduce((a, s) => a + s.images, 0) : 0
  // Errors stop the export, so they come first in the column, where a long label list can't push them
  // out of a short window; the warnings and notes come after the counts
  const errors = check ? check.groups.filter((g) => g.severity === 'error') : []
  const rest = check ? check.groups.filter((g) => g.severity !== 'error') : []
  const one = check?.counts.error === 1
  const why = sharesBad ? sharesBad : !check ? ''
    : check.blocked ? `Fix ${one ? 'the error' : 'the errors'}${out?.problem ? ' and choose another folder' : ' first'}`
      : !out ? 'Choose a folder to export to' : out.problem ? 'Choose another folder' : ''
  // The buttons below are keyed, so the one with the focus never turns into another under it, as
  // Stop would into Close when the export ends: the window takes the focus instead

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className={`dialog export-dialog ${done ? 'done' : ''}`} role="dialog" aria-modal="true" aria-label="Export dataset" tabIndex={-1} ref={box}
        onKeyDown={(e) => e.stopPropagation()}>
        <h3>Export dataset</h3>
        {done ? <Summary result={done} /> : (
          <div className="export-body">
            <fieldset className="export-options" disabled={running}>
              <h4>Format</h4>
              <div className="export-formats" role="radiogroup" aria-label="Format">
                {FORMATS.map((f) => (
                  <label key={f.id} className={`export-format ${kept.format === f.id ? 'on' : ''}`}>
                    <input type="radio" name="export-format" checked={kept.format === f.id} onChange={() => set({ format: f.id })} />
                    <span><b>{f.name}</b><small>{f.holds}</small></span>
                  </label>
                ))}
              </div>

              <h4>Folder</h4>
              <div className="export-file">
                {out ? <span title={out.path}><b>{baseName(out.path)}</b> <span className="muted">in {parentOf(out.path)}</span></span> : <span className="muted">A new or empty folder</span>}
                <button className="ghost" onClick={chooseFolder}><FolderOpen size={14} /> Choose…</button>
              </div>
              {out?.problem && <p className="bad">{out.problem}</p>}

              <h4>Split</h4>
              <div className="seg small" role="radiogroup" aria-label="Split">
                {SPLITS.map((s) => (
                  <button key={s.id} role="radio" aria-checked={kept.mode === s.id} className={kept.mode === s.id ? 'on' : ''} onClick={() => set({ mode: s.id })}>{s.name}</button>
                ))}
              </div>
              {kept.mode === 'ratio' && (
                <div className="export-shares">
                  {(['train', 'val', 'test', 'seed'] as const).map((k) => (
                    <label key={k}><span>{k === 'seed' ? 'Seed' : `${k[0].toUpperCase()}${k.slice(1)} %`}</span>
                      <input type="number" min={0} max={k === 'seed' ? undefined : 100} step={1} value={Number.isFinite(kept[k]) ? kept[k] : ''}
                        onChange={(e) => set({ [k]: e.target.value === '' ? NaN : Number(e.target.value) } as Partial<Kept>)} aria-label={k === 'seed' ? 'Seed' : `${k} percent`} />
                    </label>
                  ))}
                </div>
              )}
              <p className="export-hint">
                {kept.mode === 'none' ? (coco ? 'One annotations file, called default.' : 'images/ and labels/ hold every image.')
                  : kept.mode === 'ratio' ? 'Images are sorted by path, then shuffled with the seed: the same seed gives the same split.'
                    : images === 'leave' ? "An image's first train, val, valid, validation or test folder decides its split. Images under none of them go to train."
                      : "An image's first train, val, valid, validation or test folder decides its split, and moves to the front of its path: a/val/b.jpg goes to images/val/a/b.jpg. Images under none of them go to train."}
              </p>
              {sharesBad && <p className="bad">{sharesBad}</p>}

              <h4>Images</h4>
              <div className="seg small" role="radiogroup" aria-label="Images">
                {IMAGES.filter((m) => coco || m.id !== 'leave').map((m) => (
                  <button key={m.id} role="radio" aria-checked={images === m.id} className={images === m.id ? 'on' : ''} onClick={() => set({ images: m.id })}>{m.name}</button>
                ))}
              </div>
              <p className="export-hint">{IMAGES.find((m) => m.id === images)!.hint}</p>
              <label className="check"><input type="checkbox" checked={kept.backgrounds} onChange={(e) => set({ backgrounds: e.target.checked })} />Include never-labelled images as backgrounds</label>
              <label className="check"><input type="checkbox" checked={kept.leaveAi} onChange={(e) => set({ leaveAi: e.target.checked })} />Leave unaccepted AI proposals out</label>
              <p className="export-hint">Images labelled with no shapes always go in, as backgrounds.</p>

              {posed && (
                <>
                  <h4>Keypoints file</h4>
                  <div className="export-file">
                    <span className={keypoints ? '' : 'muted'} title={keypoints?.file}>{keypoints ? baseName(keypoints.file) : 'Each class and its keypoints, in order'}</span>
                    <button className="ghost" onClick={() => pickKeypoints(false)}>Choose…</button>
                  </div>
                  {keypoints && (
                    <p className="export-hint">
                      {keypoints.classes.map((c) => `${c.name}: ${count(c.keypoints, 'keypoint')}`).join(' · ')}
                      {keypoints.flip ? ' · flip pairs' : ''}{keypoints.skeleton ? ' · skeleton' : ''}
                    </p>
                  )}
                  <button className="ghost export-make" onClick={() => pickKeypoints(true)}>Make a keypoints file…</button>
                  <p className="export-hint">X-AnyLabeling's pose file. Making one lists each box's label with the labels of the points sharing its group_id.</p>
                </>
              )}
            </fieldset>

            <section className="export-check" aria-label="What the export holds" aria-busy={checking}>
              {checking ? (
                <Busy progress={progress} />
              ) : check ? (
                <p className="export-lead">{count(total, 'image')} and {count(check.splits.reduce((a, s) => a + s.objects, 0), 'object')}{check.splits.some((s) => s.posed) ? `, ${count(check.splits.reduce((a, s) => a + s.posed, 0), 'posed object')}` : ''}</p>
              ) : (
                <p className="export-lead muted">Not checked</p>
              )}
              {check && (
                <div className={checking || sharesBad ? 'export-stale' : ''}>
                  {errors.length > 0 && (
                    <div className="export-errors" ref={errorsBox}>
                      {errors.map((g, i) => <Group key={`${check.id}:${g.key}`} id={check.id} group={g} open={i === 0} fixes={fixes[g.code]} />)}
                    </div>
                  )}
                  <Counts splits={check.splits} labels={check.labels} />
                  {check.notes.map((t) => <p key={t} className="export-hint">{t}</p>)}
                  <h4 className="export-issues-title">
                    Problems
                    <span className="muted">{[check.counts.error && count(check.counts.error, 'error'), check.counts.warning && count(check.counts.warning, 'warning'), check.counts.info && count(check.counts.info, 'note')].filter(Boolean).join(' · ') || 'none'}</span>
                  </h4>
                  {rest.map((g) => <Group key={`${check.id}:${g.key}`} id={check.id} group={g} open={false} />)}
                  {!rest.length && <p className="muted">{errors.length ? `Once ${one ? 'the error is' : 'the errors are'} fixed, everything goes in as it is.` : 'Everything goes in as it is.'}</p>}
                </div>
              )}
            </section>
          </div>
        )}
        {(error || checkError) && <p className="error static" role="alert">{error || checkError}</p>}
        {note && !error && !checkError && <p className="export-note">{note}</p>}
        <div className="actions">
          {running ? (
            <>
              <span className="export-status"><Busy progress={progress} inline /></span>
              <button key="stop" className="ghost batch" onClick={() => window.api.exportStop()}>
                <i style={{ width: `${progress && progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
                <X size={15} /> Stop
              </button>
            </>
          ) : done ? (
            <>
              <button key="reveal" className="ghost" onClick={() => window.api.exportReveal(done.out)}><FolderOpen size={15} /> Show in folder</button>
              <button key="close" className="primary" onClick={onClose}>Close</button>
            </>
          ) : (
            <>
              {why && (
                <span className="export-status" id="export-why">
                  {check?.blocked && !sharesBad ? <button className="link" onClick={showErrors}>{why}</button> : <span className="muted">{why}</span>}
                </span>
              )}
              <button key="cancel" className="ghost" onClick={onClose}>Cancel</button>
              <button key="export" className="primary" onClick={run} disabled={!ready} aria-describedby={why ? 'export-why' : undefined}>
                {ready ? `Export ${count(total, 'image')}` : 'Export'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function Busy({ progress, inline }: { progress: ExportProgress | null; inline?: boolean }) {
  const text = progress ? `${PHASES[progress.phase]}${progress.total > 1 ? `: ${n(progress.done)} of ${n(progress.total)}` : '…'}` : inline ? 'Exporting…' : 'Checking…'
  return (
    <div className={inline ? 'export-busy inline' : 'export-busy'}>
      <span>{text}</span>
      {!inline && <div className="progress"><i className="done" style={{ width: `${progress?.total ? (progress.done / progress.total) * 100 : 0}%` }} /></div>}
    </div>
  )
}

/** Images and objects in each split, and each label's objects in them */
function Counts({ splits, labels }: { splits: SplitCount[]; labels: { label: string; objects: number[] }[] }) {
  const [all, setAll] = useState(false)
  const many = splits.length > 1
  const shown = all ? labels : labels.slice(0, 12)
  const sum = (v: number[]) => v.reduce((a, b) => a + b, 0)
  return (
    <div className="export-counts">
      <table className="export-table">
        <thead>
          <tr><th />{splits.map((s) => <th key={s.folder}>{s.name}</th>)}{many && <th>All</th>}</tr>
        </thead>
        <tbody>
          <tr className="total"><td>Images</td>{splits.map((s) => <td key={s.folder}>{n(s.images)}</td>)}{many && <td>{n(sum(splits.map((s) => s.images)))}</td>}</tr>
          <tr className="total"><td>Backgrounds</td>{splits.map((s) => <td key={s.folder}>{n(s.backgrounds)}</td>)}{many && <td>{n(sum(splits.map((s) => s.backgrounds)))}</td>}</tr>
          <tr className="total"><td>Objects</td>{splits.map((s) => <td key={s.folder}>{n(s.objects)}</td>)}{many && <td>{n(sum(splits.map((s) => s.objects)))}</td>}</tr>
          {shown.map((l) => (
            <tr key={l.label} className={sum(l.objects) ? '' : 'none'}>
              <td title={l.label}><span className="name">{l.label}</span></td>
              {l.objects.map((v, i) => <td key={i}>{n(v)}</td>)}
              {many && <td>{n(sum(l.objects))}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {labels.length > shown.length && <button className="link export-more" onClick={() => setAll(true)}>Show all {n(labels.length)} labels</button>}
    </div>
  )
}

/** A kind of issue: its title and count, opening to its messages a page at a time, and what fixes it */
function Group({ id, group, open: first, fixes = [] }: { id: number; group: IssueGroup; open: boolean; fixes?: Fix[] }) {
  const [open, setOpen] = useState(first)
  const [rows, setRows] = useState<Issue[]>(group.issues)
  const [busy, setBusy] = useState(false)
  const more = async () => {
    setBusy(true)
    try {
      const next = await window.api.exportIssues(id, group.key, rows.length)
      setRows((r) => [...r, ...next])
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className={`export-group is-${group.severity}`}>
      <button className="export-group-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {ICONS[group.severity]}
        <span className="name">{group.title}</span>
        <b>{n(group.count)}</b>
      </button>
      {open && (
        <ul className="export-rows">
          {rows.map((r, i) => <li key={i}>{r.message}</li>)}
          {rows.length < group.count && (rows.length < MOST
            ? <li><button className="link" onClick={more} disabled={busy}>{showMore(group.count - rows.length, 200)}</button></li>
            : <li className="muted">The first {n(rows.length)} of {n(group.count)} are shown.</li>)}
        </ul>
      )}
      {fixes.length > 0 && <div className="export-fixes">{fixes.map((f) => <button key={f.label} className="ghost" onClick={f.run}>{f.label}</button>)}</div>}
    </div>
  )
}

/** What the export wrote, and what it went ahead with */
function Summary({ result }: { result: ExportResult }) {
  const { images: im } = result
  const objects = result.splits.reduce((a, s) => a + s.objects, 0)
  const total = result.splits.reduce((a, s) => a + s.images, 0)
  const how = [
    im.copied && `${n(im.copied)} copied`, im.linked && `${n(im.linked)} hard-linked`,
    im.fellBack && `${n(im.fellBack)} copied where a link couldn't be made`, im.left && `${n(im.left)} left in place`
  ].filter(Boolean).join(', ')
  return (
    <div className="export-done">
      <p className="export-lead"><CircleCheck size={18} className="export-sev is-ok" /> Exported {count(total, 'image')} and {count(objects, 'object')}</p>
      <p className="export-path" title={result.out}>{result.out}</p>
      <Counts splits={result.splits} labels={[]} />
      <div className="kv">
        <div><span>Image files</span><b>{how || 'none'}</b></div>
        <div><span>Files written</span><b>{n(result.files)}</b></div>
        {result.written.map((f) => <div key={f}><span>{f.endsWith('.yaml') ? 'Dataset file' : 'Annotations'}</span><b>{f}</b></div>)}
      </div>
      {result.accepted.length > 0 && (
        <>
          <h4>Went ahead with</h4>
          <ul className="export-accepted">
            {result.accepted.map((a) => <li key={`${a.severity}:${a.title}`}>{ICONS[a.severity]}<span className="name">{a.title}</span><b>{n(a.count)}</b></li>)}
          </ul>
        </>
      )}
      {result.notes.map((t) => <p key={t} className="export-hint">{t}</p>)}
    </div>
  )
}
