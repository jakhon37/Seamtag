import { ChevronDown, ChevronRight, CircleAlert, Info, TriangleAlert, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Severity } from '../../shared/dataset'
import type { CheckSummary, LabelFormat, ProblemPage, ProblemRow, Progress, RenameDone, RenamePlanView } from './types'

// The Problems dock: what Dataset > Check labels found in the folder, grouped by check, the
// worst first. The main process keeps the check (src/main/checking.ts); the dock asks it for
// the rows it shows, a page of each group at a time, so 50,000 problems never cross whole.

/** Rows of a group shown at first, and added by Show more, as the main process pages them */
const PAGE = 200
const SEVERITIES: { key: Severity; one: string; many: string; icon: typeof Info }[] = [
  { key: 'error', one: 'error', many: 'errors', icon: CircleAlert },
  { key: 'warning', one: 'warning', many: 'warnings', icon: TriangleAlert },
  { key: 'info', one: 'info', many: 'info', icon: Info }
]
const ICON = Object.fromEntries(SEVERITIES.map((s) => [s.key, s.icon])) as Record<Severity, typeof Info>

export const number = (n: number) => n.toLocaleString('en-GB')
export const counted = (n: number, one: string, many = `${one}s`) => `${number(n)} ${n === 1 ? one : many}`

/** What a pass in the main process is doing, in words */
export function progressText(p: Progress) {
  const of = `${number(p.done)} of ${number(p.total)}`
  switch (p.phase) {
    case 'read': return `Reading ${of} images`
    case 'compare': return 'Comparing image files'
    case 'orphans': return 'Looking for label files without images'
    case 'check': return `Checking ${of} images`
    case 'count': return `Counting ${of} images`
    case 'write': return `Writing ${of} label files`
  }
}

/** Progress with Stop, as pre-labelling shows it */
export function StopButton({ progress, onStop, label = 'Starting' }: { progress: Progress | null; onStop: () => void; label?: string }) {
  const share = progress && progress.total ? progress.done / progress.total : 0
  return (
    <button className="ghost batch" onClick={onStop}>
      <i style={{ width: `${share * 100}%` }} />
      <X size={15} /> Stop · {progress ? progressText(progress) : label}
    </button>
  )
}

type Props = {
  root: string
  /** The image open in the window, for This image only */
  current: string | null
  summary: CheckSummary | null
  /** A check running: a full one shows its progress with Stop, one after saves just says so */
  running: { quick: boolean; progress: Progress | null } | null
  /** Images saved since the last check: their rows are dimmed until they're checked again */
  stale: ReadonlySet<string>
  folded: boolean
  onFold: () => void
  onClose: () => void
  onCheck: () => void
  onStop: () => void
  /** Opens the row's image and selects its shape */
  onShow: (row: ProblemRow) => void
  onFix: (row: ProblemRow) => void
  /** Renames the row's label across the folder */
  onRename: (row: ProblemRow) => void
  /** Shows a label file without its image in its folder */
  onReveal: (file: string) => void
}

export default function Problems({ root, current, summary, running, stale, folded, onFold, onClose, onCheck, onStop, onShow, onFix, onRename, onReveal }: Props) {
  const [shown, setShown] = useState<Severity[]>(['error', 'warning'])
  const [only, setOnly] = useState(false)
  // Groups folded by hand, and rows asked for past the first page, by code
  const [shut, setShut] = useState<Record<string, boolean>>({})
  const [more, setMore] = useState<Record<string, number>>({})
  const [page, setPage] = useState<ProblemPage | null>(null)
  const image = only ? current ?? '' : null

  useEffect(() => {
    if (!summary) return setPage(null)
    let live = true
    const limits: Record<string, number> = {}
    for (const code of new Set([...Object.keys(shut), ...Object.keys(more)])) limits[code] = shut[code] ? 0 : more[code] ?? PAGE
    window.api.checkView(root, { severities: shown, image, limits }).then((p) => live && setPage(p), () => live && setPage(null))
    return () => { live = false }
  }, [root, summary, shown, image, shut, more])

  const toggle = (severity: Severity) => setShown((list) => (list.includes(severity) ? list.filter((s) => s !== severity) : [...list, severity]))
  const time = summary ? new Date(summary.at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : ''
  const saved = [...stale].length
  const counts = page?.counts ?? summary?.counts
  const total = counts ? counts.error + counts.warning + counts.info : 0

  return (
    <section className={`problems${folded ? ' folded' : ''}`} aria-label="Problems">
      <h3 className="problems-head">
        <button className="problems-fold" onClick={onFold} aria-expanded={!folded} title={folded ? 'Show the problems' : 'Fold the dock'}>
          {folded ? <ChevronRight size={14} /> : <ChevronDown size={14} />} Problems
          {folded && summary && <span className="problems-badge">{number(summary.counts.error + summary.counts.warning)}</span>}
        </button>
        <button className="icon" onClick={onClose} title="Close the Problems dock (View > Problems opens it)" aria-label="Close Problems"><X size={14} /></button>
      </h3>
      {!folded && (
        <>
          {summary && counts && (
            <div className="problems-bar">
              <div className="seg small problems-severities" role="group" aria-label="Show problems">
                {SEVERITIES.map(({ key, one, many, icon: Icon }) => {
                  const on = shown.includes(key)
                  const what = counts[key] === 1 ? one : many
                  return (
                    <button key={key} className={on ? 'on' : ''} aria-pressed={on} data-severity={key} onClick={() => toggle(key)}
                      aria-label={`${number(counts[key])} ${what}`} title={`${number(counts[key])} ${what}: ${on ? 'shown, click to hide them' : 'hidden, click to show them'}`}>
                      <Icon size={13} className={`sev-${key}`} /> {number(counts[key])}
                    </button>
                  )
                })}
              </div>
              <button className="ghost" onClick={onCheck} disabled={!!running}>Check again</button>
            </div>
          )}
          {running && !running.quick ? (
            <StopButton progress={running.progress} onStop={onStop} />
          ) : summary ? (
            <div className="problems-status">
              <span className={saved ? 'warn-text' : 'muted'}
                title={saved ? `${counted(saved, 'image')} saved since the check at ${time}: checked again a moment after the last save` : undefined}>
                {running ? 'Checking again…' : saved ? 'Out of date' : `Checked at ${time}`}
              </span>
              <label className="check problems-only">
                <input type="checkbox" checked={only} onChange={(e) => setOnly(e.target.checked)} /> This image only
              </label>
            </div>
          ) : (
            <div className="problems-empty">
              <p className="muted">Looks through every label file in the folder for files that can't be read, broken or doubtful shapes, and labels written several ways.</p>
              <button className="primary" onClick={onCheck}>Check labels</button>
            </div>
          )}
          {summary && counts && (
            <>
              <div className="problems-list" role="list" aria-label="Problems found">
                {page?.groups.map((g) => {
                  const Icon = ICON[g.severity]
                  const open = !shut[g.code]
                  return (
                    <div key={g.code} className="problems-group" role="listitem" data-code={g.code}>
                      <button className="problems-title" aria-expanded={open} onClick={() => setShut((s) => ({ ...s, [g.code]: open }))}>
                        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                        <Icon size={13} className={`sev-${g.severity}`} />
                        <span>{g.title}</span>
                        <b>{number(g.count)}</b>
                      </button>
                      {open && g.rows.map((row, k) => (
                        <Row key={k} row={row} dim={!!row.image && stale.has(row.image)} onShow={onShow} onFix={onFix} onRename={onRename} onReveal={onReveal} />
                      ))}
                      {open && g.rows.length < g.count && (
                        <button className="ghost problems-more" onClick={() => setMore((m) => ({ ...m, [g.code]: g.rows.length + PAGE }))}>
                          Show {number(Math.min(PAGE, g.count - g.rows.length))} more of {number(g.count - g.rows.length)}
                        </button>
                      )}
                    </div>
                  )
                })}
                {page && !page.groups.length && (
                  <p className="muted pad">
                    {!total ? (only ? 'No problems in this image.' : `No problems in ${counted(summary.images, 'image')}.`)
                      : `Nothing shown: ${SEVERITIES.filter((s) => counts[s.key] && !shown.includes(s.key)).map((s) => s.many).join(' and ')} are hidden.`}
                  </p>
                )}
              </div>
            </>
          )}
        </>
      )}
    </section>
  )
}

/** A label a rename can start from: one written another way, or one not in the label list */
const RENAMES = new Set(['label-variants', 'unknown-label'])

function Row({ row, dim, onShow, onFix, onRename, onReveal }: { row: ProblemRow; dim: boolean } & Pick<Props, 'onShow' | 'onFix' | 'onRename' | 'onReveal'>) {
  const opens = !!row.image
  const act = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation()
    fn()
  }
  // The whole row opens its image to the mouse; its name is the button to the keyboard
  return (
    <div className={`problem${opens ? ' opens' : ''}${dim ? ' stale' : ''}`}
      title={dim ? `${row.message}\n\nSaved since the check: this is checked again in a moment.` : row.message} onClick={() => opens && onShow(row)}>
      <div className="problem-where">
        {opens ? <button className="name problem-open" onClick={act(() => onShow(row))}>{row.image}</button>
          : <span className="name">{row.file ?? 'The folder'}</span>}
        {row.shape !== undefined && <span className="muted problem-shape">shape {row.shape + 1}{row.label?.trim() ? ` · ${row.label}` : ''}</span>}
        {row.fix && <button className="problem-act" onClick={act(() => onFix(row))} title="Fix it in the image: an edit that can be undone">Fix</button>}
        {RENAMES.has(row.code) && row.label !== undefined && (
          <button className="problem-act" onClick={act(() => onRename(row))} title={`Rename “${row.label}” in every label file of the folder`}>Rename…</button>
        )}
        {row.file && <button className="problem-act" onClick={act(() => onReveal(row.file!))} title="Show the file in its folder">Show</button>}
      </div>
      <div className="problem-what">{row.message}</div>
    </div>
  )
}

type RenameProps = {
  root: string
  from: string
  /** The name to start with: the spelling a variant looks like, or nothing */
  to: string
  /** The window's save format: an image with label files in two formats has this one renamed */
  prefer: LabelFormat
  /** Names to suggest: the label list */
  labels: string[]
  /** Accepted shapes of each label across the folder, to say what a rename into a label in use joins */
  totals: Record<string, number>
  /** With validate_label: exact, the only names allowed */
  allowed: string[] | null
  /** Saves the open image's changes, or asks, before anything is written */
  canLeave: () => Promise<boolean>
  onClose: (done?: RenameDone & { to: string }) => void
}

/**
 * Renames a label in every label file of the folder: what it would change is shown first,
 * read from the files as they are; then each file is rewritten in its own format, behind a
 * progress with Stop, its old copy in the trash.
 */
export function RenameDialog({ root, from, to: start, prefer, labels, totals, allowed, canLeave, onClose }: RenameProps) {
  const [to, setTo] = useState(start)
  const [plan, setPlan] = useState<RenamePlanView | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => window.api.onRename((p) => p.root === root && setProgress(p)), [root])
  useEffect(() => {
    let live = true
    window.api.renamePlan(root, from, prefer).then((p) => {
      if (!live) return
      if ('stopped' in p) return onClose()
      setPlan(p)
      setProgress(null)
    }, (e) => live && setError(String(e?.message ?? e)))
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, from, prefer])

  const name = to.trim() ? to : ''
  const refusal = !name ? '' : name === from ? `It's called “${from}” already.` : allowed && !allowed.includes(name)
    ? `“${name}” isn't in the label list, and only listed labels are allowed.` : ''
  const ready = !!plan && plan.images > 0 && !!name && !refusal && !running
  const run = async () => {
    if (!ready || !(await canLeave())) return
    setRunning(true)
    setError('')
    setProgress(null)
    try {
      const done = await window.api.renameRun(root, from, name, prefer)
      onClose({ ...done, to: name })
    } catch (e) {
      setRunning(false)
      setError(String((e as Error)?.message ?? e).replace(/^Error invoking remote method '[^']+': (\w*Error: )?/, ''))
    }
  }
  const close = () => {
    if (running) return
    window.api.renameStop(root)
    onClose()
  }
  const joins = name && name !== from ? totals[name] ?? 0 : 0

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <form className="dialog rename" role="dialog" aria-label="Rename a label" onSubmit={(e) => { e.preventDefault(); run() }}
        onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') close() }}>
        <h3>Rename a label</h3>
        <p className="muted">Every shape labelled <b className="rename-label">“{from}”</b> in the folder's label files gets the new name, AI proposals too.</p>
        <label>New name
          <input value={to} list="rename-labels" autoFocus disabled={running} aria-label="New name" onChange={(e) => setTo(e.target.value)} />
          <datalist id="rename-labels">
            {labels.filter((l) => l !== from).map((l) => <option key={l} value={l} />)}
          </datalist>
        </label>
        {refusal && <p className="bad" role="alert">{refusal}</p>}
        <div className="rename-plan" aria-live="polite">
          {!plan && !error && <p className="muted">{progress ? `${progressText(progress)}…` : 'Reading the label files…'}</p>}
          {plan && plan.images === 0 && <p>No label file has “{from}” now, so there's nothing to rename.</p>}
          {plan && plan.images > 0 && (
            <>
              <p>
                Renames <b>{counted(plan.shapes, 'shape')}</b> in <b>{counted(plan.images, 'image')}</b>: {plan.names.join(', ')}
                {plan.images > plan.names.length ? ` and ${number(plan.images - plan.names.length)} more` : ''}.
              </p>
              {joins > 0 && <p className="muted">“{name}” has {counted(joins, 'shape')} already: these join them.</p>}
              <p className="muted">Each changed label file is rewritten in its own format, and its old copy goes to the trash.</p>
            </>
          )}
          {plan && plan.skippedCount > 0 && (
            <>
              <p>{counted(plan.skippedCount, 'label file is', 'label files are')} left alone:</p>
              <ul className="rename-skipped">
                {plan.skipped.map((s) => <li key={s.name} className="muted">{s.reason}</li>)}
                {plan.skippedCount > plan.skipped.length && <li className="muted">and {number(plan.skippedCount - plan.skipped.length)} more</li>}
              </ul>
            </>
          )}
        </div>
        {error && <p className="bad" role="alert">{error}</p>}
        <div className="actions">
          {running ? (
            <StopButton progress={progress} onStop={() => window.api.renameStop(root)} label="Renaming" />
          ) : (
            <>
              <button type="button" className="ghost" onClick={close}>Cancel</button>
              <button className="primary" disabled={!ready}>{plan && plan.shapes ? `Rename ${counted(plan.shapes, 'shape')}` : 'Rename'}</button>
            </>
          )}
        </div>
      </form>
    </div>
  )
}
