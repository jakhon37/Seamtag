import { ChevronDown, ChevronRight, CircleAlert, Clock, Info, RefreshCw, TriangleAlert, X } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Severity } from '../../shared/dataset'
import { useModal } from './Modal'
import type { CheckSummary, LabelFormat, ProblemPage, ProblemRow, Progress, RenameDone, RenamePlanView } from './types'

// The Problems dock: what Dataset > Check labels found in the folder, grouped by check, the
// worst first. The main process keeps the check (src/main/checking.ts); the dock asks it for
// the rows it shows, a page of each group at a time, so 50,000 problems never cross whole.

/** Rows of a group shown at first, and added by Show more, as the main process pages them */
const PAGE = 200
// Information, in words: notes
const SEVERITIES: { key: Severity; one: string; many: string; icon: typeof Info }[] = [
  { key: 'error', one: 'error', many: 'errors', icon: CircleAlert },
  { key: 'warning', one: 'warning', many: 'warnings', icon: TriangleAlert },
  { key: 'info', one: 'note', many: 'notes', icon: Info }
]
const ICON = Object.fromEntries(SEVERITIES.map((s) => [s.key, s.icon])) as Record<Severity, typeof Info>

export const number = (n: number) => n.toLocaleString('en-GB')
export const counted = (n: number, one: string, many = `${one}s`) => `${number(n)} ${n === 1 ? one : many}`
/** a, b and c */
const listed = (items: string[]) => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`)
/** Show more's words, for `left` rows not shown and pages of `page`: the rest when one page holds them */
export const showMore = (left: number, page: number) =>
  left > page ? `Show ${number(page)} more (${number(left)} not shown)` : left === 1 ? 'Show 1 more' : `Show the other ${number(left)}`

/** What a pass in the main process is doing, in words */
export function progressText(p: Progress) {
  const of = `${number(p.done)} of ${number(p.total)}`
  switch (p.phase) {
    case 'read': return `Reading ${of} images`
    // Only images that share a byte size with another are read and compared
    case 'compare': return p.total ? `Comparing ${of} image files` : 'Comparing image files'
    case 'orphans': return p.total ? `Looking at ${of} label files for ones without images` : 'Looking for label files without images'
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
  /** Images saved since the last check: their rows are marked out of date until they're checked again */
  stale: { has: (name: string) => boolean; readonly size: number }
  /** The open image, while it has changes that aren't saved: its rows are marked out of date until they're saved and checked */
  unsaved: string | null
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

/** A row, the same from one page to the next */
const keyOf = (row: ProblemRow) => JSON.stringify([row.code, row.image, row.shape ?? null, row.message])

/** Where the focus is in the list: its control, the group and row it's in (a group's Show more counts as the row after its last), and which of the row's controls */
type Held = { el: HTMLElement; group: number; code: string; row: number; part: string }

/**
 * The control that takes the focus once the one holding it has left the list, as a fix or a
 * rename checked again takes its row away: the same control of the row now in its place, or
 * that row's first; past a group's last row, its Show more, else the next group's title, else
 * the row before; a group gone, the group now in its place, else the last; with none, the list,
 * which takes the focus only then, to say there's nothing left.
 */
function handOn(list: HTMLElement, at: Held): HTMLElement {
  const groups = [...list.querySelectorAll<HTMLElement>(':scope > .problems-group')]
  const first = (row: HTMLElement) => (row.tabIndex >= 0 ? row : row.querySelector<HTMLElement>('button') ?? row)
  const group = groups.find((g) => g.dataset.code === at.code)
  if (group && at.row >= 0) {
    const rows = [...group.querySelectorAll<HTMLElement>(':scope > .problem')]
    const row = rows[at.row]
    if (row) return row.querySelector<HTMLElement>(`[data-part="${at.part}"]`) ?? first(row)
    const after = group.querySelector<HTMLElement>(':scope > .problems-more') ?? groups[groups.indexOf(group) + 1]?.querySelector<HTMLElement>('.problems-title')
    return after ?? (rows.length ? first(rows[rows.length - 1]) : group.querySelector<HTMLElement>('.problems-title')!)
  }
  const instead = group ?? groups[Math.min(at.group, groups.length - 1)]
  return instead?.querySelector<HTMLElement>('.problems-title') ?? list
}

export default function Problems({
  root, current, summary, running, stale, unsaved, folded, onFold, onClose, onCheck, onStop, onShow, onFix, onRename, onReveal
}: Props) {
  const [shown, setShown] = useState<Severity[]>(['error', 'warning'])
  const [only, setOnly] = useState(false)
  // Groups folded by hand, and rows asked for past the first page, by code
  const [shut, setShut] = useState<Record<string, boolean>>({})
  const [more, setMore] = useState<Record<string, number>>({})
  const [page, setPage] = useState<ProblemPage | null>(null)
  // The row clicked last, whose message shows in full
  const [picked, setPicked] = useState('')
  const image = only ? current ?? '' : null

  useEffect(() => {
    if (!summary) return setPage(null)
    let live = true
    const limits: Record<string, number> = {}
    for (const code of new Set([...Object.keys(shut), ...Object.keys(more)])) limits[code] = shut[code] ? 0 : more[code] ?? PAGE
    window.api.checkView(root, { severities: shown, image, limits }).then((p) => live && setPage(p), () => live && setPage(null))
    return () => { live = false }
  }, [root, summary, shown, image, shut, more])

  // A keyboard user who fixed or renamed from a row keeps their place: once the check takes the row
  // away, the focus goes to what's now there, rather than back to the top of the window. The focus is
  // followed by the page's own events, as React's miss those of a window giving it back as it closes.
  const dock = useRef<HTMLElement>(null)
  const listBox = useRef<HTMLDivElement>(null)
  const held = useRef<Held | null>(null)
  useEffect(() => {
    const box = dock.current!
    const hold = (e: FocusEvent) => {
      const el = e.target as HTMLElement
      const group = listBox.current?.contains(el) ? el.closest<HTMLElement>('.problems-group') : null
      if (!group || !listBox.current) return void (held.current = null)
      const rows = [...group.querySelectorAll(':scope > .problem')]
      const row = el.closest('.problem')
      held.current = {
        el, code: group.dataset.code ?? '', group: [...listBox.current.querySelectorAll(':scope > .problems-group')].indexOf(group),
        row: row ? rows.indexOf(row) : el.matches('.problems-more') ? rows.length : -1, part: el.dataset.part ?? ''
      }
    }
    const letGo = (e: FocusEvent) => {
      const el = e.target as HTMLElement
      if (e.relatedTarget) {
        // Moved elsewhere by the user: it stays there
        if (!listBox.current?.contains(e.relatedTarget as Node)) held.current = null
        return
      }
      // A click on nothing that takes the focus lets it go too; a row taken away, or the window losing the focus, doesn't
      requestAnimationFrame(() => {
        if (held.current?.el === el && el.isConnected && document.activeElement !== el) held.current = null
      })
    }
    box.addEventListener('focusin', hold)
    box.addEventListener('focusout', letGo)
    return () => {
      box.removeEventListener('focusin', hold)
      box.removeEventListener('focusout', letGo)
    }
  }, [])
  useLayoutEffect(() => {
    const at = held.current
    if (!at || at.el.isConnected || !listBox.current) return
    held.current = null
    const now = document.activeElement
    if (!now || now === document.body) handOn(listBox.current, at).focus()
  }, [page])

  const toggle = (severity: Severity) => setShown((list) => (list.includes(severity) ? list.filter((s) => s !== severity) : [...list, severity]))
  const time = summary ? new Date(summary.at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : ''
  const counts = page?.counts ?? summary?.counts
  const total = counts ? counts.error + counts.warning + counts.info : 0
  const late = stale.size > 0 || !!unsaved
  const lateWhy = [
    ...(stale.size ? [`${counted(stale.size, 'image')} saved since the check at ${time}: checked again a moment after the last save.`] : []),
    ...(unsaved ? [`${unsaved} has changes that aren't saved: they're checked once they're saved.`] : [])
  ].join('\n')
  const hidden = counts ? SEVERITIES.filter((s) => counts[s.key] && !shown.includes(s.key)) : []
  const hiddenCount = counts ? hidden.reduce((n, s) => n + counts[s.key], 0) : 0
  const lateOf = (row: ProblemRow) => (!row.image ? null : stale.has(row.image) ? 'saved' : row.image === unsaved ? 'unsaved' : null)
  // Nothing found in the whole folder: the dock is no taller than its words
  const clean = !!summary && !(summary.counts.error + summary.counts.warning + summary.counts.info)

  return (
    <section className={`problems${folded ? ' folded' : ''}${clean ? ' clean' : ''}`} aria-label="Problems" ref={dock}>
      <h3 className="problems-head">
        <button className="problems-fold" onClick={onFold} aria-expanded={!folded} title={folded ? 'Show the problems' : 'Fold the problems'}>
          {folded ? <ChevronRight size={14} /> : <ChevronDown size={14} />} Problems
          {folded && summary && <span className="problems-badge">{number(summary.counts.error + summary.counts.warning)}</span>}
        </button>
        {!folded && summary && (
          <>
            <span className={`problems-status ${late && !running ? 'warn-text' : 'muted'}`}
              title={running ? undefined : late ? lateWhy : `Checked at ${time}: ${counted(summary.images, 'image')}`}>
              {running ? 'Checking…' : late ? 'Out of date' : `Checked ${time}`}
            </span>
            <button className="icon" onClick={onCheck} disabled={!!running} aria-label="Check again" title="Check every label file in the folder again">
              <RefreshCw size={13} />
            </button>
          </>
        )}
        <button className="icon" onClick={onClose} title="Close Problems (View > Problems opens it)" aria-label="Close Problems"><X size={14} /></button>
      </h3>
      {!folded && (
        <>
          {summary && counts && (
            <div className="problems-bar">
              <div className="seg small problems-severities" role="group" aria-label="Show problems">
                {SEVERITIES.map(({ key, one, many, icon: Icon }) => {
                  const on = shown.includes(key)
                  const them = counts[key] === 1 ? 'it' : 'them'
                  return (
                    <button key={key} className={on ? 'on' : ''} aria-pressed={on} data-severity={key} onClick={() => toggle(key)}
                      aria-label={counted(counts[key], one, many)} title={`${counted(counts[key], one, many)}: ${on ? `shown, click to hide ${them}` : `hidden, click to show ${them}`}`}>
                      <Icon size={13} className={`sev-${key}`} /> {number(counts[key])}
                    </button>
                  )
                })}
              </div>
              <label className="check problems-only" title="Only the problems of the image that's open">
                <input type="checkbox" checked={only} onChange={(e) => setOnly(e.target.checked)} /> This image only
              </label>
            </div>
          )}
          {running && !running.quick && <StopButton progress={running.progress} onStop={onStop} />}
          {!summary && !running && (
            <div className="problems-empty">
              <p className="muted">Looks through every label file in the folder for files that can't be read, broken or doubtful shapes, and labels written several ways.</p>
              <button className="primary" onClick={onCheck}>Check labels</button>
            </div>
          )}
          {summary && counts && (
            <div className="problems-list" role="list" aria-label="Problems found" ref={listBox} tabIndex={page && !page.groups.length ? -1 : undefined}>
              {page?.groups.map((g) => {
                const Icon = ICON[g.severity]
                const open = !shut[g.code]
                // Each row keeps its own element from one page to the next, and with it the focus; two alike go by their order
                const seen = new Map<string, number>()
                return (
                  <div key={g.code} className="problems-group" role="listitem" data-code={g.code}>
                    <button className="problems-title" aria-expanded={open} onClick={() => setShut((s) => ({ ...s, [g.code]: open }))}>
                      {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      <Icon size={13} className={`sev-${g.severity}`} />
                      <span>{g.title}</span>
                      <b>{number(g.count)}</b>
                    </button>
                    {open && g.rows.map((row) => {
                      const key = keyOf(row), n = seen.get(key) ?? 0
                      seen.set(key, n + 1)
                      return (
                        <Row key={n ? `${key}#${n}` : key} row={row} late={lateOf(row)} picked={picked === key} onPick={() => setPicked(key)}
                          onShow={onShow} onFix={onFix} onRename={onRename} onReveal={onReveal} />
                      )
                    })}
                    {open && g.rows.length < g.count && (
                      <button className="ghost problems-more" onClick={() => setMore((m) => ({ ...m, [g.code]: g.rows.length + PAGE }))}>
                        {showMore(g.count - g.rows.length, PAGE)}
                      </button>
                    )}
                  </div>
                )
              })}
              {page && !page.groups.length && (
                <p className="muted pad">
                  {!total ? (only ? 'No problems in this image.' : `No problems in ${counted(summary.images, 'image')}.`)
                    : `Nothing shown: ${listed(hidden.map((s) => counted(counts[s.key], s.one, s.many)))} ${hiddenCount === 1 ? 'is' : 'are'} hidden.`}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </section>
  )
}

/** A label a rename can start from: one written another way, or one not in the label list */
const RENAMES = new Set(['label-variants', 'unknown-label'])
const LATE = { saved: 'Saved since the check: this is checked again in a moment.', unsaved: "Changed since the check: this is checked again once it's saved." }
// Characters a name cut short keeps at its end, so names that differ only there, as frame numbers do, still tell apart:
// at least TAIL, and up to LONGEST to keep its last word whole, as frame_000025.jpg's 000025.jpg
const TAIL = 8
const LONGEST = 16

/** How many characters of a name its cut keeps at the end: from the start of its last word, when that's short enough */
function tailLength(chars: string[]) {
  for (let k = chars.length - TAIL; k >= Math.max(1, chars.length - LONGEST); k--) if (/[\s_-]/.test(chars[k - 1])) return chars.length - k
  return TAIL
}

/**
 * A name cut short in the middle when the row is too narrow for it, its end always shown. A
 * screen reader takes its two parts for two words, so a button holding it is named whole.
 */
function MiddleCut({ text }: { text: string }) {
  const chars = [...text]
  const tail = tailLength(chars)
  if (chars.length <= tail + 2) return <span className="cut-head">{text}</span>
  return <><span className="cut-head">{chars.slice(0, -tail).join('')}</span><span className="cut-tail">{chars.slice(-tail).join('')}</span></>
}

type RowProps = {
  row: ProblemRow
  /** Out of date: saved since the check, or changed in the window and not saved */
  late: keyof typeof LATE | null
  /** Clicked last: its message shows in full */
  picked: boolean
  onPick: () => void
} & Pick<Props, 'onShow' | 'onFix' | 'onRename' | 'onReveal'>

function Row({ row, late, picked, onPick, onShow, onFix, onRename, onReveal }: RowProps) {
  const opens = !!row.image
  const act = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation()
    fn()
  }
  // The whole row opens its image to the mouse; its name is the button to the keyboard. A row
  // with no button takes the focus itself, so its message can be read in full from the keyboard.
  // The message says which shape, or which label, so the image's name has the line to itself but for the buttons.
  return (
    <div className={`problem${opens ? ' opens' : ''}${late ? ' stale' : ''}${picked ? ' picked' : ''}`} tabIndex={opens || row.file ? undefined : 0}
      title={late ? `${row.message}\n\n${LATE[late]}` : row.message} onClick={() => {
        onPick()
        if (opens) onShow(row)
      }}>
      {late && <Clock size={12} className="problem-late" aria-label="Out of date" />}
      <div className="problem-where">
        {opens ? (
          <button className="name problem-open" data-part="name" title={row.image!} aria-label={row.image!} onClick={act(() => { onPick(); onShow(row) })}>
            <MiddleCut text={row.image!} />
          </button>
        ) : <span className="name" title={row.file}><MiddleCut text={row.file ?? 'The folder'} /></span>}
        {row.fix && <button className="problem-act" data-part="fix" onClick={act(() => onFix(row))} title="Fix it in the image: an edit that can be undone">Fix</button>}
        {RENAMES.has(row.code) && row.label !== undefined && (
          <button className="problem-act" data-part="rename" onClick={act(() => onRename(row))} title={`Rename “${row.label}” in every label file of the folder`}>Rename…</button>
        )}
        {row.file && <button className="problem-act" data-part="show" onClick={act(() => onReveal(row.file!))} title="Show the file in its folder">Show</button>}
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

  // A label is its name without spaces at either end, as the label dialog takes it
  const name = to.trim()
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
  // While the rename runs its field is disabled, and the window keeps the focus that falls from it
  const box = useRef<HTMLFormElement>(null)
  useModal(box, close)
  const joins = name && name !== from ? totals[name] ?? 0 : 0

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <form className="dialog rename" role="dialog" aria-modal="true" aria-label="Rename a label" ref={box} tabIndex={-1}
        onSubmit={(e) => { e.preventDefault(); run() }} onKeyDown={(e) => e.stopPropagation()}>
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
