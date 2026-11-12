import { type ReactNode, type RefObject, useEffect, useLayoutEffect, useRef, useState } from 'react'

// Windows over the page behave as one. Focus goes into a window as it opens and stays there, Tab
// and Shift+Tab going round its controls; the page behind it, a window under it too, is inert to
// the keyboard, the mouse and screen readers; Escape does what the window says, wherever the focus
// is; and the focus goes back where it was once the window closes. The last window opened is on top.

type Open = { box: HTMLElement; escape: () => void }
const stack: Open[] = []
// What was made inert for the window on top, given back when another is on top or none is open
let shielded: Element[] = []

function shield() {
  for (const el of shielded) el.removeAttribute('inert')
  shielded = []
  const top = stack[stack.length - 1]?.box
  // Everything beside the window's backdrop, and beside each element it sits in, up to the body
  for (let el: Element | null = top?.closest('.modal-back') ?? top ?? null; el?.parentElement && el !== document.body; el = el.parentElement) {
    for (const other of Array.from(el.parentElement.children)) {
      if (other === el || other.hasAttribute('inert')) continue
      other.setAttribute('inert', '')
      shielded.push(other)
    }
  }
}

const CONTROLS = 'a[href], button, input, select, textarea, [tabindex]'
/** The window's controls that Tab stops at, in order: of a group of radio buttons, the one chosen */
function stops(box: HTMLElement) {
  return Array.from(box.querySelectorAll<HTMLElement>(CONTROLS)).filter((el) => el.tabIndex >= 0 && !el.matches(':disabled') &&
    el.getClientRects().length > 0 && !(el instanceof HTMLInputElement && el.type === 'radio' && el.name && !el.checked))
}
const follows = (a: Node, b: Node) => !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING)

function onKey(e: KeyboardEvent) {
  const top = stack[stack.length - 1]
  if (!top) return
  if (e.key === 'Escape') {
    // The window on top takes it, once for each press
    e.stopPropagation()
    if (!e.repeat && !e.isComposing) top.escape()
  } else if (e.key === 'Tab' && !e.ctrlKey && !e.altKey && !e.metaKey) {
    const list = stops(top.box)
    const at = document.activeElement
    // Past either end, or from outside the window, Tab comes round to the other end
    if (!list.length) {
      e.preventDefault()
      top.box.focus()
    } else if (!at || !top.box.contains(at) || !list.some((c) => (e.shiftKey ? follows(c, at) : follows(at, c)))) {
      e.preventDefault()
      list[e.shiftKey ? list.length - 1 : 0].focus()
    }
  }
}

/**
 * Makes `box` a modal window; it needs tabIndex={-1}, role="dialog", aria-modal="true" and a name.
 * `onEscape` runs on Escape, wherever the focus is: it closes the window, or while a job runs does
 * what the window's Close button does.
 */
export function useModal(box: RefObject<HTMLElement>, onEscape: () => void) {
  // Where the focus was before the window, or a control of it, took it
  const [opener] = useState(() => document.activeElement)
  const escape = useRef(onEscape)
  escape.current = onEscape
  useLayoutEffect(() => {
    const el = box.current!
    const me: Open = { box: el, escape: () => escape.current() }
    stack.push(me)
    if (stack.length === 1) window.addEventListener('keydown', onKey, true)
    shield()
    if (!el.contains(document.activeElement)) el.focus({ preventScroll: true })
    return () => {
      const top = stack[stack.length - 1] === me
      stack.splice(stack.indexOf(me), 1)
      if (!stack.length) window.removeEventListener('keydown', onKey, true)
      shield()
      // The focus that was in the window goes back where it was, if that's still there
      const now = document.activeElement
      if (top && (!now || now === document.body || el.contains(now)) && opener instanceof HTMLElement && opener !== document.body && opener.isConnected) {
        opener.focus({ preventScroll: true })
      }
      const under = stack[stack.length - 1]?.box
      if (top && under && !under.contains(document.activeElement)) under.focus({ preventScroll: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  // The control that had the focus can go, or be disabled, as a job starts or ends: the window takes the focus back
  useEffect(() => {
    const el = box.current
    const now = document.activeElement
    if (el && stack[stack.length - 1]?.box === el && (!el.contains(now) || now?.matches(':disabled'))) el.focus({ preventScroll: true })
  })
}

type Props = {
  /** The window's name, as screen readers say it */
  label: string
  className: string
  onClose: () => void
  /** For a window that's a form: Enter in a field submits it */
  onSubmit?: () => void
  children: ReactNode
}

/** A window closed by Escape, by a click beside it, or by a button of its own */
export default function Modal({ label, className, onClose, onSubmit, children }: Props) {
  const box = useRef<HTMLElement>(null)
  useModal(box, onClose)
  const props = { className: `dialog ${className}`, role: 'dialog', 'aria-modal': true, 'aria-label': label, tabIndex: -1, onKeyDown: (e: React.KeyboardEvent) => e.stopPropagation() }
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      {onSubmit ? (
        <form {...props} ref={box as RefObject<HTMLFormElement>} onSubmit={(e) => { e.preventDefault(); onSubmit() }}>{children}</form>
      ) : (
        <div {...props} ref={box as RefObject<HTMLDivElement>}>{children}</div>
      )}
    </div>
  )
}
