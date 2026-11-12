import type { KeyboardEvent, ReactNode } from 'react'

// A row of buttons to choose one of works as a group of radio inputs does, as the Export window's
// formats are: it's one Tab stop, the chosen button, and the arrow keys move the choice round the
// row. While none is chosen, as while the Import window asks for the task, the Tab stop is the one
// to start from, and the arrow keys only move the focus, so the first key on the way doesn't answer
// for the user: Space or Enter chooses
type Choice<T> = { id: T; name: ReactNode; title?: string; className?: string }
const STEPS: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }

type Props<T> = {
  /** The group's name, as screen readers say it */
  label: string
  className: string
  /** The chosen one, or null while none is */
  value: T | null
  /** The Tab stop while none is chosen: the first, unless given */
  start?: T
  choices: Choice<T>[]
  onChoose: (id: T) => void
}

export default function Radios<T extends string>({ label, className, value, start, choices, onChoose }: Props<T>) {
  const none = !choices.some((c) => c.id === value)
  const first = choices.some((c) => c.id === start) ? start : choices[0]?.id
  const keys = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = STEPS[e.key]
    if (!step || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    const buttons = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)')]
    const at = buttons.indexOf(e.target as HTMLButtonElement)
    if (at < 0) return
    // The key moves the choice, not the column it's in
    e.preventDefault()
    const next = buttons[(at + step + buttons.length) % buttons.length]
    next.focus()
    if (!none) next.click()
  }
  return (
    <div className={className} role="radiogroup" aria-label={label} onKeyDown={keys}>
      {choices.map((c) => {
        const on = c.id === value
        return (
          <button key={c.id} type="button" role="radio" aria-checked={on} tabIndex={on || (none && c.id === first) ? 0 : -1} className={on ? 'on' : c.className}
            title={c.title} onClick={() => onChoose(c.id)}>
            {c.name}
          </button>
        )
      })}
    </div>
  )
}
