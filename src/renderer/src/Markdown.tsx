import type { ReactNode } from 'react'

// The little Markdown the notices are written in, read as the page they'd be on GitHub: headings,
// paragraphs, lists, tables, quotes and fenced blocks, and in a line, code, links and bold. The
// text only ever becomes React's own elements, never HTML, and a fenced block keeps its lines as
// they are, so a licence in one reads word for word.

type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph' | 'quote'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }
  | { kind: 'table'; head: string[]; rows: string[][] }
  | { kind: 'rule' }

const FENCE = /^ {0,3}(`{3,}|~{3,})/
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/
const ITEM = /^\s*([-*+]|\d+[.)])\s+(.*)$/
const RULE = /^ {0,3}([-*_])(\s*\1){2,}\s*$/
const ROW = /^\s*\|.*\|\s*$/
const DIVIDER = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/
const QUOTE = /^ {0,3}>\s?/
/** A table row's cells, the outer pipes dropped and \| kept as a pipe */
const cells = (row: string) => row.trim().replace(/^\|/, '').replace(/(?<!\\)\|$/, '').split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'))

export function blocks(md: string): Block[] {
  const lines = md.replace(/\r\n?/g, '\n').split('\n')
  const out: Block[] = []
  const starts = (l: string) => FENCE.test(l) || HEADING.test(l) || ITEM.test(l) || RULE.test(l) || QUOTE.test(l)
  for (let i = 0; i < lines.length;) {
    const line = lines[i]
    const fence = FENCE.exec(line)
    if (!line.trim()) i++
    else if (fence) {
      const body: string[] = []
      for (i++; i < lines.length && !lines[i].trimStart().startsWith(fence[1]); i++) body.push(lines[i])
      out.push({ kind: 'code', text: body.join('\n') })
      i++
    } else if (HEADING.test(line)) {
      const [, marks, text] = HEADING.exec(line)!
      out.push({ kind: 'heading', level: marks.length, text })
      i++
    } else if (RULE.test(line)) {
      out.push({ kind: 'rule' })
      i++
    } else if (ROW.test(line) && DIVIDER.test(lines[i + 1] ?? '')) {
      const head = cells(line), rows: string[][] = []
      for (i += 2; i < lines.length && ROW.test(lines[i]); i++) rows.push(cells(lines[i]))
      out.push({ kind: 'table', head, rows })
    } else if (ITEM.test(line)) {
      const ordered = /\d/.test(ITEM.exec(line)![1]), items: string[] = []
      for (; i < lines.length; i++) {
        const item = ITEM.exec(lines[i])
        // A line indented under an item goes on with it; a numbered item after bullets starts a list of its own
        if (item && /\d/.test(item[1]) === ordered) items.push(item[2])
        else if (!item && /^\s+\S/.test(lines[i])) items[items.length - 1] += ` ${lines[i].trim()}`
        else break
      }
      out.push({ kind: 'list', ordered, items })
    } else if (QUOTE.test(line)) {
      const body: string[] = []
      for (; i < lines.length && QUOTE.test(lines[i]); i++) body.push(lines[i].replace(QUOTE, '').trim())
      out.push({ kind: 'quote', text: body.join(' ') })
    } else {
      const body = [line.trim()]
      for (i++; i < lines.length && lines[i].trim() && !starts(lines[i]) && !(ROW.test(lines[i]) && DIVIDER.test(lines[i + 1] ?? '')); i++) body.push(lines[i].trim())
      out.push({ kind: 'paragraph', text: body.join(' ') })
    }
  }
  return out
}

/** A link opens in the browser when it's a web address; one to a file beside the notices is its name */
function link(url: string, children: ReactNode, key: number) {
  if (!/^https:\/\//.test(url)) return <span key={key} className="md-file">{children}</span>
  return <a key={key} href={url} onClick={(e) => { e.preventDefault(); window.api.openLink(url) }}>{children}</a>
}

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = []
  const parts = /`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)|<(https?:\/\/[^>\s]+)>|\*\*(.+?)\*\*/g
  let at = 0
  for (const m of text.matchAll(parts)) {
    const index = m.index ?? 0
    if (index > at) out.push(text.slice(at, index))
    const key = out.length
    if (m[1] !== undefined) out.push(<code key={key}>{m[1]}</code>)
    else if (m[2] !== undefined) out.push(link(m[3], inline(m[2]), key))
    else if (m[4] !== undefined) out.push(link(m[4], m[4], key))
    else out.push(<strong key={key}>{inline(m[5])}</strong>)
    at = index + m[0].length
  }
  if (at < text.length) out.push(text.slice(at))
  return out
}

/** Markdown as the page it stands for; its headings sit under the heading the page is shown with, at `level` */
export default function Markdown({ text, level }: { text: string; level: number }) {
  return (
    <div className="md">
      {blocks(text).map((b, k) => {
        switch (b.kind) {
          case 'heading': return <div key={k} role="heading" aria-level={level + b.level} className={`md-h md-h${Math.min(b.level, 3)}`}>{inline(b.text)}</div>
          case 'paragraph': return <p key={k}>{inline(b.text)}</p>
          case 'quote': return <blockquote key={k}>{inline(b.text)}</blockquote>
          case 'code': return <pre key={k}>{b.text}</pre>
          case 'rule': return <hr key={k} />
          case 'list': {
            const List = b.ordered ? 'ol' : 'ul'
            return <List key={k}>{b.items.map((item, n) => <li key={n}>{inline(item)}</li>)}</List>
          }
          case 'table': return (
            <table key={k}>
              <thead><tr>{b.head.map((c, n) => <th key={n}>{inline(c)}</th>)}</tr></thead>
              <tbody>{b.rows.map((row, r) => <tr key={r}>{b.head.map((_, n) => <td key={n}>{inline(row[n] ?? '')}</td>)}</tr>)}</tbody>
            </table>
          )
        }
      })}
    </div>
  )
}
