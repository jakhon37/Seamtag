import { useEffect, useRef, useState } from 'react'
import type { LegalKind, LegalText } from '../../shared/legal'
import Markdown from './Markdown'
import Modal from './Modal'
import { plainError } from './types'

const TITLES: Record<LegalKind, string> = { licence: 'Licence', notices: 'Third-party notices' }

/**
 * Seamtag's licence, or the notices of the code and components it includes, as the files shipped
 * with it have them: a Markdown file as its page would read, the others as plain text
 */
export default function Legal({ kind, onClose }: { kind: LegalKind; onClose: () => void }) {
  const [files, setFiles] = useState<LegalText[] | null>(null)
  const [error, setError] = useState('')
  const text = useRef<HTMLDivElement>(null)
  useEffect(() => {
    window.api.legal(kind).then(setFiles, (e) => setError(plainError(e)))
  }, [kind])
  // The keys scroll the text once it's there
  useEffect(() => {
    if (files) text.current?.focus({ preventScroll: true })
  }, [files])
  return (
    <Modal label={TITLES[kind]} className="legal" onClose={onClose}>
      <h3>{TITLES[kind]}</h3>
      <div className="legal-text" ref={text} tabIndex={0} role="region" aria-label={files?.map((f) => f.name).join(' and ') ?? TITLES[kind]}>
        {error ? <p className="bad" role="alert">{error}</p> : !files && <p className="muted">Reading…</p>}
        {files?.map((f) => (
          <section key={f.name} data-file={f.name}>
            <h4>{f.name}</h4>
            {!('text' in f) ? <p className="muted">{f.missing}</p> : /\.md$/i.test(f.name) ? <Markdown text={f.text} level={4} /> : <pre>{f.text}</pre>}
          </section>
        ))}
      </div>
      <div className="actions"><button className="primary" onClick={onClose}>Close</button></div>
    </Modal>
  )
}
