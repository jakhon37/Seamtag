import { describe, expect, test } from 'vitest'
import { blocks } from '../../src/renderer/src/Markdown'

// How Help > About Seamtag > Third-party notices reads NOTICE.md: as its page, the licences in
// fenced blocks line for line
describe('blocks', () => {
  test('headings, paragraphs, a table, a list and a fenced licence', () => {
    const md = [
      '# Notices', '', '## Seamtag  ', '', 'Copyright © 2026 Someone.', 'It is in [LICENSE](LICENSE).', '',
      '| File | From |', '|---|:--|', '| `a.ts` | `a.py`: one \\| two |', '| `b.ts` |', '',
      '- one', '- two', '  carried on', '1. first', '',
      '> quoted', '> on', '', '---', '',
      '```text', 'Line 1', '    indented', '', '| not | a table |', '# not a heading', '```', 'After the block.'
    ].join('\r\n')
    expect(blocks(md)).toEqual([
      { kind: 'heading', level: 1, text: 'Notices' },
      { kind: 'heading', level: 2, text: 'Seamtag' },
      { kind: 'paragraph', text: 'Copyright © 2026 Someone. It is in [LICENSE](LICENSE).' },
      { kind: 'table', head: ['File', 'From'], rows: [['`a.ts`', '`a.py`: one | two'], ['`b.ts`']] },
      { kind: 'list', ordered: false, items: ['one', 'two carried on'] },
      { kind: 'list', ordered: true, items: ['first'] },
      { kind: 'quote', text: 'quoted on' },
      { kind: 'rule' },
      { kind: 'code', text: 'Line 1\n    indented\n\n| not | a table |\n# not a heading' },
      { kind: 'paragraph', text: 'After the block.' }
    ])
  })

  test('a paragraph ends where a heading, list or table begins', () => {
    expect(blocks('Text\n## Next\nmore\n- item\nmore\n| a | b |\n|---|---|\n| 1 | 2 |')).toEqual([
      { kind: 'paragraph', text: 'Text' },
      { kind: 'heading', level: 2, text: 'Next' },
      { kind: 'paragraph', text: 'more' },
      { kind: 'list', ordered: false, items: ['item'] },
      { kind: 'paragraph', text: 'more' },
      { kind: 'table', head: ['a', 'b'], rows: [['1', '2']] }
    ])
  })

  test('a fence left open runs to the end, and a longer fence holds a shorter one', () => {
    expect(blocks('````\n```\ninside\n```\n````')).toEqual([{ kind: 'code', text: '```\ninside\n```' }])
    expect(blocks('```\nno end')).toEqual([{ kind: 'code', text: 'no end' }])
  })
})
