import { expect, type Locator, type Page, test } from '@playwright/test'
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { doc, finish, launch, onImage, project, saved, shapesIn, square } from './helpers'

// The app's smaller windows behave as the dataset windows do: each takes the focus as it opens and
// keeps it, Tab going round its controls, the page behind it inert, and Escape closes it
const menu = async (page: Page, top: string, item: string) => {
  await page.getByRole('menuitem', { name: top, exact: true }).click()
  await page.getByRole('menuitem', { name: item, exact: true }).click()
}

test('About, Keyboard shortcuts, AI models and Brightness and contrast hold the focus, and Escape closes each', async () => {
  const dir = await project('windows', { a: doc('a', [square('cat', 20, 20, 120, 120)]) })
  const { app, page, errors } = await launch(dir)
  const holds = () => page.evaluate(() => !!document.querySelector('[role="dialog"]')?.contains(document.activeElement))
  for (const [top, item, name] of [
    ['Help', 'About Seamtag', 'About'], ['Help', 'Keyboard shortcuts', 'Keyboard shortcuts'], ['AI', 'AI models…', 'AI models'],
    ['View', 'Brightness and contrast…', 'Brightness and contrast']
  ]) {
    await menu(page, top, item)
    const dialog = page.getByRole('dialog', { name })
    await expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(await holds()).toBe(true)
    expect(await page.evaluate(() => !!document.querySelector('.topbar')!.closest('[inert]'))).toBe(true)
    for (let k = 0; k < 5; k++) {
      await page.keyboard.press('Tab')
      expect(await holds()).toBe(true)
    }
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    expect(await page.evaluate(() => document.querySelectorAll('[inert]').length)).toBe(0)
    // The menu item that opened it has gone with the menu: the image has the focus, where the keys go
    await expect(page.getByTestId('canvas')).toBeFocused()
  }
  // The folder is as it was
  await expect(page.locator('.files .file .name')).toHaveText(['a.png'])
  await finish(app, errors)
})

/** What a Markdown file shows once read as a page: its headings, links and fenced blocks, and how many tables */
function markdownParts(md: string) {
  const plain = (t: string) => t.replace(/`([^`]+)`/g, '$1').replace(/\[([^\]]+)\]\([^)\s]+\)/g, '$1').replace(/<(https?:\/\/[^>\s]+)>/g, '$1').replace(/\*\*(.+?)\*\*/g, '$1')
  const headings: string[] = [], blocks: string[] = [], links: string[] = []
  let tables = 0, fence: string[] | null = null
  const lines = md.replace(/\r\n?/g, '\n').split('\n')
  lines.forEach((line, k) => {
    if (/^ {0,3}(`{3,}|~{3,})/.test(line)) {
      if (fence) blocks.push(fence.join('\n'))
      fence = fence ? null : []
    } else if (fence) fence.push(line)
    else {
      const h = /^ {0,3}#{1,6}\s+(.*?)\s*$/.exec(line)
      if (h) headings.push(plain(h[1]))
      if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(lines[k + 1] ?? '')) tables++
      for (const m of line.matchAll(/\]\((https:\/\/[^)\s]+)\)|<(https:\/\/[^>\s]+)>/g)) links.push(m[1] ?? m[2])
    }
  })
  return { headings, blocks, links, tables }
}

/** A notices file as the viewer shows it: Markdown as its page, the rest as the file's own text */
async function showsFile(section: Locator, file: string, text: string) {
  await expect(section.getByRole('heading', { level: 4 })).toHaveText(file)
  if (!file.endsWith('.md')) {
    expect(await section.locator('pre').textContent()).toBe(text)
    return
  }
  const md = markdownParts(text)
  await expect(section.locator('.md [role="heading"]')).toHaveText(md.headings)
  // Fenced blocks, the licences among them, keep every line as it is
  expect(await section.locator('.md pre').allTextContents()).toEqual(md.blocks)
  await expect(section.locator('.md table')).toHaveCount(md.tables)
  expect(await section.locator('.md a').evaluateAll((as) => as.map((a) => a.getAttribute('href')))).toEqual(md.links)
  // None of the Markdown's marks are left in what it reads as, outside the fenced blocks
  const prose = await section.locator('.md').evaluate((el) => {
    const copy = el.cloneNode(true) as HTMLElement
    copy.querySelectorAll('pre').forEach((pre) => pre.remove())
    return copy.textContent ?? ''
  })
  expect(prose).not.toMatch(/\]\(|\|\s*-{3}|```|^#|\s#{1,6}\s|\*\*/m)
}

test('About says whose Seamtag is and on what terms, and shows its licence and the notices of what it includes', async () => {
  const root = path.resolve(__dirname, '..')
  const { app, page, errors } = await launch(null)
  await menu(page, 'Help', 'About Seamtag')
  const about = page.getByRole('dialog', { name: 'About' })
  const { version } = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'))
  await expect(about.getByRole('heading', { level: 1 })).toHaveText(`Seamtag ${version}`)
  await expect(about).toContainText('Copyright © 2026 Jakhongir Nodirov')
  await expect(about).toContainText('Free software under the GNU GPL version 3 only (GPL-3.0-only), with ABSOLUTELY NO WARRANTY.')
  await expect(about).toContainText('Includes code from other projects under their own licences.')
  // libvips's copyright among Seamtag's, and where the GPL and the LGPL are (LGPL-3.0 section 4(c))
  await expect(about).toContainText('Uses libvips, Copyright (C) 1991-2005 The National Gallery and others, under the GNU LGPL version 3 or later (LGPL-3.0-or-later). The GNU GPL is under Licence, the LGPL under Third-party notices.')

  // Each button shows the files as the app has them, here the repository's own, over the About
  // window; Escape closes it and gives the focus back to the button
  const shows = async (button: string, files: string[]) => {
    const opener = about.getByRole('button', { name: button, exact: true })
    await opener.click()
    const dialog = page.getByRole('dialog', { name: button })
    await expect(dialog.locator('section')).toHaveCount(files.length)
    for (const file of files) {
      const section = dialog.locator(`section[data-file="${file}"]`)
      // THIRD-PARTY-NOTICES.txt is made by npm run build, which a test build may leave out
      if (existsSync(path.join(root, file))) await showsFile(section, file, readFileSync(path.join(root, file), 'utf8'))
      else await expect(section).toContainText(`${file} isn't in ${root}.`)
    }
    // The keys scroll the text
    await expect(dialog.locator('.legal-text')).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    await expect(about).toBeVisible()
    await expect(opener).toBeFocused()
  }
  await shows('Licence', ['LICENSE'])
  await shows('Third-party notices', ['NOTICE.md', 'THIRD-PARTY-NOTICES.txt'])
  await page.keyboard.press('Escape')
  await expect(about).toHaveCount(0)
  await finish(app, errors)
})

test('Third-party notices reads NOTICE.md as its page: headings, tables, lists, links and fenced licences', async () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-legal-'))
  const notice = [
    '# Notices', '', '## Seamtag', '',
    'Copyright © 2026 Someone. It is in [LICENSE](LICENSE), and the GPL is at <https://www.gnu.org/licenses/>.', '',
    '| File | From |', '|---|---|', '| `src/a.ts` | `a.py`: one thing |', '| `src/b.ts` | **two** things, in [b](https://example.org/b) |', '',
    '- one item', '- two items', '  carried on', '',
    '### A licence (MIT)', '', '```text', 'Licence line 1', '    indented line 2', '', '| not | a table |', '```', ''
  ].join('\n')
  const plain = 'Plain *text* | kept as it is\n  with its spaces\n'
  writeFileSync(path.join(dir, 'NOTICE.md'), notice)
  writeFileSync(path.join(dir, 'THIRD-PARTY-NOTICES.txt'), plain)
  writeFileSync(path.join(dir, 'LICENSE'), 'GNU GENERAL PUBLIC LICENSE\n')
  const { app, page, errors } = await launch(null, [], undefined, { SEAMTAG_LEGAL_DIR: dir })
  try {
    await app.evaluate(({ shell }) => {
      const g = globalThis as any
      g.opened = []
      shell.openExternal = (async (url: string) => void g.opened.push(url)) as any
    })
    await menu(page, 'Help', 'About Seamtag')
    await page.getByRole('dialog', { name: 'About' }).getByRole('button', { name: 'Third-party notices', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Third-party notices' })
    const md = dialog.locator('section[data-file="NOTICE.md"] .md')
    await showsFile(dialog.locator('section[data-file="NOTICE.md"]'), 'NOTICE.md', notice)
    await showsFile(dialog.locator('section[data-file="THIRD-PARTY-NOTICES.txt"]'), 'THIRD-PARTY-NOTICES.txt', plain)
    // The headings sit under the file's own, in order
    await expect(md.getByRole('heading', { level: 5 })).toHaveText(['Notices'])
    await expect(md.getByRole('heading', { level: 6 })).toHaveText(['Seamtag'])
    await expect(md.getByRole('heading', { level: 7 })).toHaveText(['A licence (MIT)'])
    await expect(md.locator('p')).toHaveText(['Copyright © 2026 Someone. It is in LICENSE, and the GPL is at https://www.gnu.org/licenses/.'])
    await expect(md.locator('th')).toHaveText(['File', 'From'])
    await expect(md.locator('td')).toHaveText(['src/a.ts', 'a.py: one thing', 'src/b.ts', 'two things, in b'])
    await expect(md.locator('td code')).toHaveText(['src/a.ts', 'a.py', 'src/b.ts'])
    await expect(md.locator('td strong')).toHaveText(['two'])
    await expect(md.locator('li')).toHaveText(['one item', 'two items carried on'])
    expect(await md.locator('pre').textContent()).toBe('Licence line 1\n    indented line 2\n\n| not | a table |')
    // A web address opens in the browser; a file beside the notices is only named
    await md.getByRole('link', { name: 'https://www.gnu.org/licenses/' }).click()
    await expect.poll(() => app.evaluate(() => (globalThis as any).opened)).toEqual(['https://www.gnu.org/licenses/'])
    await expect(md.getByRole('link', { name: 'LICENSE', exact: true })).toHaveCount(0)
    await expect(md.locator('.md-file')).toHaveText(['LICENSE'])
    await expect(dialog).toBeVisible()
    await finish(app, errors)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('the points and colour prompts take the focus, keep it, submit with Enter and close with Escape', async () => {
  const dir = await project('windows-prompts', { a: doc('a', [square('cat', 20, 20, 120, 120)]) })
  const { app, page, errors } = await launch(dir)
  const holds = () => page.evaluate(() => !!document.querySelector('[role="dialog"]')?.contains(document.activeElement))
  // Ctrl+M with the pointer on a polygon's edge asks how many points to add along it
  await page.keyboard.press('v')
  const prompt = page.getByRole('dialog', { name: 'Add points to edge' })
  const ask = async () => {
    await page.mouse.move(...await onImage(page, 70, 20))
    await page.keyboard.press('ControlOrMeta+m')
    await expect(prompt).toBeVisible()
  }
  await ask()
  await expect(prompt).toHaveAttribute('aria-modal', 'true')
  await expect(prompt.getByRole('spinbutton')).toBeFocused()
  for (let k = 0; k < 4; k++) {
    await page.keyboard.press('Tab')
    expect(await holds()).toBe(true)
  }
  await page.keyboard.press('Escape')
  await expect(prompt).toHaveCount(0)
  await ask()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.type('2')
  await page.keyboard.press('Enter')
  await expect(prompt).toHaveCount(0)
  await saved(page)
  expect(shapesIn(dir, 'a')[0].points).toHaveLength(6)

  // The colour boxes are drawn in, from LabelImg's Ctrl+L
  await page.keyboard.press('ControlOrMeta+l')
  const colour = page.getByRole('dialog', { name: 'Box line colour' })
  await expect(colour).toHaveAttribute('aria-modal', 'true')
  await expect(colour.getByLabel('Colour')).toBeFocused()
  for (let k = 0; k < 5; k++) {
    await page.keyboard.press('Tab')
    expect(await holds()).toBe(true)
  }
  await page.keyboard.press('Escape')
  await expect(colour).toHaveCount(0)
  await finish(app, errors)
})
