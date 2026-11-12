import { type ElectronApplication, expect, type Locator, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { BASE, doc, finish, H, launch, open, project, read, saved, shapesIn, square, W } from './helpers'

// Dataset > Check labels, the Problems dock, its fixes and renames, and Dataset > Statistics,
// on a folder with one of each planted problem
const SHOTS = process.env.E2E_SHOTS || path.join(BASE, 'dataset-shots')
const rect = (label: string, x0: number, y0: number, x1: number, y1: number) => ({
  label, shape_type: 'rectangle', group_id: null, description: '', points: [[x0, y0], [x1, y1]], flags: {}
})

async function planted(folder: string, classes?: { name: string; color: string }[]) {
  const dir = await project(folder, {
    // Two copies of one shape
    a: doc('a', [square('cat', 20, 20, 120, 120), square('cat', 20, 20, 120, 120)]),
    // A point repeated
    b: doc('b', [{ ...square('cat', 50, 50, 150, 150), points: [[50, 50], [150, 50], [150, 50], [150, 150], [50, 150]] }]),
    // A box 30 px past the right and bottom edges
    c: doc('c', [rect('dog', 300, 200, 430, 330)]),
    // Variants of a label, and a label not in the list
    d: doc('d', [square('Cat', 60, 60, 160, 160)]),
    e: doc('e', [square('zebra', 60, 60, 160, 160)]),
    f: '{"shapes": [',
    // An AI proposal not yet accepted
    g: doc('g', [square('dog', 100, 100, 200, 200, 0.4)]),
    h: null
  }, classes)
  await paint(dir, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
  // A label file whose image is gone
  writeFileSync(path.join(dir, 'gone.json'), doc('gone', [square('cat', 1, 1, 9, 9)]))
  return dir
}

/** Each image a colour of its own: the helper's are all one grey, which the check reports as copies of one image */
async function paint(dir: string, names: string[]) {
  for (const [k, name] of names.entries()) {
    await sharp({ create: { width: W, height: H, channels: 3, background: { r: 40 + k * 20, g: 80, b: 100 } } }).png().toFile(path.join(dir, `${name}.png`))
  }
}

/** A trash folder of the test's own, as e2e/labelimg.spec.ts makes it */
async function fakeTrash(app: ElectronApplication, folder: string) {
  rmSync(folder, { recursive: true, force: true })
  mkdirSync(folder, { recursive: true })
  await app.evaluate(({ shell }, into) => {
    const fs = process.mainModule!.require('fs')
    const path = process.mainModule!.require('path')
    const trashed: string[] = (globalThis as any).trashed = []
    shell.trashItem = (async (file: string) => {
      fs.renameSync(file, path.join(into, `${trashed.length}-${path.basename(file)}`))
      trashed.push(file)
    }) as any
  }, folder)
  return () => app.evaluate(() => (globalThis as any).trashed as string[])
}
const menu = async (page: Page, top: string, item: string) => {
  await page.getByRole('menuitem', { name: top, exact: true }).click()
  await page.getByRole('menuitem', { name: item, exact: true }).click()
}
const darkMode = async (page: Page) => {
  await page.getByRole('menuitem', { name: 'View', exact: true }).click()
  await page.getByRole('menuitemcheckbox', { name: 'Dark mode' }).click()
}
// The window, and the page's viewport, which the helpers set to 1500 x 940
const resize = async (app: ElectronApplication, page: Page, width: number, height: number) => {
  await app.evaluate(({ BrowserWindow }, [w, h]) => BrowserWindow.getAllWindows()[0].setSize(w, h), [width, height])
  await page.setViewportSize({ width, height }).catch(() => {})
  await page.waitForTimeout(400)
  // A window with a frame has a little less inside it than its size
  const [w, h] = await page.evaluate(() => [innerWidth, innerHeight])
  expect(w <= width && w >= width - 20 && h <= height && h >= height - 60).toBe(true)
}
/** Nothing spills sideways out of the page or the given part of it */
const fits = (page: Page, selector: string) => page.evaluate((sel) => {
  const box = document.querySelector(sel)!.getBoundingClientRect()
  const inside = [...document.querySelectorAll(`${sel} *`)].every((el) => {
    const r = el.getBoundingClientRect()
    return !r.width || (r.left >= box.left - 1 && r.right <= box.right + 1)
  })
  return { inside, page: document.documentElement.scrollWidth <= innerWidth, onScreen: box.bottom <= innerHeight + 1 && box.right <= innerWidth + 1 }
}, selector)
const shot = (page: Page, name: string) => page.screenshot({ path: path.join(SHOTS, name) })
/**
 * The contrast of an element's words with what lies behind them, as WCAG works it out: its colour,
 * faded by its own and its ancestors' opacity, over the nearest background that isn't transparent
 */
const contrast = (target: Locator) => target.evaluate((el) => {
  const rgba = (c: string) => (c.match(/[\d.]+/g) ?? []).map(Number)
  const lum = (rgb: number[]) => {
    const [r, g, b] = rgb.map((v) => (v / 255 <= 0.04045 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  }
  let opacity = 1, back = [255, 255, 255]
  for (let e: Element | null = el; e; e = e.parentElement) opacity *= Number(getComputedStyle(e).opacity)
  for (let e: Element | null = el; e; e = e.parentElement) {
    const [r, g, b, a = 1] = rgba(getComputedStyle(e).backgroundColor)
    if (a > 0) {
      back = [r, g, b]
      break
    }
  }
  const [r, g, b, a = 1] = rgba(getComputedStyle(el).color)
  const fore = [r, g, b].map((v, k) => v * a * opacity + back[k] * (1 - a * opacity))
  const [hi, lo] = [lum(fore), lum(back)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
})
/** Whether the focus is inside the element */
const holds = (page: Page, selector: string) => page.evaluate((sel) => !!document.querySelector(sel)?.contains(document.activeElement), selector)
/** Whether the element is inert, or inside an element that is */
const inert = (page: Page, selector: string) => page.evaluate((sel) => !!document.querySelector(sel)?.closest('[inert]'), selector)
// The dock's title row once a check is in
const CHECKED = /^Checked \d\d:\d\d$/

test('Check labels lists the problems; rows open their shape, fixes undo, saves check again, a label renames across the folder', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const dir = await planted('dataset')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  const trashed = await fakeTrash(app, path.join(BASE, 'dataset-trash'))
  await app.evaluate(({ shell }) => {
    const shown: string[] = (globalThis as any).shown = []
    shell.showItemInFolder = ((file: string) => void shown.push(file)) as any
  })
  const dock = page.getByRole('region', { name: 'Problems' })
  const group = (code: string) => dock.locator(`.problems-group[data-code="${code}"]`)
  const filter = page.locator('.filters button', { hasText: 'Problems' })
  // Until a check has run, the Problems filter says why it's off
  await expect(filter).toBeDisabled()
  await expect(filter).toHaveAttribute('title', /Dataset > Check labels/)

  await menu(page, 'Dataset', 'Check labels')
  await expect(dock).toBeVisible()
  await expect(dock.locator('.problems-group')).toHaveCount(8)
  // Errors first, then warnings, each group with its count; information is hidden at first
  await expect(dock.locator('.problems-title')).toHaveText([
    "Label files that can't be read1", 'Points repeated one after another1', 'Shapes reaching outside the image1', 'Exact copies of a shape1',
    'Labels not in the label list1', 'One label written several ways1', 'AI proposals not yet accepted1', 'Label files without their image1'
  ])
  const severity = (key: string) => dock.locator(`.problems-severities [data-severity="${key}"]`)
  await expect(severity('error')).toHaveAccessibleName('1 error')
  await expect(severity('warning')).toHaveAccessibleName('7 warnings')
  await expect(severity('info')).toHaveAccessibleName('1 note')
  await expect(severity('info')).toHaveAttribute('aria-pressed', 'false')
  await expect(dock.locator('.problems-status')).toHaveText(CHECKED)
  await expect(group('label-variants').locator('.problem')).toContainText('Label “Cat” looks like “cat” from the label list')
  await shot(page, 'dock-1500-dark.png')
  expect(await fits(page, '.panel.right')).toEqual({ inside: true, page: true, onScreen: true })
  await darkMode(page)
  await shot(page, 'dock-1500-light.png')
  await resize(app, page, 1000, 640)
  await shot(page, 'dock-1000-light.png')
  expect(await fits(page, '.panel.right')).toEqual({ inside: true, page: true, onScreen: true })
  expect(await fits(page, '.panel.left')).toEqual({ inside: true, page: true, onScreen: true })
  await darkMode(page)
  await shot(page, 'dock-1000-dark.png')
  await resize(app, page, 1500, 940)

  // Information shows when asked for, and one image's problems alone
  await severity('info').click()
  await expect(group('never-labelled')).toContainText('h.png has no label file yet')
  await severity('info').click()
  await expect(group('never-labelled')).toHaveCount(0)

  // A row opens its image with its shape selected
  await group('outside-image').locator('.problem').click()
  await expect(page.locator('.file.on')).toContainText('c.png')
  await expect(page.locator('.shape.on')).toContainText('dog')
  await dock.getByLabel('This image only').check()
  await expect(dock.locator('.problems-group')).toHaveCount(1)
  await dock.getByLabel('This image only').uncheck()
  await group('duplicate-shape').locator('.problem').click()
  await expect(page.locator('.file.on')).toContainText('a.png')
  await expect(page.locator('.shape').nth(1)).toHaveClass(/\bon\b/)
  await expect(page.locator('.shape.on')).toHaveCount(1)

  // Fix takes the copy out as an edit: it's saved, checked again, and Ctrl+Z brings it back
  await group('duplicate-shape').getByRole('button', { name: 'Fix', exact: true }).click()
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(page.locator('.loading.notice')).toContainText('Deleted 1 exact copy of a shape')
  await saved(page)
  await expect.poll(() => shapesIn(dir, 'a').length).toBe(1)
  await expect(group('duplicate-shape')).toHaveCount(0)
  await page.keyboard.press('ControlOrMeta+z')
  await expect(page.locator('.shape')).toHaveCount(2)
  await saved(page)
  await expect.poll(() => shapesIn(dir, 'a').length).toBe(2)
  await expect(group('duplicate-shape')).toHaveCount(1)

  // Fix on another image's problem opens it first
  await group('outside-image').getByRole('button', { name: 'Fix', exact: true }).click()
  await expect(page.locator('.file.on')).toContainText('c.png')
  await saved(page)
  await expect.poll(() => shapesIn(dir, 'c')[0].points).toEqual([[300, 200], [400, 300]])
  await expect(group('outside-image')).toHaveCount(0)

  // The Files dock lists the images with an error or a warning
  await expect(filter).toBeEnabled()
  await expect(filter).toContainText('6')
  await filter.click()
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png', 'd.png', 'e.png', 'f.png', 'g.png'])
  await page.locator('.filters button', { hasText: 'All' }).click()

  // An edit saved in the window puts its rows out of date until they're checked again, a moment later
  await open(page, 'b.png')
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect(group('repeated-points').locator('.problem.stale')).toHaveCount(1)
  await expect(dock.locator('.problems-status')).toContainText('Out of date')
  await expect(group('repeated-points')).toHaveCount(0)
  await expect(dock.locator('.problems-status')).toHaveText(CHECKED)

  // A label file without its image is shown in its folder
  await group('orphan-label-file').getByRole('button', { name: 'Show', exact: true }).click()
  await expect.poll(() => app.evaluate(() => (globalThis as any).shown)).toEqual([path.join(dir, 'gone.json')])

  // Rename Cat to cat across the folder: the plan first, then the files, the old ones in the trash
  await group('label-variants').getByRole('button', { name: 'Rename…', exact: true }).click()
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await expect(rename.getByLabel('New name')).toHaveValue('cat')
  await expect(rename).toContainText('Renames 1 shape in 1 image: d.png.')
  await expect(rename).toContainText("1 label file is left alone:f.json can't be read, so it's left unchanged")
  await expect(rename).toContainText('“cat” has 2 shapes already: these join them.')
  await shot(page, 'rename-1500-dark.png')
  // A space typed at the end isn't part of a label
  await rename.getByLabel('New name').fill('cat ')
  await expect(rename.getByRole('alert')).toHaveCount(0)
  await rename.getByRole('button', { name: 'Rename 1 shape' }).click()
  await expect(rename).toHaveCount(0)
  await expect(page.locator('.loading.notice')).toContainText('Renamed 1 shape in 1 image from “Cat” to “cat”. 1 label file was left alone.')
  expect(shapesIn(dir, 'd')[0].label).toBe('cat')
  expect(await trashed()).toEqual([path.join(dir, 'd.json')])
  expect(existsSync(path.join(BASE, 'dataset-trash', '0-d.json'))).toBe(true)
  await expect(group('label-variants')).toHaveCount(0)
  await expect(page.locator('.class', { hasText: /^cat/ })).toContainText('· 3')
  // Left: the copy put back, the zebra, the AI proposal and the file without its image
  await expect(severity('warning')).toHaveAccessibleName('4 warnings')

  // A label added puts the labels shown into the label list, zebra too: that's checked again, with no save
  await page.getByPlaceholder('New label').fill('horse')
  await page.keyboard.press('Enter')
  await expect(group('unknown-label')).toHaveCount(0)
  await expect(severity('warning')).toHaveAccessibleName('3 warnings')
  await finish(app, errors)
})

test('Statistics count the folder, sort its labels, recount at a training size and save a CSV', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const dir = await planted('dataset-stats')
  const csv = path.join(BASE, 'dataset-stats.csv')
  rmSync(csv, { force: true })
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  await app.evaluate(({ dialog }, file) => {
    dialog.showSaveDialog = (async () => ({ canceled: false, filePath: file })) as any
  }, csv)
  await menu(page, 'Dataset', 'Statistics…')
  const stats = page.getByRole('dialog', { name: 'Statistics' })
  const tile = (key: string) => stats.locator(`[data-tile="${key}"] b`)
  await expect(tile('total')).toHaveText('8')
  const tiles = { labelled: '5', empty: '0', never: '1', verified: '0', broken: '1', unreadable: '0', aiPending: '1' }
  for (const [key, value] of Object.entries(tiles)) await expect(tile(key)).toHaveText(value)
  // Most objects first: AI proposals aren't objects until accepted
  const rows = stats.locator('.stats-labels tbody tr')
  await expect(rows).toHaveCount(4)
  await expect(rows.first()).toHaveAttribute('data-label', 'cat')
  await expect(rows.first().locator('td')).toHaveText(['cat', '3', '2', '3', '0', 'Rename…'])
  await expect(stats.locator('.stats-labels thead th')).toHaveText(['Label', 'Objects', 'Images', 'Polygon', 'Rectangle', ''])
  await stats.getByRole('button', { name: 'Objects', exact: true }).click()
  await expect(stats.locator('th[aria-sort="ascending"]')).toHaveText('Objects')
  await expect(rows.first()).toHaveAttribute('data-label', 'Cat')
  await stats.getByRole('button', { name: 'Objects', exact: true }).click()
  // One image with no objects (only an AI proposal), four with one, one with two
  await expect(stats.locator('.stats-column')).toHaveText(['10', '41', '12'])
  const sizes = stats.locator('.stats-sizes tbody tr')
  await expect(sizes).toHaveText(['Small under 32 × 32 px00', 'Medium 00', 'Large 96 × 96 px and over66'])
  await expect(stats.locator('.stats-image-sizes')).toHaveText('400 × 300 8 images')
  await shot(page, 'stats-1500-dark.png')
  expect(await fits(page, '.dialog.stats')).toEqual({ inside: true, page: true, onScreen: true })
  // The dialog is modal: the theme is changed with it closed, and it reads the folder again on opening
  const reopen = async (then: () => Promise<void>) => {
    await stats.getByRole('button', { name: 'Close' }).click()
    await then()
    await menu(page, 'Dataset', 'Statistics…')
    await expect(tile('total')).toHaveText('8')
  }

  // At 32 px each 100 px box shrinks to 8 px: small
  await stats.getByLabel('Training size').fill('32')
  await expect(sizes.last()).toHaveText('Large 96 × 96 px and over60')
  await expect(sizes.first()).toHaveText('Small under 32 × 32 px06')
  await stats.getByLabel('Training size').fill('640')
  await expect(sizes.last()).toHaveText('Large 96 × 96 px and over66')

  await stats.getByRole('button', { name: 'Save as CSV…' }).click()
  await expect(stats).toContainText('Saved as dataset-stats.csv')
  expect(readFileSync(csv, 'utf8')).toBe('label,objects,images,polygon,rectangle\r\ncat,3,2,3,0\r\nCat,1,1,1,0\r\ndog,1,1,0,1\r\nzebra,1,1,1,0\r\n')

  await reopen(() => darkMode(page))
  await shot(page, 'stats-1500-light.png')
  await resize(app, page, 1000, 640)
  await shot(page, 'stats-1000-light.png')
  expect(await fits(page, '.dialog.stats')).toEqual({ inside: true, page: true, onScreen: true })
  await reopen(() => darkMode(page))
  await shot(page, 'stats-1000-dark.png')
  await resize(app, page, 1500, 940)

  // A row's Rename… opens the rename over the statistics
  await rows.filter({ hasText: 'zebra' }).getByRole('button', { name: 'Rename…', exact: true }).click()
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await expect(rename).toContainText('Renames 1 shape in 1 image: e.png.')
  await rename.getByLabel('New name').fill('horse')
  await expect(rename.getByRole('button', { name: 'Rename 1 shape' })).toBeEnabled()
  await rename.getByRole('button', { name: 'Cancel' }).click()
  await expect(rename).toHaveCount(0)
  await stats.getByRole('button', { name: 'Close' }).click()
  await expect(stats).toHaveCount(0)
  expect(JSON.parse(readFileSync(path.join(dir, 'e.json'), 'utf8')).shapes[0].label).toBe('zebra')
  // Nothing else of the folder changed
  expect(existsSync(path.join(dir, 'gone.json'))).toBe(true)
  await finish(app, errors)
})

test("a label list that allows only its labels refuses another name in a rename", async () => {
  const dir = await planted('dataset-exact')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog', '--validatelabel', 'exact'])
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  await dock.locator('.problems-group[data-code="unknown-label"]').getByRole('button', { name: 'Rename…', exact: true }).click()
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await expect(rename).toContainText('Renames 1 shape in 1 image: e.png.')
  await rename.getByLabel('New name').fill('horse')
  await expect(rename.getByRole('alert')).toHaveText("“horse” isn't in the label list, and only listed labels are allowed.")
  await expect(rename.getByRole('button', { name: 'Rename 1 shape' })).toBeDisabled()
  await rename.getByLabel('New name').fill('dog')
  await expect(rename.getByRole('button', { name: 'Rename 1 shape' })).toBeEnabled()
  await page.keyboard.press('Escape')
  await expect(rename).toHaveCount(0)
  expect(JSON.parse(readFileSync(path.join(dir, 'e.json'), 'utf8')).shapes[0].label).toBe('zebra')
  // The dock closes, and View > Problems opens it again with the check kept
  await dock.getByRole('button', { name: 'Close Problems' }).click()
  await expect(dock).toHaveCount(0)
  await page.getByRole('menuitem', { name: 'View', exact: true }).click()
  await page.getByRole('menuitemcheckbox', { name: 'Problems' }).click()
  await expect(dock.locator('.problems-group')).toHaveCount(8)
  await finish(app, errors)
})

test("Don't save before a check or the statistics takes the changes back, so no later save writes them", async () => {
  const dir = await project('dataset-dont-save', {
    a: doc('a', [square('cat', 20, 20, 120, 120), square('dog', 200, 100, 300, 200)]),
    b: doc('b', [square('cat', 50, 50, 150, 150)])
  })
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog', '--no-auto-save'])
  await app.evaluate(({ dialog }) => {
    const asked: string[] = (globalThis as any).asked = []
    dialog.showMessageBox = (async (_w: unknown, o: { message: string }) => {
      asked.push(o.message)
      return { response: 1, checkboxChecked: false }
    }) as any
  })
  const asked = () => app.evaluate(() => (globalThis as any).asked as string[])
  const state = page.getByTestId('save-state')
  await expect(page.locator('.file.on')).toContainText('a.png')
  await page.locator('.shape', { hasText: 'cat' }).click()
  await page.keyboard.press('Delete')
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(state).toHaveText('Not saved')
  await menu(page, 'Dataset', 'Check labels')
  await expect.poll(asked).toEqual(['Save the changes to a.png?'])
  // Let go: the image is read again as saved, with nothing of it to undo
  await expect(page.locator('.shape')).toHaveCount(2)
  await expect(page.getByRole('region', { name: 'Problems' }).locator('.problems-status')).toHaveText(CHECKED)
  await expect(state).toHaveText('Saved')
  await page.keyboard.press('ControlOrMeta+z')
  await page.waitForTimeout(300)
  await expect(state).toHaveText('Saved')
  await expect(page.locator('.shape')).toHaveCount(2)
  // A later edit saved by hand saves itself alone
  await page.locator('.shape', { hasText: 'dog' }).click()
  await page.keyboard.press('Delete')
  await page.keyboard.press('ControlOrMeta+s')
  await saved(page)
  expect(shapesIn(dir, 'a').map((s: { label: string }) => s.label)).toEqual(['cat'])
  // The same before Statistics
  await page.locator('.shape', { hasText: 'cat' }).click()
  await page.keyboard.press('Delete')
  await expect(page.locator('.shape')).toHaveCount(0)
  await menu(page, 'Dataset', 'Statistics…')
  const stats = page.getByRole('dialog', { name: 'Statistics' })
  await expect(stats.locator('[data-tile="labelled"] b')).toHaveText('2')
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(state).toHaveText('Saved')
  expect(await asked()).toEqual(['Save the changes to a.png?', 'Save the changes to a.png?'])
  await stats.getByRole('button', { name: 'Close' }).click()
  expect(shapesIn(dir, 'a').map((s: { label: string }) => s.label)).toEqual(['cat'])
  await finish(app, errors)
})

test('saving by hand, a fixed problem stays out of date until the image is saved, then goes', async () => {
  const dir = await planted('dataset-by-hand')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog', '--no-auto-save'])
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  const copies = dock.locator('.problems-group[data-code="duplicate-shape"]')
  await copies.locator('.problem').click()
  await expect(page.locator('.file.on')).toContainText('a.png')
  await copies.getByRole('button', { name: 'Fix', exact: true }).click()
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(copies.locator('.problem.stale')).toHaveCount(1)
  // No check of the saved file, which still has the copy, takes that back
  await page.waitForTimeout(2500)
  await expect(copies.locator('.problem.stale')).toHaveCount(1)
  // The notice names the undo key as the menus do, ⌘Z on a Mac
  const undo = process.platform === 'darwin' ? '⌘Z' : 'Ctrl+Z'
  await expect(page.locator('.loading.notice')).toContainText(`Deleted 1 exact copy of a shape. ${undo} undoes it. It's checked again once saved.`)
  await expect(dock.locator('.problems-status')).toHaveText('Out of date')
  await expect(dock.locator('.problems-status')).toHaveAttribute('title', "a.png has changes that aren't saved: they're checked once they're saved.")
  // The row keeps its words as readable as any other, with a clock in its margin; Out of date reads in both themes too
  const late = copies.locator('.problem.stale')
  await expect(late.locator('.problem-late')).toBeVisible()
  mkdirSync(SHOTS, { recursive: true })
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1500, 940], [1000, 640], [960, 600]]) {
      await resize(app, page, width, height)
      await late.scrollIntoViewIfNeeded()
      for (const words of [late.locator('.problem-open'), late.locator('.problem-what'), late.locator('.problem-shape'), dock.locator('.problems-status')]) {
        expect(await contrast(words)).toBeGreaterThanOrEqual(4.5)
      }
      await shot(page, `dock-not-saved-${width}-${theme}.png`)
    }
    if (theme === 'dark') await darkMode(page)
  }
  await resize(app, page, 1500, 940)
  await copies.getByRole('button', { name: 'Fix', exact: true }).click()
  await expect(page.locator('.loading.notice')).toContainText("That problem isn't in the image as it is now. It's checked again once the image is saved.")
  expect(shapesIn(dir, 'a')).toHaveLength(2)
  await page.keyboard.press('ControlOrMeta+s')
  await saved(page)
  await expect(copies).toHaveCount(0)
  await expect(dock.locator('.problems-status')).toHaveText(CHECKED)
  expect(shapesIn(dir, 'a')).toHaveLength(1)
  await finish(app, errors)
})

test('at the smallest window the dock lists several problems, the filters fit, and a picked row reads in full', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const dir = await planted('dataset-small')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  await expect(dock.locator('.problems-group')).toHaveCount(8)
  await resize(app, page, 960, 600)
  // a.png's two shapes show whole, and the problems get the rest of the column
  const room = await page.evaluate(() => {
    const list = document.querySelector('.problems-list')!, shapes = document.querySelector('.shapes')!
    return { list: list.clientHeight, shapes: shapes.clientHeight, need: shapes.scrollHeight }
  })
  // 83 px with the time in a row of its own and the Shapes list taking two fifths whatever it held
  expect(room.list).toBeGreaterThanOrEqual(115)
  expect(room.shapes).toBeGreaterThanOrEqual(room.need)
  expect(await fits(page, '.panel.right')).toEqual({ inside: true, page: true, onScreen: true })
  await shot(page, 'dock-960-dark.png')

  // A message cut to two lines shows whole once its row is picked, or holds the focus
  const cut = (code: string) => dock.locator(`.problems-group[data-code="${code}"] .problem-what`).first()
    .evaluate((el) => el.scrollHeight > el.clientHeight + 1)
  expect(await cut('label-file-unreadable')).toBe(true)
  await dock.locator('.problems-group[data-code="label-file-unreadable"] .problem').click()
  await expect(page.locator('.file.on')).toContainText('f.png')
  await expect(dock.locator('.problem.picked')).toHaveCount(1)
  expect(await cut('label-file-unreadable')).toBe(false)
  expect(await cut('repeated-points')).toBe(true)
  await dock.locator('.problems-group[data-code="repeated-points"] .problem-open').focus()
  await expect.poll(() => cut('repeated-points')).toBe(false)
  await shot(page, 'dock-960-picked-dark.png')
  await darkMode(page)
  await shot(page, 'dock-960-light.png')

  // One image's notes alone: hidden, and the dock says so in words
  await open(page, 'h.png')
  await dock.getByLabel('This image only').check()
  await expect(dock.locator('.problems-list')).toHaveText('Nothing shown: 1 note is hidden.')
  await shot(page, 'dock-960-notes-light.png')
  await dock.getByLabel('This image only').uncheck()

  // With the counts it has, no filter's name is cut short; with counts of five digits, the five keep inside the Files dock
  const whole = (counts?: string[]) => page.evaluate((texts) => {
    if (texts) document.querySelectorAll('.filters b').forEach((b, k) => (b.textContent = texts[k]))
    return [...document.querySelectorAll('.filters button > span')].map((s) => s.scrollWidth <= s.clientWidth)
  }, counts)
  expect(await whole()).toEqual([true, true, true, true, true])
  // Counts past 9,999 show as thousands, so four digits in each is the widest they get: the names stay whole
  expect(await whole(['9999', '9999', '9999', '9999', '9999'])).toEqual([true, true, true, true, true])
  expect(await whole(['100k', '9999', '87k', '12k', '9999'])).toEqual([true, true, true, true, true])
  await page.screenshot({ path: path.join(SHOTS, 'filters-960-light.png'), clip: { x: 0, y: 40, width: 280, height: 160 } })
  const row = await page.evaluate(() => {
    const filters = document.querySelector('.filters') as HTMLElement
    for (const b of filters.querySelectorAll('b')) b.textContent = '99999'
    const right = Math.max(...[...filters.querySelectorAll('button')].map((b) => b.getBoundingClientRect().right))
    return { scroll: filters.scrollWidth, client: filters.clientWidth, right, edge: filters.getBoundingClientRect().right }
  })
  expect(row.scroll).toBeLessThanOrEqual(row.client)
  expect(row.right).toBeLessThanOrEqual(row.edge + 0.5)
  await page.screenshot({ path: path.join(SHOTS, 'filters-960-light-99999.png'), clip: { x: 0, y: 40, width: 280, height: 160 } })
  await finish(app, errors)
})

test('the label list follows a rename once no label file has the old name', async () => {
  const dir = await planted('dataset-classes', [{ name: 'Cat', color: '#c71585' }])
  const bin = path.join(BASE, 'dataset-classes-trash')
  rmSync(bin, { recursive: true, force: true })
  mkdirSync(bin, { recursive: true })
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  // The trash refuses d.json once
  await app.evaluate(({ shell }, into) => {
    const fs = process.mainModule!.require('fs')
    const path = process.mainModule!.require('path')
    let refused = false
    shell.trashItem = (async (file: string) => {
      if (!refused && path.basename(file) === 'd.json') {
        refused = true
        throw new Error('The trash is full')
      }
      fs.renameSync(file, path.join(into, path.basename(file)))
    }) as any
  }, bin)
  const classes = () => JSON.parse(readFileSync(path.join(dir, '.myseg', 'project.json'), 'utf8')).classes
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  const renameCat = async () => {
    await dock.locator('.problems-group[data-code="label-variants"]').getByRole('button', { name: 'Rename…', exact: true }).click()
    await expect(rename.getByLabel('New name')).toHaveValue('cat')
    await rename.getByRole('button', { name: 'Rename 1 shape' }).click()
    await expect(rename).toHaveCount(0)
  }
  await renameCat()
  await expect(page.locator('.error')).toContainText("1 label file couldn't be renamed. d.png: d.json couldn't go to the trash")
  await expect(page.locator('.loading.notice')).toContainText('Renamed 0 shapes in 0 images from “Cat” to “cat”.')
  // The label list is saved a moment after it changes: it hasn't
  await page.waitForTimeout(500)
  expect(classes()).toEqual([{ name: 'Cat', color: '#c71585' }])
  await expect(page.locator('.loading.notice')).toContainText('“Cat” stays in the label list while 1 image has it.')
  expect(shapesIn(dir, 'd')[0].label).toBe('Cat')
  await expect(page.locator('section[aria-label="Labels"] .class .name', { hasText: /^Cat$/ })).toHaveCount(1)
  // Again, with the trash taking it: the label list's Cat becomes cat, its colour kept
  await renameCat()
  await expect(page.locator('.loading.notice')).toContainText('Renamed 1 shape in 1 image from “Cat” to “cat”.')
  expect(shapesIn(dir, 'd')[0].label).toBe('cat')
  await expect.poll(classes).toEqual([{ name: 'cat', color: '#c71585' }])
  await expect(page.locator('section[aria-label="Labels"] .class .name', { hasText: /^Cat$/ })).toHaveCount(0)
  await finish(app, errors)
})

test('after a rename the label list no longer offers the old name, and new shapes still get the label picked for them', async () => {
  const dir = await project('dataset-label-list', {
    a: doc('a', [square('dgo', 20, 20, 120, 120)]), b: doc('b', [square('dog', 30, 30, 90, 90)]), c: doc('c', [square('cow', 40, 40, 140, 140)])
  }, [{ name: 'dgo', color: '#aa0000' }, { name: 'cow', color: '#00aa00' }, { name: 'dog', color: '#0000aa' }, { name: 'cat', color: '#aaaa00' }])
  const { app, page, errors } = await launch(dir)
  await fakeTrash(app, path.join(BASE, 'dataset-label-list-trash'))
  const names = page.locator('section[aria-label="Labels"] .class .name')
  const active = page.locator('section[aria-label="Labels"] .class.on .name')
  const rename = async (from: string, to: string) => {
    await menu(page, 'Dataset', 'Statistics…')
    const stats = page.getByRole('dialog', { name: 'Statistics' })
    await stats.locator(`.stats-labels tbody tr[data-label="${from}"]`).getByRole('button', { name: 'Rename…' }).click()
    const dialog = page.getByRole('dialog', { name: 'Rename a label' })
    await expect(dialog).toContainText('Renames 1 shape in 1 image')
    await dialog.getByLabel('New name').fill(to)
    await dialog.getByRole('button', { name: 'Rename 1 shape' }).click()
    await expect(dialog).toHaveCount(0)
    await expect(stats.locator(`.stats-labels tbody tr[data-label="${from}"]`)).toHaveCount(0)
    await page.keyboard.press('Escape')
    await expect(stats).toHaveCount(0)
  }
  await expect(names).toHaveText(['dgo', 'cow', 'dog', 'cat'])
  // The typo is the label new shapes get, on key 1
  await page.keyboard.press('1')
  await expect(active).toHaveText('dgo')
  await rename('dgo', 'dog')
  await expect(page.locator('.loading.notice')).toContainText('Renamed 1 shape in 1 image from “dgo” to “dog”.')
  // It's gone from the list and its number key, and new shapes get the name it was given
  await expect(names).toHaveText(['cow', 'dog', 'cat'])
  await expect(active).toHaveText('dog')
  // A label above it in the list goes: new shapes still get dog, not the label that moves up into its place
  await rename('cow', 'cat')
  await expect(names).toHaveText(['dog', 'cat'])
  await expect(active).toHaveText('dog')
  await page.keyboard.press('2')
  await expect(active).toHaveText('cat')
  expect([shapesIn(dir, 'a')[0].label, shapesIn(dir, 'c')[0].label]).toEqual(['dog', 'cat'])
  await expect.poll(() => JSON.parse(readFileSync(path.join(dir, '.myseg', 'project.json'), 'utf8')).classes.map((c: { name: string }) => c.name)).toEqual(['dog', 'cat'])
  await finish(app, errors)
})

test('Statistics and a rename over it hold the focus: Tab goes round each, Escape closes the one on top, and the page behind takes no keys', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const dir = await planted('dataset-held')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  // Ctrl+O or Ctrl+U behind a window would ask for a file or a folder
  await app.evaluate(({ dialog }) => {
    const g = globalThis as any
    g.asked = 0
    dialog.showOpenDialog = (async () => {
      g.asked++
      return { canceled: true, filePaths: [] }
    }) as any
  })
  const stats = page.getByRole('dialog', { name: 'Statistics' })
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await menu(page, 'Dataset', 'Statistics…')
  await expect(stats.locator('[data-tile="total"] b')).toHaveText('8')
  await expect(stats).toHaveAttribute('aria-modal', 'true')
  expect(await holds(page, '.dialog.stats')).toBe(true)
  // The page behind is inert, and the app's shortcuts wait: the folder stays open
  expect(await inert(page, '.panel.left')).toBe(true)
  expect(await inert(page, '.topbar')).toBe(true)
  for (const keys of ['ControlOrMeta+w', 'ControlOrMeta+o', 'ControlOrMeta+u']) await page.keyboard.press(keys)
  await page.waitForTimeout(300)
  await expect(stats).toBeVisible()
  await expect(page.locator('.file')).toHaveCount(8)
  expect(await app.evaluate(() => (globalThis as any).asked)).toBe(0)
  // Tab goes round the window's controls and never leaves it
  for (let k = 0; k < 14; k++) {
    await page.keyboard.press('Tab')
    expect(await holds(page, '.dialog.stats')).toBe(true)
  }
  const first = stats.getByRole('button', { name: 'Label', exact: true })
  await first.focus()
  await page.keyboard.press('Shift+Tab')
  await expect(stats.getByRole('button', { name: 'Close' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(first).toBeFocused()

  // A row's Rename… takes the focus and the keys, and the statistics under it go inert
  const zebra = stats.locator('.stats-labels tbody tr[data-label="zebra"]').getByRole('button', { name: 'Rename…' })
  await zebra.click()
  await expect(rename).toContainText('Renames 1 shape in 1 image: e.png.')
  await expect(rename.getByLabel('New name')).toBeFocused()
  await page.keyboard.type('pony')
  await expect(rename.getByLabel('New name')).toHaveValue('pony')
  expect(await inert(page, '.dialog.stats')).toBe(true)
  for (let k = 0; k < 6; k++) {
    await page.keyboard.press('Tab')
    expect(await holds(page, '.dialog.rename')).toBe(true)
  }
  // Escape closes the rename alone, and the focus goes back to the button that opened it
  await page.keyboard.press('Escape')
  await expect(rename).toHaveCount(0)
  await expect(zebra).toBeFocused()
  expect(await inert(page, '.dialog.stats')).toBe(false)
  await page.keyboard.press('Escape')
  await expect(stats).toHaveCount(0)
  expect(await page.evaluate(() => document.querySelectorAll('[inert]').length)).toBe(0)
  expect(JSON.parse(readFileSync(path.join(dir, 'e.json'), 'utf8')).shapes[0].label).toBe('zebra')

  // Both, at the smallest window and a large one, dark and light: each opens holding the focus, and Escape closes it
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1500, 940], [960, 600]]) {
      await resize(app, page, width, height)
      await menu(page, 'Dataset', 'Statistics…')
      await expect(stats.locator('[data-tile="total"] b')).toHaveText('8')
      expect(await holds(page, '.dialog.stats')).toBe(true)
      await shot(page, `held-stats-${width}-${theme}.png`)
      await zebra.click()
      await expect(rename).toContainText('Renames 1 shape in 1 image: e.png.')
      await shot(page, `held-rename-${width}-${theme}.png`)
      await page.keyboard.press('Escape')
      await page.keyboard.press('Escape')
      await expect(stats).toHaveCount(0)
    }
    if (theme === 'dark') await darkMode(page)
  }
  await finish(app, errors)
})

test('a rename holds its window while it runs: Ctrl+W and Escape wait for it, and the folder stays open', async () => {
  const files: Record<string, string> = {}
  for (let i = 0; i < 30; i++) files[`r${i}`] = doc(`r${i}`, [square('dgo', 20, 20, 120, 120)])
  const dir = await project('dataset-rename-held', files)
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  // A trash that takes its time, so the rename is still running when the keys come
  await app.evaluate(({ shell }) => {
    const fs = process.mainModule!.require('fs')
    shell.trashItem = (async (file: string) => {
      await new Promise((done) => setTimeout(done, 100))
      fs.rmSync(file)
    }) as any
  })
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  await dock.locator('.problems-group[data-code="unknown-label"]').getByRole('button', { name: 'Rename…', exact: true }).first().click()
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await expect(rename).toContainText('Renames 30 shapes in 30 images')
  await rename.getByLabel('New name').fill('dog')
  await page.keyboard.press('Enter')
  await expect(rename.getByRole('button', { name: /^Stop/ })).toBeVisible()
  // The field it ran from is disabled: the window keeps the focus, so the keys stay with it
  await expect.poll(() => holds(page, '.dialog.rename')).toBe(true)
  await page.keyboard.press('ControlOrMeta+w')
  await page.keyboard.press('Escape')
  await page.waitForTimeout(300)
  await expect(rename).toBeVisible()
  await expect(rename).toHaveCount(0, { timeout: 30000 })
  await expect(page.locator('.loading.notice')).toContainText('Renamed 30 shapes in 30 images from “dgo” to “dog”.')
  await expect(page.locator('.file')).toHaveCount(30)
  for (let i = 0; i < 30; i++) expect(shapesIn(dir, `r${i}`)[0].label).toBe('dog')
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('Tab moves through the Problems dock, and skips to an AI shape only from the canvas', async () => {
  const dir = await planted('dataset-tab')
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog'])
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  const copies = dock.locator('.problems-group[data-code="duplicate-shape"]')
  const fix = copies.getByRole('button', { name: 'Fix', exact: true })
  await copies.locator('.problem-open').click()
  await expect(page.locator('.file.on')).toContainText('a.png')
  await expect(copies.locator('.problem-open')).toBeFocused()
  // From a row's image name to its Fix, then to the next group, and back: a.png stays open
  await page.keyboard.press('Tab')
  await expect(fix).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(dock.locator('.problems-group[data-code="unknown-label"] .problems-title')).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(fix).toBeFocused()
  await page.waitForTimeout(300)
  await expect(page.locator('.file.on')).toContainText('a.png')
  await expect(page.locator('.shape')).toHaveCount(2)
  // On the canvas, Tab skips to the next AI shape to review: g.png's
  await page.getByTestId('canvas').focus()
  await page.keyboard.press('Tab')
  await expect(page.locator('.file.on')).toContainText('g.png')
  await expect(page.locator('.shape.on')).toContainText('dog')
  await finish(app, errors)
})

// labelme's label list for its instance segmentation example (examples/instance_segmentation/labels.txt): it fills the Labels list
const VOC = '__ignore__,_background_,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,potted plant,sheep,sofa,train,tv/monitor'

/**
 * The Shapes section over the Problems dock: whether any of it lies past its own bottom or over the
 * dock, how many of its rows show, whether they and Delete shape take a click where they show, how
 * tall the Labels list is, and whether the column has to scroll
 */
const shapesColumn = (page: Page) => page.evaluate(() => {
  const section = document.querySelector('section[aria-label="Shapes"]')!, list = section.querySelector('.shapes')!
  const panel = document.querySelector('.panel.right')!, dock = document.querySelector('section.problems')!.getBoundingClientRect()
  const box = section.getBoundingClientRect(), view = list.getBoundingClientRect()
  const takes = (el: Element, top: number, bottom: number) => {
    const r = el.getBoundingClientRect()
    return el.contains(document.elementFromPoint(r.left + r.width / 2, (Math.max(r.top, top) + Math.min(r.bottom, bottom)) / 2))
  }
  const rows = [...list.querySelectorAll('.shape')].filter((row) => {
    const r = row.getBoundingClientRect()
    return r.bottom > view.top + 2 && r.top < view.bottom - 2
  })
  const remove = section.querySelector('.row.selection .danger')
  return {
    spills: Math.max(...[...section.children].map((el) => el.getBoundingClientRect().bottom)) > box.bottom + 1,
    overDock: box.bottom > dock.top,
    rows: rows.length,
    rowsTake: rows.every((row) => takes(row, view.top, view.bottom)),
    remove: remove && takes(remove, -Infinity, Infinity),
    labels: document.querySelector('.panel.right .classes')!.clientHeight,
    scrolls: panel.scrollHeight > panel.clientHeight + 1
  }
})

test('on a short window the Shapes list keeps off the Problems dock, open or folded, with a long label list and a shape picked', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const files: Record<string, string | null> = {
    a: doc('a', ['person', 'dog', 'cat', 'car', 'chair', 'sofa', 'bus'].map((l, k) => square(l, 10 + k * 30, 10 + k * 20, 80 + k * 30, 80 + k * 20))),
    b: doc('b', [square('zebra', 60, 60, 160, 160)]), c: doc('c', [square('Cat', 60, 60, 160, 160)]),
    d: doc('d', [square('dog', 100, 100, 200, 200, 0.4)]), e: doc('e', [rect('bus', 300, 200, 430, 330)]), f: '{"shapes": [', g: null
  }
  const dir = await project('dataset-column', files)
  await paint(dir, Object.keys(files))
  const { app, page, errors } = await launch(dir, ['--labels', VOC])
  // The list's 22, and the files' Cat and zebra
  await expect(page.locator('section[aria-label="Labels"] .class')).toHaveCount(24)
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  await expect(dock.locator('.problems-status')).toHaveText(CHECKED)
  await expect(page.locator('.file.on')).toContainText('a.png')
  const states: [string, () => Promise<void>][] = [
    ['open', async () => {}],
    // Difficult and Delete shape show under the list
    ['picked', () => page.locator('.shapes .shape').first().click()],
    ['folded', async () => {
      await page.keyboard.press('Escape')
      await expect(page.locator('.row.selection')).toHaveCount(0)
      await dock.locator('.problems-fold').click()
    }]
  ]
  for (const theme of ['dark', 'light']) {
    for (const [state, make] of states) {
      await resize(app, page, 1500, 940)
      await make()
      for (const [width, height] of [[960, 600], [1000, 640], [1500, 940]]) {
        await resize(app, page, width, height)
        const { rows, labels, ...column } = await shapesColumn(page)
        expect(column).toEqual({ spills: false, overDock: false, rowsTake: true, remove: state === 'picked' ? true : null, scrolls: false })
        // Two rows of shapes at the least, and of labels too
        expect(rows).toBeGreaterThanOrEqual(2)
        expect(labels).toBeGreaterThanOrEqual(58)
        await shot(page, `column-${state}-${width}-${theme}.png`)
      }
    }
    // As it began: the dock open
    await dock.locator('.problems-fold').click()
    if (theme === 'dark') await darkMode(page)
  }
  await finish(app, errors)
})

test("problem rows keep their image's name, cut in the middle, and their buttons, whatever the names and labels; Enter and Space press them", async () => {
  mkdirSync(SHOTS, { recursive: true })
  const one = (label: string) => [square(label, 20, 20, 120, 120)]
  const files: Record<string, string | null> = {
    broken: '{"shapes": [',
    camera_north_gate_2024_06_01_frame_000363: doc('camera_north_gate_2024_06_01_frame_000363', one('surfboard')),
    camera_north_gate_2024_06_01_frame_000403: doc('camera_north_gate_2024_06_01_frame_000403', one('Vehicle registration plate')),
    cats_on_sofa: doc('cats_on_sofa', one('Cat')),
    crossing_000077: doc('crossing_000077', [...one('traffic light for pedestrians'), ...one('traffic light for pedestrians')]),
    forklift_yard_0001: doc('forklift_yard_0001', one('forklift carrying a pallet of boxes near the north loading gate')),
    // 206 copies of a shape: with the crossing's, 206 rows, past the dock's first 200
    many: doc('many', Array.from({ length: 206 }, () => square('dog', 50, 50, 90, 90))),
    never: null
  }
  const dir = await project('dataset-rows', files)
  await paint(dir, Object.keys(files))
  const { app, page, errors } = await launch(dir, ['--labels', 'cat,dog,traffic light for pedestrians'])
  await menu(page, 'Dataset', 'Check labels')
  const dock = page.getByRole('region', { name: 'Problems' })
  await expect(dock.locator('.problems-status')).toHaveText(CHECKED)
  const row = (image: string) => dock.locator('.problem', { hasText: image }).first()
  /** A row's name: its width, and whether its end shows; each of its buttons: inside the row, taking a click, and 24 px high */
  const measure = (target: Locator) => target.evaluate((el) => {
    const box = el.getBoundingClientRect(), name = el.querySelector('.name')!.getBoundingClientRect(), tail = el.querySelector('.cut-tail')
    return {
      name: name.width,
      end: tail ? tail.getBoundingClientRect().right <= name.right + 0.5 : el.querySelector('.cut-head')!.scrollWidth <= Math.ceil(name.width),
      buttons: [...el.querySelectorAll('.problem-act')].map((b) => {
        const r = b.getBoundingClientRect()
        return r.right <= box.right + 0.5 && b.contains(document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2)) && r.height >= 24
      })
    }
  })
  const images = ['camera_north_gate_2024_06_01_frame_000363.png', 'camera_north_gate_2024_06_01_frame_000403.png', 'cats_on_sofa.png', 'crossing_000077.png', 'forklift_yard_0001.png']
  const check = async () => {
    for (const image of images) {
      await row(image).scrollIntoViewIfNeeded()
      await expect(row(image).locator('.problem-open')).toHaveAttribute('title', image)
      await expect(row(image).locator('.problem-open')).toHaveAccessibleName(image)
      const { name, end, buttons } = await measure(row(image))
      expect(name).toBeGreaterThanOrEqual(60)
      expect([end, buttons]).toEqual([true, [true]])
    }
    // Names that differ only at the end still tell apart
    await expect(row('frame_000363').locator('.cut-tail')).toHaveText('0363.png')
    await expect(row('frame_000403').locator('.cut-tail')).toHaveText('0403.png')
    expect(await fits(page, '.panel.right')).toEqual({ inside: true, page: true, onScreen: true })
  }
  await check()
  // The title row's buttons are 24 px or more, as WCAG 2.2 asks of a target
  for (const name of ['Check again', 'Close Problems']) {
    const box = (await dock.getByRole('button', { name, exact: true }).boundingBox())!
    expect([box.width, box.height].every((side) => side >= 24)).toBe(true)
  }

  // Enter and Space press a row's buttons: Rename… opens the rename, and the image's name opens it, without verifying it
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  const forklift = row('forklift_yard_0001.png').getByRole('button', { name: 'Rename…', exact: true })
  for (const key of ['Enter', 'Space']) {
    await forklift.focus()
    await page.keyboard.press(key)
    await expect(rename).toContainText('Renames 1 shape in 1 image: forklift_yard_0001.png.')
    await page.keyboard.press('Escape')
    await expect(rename).toHaveCount(0)
    await expect(forklift).toBeFocused()
  }
  await row('frame_000403').locator('.problem-open').focus()
  await page.keyboard.press('Enter')
  await expect(page.locator('.file.on')).toContainText('camera_north_gate_2024_06_01_frame_000403.png')
  await row('frame_000363').locator('.problem-open').focus()
  await page.keyboard.press('Space')
  await expect(page.locator('.file.on')).toContainText('camera_north_gate_2024_06_01_frame_000363.png')
  await page.waitForTimeout(300)
  await expect(page.locator('.status')).not.toContainText('Verified')
  expect(read(dir, 'camera_north_gate_2024_06_01_frame_000363').verified).toBeUndefined()
  const only = dock.getByLabel('This image only')
  await only.focus()
  await page.keyboard.press('Space')
  await expect(only).toBeChecked()
  await expect(dock.locator('.problems-group')).toHaveCount(1)
  await page.keyboard.press('Space')
  await expect(only).not.toBeChecked()

  // A long group shows the rest when one page holds it
  const copies = dock.locator('.problems-group[data-code="duplicate-shape"]')
  await expect(copies.locator('.problems-title b')).toHaveText('206')
  await expect(copies.locator('.problems-more')).toHaveText('Show the other 6')
  await copies.locator('.problems-more').click()
  await expect(copies.locator('.problem')).toHaveCount(206)
  await expect(copies.locator('.problems-more')).toHaveCount(0)

  // A picked row's buttons keep their contrast on its colour, at each size, in both themes
  await row('crossing_000077.png').click()
  await expect(row('crossing_000077.png')).toHaveClass(/\bpicked\b/)
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1500, 940], [1000, 640], [960, 600]]) {
      await resize(app, page, width, height)
      await check()
      await row('crossing_000077.png').scrollIntoViewIfNeeded()
      for (const part of ['.problem-open', '.problem-shape', '.problem-act', '.problem-what']) {
        expect(await contrast(row('crossing_000077.png').locator(part))).toBeGreaterThanOrEqual(4.5)
      }
      await shot(page, `rows-${width}-${theme}.png`)
    }
    if (theme === 'dark') await darkMode(page)
  }
  await resize(app, page, 1500, 940)

  // With every severity hidden, the dock lists what is
  for (const severity of ['error', 'warning']) await dock.locator(`.problems-severities [data-severity="${severity}"]`).click()
  await expect(dock.locator('.problems-list')).toHaveText(/^Nothing shown: 1 error, [\d,]+ warnings and [\d,]+ notes are hidden\.$/)
  await finish(app, errors)
})
