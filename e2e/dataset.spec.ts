import { type ElectronApplication, expect, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { BASE, doc, finish, H, launch, open, project, saved, shapesIn, square, W } from './helpers'

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
  // Each image its own: the helper's are all one grey, which is a problem of its own
  for (const [k, name] of ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].entries()) {
    await sharp({ create: { width: W, height: H, channels: 3, background: { r: 40 + k * 20, g: 80, b: 100 } } }).png().toFile(path.join(dir, `${name}.png`))
  }
  // A label file whose image is gone
  writeFileSync(path.join(dir, 'gone.json'), doc('gone', [square('cat', 1, 1, 9, 9)]))
  return dir
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
  await expect(group('label-variants').locator('.problem')).toContainText('Label "Cat" looks like "cat" from the label list')
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
  await expect(page.locator('.loading.notice')).toContainText("Deleted 1 exact copy of a shape. Ctrl+Z undoes it. It's checked again once saved.")
  await expect(dock.locator('.problems-status')).toHaveText('Out of date')
  await expect(dock.locator('.problems-status')).toHaveAttribute('title', "a.png has changes that aren't saved: they're checked once they're saved.")
  mkdirSync(SHOTS, { recursive: true })
  await shot(page, 'dock-not-saved-1500-dark.png')
  await darkMode(page)
  await shot(page, 'dock-not-saved-1500-light.png')
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
  // Again, with the trash taking it: the label list's Cat becomes cat, its colour kept
  await renameCat()
  await expect(page.locator('.loading.notice')).toContainText('Renamed 1 shape in 1 image from “Cat” to “cat”.')
  expect(shapesIn(dir, 'd')[0].label).toBe('cat')
  await expect.poll(classes).toEqual([{ name: 'cat', color: '#c71585' }])
  await finish(app, errors)
})
