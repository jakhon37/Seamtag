import { type ElectronApplication, expect, type Locator, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { maskOutlines } from '../src/shared/ultralytics'
import { BASE, doc, finish, H, launch, project, square, W } from './helpers'

// File > Export dataset… through the window: every format, from a folder with every kind of shape
// in train, val and test subfolders, checked in the files written. tests/parity/coco_reads.py and
// tests/parity/ultralytics_reads.py then read the datasets in export-out with pycocotools and
// Ultralytics' own code: each YOLO dataset gets an <name>.expected.json beside it for that.

const DIR = path.join(BASE, 'export')
const OUT = path.join(BASE, 'export-out')
const SHOTS = process.env.E2E_SHOTS || OUT
const shape = (label: string, shape_type: string, points: number[][], extra: Record<string, unknown> = {}) =>
  ({ label, shape_type, points, group_id: null, description: '', flags: {}, mask: null, ...extra })
// A mask of two separate blobs, 40 x 30 at (40, 40)
const MASK = { x: 40, y: 40, width: 40, height: 30, data: new Uint8Array(40 * 30) }
for (let y = 0; y < 30; y++) for (let x = 0; x < 40; x++) MASK.data[y * 40 + x] = (y < 10 && x < 10) || (y >= 15 && x >= 20) ? 1 : 0
const DOG_GROUP = [[[200, 50], [260, 50], [230, 100]], [[280, 60], [340, 60], [310, 110]]]
// One object with a part past the image's right edge: kept inside the image, its closest point to the other part is
// another one, so the row is joined as Ultralytics would join the parts COCO's export keeps
const CAR_GROUP = [[[300, 160], [360, 160], [330, 205]], [[480, 151], [390, 109], [395, 280]]]
const NAMES = ['cat', 'dog', 'car', 'ball', 'person']

async function makeFolder() {
  rmSync(DIR, { recursive: true, force: true })
  rmSync(OUT, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  const mask = (await sharp(Buffer.from(MASK.data), { raw: { width: 40, height: 30, channels: 1 } }).png().toBuffer()).toString('base64')
  const files: Record<string, unknown[] | null> = {
    'train/a.png': [
      shape('cat', 'polygon', [[20, 20], [120, 30], [90, 110], [30, 90]]), shape('dog', 'rectangle', [[150, 40], [260, 140]]),
      shape('car', 'oriented_rectangle', [[300, 60], [340, 20], [380, 60], [340, 100]]), shape('ball', 'circle', [[200, 220], [230, 220]]),
      shape('car', 'oriented_rectangle', [[300, 160], [340, 120], [380, 160], [340, 200]]), shape('ball', 'circle', [[100, 220], [120, 220]])
    ],
    'train/sub/b.png': [
      shape('cat', 'mask', [[40, 40], [79, 69]], { mask }),
      shape('dog', 'polygon', DOG_GROUP[0], { group_id: 3 }), shape('dog', 'polygon', DOG_GROUP[1], { group_id: 3 }),
      shape('cat', 'line', [[10, 280], [100, 280]]),
      shape('car', 'polygon', CAR_GROUP[0], { group_id: 5 }), shape('car', 'polygon', CAR_GROUP[1], { group_id: 5 })
    ],
    'val/c.png': [
      shape('person', 'rectangle', [[100, 50], [200, 250]], { group_id: 1 }), shape('nose', 'point', [[150, 80]], { group_id: 1 }),
      shape('left_eye', 'point', [[140, 70]], { group_id: 1 }), shape('right_eye', 'point', [[160, 70]], { group_id: 1, flags: { difficult: true } }),
      shape('person', 'rectangle', [[250, 60], [350, 260]], { group_id: 2 }), shape('nose', 'point', [[300, 90]], { group_id: 2 }),
      shape('right_eye', 'point', [[310, 80]], { group_id: 2 })
    ],
    'val/d.png': [],
    'e.png': null,
    // Each label two objects or none in each split: COCOeval scores a label of one object a hair under AP 1, as it adds
    // np.spacing(1) to the count it divides by, and tests/parity/coco_reads.py wants 1 exactly
    'test/f.png': [
      shape('cat', 'polygon', [[50, 50], [150, 60], [100, 150]]), shape('dog', 'polygon', [[200, 50], [300, 60], [250, 150]], { flags: { ai: true }, score: 0.7 }),
      shape('dog', 'rectangle', [[250, 160], [350, 250]]), shape('cat', 'polygon', [[50, 200], [150, 210], [100, 290]]), shape('dog', 'rectangle', [[20, 160], [60, 200]])
    ]
  }
  for (const [name, shapes] of Object.entries(files)) {
    const file = path.join(DIR, name)
    mkdirSync(path.dirname(file), { recursive: true })
    await sharp({ create: { width: W, height: H, channels: 3, background: '#445566' } }).png().toFile(file)
    if (shapes) {
      writeFileSync(file.replace(/\.png$/, '.json'), JSON.stringify({
        version: '5.5.0', flags: {}, shapes, imagePath: path.basename(name), imageData: null, imageWidth: W, imageHeight: H
      }))
    }
  }
}

/** Native dialogs answer from lists the test fills: folders and files to open, files to save as; shown folders are kept */
async function stubDialogs(app: ElectronApplication) {
  await app.evaluate(({ dialog, shell }) => {
    const g = globalThis as any
    g.opens = []
    g.saves = []
    g.revealed = []
    dialog.showOpenDialog = (async () => (g.opens.length ? { canceled: false, filePaths: [g.opens.shift()] } : { canceled: true, filePaths: [] })) as any
    dialog.showSaveDialog = (async () => (g.saves.length ? { canceled: false, filePath: g.saves.shift() } : { canceled: true })) as any
    shell.showItemInFolder = ((p: string) => void g.revealed.push(p)) as any
  })
}
const answer = (app: ElectronApplication, list: 'opens' | 'saves', value: string) => app.evaluate((_e, [l, v]) => void (globalThis as any)[l].push(v), [list, value])

async function openExport(page: Page) {
  await page.getByRole('menuitem', { name: 'File' }).click()
  await page.getByRole('menuitem', { name: 'Export dataset…' }).click()
  const dialog = page.getByRole('dialog', { name: 'Export dataset' })
  await expect(dialog).toBeVisible()
  await checked(dialog)
  return dialog
}
/** Waits out the check that follows every change */
async function checked(dialog: Locator) {
  await expect(dialog.locator('.export-check')).toHaveAttribute('aria-busy', 'false', { timeout: 30000 })
  await expect(dialog.locator('.export-lead')).toBeVisible()
}
/** A change of options, and the check it starts, which waits at least 200 ms before it runs */
async function change(dialog: Locator, act: () => Promise<unknown>) {
  await act()
  await expect(dialog.locator('.export-check')).toHaveAttribute('aria-busy', 'true')
  await checked(dialog)
}
const choose = (dialog: Locator, format: string) => change(dialog, () => dialog.locator('label.export-format', { hasText: format }).click())
async function exportTo(app: ElectronApplication, dialog: Locator, name: string) {
  const out = path.join(OUT, name)
  await answer(app, 'opens', out)
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.export-file').first()).toContainText(name)
  await dialog.getByRole('button', { name: /^Export \d/ }).click()
  await expect(dialog.locator('.export-done')).toBeVisible({ timeout: 60000 })
  return out
}
async function closeDialog(dialog: Locator) {
  await dialog.locator('.actions').getByRole('button', { name: 'Close' }).click()
  await expect(dialog).toHaveCount(0)
}
const files = (dir: string): string[] => readdirSync(dir, { recursive: true, withFileTypes: true }).filter((e) => e.isFile())
  .map((e) => path.relative(dir, path.join(e.parentPath, e.name)).split(path.sep).join('/')).sort()
const json = (dir: string, file: string) => JSON.parse(readFileSync(path.join(dir, file), 'utf8'))
const rows = (dir: string, file: string) => readFileSync(path.join(dir, file), 'utf8').split('\n').filter(Boolean)
/** Nothing past the window's edges, and the export button where it can be clicked */
async function fits(page: Page) {
  const box = await page.evaluate(() => {
    const r = document.querySelector('.export-dialog')!.getBoundingClientRect()
    const b = [...document.querySelectorAll('.export-dialog .actions button')].map((e) => e.getBoundingClientRect())
    return { inside: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight, buttons: b.every((x) => x.right <= r.right && x.bottom <= r.bottom), page: document.documentElement.scrollWidth <= innerWidth }
  })
  expect(box).toEqual({ inside: true, buttons: true, page: true })
}

/** What tests/parity/ultralytics_reads.py checks a YOLO dataset against: its task and names, and each image's rows as written */
function expectYolo(dir: string, task: string, names: string[], extra: Record<string, unknown> = {}, joined: Record<string, unknown[]> = {}) {
  const images: Record<string, unknown> = {}
  for (const rel of files(dir).filter((f) => f.startsWith('images/'))) {
    const label = rel.replace(/^images\//, 'labels/').replace(/\.[^./]+$/, '.txt')
    const lines = rows(dir, label)
    images[rel] = { status: lines.length ? 'ok' : 'empty', rows: lines.map((l) => l.split(' ').map(Number)), ...(joined[rel] ? { joined: joined[rel] } : {}) }
  }
  writeFileSync(`${dir}.expected.json`, JSON.stringify({ task, names, kpt_shape: null, flip_idx: null, images, ...extra }, null, 1))
}

test('export every format through the window, with a check of what goes in before anything is written', async () => {
  await makeFolder()
  const { app, page, errors } = await launch(DIR, ['--labels', 'cat,dog'])
  await stubDialogs(app)

  // COCO: the check finds the AI proposal not yet accepted, which stops the export
  let dialog = await openExport(page)
  await expect(dialog.locator('label.export-format.on')).toContainText('COCO')
  const error = dialog.locator('.export-group.is-error')
  await expect(error).toContainText('AI proposals not yet accepted')
  await expect(error.locator('.export-rows')).toContainText('test/f.png has 1 AI proposal not yet accepted')
  await expect(dialog.locator('.export-group', { hasText: "Lines and points COCO can't hold" })).toContainText('6')
  await expect(dialog.locator('.actions')).toContainText('Fix the error first')
  await expect(dialog.getByRole('button', { name: 'Export', exact: true })).toBeDisabled()
  await page.screenshot({ path: `${SHOTS}/export-check-dark-1500.png` })
  await fits(page)

  // Left out, the proposal goes; the folders decide the split; a folder inside the images is refused
  await change(dialog, () => dialog.getByLabel('Leave unaccepted AI proposals out').check())
  await change(dialog, () => dialog.getByRole('radio', { name: 'Keep the folders' }).click())
  await expect(dialog.locator('.export-group.is-error')).toHaveCount(0)
  await expect(dialog.locator('.export-table thead')).toContainText('trainvaltestAll')
  await expect(dialog.locator('.export-table tr', { hasText: 'Images' })).toContainText('2215')
  await answer(app, 'opens', path.join(DIR, 'inside'))
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.bad')).toHaveText("That's inside the image folder: choose a new or empty folder outside it")
  await expect(dialog.getByRole('button', { name: 'Export', exact: true })).toBeDisabled()
  const coco = await exportTo(app, dialog, 'coco')
  await expect(dialog.locator('.export-lead')).toContainText('Exported 5 images and 15 objects')
  await page.screenshot({ path: `${SHOTS}/export-done-dark-1500.png` })
  await dialog.getByRole('button', { name: 'Show in folder' }).click()
  await expect.poll(() => app.evaluate(() => (globalThis as any).revealed)).toEqual([coco])
  await closeDialog(dialog)
  expect(files(coco)).toEqual([
    'annotations/instances_test.json', 'annotations/instances_train.json', 'annotations/instances_val.json',
    'images/test/f.png', 'images/train/a.png', 'images/train/sub/b.png', 'images/val/c.png', 'images/val/d.png'
  ])
  const train = json(coco, 'annotations/instances_train.json')
  expect(train.categories.map((c: any) => c.name)).toEqual(NAMES)
  expect(train.images.map((i: any) => i.file_name)).toEqual(['images/train/a.png', 'images/train/sub/b.png'])
  expect(train.annotations.map((a: any) => train.categories[a.category_id - 1].name)).toEqual(['cat', 'dog', 'car', 'ball', 'car', 'ball', 'cat', 'dog', 'car'])
  // The two polygons of group 3 are one object of two parts, and the mask one of its own; group 5's part past the edge is kept inside the image
  expect(train.annotations[7].segmentation.length).toBe(2)
  expect(typeof train.annotations[6].segmentation.counts).toBe('string')
  expect(train.annotations[8].segmentation[1]).toEqual([400, 151, 390, 109, 395, 280])
  expect(json(coco, 'annotations/instances_test.json').annotations.length).toBe(4)
  expect(existsSync(path.join(DIR, 'inside'))).toBe(false)

  // COCO with keypoints: a keypoints file made from the folder's posed objects
  dialog = await openExport(page)
  await choose(dialog, 'COCO with keypoints')
  await expect(dialog.locator('.export-group.is-error')).toContainText('No keypoints file')
  const keypoints = path.join(OUT, 'keypoints.yaml')
  await answer(app, 'saves', keypoints)
  await change(dialog, () => dialog.getByRole('button', { name: 'Make a keypoints file…' }).click())
  await expect(dialog.locator('.export-options')).toContainText('person: 3 keypoints · flip pairs')
  expect(readFileSync(keypoints, 'utf8')).toContain('classes:\n  person:\n    - nose\n    - left_eye\n    - right_eye\n')
  const posed = await exportTo(app, dialog, 'coco-keypoints')
  await closeDialog(dialog)
  const people = json(posed, 'annotations/person_keypoints_val.json')
  expect(people.categories).toEqual([{ id: 5, name: 'person', supercategory: '', keypoints: ['nose', 'left_eye', 'right_eye'], skeleton: [] }])
  expect(people.annotations.map((a: any) => [a.keypoints, a.num_keypoints])).toEqual([[[150, 80, 2, 140, 70, 2, 160, 70, 1], 3], [[300, 90, 2, 0, 0, 0, 310, 80, 2], 2]])
  expect(json(posed, 'annotations/instances_val.json').annotations.length).toBe(2)

  // YOLO segment, in light mode: the parts of an object are one outline
  await page.getByRole('menuitem', { name: 'View' }).click()
  await page.getByRole('menuitemcheckbox', { name: 'Dark mode' }).click()
  dialog = await openExport(page)
  await choose(dialog, 'YOLO segment')
  await expect(dialog.locator('.export-group', { hasText: 'Objects of several parts joined' })).toContainText('3')
  await dialog.locator('.export-group-head').first().click()
  await page.screenshot({ path: `${SHOTS}/export-check-light-1500.png` })
  const segment = await exportTo(app, dialog, 'yolo-segment')
  await page.screenshot({ path: `${SHOTS}/export-done-light-1500.png` })
  await closeDialog(dialog)
  expect(files(segment).filter((f) => !f.startsWith('images/'))).toEqual([
    'data.yaml', 'labels/test/f.txt', 'labels/train/a.txt', 'labels/train/sub/b.txt', 'labels/val/c.txt', 'labels/val/d.txt'
  ])
  expect(readFileSync(path.join(segment, 'data.yaml'), 'utf8')).toContain('train: images/train\nval: images/val\ntest: images/test\n')
  expect(rows(segment, 'labels/train/sub/b.txt').map((r) => r.split(' ')[0])).toEqual(['0', '1', '2'])
  expect(rows(segment, 'labels/val/d.txt')).toEqual([])
  expectYolo(segment, 'segment', NAMES, {}, {
    'images/train/sub/b.png': [{ class: 0, parts: maskOutlines(MASK).outer }, { class: 1, parts: DOG_GROUP }, { class: 2, parts: CAR_GROUP }]
  })

  // YOLO detect in a small window, still light
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1000, 640))
  await page.setViewportSize({ width: 1000, height: 640 }).catch(() => {})
  await page.waitForTimeout(400)
  dialog = await openExport(page)
  await choose(dialog, 'YOLO detect')
  await dialog.locator('.export-group-head').first().click()
  await page.screenshot({ path: `${SHOTS}/export-check-light-1000.png` })
  await fits(page)
  const detect = await exportTo(app, dialog, 'yolo-detect')
  await page.screenshot({ path: `${SHOTS}/export-done-light-1000.png` })
  await fits(page)
  await closeDialog(dialog)
  expect(rows(detect, 'labels/train/a.txt').length).toBe(6)
  expect(rows(detect, 'labels/train/sub/b.txt')).toEqual([
    '0 0.150000 0.183333 0.100000 0.100000', '1 0.675000 0.266667 0.350000 0.200000', '2 0.875000 0.648333 0.250000 0.570000'
  ])
  expectYolo(detect, 'detect', NAMES)

  // YOLO OBB, dark again
  await page.getByRole('menuitem', { name: 'View' }).click()
  await page.getByRole('menuitemcheckbox', { name: 'Dark mode' }).click()
  dialog = await openExport(page)
  await choose(dialog, 'YOLO OBB')
  await dialog.locator('.export-group-head').first().click()
  await page.screenshot({ path: `${SHOTS}/export-check-dark-1000.png` })
  await fits(page)
  const obb = await exportTo(app, dialog, 'yolo-obb')
  await page.screenshot({ path: `${SHOTS}/export-done-dark-1000.png` })
  await closeDialog(dialog)
  expect(rows(obb, 'labels/train/a.txt')[2]).toBe('2 0.750000 0.200000 0.850000 0.066667 0.950000 0.200000 0.850000 0.333333')
  expectYolo(obb, 'obb', NAMES)

  // YOLO pose with the keypoints file chosen, hard links where they can be made, in the smallest window
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 600))
  await page.setViewportSize({ width: 960, height: 600 }).catch(() => {})
  await page.waitForTimeout(400)
  dialog = await openExport(page)
  await fits(page)
  await choose(dialog, 'YOLO pose')
  await answer(app, 'opens', keypoints)
  await change(dialog, () => dialog.getByRole('button', { name: 'Choose…' }).nth(1).click())
  await expect(dialog.locator('.export-options')).toContainText('person: 3 keypoints')
  await dialog.getByRole('radio', { name: 'Hard link' }).click()
  const pose = await exportTo(app, dialog, 'yolo-pose')
  await expect(dialog.locator('.export-done')).toContainText('hard-linked')
  await closeDialog(dialog)
  expect(rows(pose, 'labels/val/c.txt')).toEqual([
    '0 0.375000 0.500000 0.250000 0.666667 0.375000 0.266667 2 0.350000 0.233333 2 0.400000 0.233333 1',
    '0 0.750000 0.533333 0.250000 0.666667 0.750000 0.300000 2 0 0 0 0.775000 0.266667 2'
  ])
  expect(readFileSync(path.join(pose, 'data.yaml'), 'utf8')).toContain('kpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n')
  expectYolo(pose, 'pose', ['person'], { kpt_shape: [3, 3], flip_idx: [0, 2, 1] })

  // An export that fails says why, and still does once the window has read the folder and checked it again
  dialog = await openExport(page)
  await choose(dialog, 'YOLO detect')
  const taken = path.join(OUT, 'taken')
  await answer(app, 'opens', taken)
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.export-file').first()).toContainText('taken')
  // Something else saves a file there meanwhile
  mkdirSync(taken)
  writeFileSync(path.join(taken, 'notes.txt'), 'mine')
  await change(dialog, () => dialog.getByRole('button', { name: /^Export \d/ }).click())
  await expect(dialog.getByRole('alert')).toHaveText("That folder isn't empty: choose a new or empty folder, so nothing in it is written over")
  await expect(dialog.locator('.export-file').first()).toContainText('A new or empty folder')
  expect(readdirSync(taken)).toEqual(['notes.txt'])

  // A check that fails leaves nothing to export from: the last check's counts go, and Export waits for one that works
  await answer(app, 'opens', path.join(OUT, 'unused'))
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.getByRole('alert')).toHaveCount(0)
  await expect(dialog.getByRole('button', { name: /^Export \d/ })).toBeEnabled()
  await app.evaluate(({ ipcMain }) => {
    ipcMain.removeHandler('export:check')
    ipcMain.handle('export:check', () => {
      throw new Error('The folder went away')
    })
  })
  await change(dialog, () => dialog.getByLabel('Include never-labelled images as backgrounds').click())
  await expect(dialog.getByRole('alert')).toHaveText('The folder went away')
  await expect(dialog.locator('.export-lead')).toHaveText('Not checked')
  await expect(dialog.locator('.export-table')).toHaveCount(0)
  await expect(dialog.getByRole('button', { name: 'Export', exact: true })).toBeDisabled()
  await page.screenshot({ path: `${SHOTS}/export-failed-dark-960.png` })
  await fits(page)
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  expect(existsSync(path.join(OUT, 'unused'))).toBe(false)

  // Nothing was written in the image folder
  expect(files(DIR).filter((f) => !/\.(png|json)$/.test(f))).toEqual([])
  await finish(app, errors)
})

/** A file dropped on the element, as the system's drag and drop gives it */
async function drop(page: Page, file: string, on: string) {
  await page.evaluate(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.id = 'dropped'
    input.hidden = true
    document.body.append(input)
  })
  await page.setInputFiles('#dropped', file)
  await page.evaluate((selector) => {
    const data = new DataTransfer()
    data.items.add((document.getElementById('dropped') as HTMLInputElement).files![0])
    const target = document.querySelector(selector)!
    target.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: data }))
    target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: data }))
    document.getElementById('dropped')!.remove()
  }, on)
}
const holds = (page: Page) => page.evaluate(() => !!document.querySelector('.export-dialog')?.contains(document.activeElement))

test('the Export window holds the focus through an export: Tab goes round it, Escape closes its summary, and a file dropped on it opens nothing', async () => {
  const dir = await project('export-held', { a: doc('a', [square('cat', 20, 20, 120, 120)]), b: doc('b', [square('dog', 30, 30, 90, 90)]) })
  const other = await project('export-held-other', { x: doc('x', [square('cow', 10, 10, 50, 50)]), y: null })
  const out = path.join(OUT, 'held')
  rmSync(out, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  // The export starts a moment late, so the window can be seen running
  await app.evaluate(({ ipcMain }) => {
    const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: unknown[]) => unknown>
    const run = handlers.get('export:run')!
    handlers.set('export:run', async (...args: unknown[]) => {
      await new Promise((done) => setTimeout(done, 1500))
      return run(...args)
    })
  })
  const dialog = await openExport(page)
  await expect(dialog).toHaveAttribute('aria-modal', 'true')
  expect(await holds(page)).toBe(true)
  // Tab goes round: from the last control to the first, and back with Shift+Tab
  const cancel = dialog.getByRole('button', { name: 'Cancel' })
  await cancel.focus()
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('button', { name: 'Close' })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(cancel).toBeFocused()
  for (let k = 0; k < 20; k++) {
    await page.keyboard.press('Tab')
    expect(await holds(page)).toBe(true)
  }
  // A file dropped on the window leaves the folder open behind it as it is
  const top = (await page.locator('.topbar .muted').first().textContent())!
  await drop(page, path.join(other, 'x.png'), '.export-dialog')
  await page.waitForTimeout(500)
  await expect(page.locator('.topbar .muted').first()).toHaveText(top)
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png'])

  await answer(app, 'opens', out)
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  const go = dialog.getByRole('button', { name: 'Export 2 images' })
  await expect(go).toBeEnabled()
  await go.focus()
  await page.keyboard.press('Enter')
  // Export gives way to Stop: the window keeps the focus, Stop is a Shift+Tab away, and Ctrl+W waits
  const stop = dialog.getByRole('button', { name: /Stop/ })
  await expect(stop).toBeVisible()
  await expect.poll(() => holds(page)).toBe(true)
  await page.screenshot({ path: `${SHOTS}/export-running-dark-1500.png` })
  await page.keyboard.press('Shift+Tab')
  await expect(stop).toBeFocused()
  await page.keyboard.press('ControlOrMeta+w')
  await page.keyboard.press('Escape')
  await expect(dialog.locator('.export-done')).toBeVisible({ timeout: 60000 })
  await expect(dialog.locator('.export-lead')).toContainText('Exported 2 images and 2 objects')
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png'])
  // The summary holds the focus too, not on its Close where Stop was, so a late press on Stop closes nothing; Escape closes it
  expect(await holds(page)).toBe(true)
  await expect(dialog.locator('.actions').getByRole('button', { name: 'Close' })).not.toBeFocused()
  await page.screenshot({ path: `${SHOTS}/export-held-done-dark-1500.png` })
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  expect(files(out)).toEqual(['annotations/instances_default.json', 'images/default/a.png', 'images/default/b.png'])

  // At the smallest window and a large one, dark and light, it opens holding the focus, and Escape closes it
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1500, 940], [960, 600]]) {
      await app.evaluate(({ BrowserWindow }, [w, h]) => BrowserWindow.getAllWindows()[0].setSize(w, h), [width, height])
      await page.setViewportSize({ width, height }).catch(() => {})
      await page.waitForTimeout(400)
      const opened = await openExport(page)
      expect(await holds(page)).toBe(true)
      await fits(page)
      await page.screenshot({ path: `${SHOTS}/export-held-${theme}-${width}.png` })
      await page.keyboard.press('Escape')
      await expect(opened).toHaveCount(0)
    }
    if (theme === 'dark') {
      await page.getByRole('menuitem', { name: 'View' }).click()
      await page.getByRole('menuitemcheckbox', { name: 'Dark mode' }).click()
    }
  }
  await finish(app, errors)
})
