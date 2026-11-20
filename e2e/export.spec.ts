import { type ElectronApplication, expect, type Locator, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { maskOutlines } from '../src/shared/ultralytics'
import { contrast } from './contrast'
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
  // The error comes first in the column, above the counts, with the button that fixes it
  await expect(dialog.locator('.export-check > div > :first-child')).toHaveClass('export-errors')
  await expect(error.getByRole('button', { name: 'Leave them out' })).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/export-check-dark-1500.png` })
  await fits(page)

  // Left out, the proposal goes; the folders decide the split; a folder inside the images is refused
  await change(dialog, () => dialog.getByLabel('Leave unaccepted AI proposals out').check())
  await change(dialog, () => dialog.getByRole('radio', { name: 'Keep the folders' }).click())
  await expect(dialog.locator('.export-group.is-error')).toHaveCount(0)
  await expect(dialog.locator('.export-table thead')).toContainText('trainvaltestAll')
  await expect(dialog.locator('.export-table tr', { hasText: 'Images' })).toContainText('2215')
  await expect(dialog.locator('.export-options')).toContainText(
    "An image's first train, val, valid, validation or test folder decides its split, and moves to the front of its path: a/val/b.jpg goes to images/val/a/b.jpg. Images under none of them go to train.")
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

  // COCO with keypoints: the folder's own, COCO's person as it has no keypoints file, then a file made from its posed objects
  dialog = await openExport(page)
  await choose(dialog, 'COCO with keypoints')
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await expect(dialog.locator('.export-options')).toContainText("COCO's person, built in")
  await expect(dialog.locator('.export-options')).toContainText('person: 17 keypoints · flip pairs · skeleton')
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

test('the Export window holds the focus through an export: Tab goes round it, a choice is one stop, Escape stops an export and closes its summary, and a file dropped on it opens nothing', async () => {
  const dir = await project('export-held', { a: doc('a', [square('cat', 20, 20, 120, 120)]), b: doc('b', [square('dog', 30, 30, 90, 90)]) })
  const other = await project('export-held-other', { x: doc('x', [square('cow', 10, 10, 50, 50)]), y: null })
  const out = path.join(OUT, 'held')
  rmSync(out, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  // The images are copied slowly while the test says so, so the window can be seen running, and stopped
  await app.evaluate(() => {
    const fs = process.mainModule!.require('fs').promises
    const g = globalThis as any
    const copyFile = fs.copyFile
    g.slow = 0
    fs.copyFile = async (...args: unknown[]) => {
      if (g.slow) await new Promise((done) => setTimeout(done, g.slow))
      return copyFile.apply(fs, args)
    }
  })
  const slow = (ms: number) => app.evaluate((_e, v) => void ((globalThis as any).slow = v), ms)
  const dialog = await openExport(page)
  await expect(dialog).toHaveAttribute('aria-modal', 'true')
  expect(await holds(page)).toBe(true)
  // No close button of its own in the title, as the app's other windows have none
  await expect(dialog.getByRole('button', { name: 'Close' })).toHaveCount(0)
  // Tab goes round: from the last control to the first, the chosen format, and back with Shift+Tab
  const cancel = dialog.getByRole('button', { name: 'Cancel' })
  await cancel.focus()
  await page.keyboard.press('Tab')
  await expect(dialog.locator('input[name="export-format"]:checked')).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(cancel).toBeFocused()
  for (let k = 0; k < 20; k++) {
    await page.keyboard.press('Tab')
    expect(await holds(page)).toBe(true)
  }
  // Split and Images are a Tab stop each, at the choice made, as the formats are, and the arrow keys move the choice round them
  const split = dialog.getByRole('radiogroup', { name: 'Split' }), kept = dialog.getByRole('radiogroup', { name: 'Images' })
  const [none, ratio] = [split.getByRole('radio', { name: 'None' }), split.getByRole('radio', { name: 'By ratio' })]
  const [copy, link, leave] = ['Copy', 'Hard link', 'Leave in place'].map((name) => kept.getByRole('radio', { name }))
  await dialog.getByRole('button', { name: 'Choose…' }).first().focus()
  for (const stop of [none, copy, dialog.getByLabel('Include never-labelled images as backgrounds')]) {
    await page.keyboard.press('Tab')
    await expect(stop).toBeFocused()
  }
  await page.keyboard.press('Shift+Tab')
  await page.keyboard.press('ArrowRight')
  await expect(link).toBeFocused()
  await expect(link).toHaveAttribute('aria-checked', 'true')
  await page.keyboard.press('ArrowLeft')
  // Round from the first to the last
  await change(dialog, () => page.keyboard.press('ArrowLeft'))
  await expect(leave).toBeFocused()
  await expect(leave).toHaveAttribute('aria-checked', 'true')
  await change(dialog, () => page.keyboard.press('ArrowDown'))
  await expect(copy).toHaveAttribute('aria-checked', 'true')
  await page.keyboard.press('Shift+Tab')
  await expect(none).toBeFocused()
  await change(dialog, () => page.keyboard.press('ArrowDown'))
  await expect(ratio).toBeFocused()
  await expect(ratio).toHaveAttribute('aria-checked', 'true')
  await change(dialog, () => page.keyboard.press('ArrowUp'))
  await expect(none).toHaveAttribute('aria-checked', 'true')
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
  await slow(3000)
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
  // Escape stops it, as Stop does, and as it stops an import: the window stays, and nothing is written
  await page.keyboard.press('Escape')
  await expect(dialog.locator('.export-note')).toHaveText('Stopped: nothing was written.', { timeout: 30000 })
  expect(existsSync(out)).toBe(false)
  expect(await holds(page)).toBe(true)
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png'])
  await slow(0)
  await go.click()
  await expect(dialog.locator('.export-done')).toBeVisible({ timeout: 60000 })
  await expect(dialog.locator('.export-lead')).toContainText('Exported 2 images and 2 objects')
  // The summary holds the focus too, not on its Close where Stop was, so a late press on Stop closes nothing
  expect(await holds(page)).toBe(true)
  await expect(dialog.locator('.actions').getByRole('button', { name: 'Close' })).not.toBeFocused()
  await page.screenshot({ path: `${SHOTS}/export-held-done-dark-1500.png` })
  // Escape closes it, and the focus goes to the image, as the menu item that opened the window has gone
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(page.getByTestId('canvas')).toBeFocused()
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

/** Whether the element lies wholly inside the scrolling column's visible part */
async function inView(column: Locator, part: Locator) {
  const [c, r] = [await column.boundingBox(), await part.boundingBox()]
  return !!c && !!r && r.y >= c.y - 0.5 && r.y + r.height <= c.y + c.height + 0.5
}
async function resize(app: ElectronApplication, page: Page, width: number, height: number) {
  await app.evaluate(({ BrowserWindow }, [w, h]) => BrowserWindow.getAllWindows()[0].setSize(w, h), [width, height])
  await page.setViewportSize({ width, height }).catch(() => {})
  await page.waitForTimeout(400)
}

test('in the smallest window an error that stops the export shows first, with the button that fixes it, and the footer goes back to it', async () => {
  // Fourteen labels, whose counts once pushed the error out of view, an AI proposal not yet accepted, a line COCO can't hold, and a posed object
  const LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
  const files: Record<string, string> = {}
  for (let i = 0; i < 10; i++) {
    const shapes: unknown[] = [0, 1, 2].map((k) => square(LABELS[(i * 3 + k) % LABELS.length], 10 + k * 100, 10, 90 + k * 100, 90))
    if (i === 4) shapes.push(square('person', 20, 150, 120, 250, 0.8))
    if (i === 2) shapes.push(shape('cat', 'line', [[10, 280], [100, 280]]))
    if (i === 8) shapes.push(shape('person', 'rectangle', [[100, 120], [200, 280]], { group_id: 1 }), shape('nose', 'point', [[150, 150]], { group_id: 1 }), shape('left_eye', 'point', [[140, 140]], { group_id: 1 }))
    files[`img${i}`] = doc(`img${i}`, shapes)
  }
  const dir = await project('export-small', files)
  mkdirSync(OUT, { recursive: true })
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  await resize(app, page, 960, 600)
  const dialog = await openExport(page)
  const column = dialog.locator('.export-check')
  const error = dialog.locator('.export-errors .export-group')
  await expect(error).toHaveCount(1)
  await expect(error.locator('.export-group-head')).toContainText('AI proposals not yet accepted')
  // The error, its message and what fixes it are in view, above the counts
  const head = error.locator('.export-group-head'), leave = error.getByRole('button', { name: 'Leave them out' })
  for (const part of [head, error.locator('.export-rows li').first(), leave]) expect(await inView(column, part)).toBe(true)
  await expect(dialog.getByRole('button', { name: 'Show all 14 labels' })).toBeAttached()
  // The footer says what's wrong as a button that goes to it, and the Export button points at it
  const status = dialog.getByRole('button', { name: 'Fix the error first' })
  await expect(dialog.getByRole('button', { name: 'Export', exact: true })).toHaveAttribute('aria-describedby', 'export-why')
  await expect(dialog.locator('#export-why')).toHaveText('Fix the error first')
  await page.screenshot({ path: `${SHOTS}/export-error-dark-960.png` })
  await column.evaluate((el) => { el.scrollTop = el.scrollHeight })
  expect(await inView(column, head)).toBe(false)
  await status.click()
  await expect(head).toBeFocused()
  expect(await inView(column, head)).toBe(true)
  // A folder that can't be used as well: the footer names both
  await answer(app, 'opens', dir)
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.export-options .bad')).toBeVisible()
  await expect(dialog.locator('#export-why')).toHaveText('Fix the error and choose another folder')

  // The chosen format's words, and a warning's triangle, read in both themes
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    expect(await contrast(dialog.locator('.export-format.on small'))).toBeGreaterThanOrEqual(4.5)
    expect(await contrast(dialog.locator('.export-group.is-warning .export-sev'))).toBeGreaterThanOrEqual(3)
    expect(await contrast(dialog.locator('#export-why button'))).toBeGreaterThanOrEqual(4.5)
  }
  await page.screenshot({ path: `${SHOTS}/export-error-light-960.png` })
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark' })

  // Leave them out: the proposal goes, and the option it stands for is ticked
  await change(dialog, () => leave.click())
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await expect(dialog.getByLabel('Leave unaccepted AI proposals out')).toBeChecked()
  await expect(dialog.locator('#export-why')).toHaveText('Choose another folder')
  expect(await page.evaluate(() => !!document.querySelector('.export-dialog')?.contains(document.activeElement))).toBe(true)

  // YOLO pose starts from the folder's keypoints, COCO's person as it has no file, so nothing stops it; making a file is under them
  await change(dialog, () => dialog.locator('label.export-format', { hasText: 'YOLO pose' }).click())
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await expect(dialog.locator('.export-options')).toContainText('person: 17 keypoints · flip pairs · skeleton')
  const make = dialog.getByRole('button', { name: 'Make a keypoints file…' })
  await make.scrollIntoViewIfNeeded()
  await page.screenshot({ path: `${SHOTS}/export-keypoints-dark-960.png` })
  const keypoints = path.join(OUT, 'small-keypoints.yaml')
  rmSync(keypoints, { force: true })
  await answer(app, 'saves', keypoints)
  await change(dialog, () => make.click())
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await expect(dialog.locator('.export-options')).toContainText('person: 2 keypoints')
  // What was made, and what's left to do with it, is said under the Keypoints file, brought into view
  const made = dialog.locator('.export-options .export-hint', { hasText: 'small-keypoints.yaml lists' })
  await expect(made).toHaveText('small-keypoints.yaml lists the point labels found with each box in the folder: add any keypoint names that are missing, then choose the file again.')
  await expect.poll(() => inView(dialog.locator('.export-options'), made)).toBe(true)
  await page.screenshot({ path: `${SHOTS}/export-keypoints-made-dark-960.png` })
  // A format that takes no keypoints file says nothing of it
  await choose(dialog, 'YOLO detect')
  await expect(dialog.getByText('lists the point labels')).toHaveCount(0)
  await choose(dialog, 'YOLO pose')
  await expect(made).toBeVisible()

  // On a large screen the window grows past 860 px, so the posed formats' options need no scrolling
  await answer(app, 'opens', path.join(OUT, 'small'))
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.export-options .bad')).toHaveCount(0)
  await resize(app, page, 1920, 1080)
  const options = await dialog.locator('.export-options').evaluate((el) => [el.scrollHeight, el.clientHeight])
  expect(options[0]).toBeLessThanOrEqual(options[1])
  await page.screenshot({ path: `${SHOTS}/export-pose-dark-1920.png` })
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  await finish(app, errors)
})

test("a label file that can't be read stops the export, says how to clear it, and Check again reads it afresh once it's repaired", async () => {
  const dir = await project('export-broken', { a: doc('a', [square('cat', 20, 20, 120, 100)]), b: doc('b', [square('dog', 50, 50, 150, 150)]) })
  // b's label file cut off in the middle, as by a crash
  writeFileSync(path.join(dir, 'b.json'), doc('b', [square('dog', 50, 50, 150, 150)]).slice(0, 60))
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  await resize(app, page, 960, 600)
  const dialog = await openExport(page)
  const error = dialog.locator('.export-errors .export-group')
  const said = /^b\.png: its label file, b\.json, can't be read \(.+\), so its labels can't be exported: repair or delete the label file, then check again$/
  // b isn't counted, as a background or otherwise, while its labels can't be exported
  const backgrounds = dialog.locator('.export-table tr', { hasText: 'Backgrounds' }).locator('td').nth(1)
  for (const format of ['COCO', 'YOLO detect']) {
    if (format !== 'COCO') await choose(dialog, format)
    await expect(error).toHaveCount(1)
    await expect(error.locator('.export-group-head')).toContainText("Label files that can't be read")
    await expect(error.locator('.export-rows li')).toHaveText(said)
    await expect(error.getByRole('button', { name: 'Check again' })).toBeVisible()
    await expect(dialog.locator('#export-why')).toHaveText('Fix the error first')
    await expect(dialog.locator('.export-lead')).toHaveText('1 image and 1 object')
    await expect(backgrounds).toHaveText('0')
  }
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    for (const [width, height] of [[960, 600], [1500, 940]]) {
      await resize(app, page, width, height)
      await fits(page)
      await page.screenshot({ path: `${SHOTS}/export-broken-${theme}-${width}.png` })
    }
  }
  // Checked again before it's repaired, it still stops the export
  await change(dialog, () => error.getByRole('button', { name: 'Check again' }).click())
  await expect(error.locator('.export-rows li')).toHaveText(said)
  // Repaired in another app while the window is open: Check again reads it, and the export can go ahead
  writeFileSync(path.join(dir, 'b.json'), doc('b', [square('dog', 50, 50, 150, 150)]))
  await change(dialog, () => error.getByRole('button', { name: 'Check again' }).click())
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await expect(dialog.locator('.export-lead')).toHaveText('2 images and 2 objects')
  await expect(dialog.locator('#export-why')).toHaveText('Choose a folder to export to')
  expect(await page.evaluate(() => !!document.querySelector('.export-dialog')?.contains(document.activeElement))).toBe(true)
  await page.screenshot({ path: `${SHOTS}/export-broken-repaired-light-1500.png` })
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  await finish(app, errors)
})

test("an opened group's messages grow in the column, which alone scrolls, an error's fix stays under its title, and the summary's numbers end where its buttons do", async () => {
  // Twelve shapes past the image's bottom edge, a message each, and twelve label files cut off, an error each
  const files: Record<string, string> = { a: doc('a', Array.from({ length: 12 }, (_, k) => square('cat', 10 + k * 30, 250, 35 + k * 30, 340))) }
  for (let k = 0; k < 12; k++) files[`b${k}`] = doc(`b${k}`, [square('dog', 50, 50, 150, 150)]).slice(0, 60)
  const dir = await project('export-rows', files)
  const out = path.join(OUT, 'rows')
  rmSync(out, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  await resize(app, page, 960, 600)
  const dialog = await openExport(page)
  const column = dialog.locator('.export-check')
  const scrolls = (part: Locator) => part.evaluate((el) => el.scrollHeight > el.clientHeight + 1)
  // The error comes open, its messages in the column, with Check again under its title, in view
  const error = dialog.locator('.export-errors .export-group')
  await expect(error.locator('.export-rows li')).toHaveCount(12)
  expect(await scrolls(error.locator('.export-rows'))).toBe(false)
  expect(await scrolls(column)).toBe(true)
  expect(await inView(column, error.getByRole('button', { name: 'Check again' }))).toBe(true)
  await page.screenshot({ path: `${SHOTS}/export-rows-error-dark-960.png` })
  const group = dialog.locator('.export-group', { hasText: "Shapes past the image's edge" })
  await group.locator('.export-group-head').click()
  await expect(group.locator('.export-rows li')).toHaveCount(12)
  expect(await scrolls(group.locator('.export-rows'))).toBe(false)
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    await group.locator('.export-rows li').nth(4).scrollIntoViewIfNeeded()
    await page.screenshot({ path: `${SHOTS}/export-rows-${theme}-960.png` })
  }
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark' })

  // Repaired and checked again, the export goes ahead
  for (let k = 0; k < 12; k++) writeFileSync(path.join(dir, `b${k}.json`), doc(`b${k}`, [square('dog', 50, 50, 150, 150)]))
  await change(dialog, () => error.getByRole('button', { name: 'Check again' }).click())
  await expect(dialog.locator('.export-errors')).toHaveCount(0)
  await answer(app, 'opens', out)
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await dialog.getByRole('button', { name: 'Export 13 images' }).click()
  await expect(dialog.locator('.export-done')).toBeVisible({ timeout: 60000 })
  const right = (part: Locator) => part.evaluate((el) => Math.round(el.getBoundingClientRect().right))
  // Where the words or numbers themselves end
  const ends = (part: Locator) => part.evaluate((el) => {
    const words = document.createRange()
    words.selectNodeContents(el)
    return Math.round(words.getBoundingClientRect().right)
  })
  const close = dialog.locator('.actions').getByRole('button', { name: 'Close' })
  for (const [width, height] of [[960, 600], [1500, 940]]) {
    await resize(app, page, width, height)
    for (const part of ['.export-table', '.export-done .kv', '.export-accepted']) expect(await right(dialog.locator(part))).toBe(await right(close))
    for (const part of ['.export-table tbody tr:first-child td:last-child', '.export-done .kv b', '.export-accepted b']) expect(await ends(dialog.locator(part).first())).toBe(await right(close))
    for (const theme of ['dark', 'light']) {
      await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
      await page.screenshot({ path: `${SHOTS}/export-summary-${theme}-${width}.png` })
    }
  }
  await closeDialog(dialog)
  await finish(app, errors)
})

test("a pose export starts from the folder's own keypoints file; a file chosen instead can be put back, and one that can't be read stops it", async () => {
  const pose = [
    shape('person', 'rectangle', [[100, 50], [200, 250]], { group_id: 1 }), shape('nose', 'point', [[150, 80]], { group_id: 1 }),
    shape('left_eye', 'point', [[160, 70]], { group_id: 1 }), shape('right_eye', 'point', [[140, 70]], { group_id: 1, flags: { difficult: true } })
  ]
  const dir = await project('export-folder-keypoints', { a: doc('a', pose) })
  const own = path.join(dir, '.myseg', 'keypoints.yaml')
  mkdirSync(path.dirname(own), { recursive: true })
  writeFileSync(own, 'classes:\n  person: [nose, left_eye, right_eye]\nflip:\n  person: [[left_eye, right_eye]]\n')
  mkdirSync(OUT, { recursive: true })
  const other = path.join(OUT, 'other-keypoints.yaml')
  writeFileSync(other, 'classes:\n  person: [nose, tail]\n')
  const { app, page, errors } = await launch(dir)
  await stubDialogs(app)
  let dialog = await openExport(page)
  await choose(dialog, 'YOLO pose')
  const file = dialog.locator('.export-options .export-file').nth(1)
  await expect(file).toContainText("This folder's keypoints.yaml")
  await expect(dialog.locator('.export-options')).toContainText('person: 3 keypoints · flip pairs')
  // Another file, then the folder's again
  await answer(app, 'opens', other)
  await change(dialog, () => file.getByRole('button', { name: 'Choose…' }).click())
  await expect(file).toContainText('other-keypoints.yaml')
  await expect(dialog.locator('.export-options')).toContainText('person: 2 keypoints')
  await change(dialog, () => file.getByRole('button', { name: "This folder's" }).click())
  await expect(file).toContainText("This folder's keypoints.yaml")
  await expect(file.getByRole('button', { name: "This folder's" })).toHaveCount(0)
  const out = await exportTo(app, dialog, 'folder-pose')
  await closeDialog(dialog)
  expect(rows(out, 'labels/a.txt')).toEqual(['0 0.375000 0.500000 0.250000 0.666667 0.375000 0.266667 2 0.400000 0.233333 2 0.350000 0.233333 1'])
  expect(readFileSync(path.join(out, 'data.yaml'), 'utf8')).toContain('kpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\n')
  // A folder's file that can't be read stops the export, rather than COCO's person going in its place
  writeFileSync(own, 'classes: [nose\n')
  dialog = await openExport(page)
  await expect(file).toContainText("This folder's keypoints file")
  const error = dialog.locator('.export-errors .export-group')
  await expect(error.locator('.export-group-head')).toContainText("The keypoints file can't be read")
  await expect(error.locator('.export-rows li')).toHaveText(/^\.myseg\/keypoints\.yaml: The keypoints file isn't valid YAML/)
  await expect(error.getByRole('button', { name: 'Choose another…' })).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/export-folder-keypoints-unreadable-dark.png` })
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  await finish(app, errors)
})
