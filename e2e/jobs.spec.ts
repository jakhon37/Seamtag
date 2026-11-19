import { type ElectronApplication, expect, type Page, test } from '@playwright/test'
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import { BASE, doc, finish, H, launch, project, square, W } from './helpers'

// Jobs that write the folder's label files or an export: closing the window while one runs
// stops it and waits for it, and a rename counts its images again as it read them.

async function menu(page: Page, top: string, item: string) {
  await page.getByRole('menuitem', { name: top, exact: true }).click()
  await page.getByRole('menuitem', { name: item }).first().click()
}
/** The next file or folder dialog answers with this path */
const answer = (app: ElectronApplication, file: string) => app.evaluate(({ dialog }, f) => {
  dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [f] })) as any
}, file)
/** The window's close button, clicked twice, and the app gone */
async function closeTwice(app: ElectronApplication) {
  const closed = app.waitForEvent('close', { timeout: 60000 })
  await app.evaluate(({ BrowserWindow }) => {
    const [win] = BrowserWindow.getAllWindows()
    win.close()
    win.close()
  })
  await closed
}

/**
 * The trash as Electron's is on Linux, where it runs `gio trash` and waits for it: a process of
 * its own that moves the file after a moment, and goes on when the app has quit
 */
async function slowTrash(app: ElectronApplication, into: string) {
  rmSync(into, { recursive: true, force: true })
  mkdirSync(into, { recursive: true })
  await app.evaluate(({ shell }, into) => {
    const { spawn } = process.mainModule!.require('child_process')
    const path = process.mainModule!.require('path')
    const g = globalThis as any
    g.started = 0
    g.trashed = 0
    shell.trashItem = ((file: string) => new Promise<void>((resolve, reject) => {
      const to = path.join(into, `${g.started++}-${path.basename(file)}`)
      const move = 'setTimeout(() => require("fs").renameSync(process.argv[1], process.argv[2]), 300)'
      const child = spawn(process.execPath, ['-e', move, file, to], {
        detached: true, stdio: 'ignore', windowsHide: true, env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' }
      })
      child.on('exit', (code: number) => {
        if (code !== 0) return reject(new Error(`The trash failed with ${code}`))
        g.trashed++
        resolve()
      })
    })) as any
  }, into)
}

test('closing the window while an import replaces labels waits for the image it is on: none is left without its label file', async () => {
  const N = 40
  const files: Record<string, string> = {}
  for (let k = 0; k < N; k++) files[`i${k}`] = doc(`i${k}`, [square('cat', 20, 20, 120, 100)])
  const dir = await project('close-import', files)
  const coco = path.join(BASE, 'close-import.json')
  writeFileSync(coco, JSON.stringify({
    images: Array.from({ length: N }, (_, k) => ({ id: k + 1, file_name: `i${k}.png`, width: W, height: H })),
    categories: [{ id: 1, name: 'dog', supercategory: '' }],
    annotations: Array.from({ length: N }, (_, k) => ({
      id: k + 1, image_id: k + 1, category_id: 1, iscrowd: 0, area: 4500, bbox: [50, 50, 100, 100], segmentation: [[50, 50, 150, 60, 120, 150]]
    }))
  }))
  const { app, page } = await launch(dir)
  const bin = path.join(BASE, 'close-import-trash')
  await slowTrash(app, bin)
  await answer(app, coco)
  await menu(page, 'File', 'Import dataset…')
  const dialog = page.getByRole('dialog', { name: 'Import dataset' })
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog.locator('.import-source')).toContainText('close-import.json')
  await dialog.getByRole('radio', { name: 'Replace' }).click()
  await dialog.getByRole('button', { name: `Import ${N} images` }).click()
  // Closed as an old label file is on its way to the trash
  await expect.poll(() => app.evaluate(() => (globalThis as any).trashed), { timeout: 60000 }).toBeGreaterThan(2)
  await closeTwice(app)
  // A trash still under way has moved its file by now
  await new Promise((go) => setTimeout(go, 2000))
  const names = readdirSync(dir)
  expect(names.filter((n) => n.endsWith('.tmp'))).toEqual([])
  expect(Array.from({ length: N }, (_, k) => `i${k}.json`).filter((n) => !names.includes(n))).toEqual([])
  // It stopped part way: each image whose labels went to the trash has the imported ones
  const replaced = names.filter((n) => n.endsWith('.json') && JSON.parse(readFileSync(path.join(dir, n), 'utf8')).shapes[0].label === 'dog')
  expect(replaced.length).toBe(readdirSync(bin).length)
  expect(replaced.length).toBeLessThan(N)
})

test('closing the window during an export stops it and removes what it wrote', async () => {
  const files: Record<string, string> = {}
  for (let k = 0; k < 40; k++) files[`e${k}`] = doc(`e${k}`, [square('cat', 20, 20, 120, 100)])
  const dir = await project('close-export', files)
  const parent = path.join(BASE, 'close-export-out')
  rmSync(parent, { recursive: true, force: true })
  mkdirSync(parent, { recursive: true })
  const { app, page } = await launch(dir)
  await answer(app, path.join(parent, 'dataset'))
  // A slow drive: each image takes a moment to copy
  await app.evaluate(() => {
    const fsp = process.mainModule!.require('fs').promises
    const copy = fsp.copyFile.bind(fsp)
    const g = globalThis as any
    g.copies = 0
    fsp.copyFile = async (...args: unknown[]) => {
      await new Promise((go) => setTimeout(go, 100))
      await copy(...args)
      g.copies++
    }
  })
  await menu(page, 'File', 'Export dataset…')
  const dialog = page.getByRole('dialog', { name: 'Export dataset' })
  await expect(dialog.locator('.export-check')).toHaveAttribute('aria-busy', 'false', { timeout: 30000 })
  await dialog.getByRole('button', { name: 'Choose…' }).first().click()
  await expect(dialog.locator('.export-file').first()).toContainText('dataset')
  await dialog.getByRole('button', { name: /^Export \d/ }).click()
  await expect.poll(() => app.evaluate(() => (globalThis as any).copies), { timeout: 60000 }).toBeGreaterThan(5)
  await closeTwice(app)
  // Neither the dataset nor the hidden folder it was being made in
  expect(readdirSync(parent)).toEqual([])
})

test("a rename across the folder counts its images again in the window's save format", async () => {
  const voc = (boxes: number[][]) => `<annotation><folder>x</folder><size><width>${W}</width><height>${H}</height><depth>3</depth></size>` +
    boxes.map(([x0, y0, x1, y1]) => `<object><name>dog</name><difficult>0</difficult><bndbox><xmin>${x0}</xmin><ymin>${y0}</ymin><xmax>${x1}</xmax><ymax>${y1}</ymax></bndbox></object>`).join('') +
    '</annotation>'
  // a has LabelMe's file too, which the config's format would read first; 0 opens first, with no labels
  const dir = await project('rename-format', { '0': null, a: doc('a', [square('cat', 20, 20, 120, 100)]), b: null })
  writeFileSync(path.join(dir, 'a.xml'), voc([[10, 10, 50, 50], [60, 60, 120, 120]]))
  writeFileSync(path.join(dir, 'b.xml'), voc([[10, 10, 50, 50]]))
  const { app, page, errors } = await launch(dir)
  await page.getByLabel('Save format').selectOption('voc')
  await menu(page, 'Dataset', 'Statistics…')
  const stats = page.getByRole('dialog', { name: 'Statistics' })
  await stats.locator('.stats-labels tbody tr', { hasText: 'dog' }).getByRole('button', { name: 'Rename…', exact: true }).click()
  const rename = page.getByRole('dialog', { name: 'Rename a label' })
  await expect(rename).toContainText('Renames 3 shapes in 2 images')
  await rename.getByLabel('New name').fill('wolf')
  await rename.getByRole('button', { name: 'Rename 3 shapes' }).click()
  await expect(rename).toHaveCount(0)
  // The folder's count is of the files renamed: a's LabelMe file isn't counted in place of its VOC file
  await expect(page.locator('.class', { hasText: /^wolf/ }).locator('.muted')).toHaveText('0 · 3')
  expect(readFileSync(path.join(dir, 'a.xml'), 'utf8')).toContain('<name>wolf</name>')
  await stats.getByRole('button', { name: 'Close' }).click()
  await finish(app, errors)
})
