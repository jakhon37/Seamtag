import { type ElectronApplication, expect, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { BASE, doc, finish, H, launch, open, project, read, shapesIn, square, W } from './helpers'

// File > Import dataset: a COCO file, then a YOLO dataset, written into the open folder's label
// files, with the open image read again. Native dialogs and the trash are the test's own.
const SHOTS = process.env.E2E_SHOTS || path.join(BASE, 'import-shots')

/** A trash folder of the test's own, with a list of what went in */
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
/** The next file or folder dialog answers with this path */
const answer = (app: ElectronApplication, file: string) => app.evaluate(({ dialog }, f) => {
  dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [f] })) as any
}, file)
const png = async (file: string, width = W, height = H) => {
  mkdirSync(path.dirname(file), { recursive: true })
  await sharp({ create: { width, height, channels: 3, background: '#445566' } }).png().toFile(file)
}
const flat = (points: number[][]) => points.flat()
// The window's size, and the page's: launch() fixes the page's at 1500 x 940, which the window's alone doesn't change
async function size(app: ElectronApplication, page: Page, width: number, height: number) {
  await app.evaluate(({ BrowserWindow }, [w, h]) => BrowserWindow.getAllWindows()[0].setSize(w, h), [width, height])
  await page.setViewportSize({ width, height })
  await page.waitForTimeout(400)
}
async function menu(page: Page, top: string, item: string) {
  await page.getByRole('menuitem', { name: top, exact: true }).click()
  await page.getByRole('menuitem', { name: item }).or(page.getByRole('menuitemcheckbox', { name: item })).first().click()
}
// The window, its buttons included, is inside the page, and nothing makes the page scroll sideways
async function fitsIn(page: Page, width: number) {
  const at = await page.evaluate(() => {
    const box = document.querySelector('.dialog.import')!.getBoundingClientRect()
    const inside = [...document.querySelectorAll('.dialog.import .actions *')].every((el) => {
      const r = el.getBoundingClientRect()
      return r.left >= box.left - 0.5 && r.right <= box.right + 0.5 && r.bottom <= box.bottom + 0.5
    })
    return { inside, fits: box.top >= 0 && box.bottom <= innerHeight && box.right <= innerWidth, page: document.documentElement.scrollWidth <= innerWidth, width: innerWidth }
  })
  expect([at.inside, at.fits, at.page]).toEqual([true, true, true])
  expect(at.width).toBeLessThanOrEqual(width)
}

test('import a COCO file, then a YOLO dataset, into the open folder', async () => {
  const dir = await project('import', { a: null, b: null, c: doc('c', [square('cat', 20, 20, 120, 100)]), d: null, e: null, f: null })
  for (const sub of ['sub1', 'sub2']) await png(path.join(dir, sub, 'g.png'))
  // d has LabelImg's Pascal VOC labels
  writeFileSync(path.join(dir, 'd.xml'), '<annotation><folder>import</folder><filename>d.png</filename><size><width>400</width><height>300</height><depth>3</depth></size>' +
    '<object><name>cat</name><difficult>0</difficult><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>')
  // A COCO file outside the folder: e is turned the other way, other/g.png could be either g.png, z.png isn't here
  const src = path.join(BASE, 'import-src')
  rmSync(src, { recursive: true, force: true })
  mkdirSync(src, { recursive: true })
  const image = (id: number, file_name: string, width = W, height = H) => ({ id, file_name, width, height })
  const polygon = (id: number, image_id: number, category_id: number, points: number[][]) => ({ id, image_id, category_id, iscrowd: 0, area: 1, bbox: [0, 0, 1, 1], segmentation: [flat(points)] })
  // A 10 x 10 square at 10, 20, as pycocotools' runs down each column
  const counts = [3020, ...Array(9).fill([10, 290]).flat(), 10, 114270]
  const coco = path.join(src, 'instances.json')
  writeFileSync(coco, JSON.stringify({
    images: [image(1, 'a.png'), image(2, 'b.png'), image(3, 'c.png'), image(4, 'd.png'), image(5, 'e.png', H, W), image(6, 'other/g.png'), image(7, 'z.png')],
    categories: [{ id: 1, name: 'cat', supercategory: '' }, { id: 2, name: 'dog', supercategory: '' }],
    annotations: [
      polygon(1, 1, 2, [[50, 50], [150, 60], [120, 150]]), { ...polygon(2, 1, 9, [[1, 1], [9, 1], [5, 9]]) },
      { id: 3, image_id: 2, category_id: 1, iscrowd: 0, area: 4800, bbox: [200, 100, 80, 60], segmentation: [] },
      { id: 4, image_id: 2, category_id: 2, iscrowd: 0, area: 100, bbox: [10, 20, 10, 10], segmentation: { size: [H, W], counts } },
      polygon(5, 3, 2, [[250, 150], [350, 160], [300, 260]]), polygon(6, 4, 2, [[30, 200], [130, 210], [80, 280]]),
      polygon(7, 5, 1, [[5, 5], [50, 5], [20, 40]]), polygon(8, 6, 1, [[5, 5], [50, 5], [20, 40]]), polygon(9, 7, 1, [[5, 5], [50, 5], [20, 40]])
    ]
  }))
  const { app, page, errors } = await launch(dir, ['--labels', 'cat'])
  const trashed = await fakeTrash(app, path.join(BASE, 'import-trash'))
  const filter = (name: string) => page.locator('.filters button', { hasText: name })
  await open(page, 'c.png')
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(filter('Done')).toContainText('2')
  const dialog = page.getByRole('dialog', { name: 'Import dataset' })

  // The COCO file: read, nothing written yet
  await answer(app, coco)
  await menu(page, 'File', 'Import dataset…')
  await expect(dialog).toContainText('Nothing is written until you choose Import')
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog.locator('.import-source')).toContainText('instances.json')
  await expect(dialog).toContainText('5 of its 7 images are in this folder: 5 by their path.')
  await expect(dialog).toContainText("1 isn't here, and 1 has a name that matches more than one image: nothing is imported for them, as listed below.")
  await expect(dialog).toContainText("1 image can't take its labels, as listed below.")
  await expect(dialog).toContainText("2 images have labels already: they're left as they are.")
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' })).toContainText('2')
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' }).locator('.tag.new')).toBeVisible()
  for (const [code, n] of [['import-size', '1'], ['coco-image-missing', '1'], ['coco-image-ambiguous', '1'], ['coco-unknown-category', '1'], ['coco-box-only', '1']]) {
    await expect(dialog.locator(`.import-group[data-code="${code}"] .count`)).toHaveText(n)
  }
  await dialog.locator('.import-group[data-code="import-size"] > button').click()
  await expect(dialog.locator('.import-rows')).toContainText("EXIF orientation probably differs")
  await dialog.locator('.import-group[data-code="coco-image-ambiguous"] > button').click()
  await expect(dialog.locator('.import-rows').nth(1)).toContainText('sub1/g.png, sub2/g.png')
  expect(existsSync(path.join(dir, 'a.json'))).toBe(false)
  // The window starts dark
  await page.screenshot({ path: `${SHOTS}/import-coco-dark-1500.png` })
  await size(app, page, 1000, 640)
  await fitsIn(page, 1000)
  await page.screenshot({ path: `${SHOTS}/import-coco-dark-1000.png` })
  await size(app, page, 1500, 940)

  // Skip: the images without labels get them; c and d keep theirs
  await dialog.getByRole('button', { name: 'Import 2 images' }).click()
  await expect(dialog.locator('.import-result')).toHaveText('Wrote 2 label files with 3 shapes.')
  await expect(dialog).toContainText("dog isn't in the label list, so it shows under Labels as found in the label files.")
  await expect(dialog.locator('.import-group[data-code="labelled"] .count')).toHaveText('2')
  await dialog.locator('.import-group[data-code="import-size"] > button').click()
  await page.screenshot({ path: `${SHOTS}/import-report-dark-1500.png` })
  await size(app, page, 1000, 640)
  await page.screenshot({ path: `${SHOTS}/import-report-dark-1000.png` })
  await size(app, page, 1500, 940)
  expect(shapesIn(dir, 'a').map((s: any) => [s.label, s.shape_type, s.points])).toEqual([['dog', 'polygon', [[50, 50], [150, 60], [120, 150]]]])
  const b = shapesIn(dir, 'b')
  expect(b.map((s: any) => [s.label, s.shape_type, s.points])).toEqual([['cat', 'rectangle', [[200, 100], [280, 160]]], ['dog', 'mask', [[10, 20], [19, 29]]]])
  const mask = await sharp(Buffer.from(b[1].mask, 'base64')).raw().toBuffer({ resolveWithObject: true })
  expect([mask.info.width, mask.info.height, mask.data.every((v) => v > 0)]).toEqual([10, 10, true])
  expect([read(dir, 'a').imageWidth, read(dir, 'a').imageHeight, read(dir, 'a').imagePath]).toEqual([W, H, 'a.png'])
  expect(shapesIn(dir, 'c')).toHaveLength(1)
  expect([existsSync(path.join(dir, 'd.xml')), existsSync(path.join(dir, 'd.json')), existsSync(path.join(dir, 'e.json'))]).toEqual([true, false, false])
  await dialog.getByRole('button', { name: 'Close' }).click()
  await expect(dialog).toHaveCount(0)
  await expect(filter('Done')).toContainText('4')
  await expect(page.locator('.class', { hasText: 'dog' })).toBeVisible()

  // Add: the shapes a and b have already aren't added twice; c gets the dog; d becomes LabelMe JSON
  await answer(app, coco)
  await menu(page, 'File', 'Import dataset…')
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog).toContainText("4 images have labels already: they're left as they are.")
  await dialog.getByRole('radio', { name: 'Add shapes' }).click()
  await expect(dialog).toContainText("4 images have labels already: the imported shapes go after theirs, and shapes they have already aren't added twice.")
  await expect(dialog).toContainText('1 of them has Pascal VOC, YOLO or CreateML labels: it gets LabelMe JSON, which holds every shape type, with its boxes kept, and the old file goes to the trash.')
  await dialog.getByRole('button', { name: 'Import 4 images' }).click()
  await expect(dialog.locator('.import-result')).toHaveText('Wrote 2 label files with 2 shapes.')
  await expect(dialog).toContainText("3 shapes were there already, so they weren't added again.")
  await expect(dialog).toContainText('1 old label file went to the trash.')
  expect(shapesIn(dir, 'c').map((s: any) => [s.label, s.shape_type])).toEqual([['cat', 'polygon'], ['dog', 'polygon']])
  expect(shapesIn(dir, 'd').map((s: any) => [s.label, s.shape_type, s.points])).toEqual([['cat', 'rectangle', [[10, 10], [60, 50]]], ['dog', 'polygon', [[30, 200], [130, 210], [80, 280]]]])
  expect(shapesIn(dir, 'a')).toHaveLength(1)
  expect(await trashed()).toEqual([path.join(dir, 'd.xml')])
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  // The open image is read again, with the imported shape
  await expect(page.locator('.shape')).toHaveCount(2)
  await expect(page.locator('.shape', { hasText: 'dog' })).toBeVisible()

  // A YOLO dataset of rotated boxes: the rows fit two tasks, so the window asks
  const yolo = path.join(BASE, 'import-yolo')
  rmSync(yolo, { recursive: true, force: true })
  const turned = (cx: number, cy: number, hw: number, hh: number) => {
    const t = Math.PI / 6
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [cx + x * Math.cos(t) - y * Math.sin(t), cy + x * Math.sin(t) + y * Math.cos(t)])
  }
  const row = (cls: number, corners: number[][]) => `${cls} ${corners.map(([x, y]) => `${(x / W).toFixed(6)} ${(y / H).toFixed(6)}`).join(' ')}\n`
  for (const [name, text] of [['a', row(0, turned(100, 100, 40, 20))], ['c', row(1, turned(250, 150, 60, 30))]]) {
    await png(path.join(yolo, 'images', 'train', `${name}.png`))
    mkdirSync(path.join(yolo, 'labels', 'train'), { recursive: true })
    writeFileSync(path.join(yolo, 'labels', 'train', `${name}.txt`), text)
  }
  writeFileSync(path.join(yolo, 'data.yaml'), 'train: images/train\nval: images/train\nnames:\n  0: cat\n  1: dog\n')
  // Light from here on
  await menu(page, 'View', 'Dark mode')
  await answer(app, yolo)
  await menu(page, 'File', 'Import dataset…')
  await dialog.getByRole('button', { name: 'YOLO dataset folder…' }).click()
  await expect(dialog.locator('.import-source')).toContainText('import-yolo')
  await expect(dialog).toContainText('Both its images are in this folder: 2 by file name alone.')
  await expect(dialog).toContainText('These rows fit more than one task: click the one they were made for.')
  await expect(dialog.getByRole('radio', { name: 'OBB (guess)' })).toHaveAttribute('aria-checked', 'true')
  // Both have labels now, which Skip leaves alone; and the task comes first
  await expect(dialog.getByRole('button', { name: 'Import', exact: true })).toBeDisabled()
  await expect(dialog.locator('.import-sum')).toHaveText('Pick the task first.')
  await page.screenshot({ path: `${SHOTS}/import-yolo-light-1500.png` })
  await size(app, page, 1000, 640)
  await fitsIn(page, 1000)
  await page.screenshot({ path: `${SHOTS}/import-yolo-light-1000.png` })
  await size(app, page, 1500, 940)
  await dialog.getByRole('radio', { name: 'OBB (guess)' }).click()
  await dialog.getByRole('radio', { name: 'Replace' }).click()
  await expect(dialog).toContainText('2 images have labels already: their label files go to the trash, and the imported shapes take their place.')
  await dialog.getByRole('button', { name: 'Import 2 images' }).click()
  await expect(dialog.locator('.import-result')).toHaveText('Wrote 2 label files with 2 shapes.')
  await expect(dialog).toContainText('2 old label files went to the trash.')
  await page.screenshot({ path: `${SHOTS}/import-report-light-1500.png` })
  await size(app, page, 1000, 640)
  await page.screenshot({ path: `${SHOTS}/import-report-light-1000.png` })
  await size(app, page, 1500, 940)
  const c = shapesIn(dir, 'c')
  expect(c.map((s: any) => [s.label, s.shape_type])).toEqual([['dog', 'oriented_rectangle']])
  c[0].points.forEach(([x, y]: number[], i: number) => {
    const [ex, ey] = turned(250, 150, 60, 30)[i]
    expect(Math.abs(x - ex) + Math.abs(y - ey)).toBeLessThan(0.01)
  })
  expect(shapesIn(dir, 'a').map((s: any) => s.shape_type)).toEqual(['oriented_rectangle'])
  expect(await trashed()).toEqual(['d.xml', 'a.json', 'c.json'].map((f) => path.join(dir, f)))
  // The old c.json went to the trash whole: its cat and dog
  expect(JSON.parse(readFileSync(path.join(BASE, 'import-trash', '2-c.json'), 'utf8')).shapes).toHaveLength(2)
  await dialog.getByRole('button', { name: 'Close' }).click()
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(page.locator('.shape')).toContainText('dog')
  await expect(filter('Done')).toContainText('4')
  await finish(app, errors)
})

test("a COCO file too big for its reader's memory is refused plainly, and the window goes on", async () => {
  const dir = await project('import-big', { a: null })
  const file = path.join(BASE, 'import-src', 'big.json')
  mkdirSync(path.dirname(file), { recursive: true })
  const annotations = Array.from({ length: 60000 }, (_, i) => ({ id: i + 1, image_id: 1, category_id: 1, iscrowd: 0, area: 1, bbox: [0, 0, 1, 1], segmentation: [Array.from({ length: 40 }, (_, k) => k * 1.25)] }))
  writeFileSync(file, JSON.stringify({ images: [{ id: 1, file_name: 'a.png', width: W, height: H }], annotations, categories: [{ id: 1, name: 'cat' }] }))
  // The reader gets 64 MB, too little for the file read whole
  process.env.SEAMTAG_IMPORT_HEAP_MB = '64'
  const { app, page, errors } = await launch(dir).finally(() => { delete process.env.SEAMTAG_IMPORT_HEAP_MB })
  await answer(app, file)
  await menu(page, 'File', 'Import dataset…')
  const dialog = page.getByRole('dialog', { name: 'Import dataset' })
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog.locator('.bad')).toHaveText('big.json is too big to import here: reading it needs more than the 64 MB of memory its reader may use, so nothing is imported. A COCO file with fewer images reads in less.')
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await open(page, 'a.png')
  expect(existsSync(path.join(dir, 'a.json'))).toBe(false)
  await finish(app, errors)
})
