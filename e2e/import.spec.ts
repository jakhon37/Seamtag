import { type ElectronApplication, expect, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { contrast } from './contrast'
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
  // A COCO file outside the folder: e is turned the other way, f has no annotations, other/g.png could be either g.png, z.png isn't here
  const src = path.join(BASE, 'import-src')
  rmSync(src, { recursive: true, force: true })
  mkdirSync(src, { recursive: true })
  const image = (id: number, file_name: string, width = W, height = H) => ({ id, file_name, width, height })
  const polygon = (id: number, image_id: number, category_id: number, points: number[][]) => ({ id, image_id, category_id, iscrowd: 0, area: 1, bbox: [0, 0, 1, 1], segmentation: [flat(points)] })
  // A 10 x 10 square at 10, 20, as pycocotools' runs down each column
  const counts = [3020, ...Array(9).fill([10, 290]).flat(), 10, 114270]
  const coco = path.join(src, 'instances.json')
  writeFileSync(coco, JSON.stringify({
    images: [image(1, 'a.png'), image(2, 'b.png'), image(3, 'c.png'), image(4, 'd.png'), image(5, 'e.png', H, W), image(6, 'other/g.png'), image(7, 'z.png'), image(8, 'f.png')],
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
  await expect(dialog).toContainText('6 of its 8 images are in this folder: 6 by their path.')
  await expect(dialog).toContainText("1 isn't here, and 1 has a name that matches more than one image: nothing is imported for them, as listed below.")
  await expect(dialog).toContainText("1 image can't take its labels, as listed below.")
  await expect(dialog).toContainText("1 image gets nothing from it, so it's left as it is.")
  await expect(dialog).toContainText("2 images have labels already: they're left as they are.")
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' })).toContainText('2')
  // dog is in neither the label list nor a label file here; cat is in both
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' }).locator('.tag.new')).toBeVisible()
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' }).locator('.tag.new')).toHaveAttribute('title', "Not in the label list or this folder's label files yet")
  await expect(dialog.locator('.import-labels tr', { hasText: 'cat' }).locator('.tag.new')).toHaveCount(0)
  for (const [code, n] of [['import-size', '1'], ['coco-image-missing', '1'], ['coco-image-ambiguous', '1'], ['coco-unknown-category', '1'], ['coco-box-only', '1'], ['import-no-annotations', '1']]) {
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
  await expect(dialog).toContainText("“dog” isn't in the label list, so it shows under Labels as found in the label files.")
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
  // f is listed without annotations: it isn't given an empty label file
  expect(['d.xml', 'd.json', 'e.json', 'f.json'].map((f) => existsSync(path.join(dir, f)))).toEqual([true, false, false, false])
  await dialog.getByRole('button', { name: 'Close' }).click()
  await expect(dialog).toHaveCount(0)
  await expect(filter('Done')).toContainText('4')
  await expect(page.locator('.class', { hasText: 'dog' })).toBeVisible()

  // Add: a and b have every shape already, so they're left as they are and not counted; c gets the dog; d becomes LabelMe JSON
  await answer(app, coco)
  await menu(page, 'File', 'Import dataset…')
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog).toContainText("4 images have labels already: they're left as they are.")
  await expect(dialog.locator('.import-section h4', { hasText: 'Shapes to import' })).toHaveText('Shapes to import')
  await dialog.getByRole('radio', { name: 'Add shapes' }).click()
  await expect(dialog).toContainText("4 images have labels already: 2 get the imported shapes they don't have yet, after their own; 2 have every shape already, so they're left as they are.")
  await expect(dialog).toContainText('1 of them has Pascal VOC, YOLO or CreateML labels: it gets LabelMe JSON, which holds every shape type, with its boxes kept, and the old file goes to the trash.')
  await expect(dialog.locator('.import-section h4', { hasText: 'Shapes to import' })).toHaveText('Shapes to import · 2')
  await expect(dialog.locator('.import-labels tr', { hasText: 'dog' })).toContainText('2')
  await expect(dialog.locator('.import-sum')).toHaveText('2 images get a LabelMe JSON label file.')
  // Replace writes all four, every shape the file has for them
  await dialog.getByRole('radio', { name: 'Replace' }).click()
  await expect(dialog.locator('.import-section h4', { hasText: 'Shapes to import' })).toHaveText('Shapes to import · 5')
  await expect(dialog.getByRole('button', { name: 'Import 4 images' })).toBeVisible()
  await dialog.getByRole('radio', { name: 'Add shapes' }).click()
  await dialog.getByRole('button', { name: 'Import 2 images' }).click()
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

  // A YOLO dataset of rotated boxes: the rows fit two tasks, so the window asks. b's label file is empty
  const yolo = path.join(BASE, 'import-yolo')
  rmSync(yolo, { recursive: true, force: true })
  const turned = (cx: number, cy: number, hw: number, hh: number) => {
    const t = Math.PI / 6
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [cx + x * Math.cos(t) - y * Math.sin(t), cy + x * Math.sin(t) + y * Math.cos(t)])
  }
  const row = (cls: number, corners: number[][]) => `${cls} ${corners.map(([x, y]) => `${(x / W).toFixed(6)} ${(y / H).toFixed(6)}`).join(' ')}\n`
  for (const [name, text] of [['a', row(0, turned(100, 100, 40, 20))], ['b', ''], ['c', row(1, turned(250, 150, 60, 30))]]) {
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
  await expect(dialog).toContainText('All 3 of its images are in this folder: 3 by file name alone.')
  await expect(dialog).toContainText('Every row is the 4 corners of a rectangle, and 2 are rotated: boxes for OBB, or outlines of 4 points for Segment.')
  await expect(dialog).toContainText('These rows fit OBB and Segment: click the one they were made for.')
  // The two the rows fit stand out from the two they don't
  await expect(dialog.getByRole('radiogroup', { name: 'Task' }).locator('.fits')).toHaveText(['Segment', 'OBB (guess)'])
  // No task is chosen while the window asks: the guess is only marked as one
  const tasks = dialog.getByRole('radiogroup', { name: 'Task' })
  await expect(tasks.getByRole('radio', { checked: true })).toHaveCount(0)
  await expect(tasks.locator('.on')).toHaveCount(0)
  await expect(dialog.getByRole('radio', { name: 'OBB (guess)' })).toHaveAttribute('aria-checked', 'false')
  // Both have labels now, which Skip leaves alone; and the task comes first
  await expect(dialog.getByRole('button', { name: 'Import', exact: true })).toBeDisabled()
  await expect(dialog.locator('.import-sum')).toHaveText('Pick the task first.')
  await page.screenshot({ path: `${SHOTS}/import-yolo-light-1500.png` })
  await size(app, page, 1000, 640)
  await fitsIn(page, 1000)
  await page.screenshot({ path: `${SHOTS}/import-yolo-light-1000.png` })
  await size(app, page, 1500, 940)
  // Import can't take focus while the task is asked, so the window keeps it, and Escape closes it
  expect(await page.evaluate(() => document.querySelector('.dialog.import')!.contains(document.activeElement))).toBe(true)
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await answer(app, yolo)
  await menu(page, 'File', 'Import dataset…')
  await dialog.getByRole('button', { name: 'YOLO dataset folder…' }).click()
  await expect(dialog).toContainText('These rows fit OBB and Segment: click the one they were made for.')
  // Each group of choices is one Tab stop, at the choice made, the first while none is, and the arrow keys move the choice round it
  const policy = dialog.getByRole('radiogroup', { name: 'Images that have labels' })
  const [skip, add, replace] = ['Skip them', 'Add shapes', 'Replace'].map((name) => policy.getByRole('radio', { name }))
  const [detect, segment] = ['Detect', 'Segment'].map((name) => tasks.getByRole('radio', { name, exact: true }))
  await page.keyboard.press('Tab')
  await expect(detect).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(skip).toBeFocused()
  await page.keyboard.press('ArrowRight')
  await expect(add).toBeFocused()
  await expect(add).toHaveAttribute('aria-checked', 'true')
  await page.keyboard.press('ArrowLeft')
  await page.keyboard.press('ArrowLeft')
  await expect(replace).toBeFocused()
  await expect(replace).toHaveAttribute('aria-checked', 'true')
  await page.keyboard.press('ArrowRight')
  await expect(skip).toHaveAttribute('aria-checked', 'true')
  // A task chosen with them is read again, and has the focus back once it's read, so the arrow keys go on from it
  await page.keyboard.press('Shift+Tab')
  await expect(detect).toBeFocused()
  await page.keyboard.press('ArrowRight')
  await expect(segment).toHaveAttribute('aria-checked', 'true')
  await expect(segment).toBeFocused()
  await expect(dialog).not.toContainText('These rows fit')
  await dialog.getByRole('radio', { name: 'OBB (guess)' }).click()
  await expect(dialog.getByRole('radio', { name: 'OBB (guess)' })).toHaveAttribute('aria-checked', 'true')
  await expect(dialog).not.toContainText('These rows fit')
  await expect(dialog.locator('.import-seg .fits')).toHaveCount(0)
  await dialog.getByRole('radio', { name: 'Replace' }).click()
  // b gets no shapes from its empty label file, so Replace leaves it as it is
  await expect(dialog).toContainText("3 images have labels already: their label files go to the trash, and the imported shapes take their place. 1 of them gets no shapes from it, so it's left as it is.")
  await page.screenshot({ path: `${SHOTS}/import-replace-light-1500.png` })
  await size(app, page, 1000, 640)
  await fitsIn(page, 1000)
  await page.screenshot({ path: `${SHOTS}/import-replace-light-1000.png` })
  await size(app, page, 1500, 940)
  await dialog.getByRole('button', { name: 'Import 2 images' }).click()
  await expect(dialog.locator('.import-result')).toHaveText('Wrote 2 label files with 2 shapes.')
  await expect(dialog).toContainText('2 old label files went to the trash.')
  await expect(dialog.locator('.import-group[data-code="no-shapes"] .count')).toHaveText('1')
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
  expect(shapesIn(dir, 'b').map((s: any) => [s.label, s.shape_type])).toEqual([['cat', 'rectangle'], ['dog', 'mask']])
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

test('the Import window holds the focus: Tab goes round its controls, Escape closes it, and a file dropped on it opens nothing', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const dir = await project('import-held', { a: null, b: doc('b', [square('cat', 20, 20, 120, 100)]) })
  const other = await project('import-held-other', { x: null })
  const coco = path.join(BASE, 'import-src', 'held.json')
  mkdirSync(path.dirname(coco), { recursive: true })
  writeFileSync(coco, JSON.stringify({
    images: [{ id: 1, file_name: 'a.png', width: W, height: H }],
    annotations: [{ id: 1, image_id: 1, category_id: 1, iscrowd: 0, area: 2500, bbox: [10, 10, 50, 50], segmentation: [[10, 10, 60, 10, 60, 60, 10, 60]] }],
    categories: [{ id: 1, name: 'cat' }]
  }))
  const { app, page, errors } = await launch(dir)
  const dialog = page.getByRole('dialog', { name: 'Import dataset' })
  const holds = () => page.evaluate(() => !!document.querySelector('.dialog.import')?.contains(document.activeElement))
  await menu(page, 'File', 'Import dataset…')
  await expect(dialog).toHaveAttribute('aria-modal', 'true')
  // Shift+Tab from its first control comes round to its last, and Tab from there back to the first
  const first = dialog.getByRole('button', { name: 'COCO file…' })
  await expect(first).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(first).toBeFocused()
  for (let k = 0; k < 8; k++) {
    await page.keyboard.press('Tab')
    expect(await holds()).toBe(true)
  }
  // Ctrl+W behind it waits, and Escape closes it; the focus goes to the image, as the menu item that opened it has gone
  await page.keyboard.press('ControlOrMeta+w')
  await page.waitForTimeout(300)
  await expect(dialog).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(page.getByTestId('canvas')).toBeFocused()
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png'])

  // A file dropped on the window while it shows what it read opens nothing: the import goes into this folder
  await answer(app, coco)
  await menu(page, 'File', 'Import dataset…')
  await dialog.getByRole('button', { name: 'COCO file…' }).click()
  await expect(dialog).toContainText('Its one image is in this folder')
  // With no label list, a label the folder's label files have is nothing new
  await expect(dialog.locator('.import-labels tr', { hasText: 'cat' })).toBeVisible()
  await expect(dialog.locator('.import-labels .tag.new')).toHaveCount(0)
  await page.evaluate(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.id = 'dropped'
    input.hidden = true
    document.body.append(input)
  })
  await page.setInputFiles('#dropped', path.join(other, 'x.png'))
  await page.evaluate(() => {
    const data = new DataTransfer()
    data.items.add((document.getElementById('dropped') as HTMLInputElement).files![0])
    const target = document.querySelector('.dialog.import')!
    target.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: data }))
    target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: data }))
    document.getElementById('dropped')!.remove()
  })
  await page.waitForTimeout(500)
  await expect(page.locator('.files .file .name')).toHaveText(['a.png', 'b.png'])
  await dialog.getByRole('button', { name: 'Import 1 image' }).click()
  await expect(dialog.locator('.import-result')).toHaveText('Wrote 1 label file with 1 shape.')
  await expect(dialog).not.toContainText('label list')
  expect(await holds()).toBe(true)
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  expect(shapesIn(dir, 'a').map((s: any) => s.label)).toEqual(['cat'])

  // At the smallest window and a large one, dark and light, it opens holding the focus
  for (const theme of ['dark', 'light']) {
    for (const [width, height] of [[1500, 940], [960, 600]]) {
      await size(app, page, width, height)
      await menu(page, 'File', 'Import dataset…')
      await expect(first).toBeFocused()
      await fitsIn(page, width)
      await page.screenshot({ path: `${SHOTS}/import-held-${theme}-${width}.png` })
      await page.keyboard.press('Escape')
      await expect(dialog).toHaveCount(0)
    }
    if (theme === 'dark') await menu(page, 'View', 'Dark mode')
  }
  await finish(app, errors)
})

test("the Import window's words: counts that agree, its Problems, severities and keypoints file named as the Export window names them, and only labels new to the folder tagged", async () => {
  const dir = await project('import-words', { a: null, b: doc('b', [square('cat', 20, 20, 120, 100)]), d: null })
  const src = path.join(BASE, 'import-src')
  mkdirSync(src, { recursive: true })
  const image = (id: number, file_name: string, width = W, height = H) => ({ id, file_name, width, height })
  const polygon = (id: number, image_id: number, category_id: number) => ({ id, image_id, category_id, iscrowd: 0, area: 1, bbox: [0, 0, 1, 1], segmentation: [[10, 10, 60, 10, 40, 60]] })
  const categories = [{ id: 1, name: 'cat' }, { id: 2, name: 'zebra' }]
  // One of its three images is here
  const three = path.join(src, 'words-three.json')
  writeFileSync(three, JSON.stringify({ images: [image(1, 'a.png'), image(2, 'x.png'), image(3, 'y.png')], categories, annotations: [polygon(1, 1, 1), polygon(2, 1, 2)] }))
  // An error, warnings and a note: d.png's sides are swapped, q.png isn't here, category 9 is unknown, b.png has no annotations
  const mixed = path.join(src, 'words-mixed.json')
  writeFileSync(mixed, JSON.stringify({
    images: [image(1, 'a.png'), image(2, 'd.png', H, W), image(3, 'b.png'), image(4, 'q.png')], categories,
    annotations: [polygon(1, 1, 1), polygon(2, 1, 9), polygon(3, 2, 1)]
  }))
  // YOLO pose rows of one keypoint, which data.yaml doesn't name
  const pose = path.join(src, 'words-pose')
  rmSync(pose, { recursive: true, force: true })
  await png(path.join(pose, 'images', 'train', 'a.png'))
  mkdirSync(path.join(pose, 'labels', 'train'), { recursive: true })
  writeFileSync(path.join(pose, 'labels', 'train', 'a.txt'), '0 0.5 0.5 0.2 0.2 0.45 0.45 2\n')
  writeFileSync(path.join(pose, 'data.yaml'), 'train: images/train\nval: images/train\nkpt_shape: [1, 3]\nnames:\n  0: person\n')
  const names = path.join(src, 'words-keypoints.yaml')
  writeFileSync(names, 'classes:\n  person:\n    - nose\n')

  // No label list here
  const { app, page, errors } = await launch(dir)
  const dialog = page.getByRole('dialog', { name: 'Import dataset' })
  const read = async (file: string, button: string) => {
    await answer(app, file)
    await menu(page, 'File', 'Import dataset…')
    await dialog.getByRole('button', { name: button }).click()
    await expect(dialog.locator('.import-source')).toBeVisible()
  }
  await read(three, 'COCO file…')
  await expect(dialog).toContainText('1 of its 3 images is in this folder: 1 by its path.')
  await expect(dialog).toContainText("2 aren't here: nothing is imported for them, as listed below.")
  await expect(dialog.locator('.import-section h4', { hasText: /^Problems$/ })).toBeVisible()
  // cat is in b's label file; zebra is new to the folder
  await expect(dialog.locator('.import-labels tr', { hasText: 'cat' }).locator('.tag')).toHaveCount(0)
  const fresh = dialog.locator('.import-labels tr', { hasText: 'zebra' }).locator('.tag.new')
  await expect(fresh).toHaveAttribute('title', "Not in this folder's label files yet")
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    expect(await contrast(fresh)).toBeGreaterThanOrEqual(4.5)
  }
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark' })
  await page.keyboard.press('Escape')

  await read(mixed, 'COCO file…')
  await expect(dialog).toContainText("1 isn't here: nothing is imported for it, as listed below.")
  // The icons are named Error, Warning and Note, as in the Export window
  expect((await dialog.locator('.import-group .sev').evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')))).sort()).toEqual(['Error', 'Note', 'Warning', 'Warning'])
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    expect(await contrast(dialog.locator('.import-group.is-warning .sev').first())).toBeGreaterThanOrEqual(3)
  }
  await page.screenshot({ path: `${SHOTS}/import-words-light-1500.png` })
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark' })
  await page.keyboard.press('Escape')

  // Forty of its images aren't here: in the smallest window their group opens to its messages in the window's own scroll
  const away = path.join(src, 'words-away.json')
  writeFileSync(away, JSON.stringify({ images: [image(1, 'a.png'), ...Array.from({ length: 40 }, (_, k) => image(k + 2, `away-${k}.png`))], categories, annotations: [polygon(1, 1, 1)] }))
  await size(app, page, 960, 600)
  await read(away, 'COCO file…')
  const missing = dialog.locator('.import-group[data-code="coco-image-missing"]')
  await missing.getByRole('button', { name: /Images not in this folder/ }).click()
  await expect(missing.locator('.import-rows li')).toHaveCount(21)
  const scrolls = (selector: string) => dialog.locator(selector).evaluate((el) => el.scrollHeight > el.clientHeight + 1)
  expect(await scrolls('.import-rows')).toBe(false)
  expect(await scrolls('.import-body')).toBe(true)
  for (const theme of ['dark', 'light']) {
    await page.evaluate((t) => { document.documentElement.dataset.theme = t }, theme)
    await dialog.locator('.import-body').evaluate((el) => { el.scrollTop = el.scrollHeight })
    await page.screenshot({ path: `${SHOTS}/import-rows-${theme}-960.png` })
  }
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark' })
  await page.keyboard.press('Escape')
  await size(app, page, 1500, 940)

  // A keypoint data.yaml doesn't name, and the keypoints file that names it
  await read(pose, 'YOLO dataset folder…')
  await expect(dialog.locator('.import-line')).toContainText("None given, so it's called kp0.")
  await app.evaluate(({ dialog }, f) => {
    dialog.showOpenDialog = (async (...args: any[]) => {
      (globalThis as any).asked = args[args.length - 1]
      return { canceled: false, filePaths: [f] }
    }) as any
  }, names)
  await dialog.getByRole('button', { name: 'Keypoints file…' }).click()
  await expect(dialog.locator('.import-line')).toContainText('From words-keypoints.yaml.')
  expect(await app.evaluate(() => { const o = (globalThis as any).asked; return [o.title, o.filters[0].name] })).toEqual(['Choose a keypoints file', 'Keypoints files'])
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await finish(app, errors)
})
