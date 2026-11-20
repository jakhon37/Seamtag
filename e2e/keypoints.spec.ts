import { expect, type Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import path from 'path'
import { click, doc, drag, finish, launch, onImage, open, project, saved, shapesIn } from './helpers'

// Keypoints in the editor: a pose is a box and the points sharing its group_id, each named for a
// keypoint of COCO's person until the folder has its own keypoints file

const COCO = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
const near = (a: number[], b: number[], tolerance = 2) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance
const shape = (label: string, shape_type: string, points: number[][], group_id: number | null, extra: Record<string, unknown> = {}) =>
  ({ label, shape_type, points, group_id, description: '', flags: {}, ...extra })
/** A person from 100 to 200 across, with a nose, both shoulders and the left wrist */
const person = (group: number) => [
  shape('person', 'rectangle', [[100, 50], [200, 250]], group),
  shape('nose', 'point', [[150, 70]], group),
  shape('left_shoulder', 'point', [[130, 110]], group),
  shape('right_shoulder', 'point', [[170, 110]], group),
  shape('left_wrist', 'point', [[120, 180]], group)
]
const step = (page: Page) => page.getByTestId('pose-step')
/** How many shapes an image's label file holds, once it's written */
const count = (dir: string, name: string) => (existsSync(path.join(dir, `${name}.json`)) ? shapesIn(dir, name).length : 0)
const at = (shapes: any[], label: string) => shapes.find((s: any) => s.label === label)

test('a pose drawn with the mouse and keys: a box, then each keypoint, hidden, skipped and taken back', async () => {
  const dir = await project('kp-draw', { p: null, q: null })
  const { app, page, errors } = await launch(dir)
  await open(page, 'p.png')
  await page.keyboard.press('k')
  await expect(page.locator('.status')).toContainText('Pose')
  await drag(page, [100, 40], [300, 290])
  await expect(step(page)).toHaveText('1/17 nose')
  await click(page, 200, 70)
  // Shift+click: hidden
  await page.keyboard.down('Shift')
  await click(page, 190, 62)
  await page.keyboard.up('Shift')
  await expect(step(page)).toHaveText('3/17 right eye')
  // X: not labelled
  await page.keyboard.press('x')
  await click(page, 180, 66)
  await click(page, 220, 66)
  await click(page, 150, 120)
  await expect(step(page)).toHaveText('7/17 right shoulder')
  // Backspace: back one
  await page.keyboard.press('Backspace')
  await expect(step(page)).toHaveText('6/17 left shoulder')
  await click(page, 160, 110)
  // Enter: done, the rest not labelled
  await page.keyboard.press('Enter')
  await expect(step(page)).toHaveCount(0)
  await saved(page)
  let shapes = shapesIn(dir, 'p')
  expect(shapes.map((s: any) => s.label)).toEqual(['person', 'nose', 'left_eye', 'left_ear', 'right_ear', 'left_shoulder'])
  expect(shapes.every((s: any) => s.group_id === 1)).toBe(true)
  expect(shapes[0].shape_type).toBe('rectangle')
  expect(near(shapes[0].points[0], [100, 40]) && near(shapes[0].points[1], [300, 290])).toBe(true)
  expect(shapes.slice(1).every((s: any) => s.shape_type === 'point')).toBe(true)
  expect(shapes.map((s: any) => s.flags)).toEqual([{}, {}, { difficult: true }, {}, {}, {}])
  expect(near(at(shapes, 'left_shoulder').points[0], [160, 110])).toBe(true)
  // The pose's class is a label; its keypoints' names aren't
  await expect(page.getByRole('region', { name: 'Labels' }).locator('.class .name')).toHaveText(['person'])

  // The Shapes list folds the pose into one row, and opened, lists its keypoints, missing ones too
  const row = page.locator('.pose-row')
  await expect(row).toContainText('person (1) · 5/17')
  await expect(page.locator('.shape')).toHaveCount(0)
  await row.getByRole('button', { name: 'Open' }).click()
  await expect(page.locator('.shape .name')).toHaveText(['person (1)', 'nose', 'left_eye', 'left_ear', 'right_ear', 'left_shoulder'])
  await expect(page.locator('.shape', { hasText: 'left_eye' }).locator('.tag')).toHaveText('hidden')
  await expect(page.locator('.pose-missing')).toHaveCount(12)
  await expect(page.locator('.pose-missing .name').first()).toHaveText('right_eye')
  // A missing one is placed with the next click
  await page.locator('.pose-missing', { hasText: 'right_eye' }).click()
  await expect(step(page)).toHaveText('3/17 right eye')
  await click(page, 210, 62)
  await expect(step(page)).toHaveCount(0)
  await saved(page)
  shapes = shapesIn(dir, 'p')
  expect(at(shapes, 'right_eye')).toMatchObject({ shape_type: 'point', group_id: 1, flags: {} })
  await expect(row).toContainText('6/17')
  await expect(page.locator('.shape.on .name')).toHaveText('right_eye')
  await row.getByRole('button', { name: 'Fold' }).click()
  await expect(page.locator('.shape')).toHaveCount(0)
  // A click on the pose's box selects it whole
  await page.getByRole('toolbar').getByRole('button', { name: 'Edit', exact: true }).click()
  await click(page, 250, 250)
  await expect(page.locator('.pose-row.on')).toHaveCount(1)
  await expect(page.locator('.row.selection')).toContainText('Delete 7 shapes')

  // Carrying on a pose: a click on its box asks for what it's missing
  await page.keyboard.press('k')
  await click(page, 250, 250)
  await expect(step(page)).toHaveText('7/17 right shoulder')
  await click(page, 240, 110)
  await page.keyboard.press('Escape')
  await expect(step(page)).toHaveCount(0)
  await saved(page)
  expect(shapesIn(dir, 'p').filter((s: any) => s.group_id === 1)).toHaveLength(8)

  // LabelImg's formats keep boxes only: the pose tools are off there
  await open(page, 'q.png')
  await page.getByLabel('Save format').selectOption('voc')
  await expect(page.getByRole('toolbar').getByRole('button', { name: 'Pose', exact: true })).toBeDisabled()
  await finish(app, errors)
})

test('Q, M, Shift+M and N on a pose', async () => {
  const dir = await project('kp-keys', { a: doc('a', person(1)) })
  const { app, page, errors } = await launch(dir)
  await open(page, 'a.png')
  await page.keyboard.press('v')
  await click(page, 150, 230)
  await expect(page.locator('.pose-row.on')).toContainText('person (1) · 4/17')
  // M: left and right trade names, and stay where they are
  await page.keyboard.press('m')
  await saved(page)
  let shapes = shapesIn(dir, 'a')
  expect(shapes.map((s: any) => s.label)).toEqual(['person', 'nose', 'right_shoulder', 'left_shoulder', 'right_wrist'])
  expect(shapes.map((s: any) => s.points[0][0])).toEqual([100, 150, 130, 170, 120])
  // Shift+M: the points mirrored inside the box, keeping their names
  await page.keyboard.press('Shift+M')
  await expect.poll(() => shapesIn(dir, 'a').map((s: any) => s.points[0][0])).toEqual([100, 150, 170, 130, 180])
  expect(shapesIn(dir, 'a').map((s: any) => s.label)).toEqual(['person', 'nose', 'right_shoulder', 'left_shoulder', 'right_wrist'])
  // Q: the keypoint under the pointer hidden, then seen again
  await page.mouse.move(...(await onImage(page, 150, 70)))
  await page.keyboard.press('q')
  await expect.poll(() => at(shapesIn(dir, 'a'), 'nose').flags).toEqual({ difficult: true })
  await page.keyboard.press('q')
  await expect.poll(() => at(shapesIn(dir, 'a'), 'nose').flags).toEqual({})
  // N: the pose's keypoints in turn; a missing one is placed with the next click
  await page.mouse.move(...(await onImage(page, 300, 280)))
  await page.keyboard.press('n')
  await expect(page.locator('.shape.on .name')).toHaveText('nose')
  await page.keyboard.press('n')
  await expect(step(page)).toHaveText('2/17 left eye')
  await click(page, 140, 62)
  await expect(step(page)).toHaveCount(0)
  await saved(page)
  shapes = shapesIn(dir, 'a')
  expect(at(shapes, 'left_eye')).toMatchObject({ group_id: 1, shape_type: 'point' })
  expect(near(at(shapes, 'left_eye').points[0], [140, 62])).toBe(true)
  await expect(page.locator('.shape.on .name')).toHaveText('left_eye')
  await page.keyboard.press('Shift+N')
  await expect(page.locator('.shape.on .name')).toHaveText('nose')
  // Q on the keypoints selected, with the pointer off them
  await page.mouse.move(...(await onImage(page, 300, 280)))
  await page.keyboard.press('q')
  await expect.poll(() => at(shapesIn(dir, 'a'), 'nose').flags).toEqual({ difficult: true })
  // Backspace takes the keypoint under the pointer away, as it does a polygon's point
  await page.mouse.move(...(await onImage(page, 180, 180)))
  await page.keyboard.press('Backspace')
  await expect.poll(() => shapesIn(dir, 'a').map((s: any) => s.label)).toEqual(['person', 'nose', 'right_shoulder', 'left_shoulder', 'left_eye'])
  await expect(page.locator('.shape.on .name')).toHaveText('nose')
  await finish(app, errors)
})

test('a pose moves, copies, pastes and goes as one; its box resizes alone', async () => {
  const dir = await project('kp-copy', { c: doc('c', person(1)) })
  const { app, page, errors } = await launch(dir)
  await open(page, 'c.png')
  await page.keyboard.press('v')
  // Dragging the box moves the whole pose
  await drag(page, [150, 230], [180, 230])
  await saved(page)
  let shapes = shapesIn(dir, 'c')
  expect(near(shapes[0].points[0], [130, 50]) && near(at(shapes, 'nose').points[0], [180, 70])).toBe(true)
  // Dragging a point moves that point alone
  await drag(page, [180, 70], [185, 60])
  await saved(page)
  shapes = shapesIn(dir, 'c')
  expect(near(at(shapes, 'nose').points[0], [185, 60]) && near(shapes[0].points[0], [130, 50])).toBe(true)
  // Resizing the box changes only the box
  await drag(page, [230, 250], [260, 270])
  await saved(page)
  shapes = shapesIn(dir, 'c')
  expect(near(shapes[0].points[1], [260, 270]) && near(at(shapes, 'left_wrist').points[0], [150, 180])).toBe(true)
  // Copy and paste: the pasted pose takes the next group_id, and duplicating, the one after
  await click(page, 180, 230)
  await expect(page.locator('.pose-row.on')).toHaveCount(1)
  await page.keyboard.press('Control+c')
  await page.keyboard.press('Control+v')
  await expect(page.locator('.pose-row')).toHaveCount(2)
  await page.keyboard.press('Control+d')
  await expect(page.locator('.pose-row')).toHaveText([/person \(1\) · 4\/17/, /person \(2\) · 4\/17/, /person \(3\) · 4\/17/])
  await saved(page)
  shapes = shapesIn(dir, 'c')
  expect(shapes.map((s: any) => s.group_id)).toEqual([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
  expect(shapes.slice(10).map((s: any) => s.label)).toEqual(['person', 'nose', 'left_shoulder', 'right_shoulder', 'left_wrist'])
  // Delete takes the selected pose whole
  await page.keyboard.press('Delete')
  await expect(page.locator('.pose-row')).toHaveCount(2)
  await saved(page)
  expect(shapesIn(dir, 'c').map((s: any) => s.group_id)).toEqual([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
  await finish(app, errors)
})

const NOTE = "The weights come from OpenMMLab's RTMPose, under Apache-2.0; the terms of their training data vary."

/**
 * The app on no folder, with main's AI answering as a pose model would: ai:models lists one, its
 * download and YOLO-World's text prompt are stubbed, and every question is answered with its first
 * button. Then the folder opens, so the window lists the stubbed models.
 */
async function launchWithPose(dir: string) {
  const { app, page, errors } = await launch(null)
  await app.evaluate(({ dialog, ipcMain }, [folder, names, note]) => {
    const g = globalThis as any
    g.downloaded = false
    g.asked = []
    g.poseRequests = []
    dialog.showMessageBox = (async (_win: unknown, o: { message: string; detail: string }) => {
      g.asked.push(`${o.message} ${o.detail}`)
      return { response: 0, checkboxChecked: false }
    }) as any
    dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [folder] })) as any
    const license = { name: 'Apache License 2.0', url: 'https://example.com/license' }
    ipcMain.removeHandler('ai:models')
    ipcMain.handle('ai:models', () => [
      { name: 'rtmpose-m', title: 'RTMPose-m', size: 50_800_000, downloaded: g.downloaded, downloading: false, license, points: false, box: false, text: false, pose: true, keypoints: names, note },
      { name: 'yoloworld:latest', title: 'YOLO-World', size: 670_000_000, downloaded: true, downloading: false, license, points: false, box: false, text: true }
    ])
    ipcMain.removeHandler('ai:download')
    ipcMain.handle('ai:download', () => { g.downloaded = true })
    // Each box's keypoints spread over it: the nose missing, the left eye under the threshold, the right ankle past the image's edge
    ipcMain.removeHandler('ai:pose')
    ipcMain.handle('ai:pose', (_e, req: { boxes: number[][] }) => {
      g.poseRequests.push(req)
      return {
        names,
        poses: req.boxes.map(([x0, y0, x1, y1]) => (names as string[]).map((_, k) => (k === 0 ? null : k === 1 ? { x: x0 + 5, y: y0 + 5, score: 0.2 }
          : k === 16 ? { x: 999, y: 999, score: 0.9 } : { x: x0 + ((x1 - x0) * k) / 17, y: y0 + ((y1 - y0) * k) / 17, score: 0.5 + k / 100 })))
      }
    })
    ipcMain.removeHandler('ai:text')
    ipcMain.handle('ai:text', () => ({ shapes: [{ label: 'person', shape_type: 'rectangle', points: [[220, 40], [320, 260]], description: '{"score": 0.9, "text": "person"}' }], matched: [] }))
  }, [dir, COCO, NOTE] as const)
  await page.getByRole('button', { name: 'Open folder' }).first().click()
  await expect(page.locator('.file').first()).toBeVisible()
  const global = <T,>(key: string) => app.evaluate((_, k) => (globalThis as any)[k], key) as Promise<T>
  return { app, page, errors, global }
}

test('AI pose: a box dragged, its keypoints from the model as AI shapes, reviewed as one', async () => {
  const dir = await project('kp-ai', { d: null })
  const { app, page, errors, global } = await launchWithPose(dir)
  await open(page, 'd.png')
  await page.keyboard.press('Shift+K')
  await expect(page.locator('.status')).toContainText('AI pose')
  // Only pose models are offered, with the score a keypoint needs
  await expect(page.getByLabel('AI pose model').locator('option')).toHaveText(['RTMPose-m (download)'])
  await expect(page.getByLabel('Keypoint score threshold')).toHaveValue('0.3')
  await drag(page, [40, 30], [210, 270])
  // The download is asked first, with the model's note
  await expect.poll(() => global<string[]>('asked')).toEqual([expect.stringContaining(NOTE)])
  await expect.poll(() => count(dir, 'd')).toBe(16)
  const requests = await global<{ boxes: number[][]; model: string; name: string }[]>('poseRequests')
  expect(requests).toHaveLength(1)
  expect(requests[0].model).toBe('rtmpose-m')
  expect(requests[0].name).toBe('d.png')
  expect(requests[0].boxes[0].every((v, k) => Math.abs(v - [40, 30, 210, 270][k]) <= 2)).toBe(true)
  let shapes = shapesIn(dir, 'd')
  // The box and 15 points: the nose wasn't placed, the left eye scored under 0.3
  expect(shapes.map((s: any) => s.label)).toEqual(['person', ...COCO.slice(2)])
  expect(shapes.every((s: any) => s.group_id === 1 && s.flags.ai === true && typeof s.score === 'number')).toBe(true)
  expect(shapes[0].shape_type).toBe('rectangle')
  // Past the image's edge: hidden, at the nearest place on it
  const ankle = at(shapes, 'right_ankle')
  expect(ankle.points).toEqual([[400, 300]])
  expect(ankle.flags).toEqual({ difficult: true, ai: true })
  expect(ankle.score).toBe(0.9)
  expect(at(shapes, 'right_eye').flags).toEqual({ ai: true })
  // Review takes the pose as one: W selects it, W accepts all of it
  await expect(page.getByTestId('review-bar')).toContainText('16 AI shapes')
  await page.keyboard.press('Escape')
  await page.keyboard.press('w')
  await expect(page.locator('.pose-row.on')).toContainText('person (1) · 15/17')
  await expect(page.getByTestId('review-bar')).toContainText('this one 52%')
  await page.keyboard.press('w')
  await saved(page)
  shapes = shapesIn(dir, 'd')
  expect(shapes).toHaveLength(16)
  expect(shapes.some((s: any) => s.flags.ai || s.score !== undefined)).toBe(false)
  await expect(page.getByTestId('review-bar')).toHaveCount(0)
  await finish(app, errors)
})

test("Pose every person: on the image's boxes without keypoints, else on YOLO-World's people; rejected as one", async () => {
  const dir = await project('kp-everyone', { e: doc('e', [...person(1), shape('person', 'rectangle', [[240, 40], [360, 280]], null)]), f: null })
  const { app, page, errors, global } = await launchWithPose(dir)
  await open(page, 'e.png')
  // The posed person is left alone; the other one's box takes a group_id and AI points
  await page.getByRole('menuitem', { name: 'AI' }).click()
  await page.getByRole('menuitem', { name: 'Pose every person' }).click()
  await expect(page.locator('.loading.notice')).toContainText('Found 1 pose.')
  await expect.poll(() => count(dir, 'e')).toBe(5 + 1 + 15)
  let requests = await global<{ boxes: number[][] }[]>('poseRequests')
  expect(requests.map((r) => r.boxes)).toEqual([[[240, 40, 360, 280]]])
  let shapes = shapesIn(dir, 'e')
  expect(shapes[5]).toMatchObject({ label: 'person', group_id: 2, flags: {} })
  expect(shapes.slice(6).every((s: any) => s.group_id === 2 && s.flags.ai === true)).toBe(true)
  // Rejecting one of its points rejects all of them; the box is the user's, and stays
  await page.keyboard.press('Escape')
  await page.keyboard.press('w')
  await expect(page.locator('.pose-row.on')).toContainText('person (2) · 15/17')
  await page.keyboard.press('s')
  await expect.poll(() => count(dir, 'e')).toBe(6)
  shapes = shapesIn(dir, 'e')
  expect(shapes.some((s: any) => s.flags.ai)).toBe(false)
  // With no box of the class, YOLO-World finds the people first; they come with their poses, as AI shapes
  await open(page, 'f.png')
  await page.keyboard.press('Shift+K')
  await page.getByRole('button', { name: 'Pose every person' }).click()
  await expect(page.locator('.loading.notice')).toContainText('Found 1 pose.')
  await expect.poll(() => count(dir, 'f')).toBe(16)
  requests = await global<{ boxes: number[][] }[]>('poseRequests')
  expect(requests.at(-1)!.boxes).toEqual([[220, 40, 320, 260]])
  shapes = shapesIn(dir, 'f')
  expect(shapes.map((s: any) => s.label)).toEqual(['person', ...COCO.slice(2)])
  expect(shapes[0]).toMatchObject({ shape_type: 'rectangle', group_id: 1, flags: { ai: true }, description: '{"score": 0.9, "text": "person"}' })
  await finish(app, errors)
})

test("Dataset > Keypoints: the folder's own keypoints, made from its poses, chosen from a file, or COCO's person", async () => {
  const cat = [
    shape('cat', 'rectangle', [[100, 50], [300, 250]], 1), shape('head', 'point', [[150, 80]], 1),
    shape('left_paw', 'point', [[120, 230]], 1), shape('right_paw', 'point', [[180, 230]], 1)
  ]
  const dir = await project('kp-file', { g: doc('g', cat) })
  // A keypoints file that can't be read is said, and COCO's person is used
  mkdirSync(path.join(dir, '.myseg'), { recursive: true })
  writeFileSync(path.join(dir, '.myseg/keypoints.yaml'), 'classes: [oops\n')
  const { app, page, errors } = await launch(dir)
  await expect(page.getByRole('alert')).toContainText("so COCO's person is used")
  await page.getByRole('alert').click()
  await open(page, 'g.png')
  const labels = page.getByRole('region', { name: 'Labels' }).locator('.class .name')
  await expect(labels).toHaveText(['cat', 'head', 'left_paw', 'right_paw'])
  const menu = async (item: string) => {
    await page.getByRole('menuitem', { name: 'Dataset' }).click()
    await page.getByRole('menuitem', { name: 'Keypoints' }).hover()
    await page.getByRole('menuitem', { name: item }).click()
  }
  // Made from the folder's poses: the cat's points name keypoints, not classes, and the cat folds into a pose
  await menu("Make one from the folder's poses")
  await expect(page.locator('.loading.notice')).toContainText('Keypoints: cat, 3 points.')
  await expect(labels).toHaveText(['cat'])
  await expect(page.locator('.pose-row')).toContainText('cat (1) · 3/3')
  expect(readFileSync(path.join(dir, '.myseg/keypoints.yaml'), 'utf8')).toContain('classes:\n  cat:\n    - head\n    - left_paw\n    - right_paw\n')
  // The Pose tool asks for the folder's keypoints
  await page.keyboard.press('k')
  await drag(page, [250, 40], [380, 280])
  await expect(step(page)).toHaveText('1/3 head')
  await page.keyboard.press('Backspace')
  await expect(step(page)).toHaveCount(0)
  // Chosen from a file, copied in as the folder's own
  const chosen = path.join(dir, 'dogs.yaml')
  writeFileSync(chosen, 'classes:\n  dog:\n    - nose\n    - tail\n')
  await app.evaluate(({ dialog }, file) => {
    dialog.showOpenDialog = (async () => ({ canceled: false, filePaths: [file] })) as any
  }, chosen)
  await menu('Choose a keypoints file…')
  await expect(page.locator('.loading.notice')).toContainText('Keypoints: dog, 2 points.')
  await expect(labels).toHaveText(['cat', 'head', 'left_paw', 'right_paw'])
  expect(readFileSync(path.join(dir, '.myseg/keypoints.yaml'), 'utf8')).toContain('  dog:\n    - nose\n    - tail\n')
  // And back to COCO's person
  await menu("Use COCO's person")
  await expect(page.locator('.loading.notice')).toContainText('Keypoints: person, 17 points.')
  expect(readFileSync(path.join(dir, '.myseg/keypoints.yaml'), 'utf8')).toContain('    - right_ankle\n')
  await expect(page.locator('.pose-row')).toHaveCount(0)
  await finish(app, errors)
})
