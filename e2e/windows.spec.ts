import { expect, type Page, test } from '@playwright/test'
import { existsSync, readFileSync } from 'fs'
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

test('About says whose Seamtag is and on what terms, and shows its licence and the notices of what it includes', async () => {
  const root = path.resolve(__dirname, '..')
  const { app, page, errors } = await launch(null)
  await menu(page, 'Help', 'About Seamtag')
  const about = page.getByRole('dialog', { name: 'About' })
  const { version } = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'))
  await expect(about.getByRole('heading', { level: 1 })).toHaveText(`Seamtag ${version}`)
  await expect(about).toContainText('Copyright © 2026 Jakhongir Nodirov')
  await expect(about).toContainText('Free software under the GNU GPL version 3 only, with ABSOLUTELY NO WARRANTY.')
  await expect(about).toContainText('Includes code from other projects under their own licences.')

  // Each button shows the files as the app has them, here the repository's own, over the About
  // window; Escape closes it and gives the focus back to the button
  const shows = async (button: string, files: string[]) => {
    const opener = about.getByRole('button', { name: button, exact: true })
    await opener.click()
    const dialog = page.getByRole('dialog', { name: button })
    await expect(dialog.locator('section')).toHaveCount(files.length)
    for (const file of files) {
      const section = dialog.locator(`section[data-file="${file}"]`)
      await expect(section.getByRole('heading')).toHaveText(file)
      // THIRD-PARTY-NOTICES.txt is made by npm run build, which a test build may leave out
      if (existsSync(path.join(root, file))) expect(await section.locator('pre').textContent()).toBe(readFileSync(path.join(root, file), 'utf8'))
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
