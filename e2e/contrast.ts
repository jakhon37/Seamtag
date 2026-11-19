import type { Locator } from '@playwright/test'

/**
 * The contrast of an element's words, or its icon's lines, with what lies behind them, as WCAG works
 * it out: its colour, faded by its own and its ancestors' opacity, over the nearest background that
 * isn't transparent
 */
export const contrast = (target: Locator) => target.evaluate((el) => {
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
