import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, test } from 'vitest'
import type { MaskBits } from '../../src/shared/dataset'
import {
  pasteMask, type Rle, rleArea, rleDecode, rleEncode, rleFrBbox, rleFrMaskBits, rleFrPoly, rleFrString, rleLength, rleMerge,
  rleMove, rleToBbox, rleToMaskBits, rleToString
} from '../../src/shared/rle'

// What pycocotools 2.0.11 makes of the same inputs: tests/fixtures/coco/make_rle_cases.py
const ref = JSON.parse(readFileSync(path.join(__dirname, '../fixtures/coco/rle_cases.json'), 'utf8'))

// Rows of 0 and 1, top to bottom, to a column-major mask
function columns(rows: string[]) {
  const h = rows.length, w = rows[0].length
  const mask = new Uint8Array(h * w)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) mask[x * h + y] = rows[y][x] === '1' ? 1 : 0
  return mask
}

function disc(h: number, w: number, [cy, cx, r]: number[]) {
  const mask = new Uint8Array(h * w)
  for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) mask[x * h + y] = (y - cy) ** 2 + (x - cx) ** 2 <= r * r ? 1 : 0
  return mask
}

// Where two masks first differ, or -1: quicker than toEqual on masks of a million pixels
function firstDifference(a: Uint8Array, b: Uint8Array) {
  if (a.length !== b.length) return Math.min(a.length, b.length)
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i
  return -1
}

function expectDescribed(r: Rle, c: { counts: string; area: number; bbox: number[] }) {
  expect(rleToString(r)).toBe(c.counts)
  expect(rleArea(r)).toBe(c.area)
  expect(rleToBbox(r)).toEqual(c.bbox)
}

test('the fixture comes from pycocotools 2.0.11', () => {
  expect(ref.pycocotools).toBe('2.0.11')
})

describe("masks encode as pycocotools' encode does", () => {
  for (const c of ref.masks) {
    test(c.name, () => {
      const mask = c.rows ? columns(c.rows) : disc(c.h, c.w, c.disc)
      const r = rleEncode(mask, c.h, c.w)
      expectDescribed(r, c)
      // and back: the string to the same runs, the runs to the same pixels
      const back = rleFrString(c.counts, c.h, c.w)
      expect(back.counts).toEqual(r.counts)
      expect(firstDifference(rleDecode(back), mask)).toBe(-1)
    })
  }
})

describe("polygons fill as pycocotools' frPoly fills them", () => {
  for (const c of ref.polygons) test(c.name, () => expectDescribed(rleFrPoly(c.poly, c.h, c.w), c))
})

describe("boxes fill as pycocotools' frBbox fills them", () => {
  for (const c of ref.boxes) test(c.name, () => expectDescribed(rleFrBbox(c.box, c.h, c.w), c))
})

describe("masks merge as pycocotools' merge merges them", () => {
  for (const c of ref.merges) {
    test(c.name, () => {
      const rles = c.polys.map((p: number[]) => rleFrPoly(p, c.h, c.w))
      expectDescribed(rleMerge(rles), c.union)
      expectDescribed(rleMerge(rles, true), c.intersection)
    })
  }
})

describe("a polygon list fills as pycocotools' annToRLE fills it", () => {
  for (const c of ref.segmentations) {
    test(c.name, () => expectDescribed(rleMerge(c.segmentation.map((p: number[]) => rleFrPoly(p, c.h, c.w))), c))
  }
})

describe('runs are written, measured and boxed as pycocotools does it, odd ones too', () => {
  for (const c of ref.runs) {
    test(c.name, () => {
      const r = { h: c.h, w: c.w, counts: c.runs }
      expectDescribed(r, c)
      expect(rleFrString(c.counts, c.h, c.w).counts).toEqual(c.runs)
      if (c.decodes === false) expect(() => rleDecode(r)).toThrow('more than its')
      if (c.rows) expect(rleDecode(r)).toEqual(columns(c.rows))
    })
  }
})

test('runs that stop short of the mask leave the rest clear', () => {
  expect([...rleDecode({ h: 2, w: 2, counts: [1, 1] })]).toEqual([0, 1, 0, 0])
})

test('compressed counts with a character outside COCO\'s or cut off in the middle are refused', () => {
  expect(() => rleFrString('3!1', 3, 4)).toThrow('isn\'t one of')
  expect(() => rleFrString('3p', 3, 4)).toThrow('isn\'t one of')
  // 0x20 on the last character says another one follows
  expect(() => rleFrString('3P', 3, 4)).toThrow('middle of a number')
})

test('runs past 2^31 pixels keep their bits through the compressed string', () => {
  const r = { h: 70000, w: 70000, counts: [2 ** 31 + 5, 3, 70000 * 70000 - 2 ** 31 - 8] }
  expect(rleFrString(rleToString(r), r.h, r.w).counts).toEqual(r.counts)
})

test('masks of different sizes, or runs that miss pixels, are not merged', () => {
  const a = rleFrPoly([0, 0, 5, 0, 0, 5], 10, 10)
  expect(() => rleMerge([a, rleFrPoly([0, 0, 5, 0, 0, 5], 10, 11)])).toThrow("can't be merged")
  expect(() => rleMerge([a, { h: 10, w: 10, counts: [50, 20] }])).toThrow('cover 70 pixels')
})

test('a polygon whose outline is longer than it may be is refused before it takes the memory', () => {
  // Corner to corner of a 1000 x 1000 image 10 times: 10 x 5000 points at 5 times the size, and the 10 corners
  const zigzag = Array.from({ length: 10 }, (_, i) => (i % 2 ? [1000, 1000] : [0, 0])).flat()
  expect(rleFrPoly(zigzag, 1000, 1000, 50010).counts).toEqual(rleFrPoly(zigzag, 1000, 1000).counts)
  expect(() => rleFrPoly(zigzag, 1000, 1000, 50009)).toThrow('Its outline is 50010 points long at 5 times the size, more than the 50009 it may be')
})

describe('mask shapes in an image', () => {
  // A seeded generator, so the cases are the same on every run
  let seed = 7
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  const bits = (x: number, y: number, width: number, height: number): MaskBits => {
    const data = new Uint8Array(width * height)
    for (let i = 0; i < data.length; i++) data[i] = random() < 0.4 ? 1 : 0
    return { x, y, width, height, data }
  }

  test('pasting and encoding the image-sized mask gives the runs made from the mask alone', () => {
    const cases: [MaskBits, number, number][] = [
      [bits(2, 3, 5, 4), 12, 10], [bits(-3, -2, 8, 7), 12, 10], [bits(8, 9, 6, 6), 12, 10], [bits(0, 0, 10, 12), 12, 10],
      [bits(20, 2, 3, 3), 12, 10], [bits(2, 20, 3, 3), 12, 10], [bits(-6, 4, 3, 3), 12, 10], [bits(0, 0, 0, 0), 5, 5], [bits(1, 1, 1, 1), 3, 3]
    ]
    for (let i = 0; i < 30; i++) cases.push([bits(Math.floor(random() * 30) - 10, Math.floor(random() * 30) - 10, 1 + Math.floor(random() * 20), 1 + Math.floor(random() * 20)), 1 + Math.floor(random() * 25), 1 + Math.floor(random() * 25)])
    for (const [b, h, w] of cases) {
      const mask = new Uint8Array(h * w)
      const cut = pasteMask(b, mask, h, w)
      const r = rleFrMaskBits(b, h, w)
      expect(r.counts).toEqual(rleEncode(mask, h, w).counts)
      expect(rleLength(r)).toBe(h * w)
      let set = 0
      for (const v of b.data) set += v
      expect(rleArea(r) + cut).toBe(set)
    }
  })

  test('cutting a mask shape back out of the image gives its pixels, and 0 past the edges', () => {
    const b = bits(3, 2, 6, 5)
    const r = rleFrMaskBits(b, 10, 12)
    expect(rleToMaskBits(r, b)).toEqual(b)
    const wider = rleToMaskBits(r, { x: -2, y: -1, width: 16, height: 13 })
    for (let y = 0; y < 13; y++) {
      for (let x = 0; x < 16; x++) {
        const bx = x - 2 - 3, by = y - 1 - 2
        const inside = bx >= 0 && by >= 0 && bx < 6 && by < 5
        expect(wider.data[y * 16 + x], `${x}, ${y}`).toBe(inside ? b.data[by * 6 + bx] : 0)
      }
    }
  })

  test("a box of a mask moved into an image is, as runs, the box's pixels cut out, placed and encoded, without making them", () => {
    const masks: Rle[] = [rleFrMaskBits(bits(0, 0, 9, 7), 7, 9), rleFrPoly([1, 1, 7, 2, 4, 6], 7, 9), { h: 7, w: 9, counts: [63] }, { h: 7, w: 9, counts: [0, 63] }]
    for (let i = 0; i < 20; i++) {
      const h = 1 + Math.floor(random() * 12), w = 1 + Math.floor(random() * 12)
      masks.push(rleFrMaskBits(bits(0, 0, w, h), h, w))
    }
    let cases = 0
    for (const r of masks) {
      for (let i = 0; i < 25; i++) {
        const int = (from: number, to: number) => from + Math.floor(random() * (to - from + 1))
        // Boxes inside, across and past the mask's edges, placed inside, across and past the image's
        const box = { x: int(-4, r.w + 2), y: int(-4, r.h + 2), width: int(0, r.w + 4), height: int(0, r.h + 4) }
        const h = int(1, 14), w = int(1, 14), x = int(-8, w + 2), y = int(-8, h + 2)
        const want = rleFrMaskBits({ ...rleToMaskBits(r, box), x, y }, h, w)
        expect(rleMove(r, box, x, y, h, w), JSON.stringify({ r, box, x, y, h, w })).toEqual(want)
        cases++
      }
    }
    expect(cases).toBe(600)
    // Runs that stop short of the mask leave the rest clear, as rleToMaskBits does
    expect(rleMove({ h: 2, w: 2, counts: [1, 1] }, { x: 0, y: 0, width: 2, height: 2 }, 0, 0, 2, 2).counts).toEqual([1, 1, 2])
  })

  test("a mask moved into an image takes steps for the image's columns and the mask's runs, not a byte a pixel", () => {
    // 2^40 pixels of mask, all set, with its top left 3 across and 4 down a 10 x 10 image
    const r = { h: 2 ** 20, w: 2 ** 20, counts: [0, 2 ** 40] }
    expect(rleMove(r, { x: 0, y: 0, width: 2 ** 20, height: 2 ** 20 }, 3, 4, 10, 10).counts).toEqual([34, 6, ...Array(6).fill([4, 6]).flat()])
  })

  test('without a box, the cut-out is the set pixels\' own box', () => {
    const r = rleFrPoly([2, 3, 9, 3, 9, 7, 2, 7], 10, 12)
    const [x, y, width, height] = rleToBbox(r)
    const m = rleToMaskBits(r)
    expect([m.x, m.y, m.width, m.height]).toEqual([x, y, width, height])
    expect(m.data.every((v) => v === 1)).toBe(true)
    expect(rleToMaskBits({ h: 3, w: 3, counts: [9] })).toEqual({ x: 0, y: 0, width: 0, height: 0, data: new Uint8Array(0) })
  })
})
