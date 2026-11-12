/*!
 * COCO's run-length encoded masks, ported from pycocotools 2.0.11's common/maskApi.c
 * (https://github.com/ppwwyyxx/cocoapi, the source of the pycocotools package), so the areas,
 * boxes, runs and RLE strings Seamtag writes are pycocotools' own, to the pixel and the byte.
 * Under its Simplified BSD licence:
 *
 * Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */
import type { MaskBits } from './dataset'

// Plain TypeScript with no imports at run time, so the main process, the window and the tests
// all use it. Masks here are COCO's: h rows by w columns, read down each column from the left.

/**
 * A mask as COCO keeps it: the lengths of the runs of 0s and 1s down its columns, in turn,
 * starting with 0s, so a mask whose first pixel is set starts with a run of none
 */
export type Rle = { h: number; w: number; counts: number[] }

// C's (int) cast of a double cuts towards zero. x86-64's cvttsd2si, which pycocotools' wheels
// run, gives INT_MIN for NaN and values out of range; rleFrPoly makes a NaN at a repeated point.
const int = (x: number) => (x > -2147483649 && x < 2147483648 ? x | 0 : -2147483648)

/** rleEncode: a column-major mask to its runs. The mask holds 0s and 1s, as COCO's do. */
export function rleEncode(mask: ArrayLike<number>, h: number, w: number): Rle {
  const counts: number[] = []
  let p = 0, c = 0
  for (let j = 0, a = h * w; j < a; j++) {
    if (mask[j] !== p) { counts.push(c); c = 0; p = mask[j] }
    c++
  }
  counts.push(c)
  return { h, w, counts }
}

/**
 * rleDecode: the runs back to a column-major mask of 0s and 1s. Runs that overflow the mask
 * throw, as pycocotools 2.0.11 refuses them; runs that stop short leave the rest 0, where
 * pycocotools leaves whatever was in memory.
 */
export function rleDecode(r: Rle): Uint8Array {
  const out = new Uint8Array(r.h * r.w)
  let at = 0, v = 0
  for (const n of r.counts) {
    if (at + n > out.length) throw new Error(`Its runs add up to more than its ${r.h} x ${r.w} pixels`)
    if (v) out.fill(1, at, at + n)
    at += n
    v ^= 1
  }
  return out
}

/** rleArea: how many pixels are set */
export function rleArea(r: Rle): number {
  let a = 0
  // uint in C
  for (let j = 1; j < r.counts.length; j += 2) a = (a + r.counts[j]) >>> 0
  return a
}

/** rleToBbox as pycocotools 2.0.11 fixed it: [x, y, width, height] of the set pixels, whole pixels; all 0 for an empty mask */
export function rleToBbox(r: Rle): [number, number, number, number] {
  const { h, w, counts } = r
  // An odd last run is 0s
  const m = Math.floor(counts.length / 2) * 2
  if (m === 0) return [0, 0, 0, 0]
  let xs = w, ys = h, xe = 0, ye = 0, cc = 0
  for (let j = 0; j < m; j++) {
    const start = cc
    cc = (cc + counts[j]) >>> 0
    if (j % 2 === 0 || counts[j] === 0) continue
    const yStart = start % h, xStart = (start - yStart) / h
    const yEnd = (cc - 1) % h, xEnd = (cc - 1 - yEnd) / h
    xs = Math.min(xs, xStart)
    xe = Math.max(xe, xEnd)
    // A run that goes on into the next column covers the whole height
    if (xStart < xEnd) { ys = 0; ye = h - 1 } else { ys = Math.min(ys, yStart); ye = Math.max(ye, yEnd) }
  }
  // uint in C, so a mask whose only runs of 1s are empty wraps round, as pycocotools' does
  return [xs, ys, (xe - xs + 1) >>> 0, (ye - ys + 1) >>> 0]
}

/**
 * rleFrPoly: a polygon (x0, y0, x1, y1, ...) filled as pycocotools fills it. The outline is
 * traced at 5 times the size with C's rounding, and a pixel is in when its centre is. Tracing
 * takes 8 bytes a point; past `most` points it throws instead.
 */
export function rleFrPoly(xy: ArrayLike<number>, h: number, w: number, most = Infinity): Rle {
  // pycocotools' frPoly passes int(len(p) / 2) points: an odd last number is dropped
  const k = Math.floor(xy.length / 2)
  if (!k) return { h, w, counts: [h * w] }
  // upsample and get discrete points densely along the entire boundary
  const scale = 5
  const x = new Int32Array(k + 1), y = new Int32Array(k + 1)
  for (let j = 0; j < k; j++) x[j] = int(scale * xy[j * 2] + 0.5)
  x[k] = x[0]
  for (let j = 0; j < k; j++) y[j] = int(scale * xy[j * 2 + 1] + 0.5)
  y[k] = y[0]
  let m = 0
  for (let j = 0; j < k; j++) m += Math.max(Math.abs(x[j] - x[j + 1]), Math.abs(y[j] - y[j + 1])) + 1
  if (m > most) throw new Error(`Its outline is ${m} points long at 5 times the size, more than the ${most} it may be`)
  const u = new Int32Array(m), v = new Int32Array(m)
  m = 0
  for (let j = 0; j < k; j++) {
    let xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1]
    const dx = Math.abs(xe - xs), dy = Math.abs(ys - ye)
    const flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye)
    if (flip) { let t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t }
    const s = dx >= dy ? (ye - ys) / dx : (xe - xs) / dy
    if (dx >= dy) {
      for (let d = 0; d <= dx; d++) {
        const t = flip ? dx - d : d
        u[m] = t + xs; v[m] = int(ys + s * t + 0.5); m++
      }
    } else {
      for (let d = 0; d <= dy; d++) {
        const t = flip ? dy - d : d
        v[m] = t + ys; u[m] = int(xs + s * t + 0.5); m++
      }
    }
  }
  // get points along the y-boundary and downsample
  const a: number[] = []
  for (let j = 1; j < m; j++) {
    if (u[j] === u[j - 1]) continue
    let xd = u[j] < u[j - 1] ? u[j] : u[j] - 1
    xd = (xd + 0.5) / scale - 0.5
    if (Math.floor(xd) !== xd || xd < 0 || xd > w - 1) continue
    let yd = v[j] < v[j - 1] ? v[j] : v[j - 1]
    yd = (yd + 0.5) / scale - 0.5
    if (yd < 0) yd = 0
    else if (yd > h) yd = h
    yd = Math.ceil(yd)
    a.push(xd * h + yd)
  }
  // compute the runs from the y-boundary points
  a.push(h * w)
  a.sort((p, q) => p - q)
  let p = 0
  for (let j = 0; j < a.length; j++) { const t = a[j]; a[j] -= p; p = t }
  // two boundaries at one place cancel out
  const counts = [a[0]]
  for (let j = 1; j < a.length;) {
    if (a[j] > 0) counts.push(a[j++])
    else if (++j < a.length) counts[counts.length - 1] += a[j++]
  }
  return { h, w, counts }
}

/** rleFrBbox: a box, [x, y, width, height], filled as the polygon of its corners */
export function rleFrBbox(bb: ArrayLike<number>, h: number, w: number): Rle {
  const xs = bb[0], xe = xs + bb[2], ys = bb[1], ye = ys + bb[3]
  return rleFrPoly([xs, ys, xs, ye, xe, ye, xe, ys], h, w)
}

/** The number of pixels the runs cover: h x w for a whole mask */
export function rleLength(r: Rle): number {
  let n = 0
  for (const c of r.counts) n += c
  return n
}

// pycocotools gives an empty 0 x 0 mask for masks of different sizes, and loops forever on runs
// that don't cover their mask; both are mistakes, so they throw here
function checkRuns(r: Rle, h: number, w: number) {
  if (r.h !== h || r.w !== w) throw new Error(`Masks of ${h} x ${w} and ${r.h} x ${r.w} pixels can't be merged`)
  if (rleLength(r) !== h * w) throw new Error(`Its runs cover ${rleLength(r)} pixels, not the ${h} x ${w} of its mask`)
}

/** rleMerge: the union of masks of one size, or where they all overlap */
export function rleMerge(rles: Rle[], intersect = false): Rle {
  if (!rles.length) return { h: 0, w: 0, counts: [] }
  const { h, w } = rles[0]
  let cnts = rles[0].counts.slice()
  if (rles.length > 1) for (const r of rles) checkRuns(r, h, w)
  for (let i = 1; i < rles.length; i++) {
    const A = cnts, B = rles[i].counts
    let ca = A[0] ?? 0, cb = B[0] ?? 0, v = 0, va = 0, vb = 0, a = 1, b = 1, cc = 0, ct = 1
    cnts = []
    while (ct > 0) {
      const c = Math.min(ca, cb)
      cc += c; ct = 0
      ca -= c
      if (!ca && a < A.length) { ca = A[a++]; va ^= 1 }
      ct += ca
      cb -= c
      if (!cb && b < B.length) { cb = B[b++]; vb ^= 1 }
      ct += cb
      const vp = v
      v = intersect ? va & vb : va | vb
      if (v !== vp || ct === 0) { cnts.push(cc); cc = 0 }
    }
  }
  return { h, w, counts: cnts }
}

/**
 * rleToString: the runs as COCO's compressed counts. Like LEB128 at 5 bits a character, in
 * ASCII 48 to 111, and from the fourth run on each is kept as its difference from the run two back.
 */
export function rleToString(r: Rle): string {
  const out: number[] = []
  const c = r.counts
  for (let i = 0; i < c.length; i++) {
    let x = c[i]
    if (i > 2) x -= c[i - 2]
    let more = true
    while (more) {
      // x & 0x1f and x >>= 5 on a signed long, as arithmetic so counts past 2^31 keep their bits
      let ch = ((x % 32) + 32) % 32
      x = Math.floor(x / 32)
      more = ch & 0x10 ? x !== -1 : x !== 0
      if (more) ch |= 0x20
      out.push(ch + 48)
    }
  }
  let s = ''
  for (let i = 0; i < out.length; i += 8192) s += String.fromCharCode(...out.slice(i, i + 8192))
  return s
}

/**
 * rleFrString: COCO's compressed counts back to runs. Throws for a character outside 0 to o or
 * a string that stops in the middle of a number, which pycocotools reads past the end of.
 */
export function rleFrString(s: string, h: number, w: number): Rle {
  const counts: number[] = []
  let p = 0
  while (p < s.length) {
    let x = 0, k = 0, more = true
    while (more) {
      if (p >= s.length) throw new Error('Its counts stop in the middle of a number')
      const c = s.charCodeAt(p) - 48
      if (c < 0 || c > 63) throw new Error(`Its counts hold ${JSON.stringify(s[p])}, which isn't one of COCO's characters`)
      // 50 bits is past any image's size, and as far as a double holds whole numbers
      if (k === 10) throw new Error('Its counts hold a number too large for any image')
      x += (c & 0x1f) * 2 ** (5 * k)
      more = (c & 0x20) !== 0
      p++; k++
      // the sign bit of the last character fills the bits above it
      if (!more && c & 0x10) x -= 2 ** (5 * k)
    }
    if (counts.length > 2) x += counts[counts.length - 2]
    // (uint) in C
    counts.push(x >>> 0)
  }
  return { h, w, counts }
}

// Round half to even, as numpy's round, which labelme uses for a mask's place
function roundHalfEven(x: number) {
  const r = Math.round(x)
  // + 0 makes Math.round's -0 a plain 0
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r) + 0
}

/**
 * A mask shape's pixels pasted into a column-major h x w mask, set pixels only. Its x and y are
 * whole pixels (halves round to even, as labelme's numpy does). Pixels past the image's edges
 * are cut off; returns how many set pixels were.
 */
export function pasteMask(bits: MaskBits, mask: Uint8Array, h: number, w: number): number {
  const x0 = roundHalfEven(bits.x), y0 = roundHalfEven(bits.y)
  let cut = 0
  for (let y = 0; y < bits.height; y++) {
    const iy = y0 + y
    for (let x = 0; x < bits.width; x++) {
      if (!bits.data[y * bits.width + x]) continue
      const ix = x0 + x
      if (ix < 0 || iy < 0 || ix >= w || iy >= h) cut++
      else mask[ix * h + iy] = 1
    }
  }
  return cut
}

/**
 * A mask shape's pixels as COCO's runs over an h x w image, cut off at its edges: what
 * rleEncode gives for pasteMask's result, without making a mask the size of the image
 */
export function rleFrMaskBits(bits: MaskBits, h: number, w: number): Rle {
  const x0 = roundHalfEven(bits.x), y0 = roundHalfEven(bits.y)
  const counts: number[] = []
  let p = 0, c = 0
  const run = (value: number, n: number) => {
    if (!n) return
    if (value !== p) { counts.push(c); c = 0; p = value }
    c += n
  }
  const top = Math.min(h, Math.max(0, y0)), bottom = Math.max(top, Math.min(h, y0 + bits.height))
  for (let ix = 0; ix < w; ix++) {
    const x = ix - x0
    if (x < 0 || x >= bits.width || top === bottom) { run(0, h); continue }
    run(0, top)
    for (let iy = top; iy < bottom; iy++) run(bits.data[(iy - y0) * bits.width + x] ? 1 : 0, 1)
    run(0, h - bottom)
  }
  counts.push(c)
  return { h, w, counts }
}

/**
 * The pixels of a mask inside a box (whole pixels of its own), placed with the box's top left at
 * x, y in an h x w image and cut off at its edges, as runs: what rleFrMaskBits gives for
 * rleToMaskBits(r, box) placed there, without making the pixels: a step for each of the image's
 * columns and each of the mask's runs, never a byte a pixel.
 */
export function rleMove(r: Rle, box: { x: number; y: number; width: number; height: number }, x: number, y: number, h: number, w: number): Rle {
  const counts: number[] = []
  let p = 0, c = 0
  const run = (value: number, n: number) => {
    if (!n) return
    if (value !== p) { counts.push(c); c = 0; p = value }
    c += n
  }
  // The mask's rows that land in the image, and the box's columns that do
  const top = Math.max(box.y, 0, box.y - y), bottom = Math.max(top, Math.min(box.y + box.height, r.h, box.y - y + h))
  const left = Math.max(0, -x), right = Math.min(box.width, w - x)
  if (right <= left) run(0, h * w)
  else {
    run(0, h * (x + left))
    // The run the mask's pixels have reached, j, and where it starts, at: each column starts past the last
    let j = 0, at = 0
    for (let k = left; k < right; k++) {
      const col = box.x + k
      if (col < 0 || col >= r.w || top === bottom) { run(0, h); continue }
      const above = y + top - box.y
      run(0, above)
      for (let a = col * r.h + top, b = col * r.h + bottom; a < b;) {
        while (j < r.counts.length && at + r.counts[j] <= a) at += r.counts[j++]
        // Runs that stop short of the mask leave the rest clear, as rleToMaskBits does
        if (j === r.counts.length) { run(0, b - a); break }
        const end = Math.min(b, at + r.counts[j])
        run(j % 2, end - a)
        a = end
      }
      run(0, h - above - (bottom - top))
    }
    run(0, h * (w - x - right))
  }
  counts.push(c)
  return { h, w, counts }
}

/**
 * The pixels of an RLE mask inside a box (the mask's own box when none is given), row by row
 * as a mask shape keeps them. Pixels of the box outside the mask are 0.
 */
export function rleToMaskBits(r: Rle, box?: { x: number; y: number; width: number; height: number }): MaskBits {
  let { x, y, width, height } = box ?? { x: 0, y: 0, width: 0, height: 0 }
  if (!box && rleArea(r)) [x, y, width, height] = rleToBbox(r)
  const data = new Uint8Array(width * height)
  const { h } = r
  let at = 0
  for (let j = 0; j < r.counts.length; j++) {
    const start = at, end = at + r.counts[j]
    at = end
    if (j % 2 === 0 || start === end) continue
    // a run of 1s, column by column
    for (let col = Math.floor(start / h); col * h < end; col++) {
      const bx = col - x
      if (bx < 0 || bx >= width) continue
      const from = Math.max(start, col * h) - col * h, to = Math.min(end, col * h + h) - col * h
      for (let row = Math.max(from, y); row < Math.min(to, y + height); row++) data[(row - y) * width + bx] = 1
    }
  }
  return { x, y, width, height, data }
}
