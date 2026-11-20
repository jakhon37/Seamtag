import AdmZip from 'adm-zip'
import { mkdtempSync, readFileSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { crc32, deflateRawSync } from 'zlib'
import { describe, expect, test } from 'vitest'
import { extractEntry } from '../../src/main/zip'

const dir = mkdtempSync(path.join(tmpdir(), 'seamtag-zip-'))
let n = 0
function saved(bytes: Buffer) {
  const file = path.join(dir, `${++n}.zip`)
  writeFileSync(file, bytes)
  return file
}
const out = () => path.join(dir, `${++n}.out`)

/** Bytes that a deflate packs a little, as an ONNX file's weights are */
function weights(length: number, seed: number) {
  const bytes = Buffer.alloc(length)
  for (let i = 0, x = seed; i < length; i++) {
    x = (x * 1103515245 + 12345) >>> 0
    bytes[i] = i % 4 === 3 ? 0x3f : x >>> 24
  }
  return bytes
}

type File = { name: string; data: Buffer; method?: number; packed?: Buffer; crc?: number; flags?: number }
/**
 * A zip laid out as Info-ZIP's zip 3.0 writes RTMPose's: a time field in each local header longer
 * than the one in its directory entry, so the data starts where the local header says, not the
 * directory. `packed`, `crc` and `flags` stand in for the true ones, to damage it.
 */
function infoZip(files: File[], comment = Buffer.alloc(0)) {
  const parts: Buffer[] = [], directory: Buffer[] = []
  let offset = 0
  const time = (extra: number) => {
    const b = Buffer.alloc(4 + extra)
    b.writeUInt16LE(0x5455, 0)
    b.writeUInt16LE(extra, 2)
    b[4] = 3
    return b
  }
  for (const f of files) {
    const method = f.method ?? 8
    const packed = f.packed ?? (method === 8 ? deflateRawSync(f.data) : f.data)
    const name = Buffer.from(f.name)
    const local = Buffer.alloc(30), entry = Buffer.alloc(46)
    const localTime = time(9), entryTime = time(5)
    for (const [header, at] of [[local, 0], [entry, 2]] as const) {
      header.writeUInt16LE(20, 4 + at)
      header.writeUInt16LE(f.flags ?? 0, 6 + at)
      header.writeUInt16LE(method, 8 + at)
      header.writeUInt32LE(f.crc ?? crc32(f.data), 14 + at)
      header.writeUInt32LE(packed.length, 18 + at)
      header.writeUInt32LE(f.data.length, 22 + at)
      header.writeUInt16LE(name.length, 26 + at)
    }
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(localTime.length, 28)
    entry.writeUInt32LE(0x02014b50, 0)
    entry.writeUInt16LE(0x031e, 4)
    entry.writeUInt16LE(entryTime.length, 30)
    entry.writeUInt32LE(offset, 42)
    parts.push(local, name, localTime, packed)
    directory.push(entry, name, entryTime)
    offset += local.length + name.length + localTime.length + packed.length
  }
  const central = Buffer.concat(directory)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(files.length, 8)
  end.writeUInt16LE(files.length, 10)
  end.writeUInt32LE(central.length, 12)
  end.writeUInt32LE(offset, 16)
  end.writeUInt16LE(comment.length, 20)
  return Buffer.concat([...parts, central, end, comment])
}

const MODEL = '20230831/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx'

describe('a zip written by another app', () => {
  test('a deflated entry in a nested folder, found by its whole path', async () => {
    const zip = new AdmZip()
    const data = weights(300_000, 1)
    zip.addFile('20230831/', Buffer.alloc(0))
    zip.addFile('end2end.onnx', weights(1000, 2))
    zip.addFile('other/end2end.onnx', weights(1000, 3))
    zip.addFile(MODEL, data)
    expect(zip.getEntry(MODEL)!.header.method).toBe(8)
    const file = out()
    await extractEntry(saved(zip.toBuffer()), MODEL, file)
    expect(readFileSync(file).equals(data)).toBe(true)
  })

  test('a stored entry', async () => {
    const zip = new AdmZip()
    const data = weights(5000, 4)
    zip.addFile('model/stored.onnx', data)
    zip.getEntry('model/stored.onnx')!.header.method = 0
    zip.addFile('model/notes.txt', Buffer.from('not this one'))
    const bytes = zip.toBuffer()
    expect(new AdmZip(bytes).getEntry('model/stored.onnx')!.header.compressedSize).toBe(5000)
    const file = out()
    await extractEntry(saved(bytes), 'model/stored.onnx', file)
    expect(readFileSync(file).equals(data)).toBe(true)
  })
})

describe("Info-ZIP's layout, as RTMPose's zips have it", () => {
  const data = weights(70_000, 5)
  const files = [
    { name: '20230831/', data: Buffer.alloc(0), method: 0 },
    { name: '20230831/deploy.json', data: Buffer.from('{"version": "1.2.0"}') },
    { name: '20230831/empty.txt', data: Buffer.alloc(0), method: 0 },
    { name: '20230831/end2end.onnx', data }
  ]
  // The comment holds an end record's signature, which mustn't be taken for the real one
  const bytes = infoZip(files, Buffer.from('made by a test PK\x05\x06 with a signature in it'))

  test('the data starts after the local header, whose extra field is longer than the directory says', async () => {
    // Another reader agrees the layout is sound (adm-zip, like Python's zipfile, takes a signature in the comment for the end)
    expect(new AdmZip(infoZip(files)).readFile('20230831/end2end.onnx')!.equals(data)).toBe(true)
    const file = out()
    await extractEntry(saved(bytes), '20230831/end2end.onnx', file)
    expect(readFileSync(file).equals(data)).toBe(true)
  })

  test('an empty entry', async () => {
    const file = out()
    await extractEntry(saved(bytes), '20230831/empty.txt', file)
    expect(readFileSync(file).length).toBe(0)
  })

  test('an entry the zip lacks', async () => {
    await expect(extractEntry(saved(bytes), 'end2end.onnx', out())).rejects.toThrow('The zip has no end2end.onnx')
    await expect(extractEntry(saved(bytes), '20230831/END2END.onnx', out())).rejects.toThrow('has no')
  })
})

describe('bad data', () => {
  const data = weights(20_000, 6)
  const name = 'm/end2end.onnx'

  test("a file that isn't a zip, or a zip cut short", async () => {
    await expect(extractEntry(saved(weights(4000, 7)), name, out())).rejects.toThrow("It isn't a zip file")
    await expect(extractEntry(saved(Buffer.alloc(0)), name, out())).rejects.toThrow("It isn't a zip file")
    const whole = infoZip([{ name, data }])
    await expect(extractEntry(saved(whole.subarray(0, whole.length - 30)), name, out())).rejects.toThrow("It isn't a zip file")
    // The end record kept, the entries' data gone
    const end = whole.subarray(whole.length - 22)
    await expect(extractEntry(saved(Buffer.concat([whole.subarray(0, 100), end])), name, out())).rejects.toThrow('cut short')
  })

  test("a damaged deflate stream, or bytes that don't match the CRC-32", async () => {
    const packed = deflateRawSync(data)
    packed.fill(0xff, 10, 40)
    await expect(extractEntry(saved(infoZip([{ name, data, packed }])), name, out())).rejects.toThrow(`${name} is damaged in the zip`)
    const stored = Buffer.from(data)
    stored[1234] ^= 1
    await expect(extractEntry(saved(infoZip([{ name, data, method: 0, packed: stored }])), name, out())).rejects.toThrow("doesn't match its size and CRC-32")
    await expect(extractEntry(saved(infoZip([{ name, data, crc: (crc32(data) ^ 1) >>> 0 }])), name, out())).rejects.toThrow("doesn't match its size and CRC-32")
  })

  test('more bytes than the directory says stops there', async () => {
    const bigger = infoZip([{ name, data: Buffer.concat([data, data]) }])
    // The sizes say 20,000, the data unpacks to 40,000
    for (const at of [22, bigger.readUInt32LE(bigger.length - 6) + 24]) bigger.writeUInt32LE(data.length, at)
    await expect(extractEntry(saved(bigger), name, out())).rejects.toThrow("doesn't match its size and CRC-32")
  })

  test("packing this reader doesn't know, and encryption", async () => {
    await expect(extractEntry(saved(infoZip([{ name, data, method: 12 }])), name, out())).rejects.toThrow('method 12')
    await expect(extractEntry(saved(infoZip([{ name, data, flags: 1 }])), name, out())).rejects.toThrow('encrypted')
  })
})
