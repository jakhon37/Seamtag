// A small zip reader, for models that are published only inside a zip, as RTMPose's ONNX files
// are: the zip's central directory says where each entry is and how it's packed, and a stored or
// deflated entry is written out through Node's zlib, checked against the size and CRC-32 the
// directory gives. Zip64, encrypted and split zips aren't read: no model's zip is any of those.
import { createReadStream, createWriteStream, promises as fs } from 'fs'
import type { FileHandle } from 'fs/promises'
import { Readable, Transform } from 'stream'
import { pipeline } from 'stream/promises'
import { crc32, createInflateRaw } from 'zlib'

/** An entry as the central directory gives it: `packed` is its size in the zip, `offset` where its local header is */
type Entry = { name: string; flags: number; method: number; crc: number; packed: number; size: number; offset: number }

const END = 0x06054b50, CENTRAL = 0x02014b50, LOCAL = 0x04034b50
const STORED = 0, DEFLATED = 8
// The end record's fixed part, which a comment of up to 65,535 bytes may follow
const END_SIZE = 22

async function readAt(handle: FileHandle, position: number, length: number) {
  const buffer = Buffer.alloc(length)
  const { bytesRead } = await handle.read(buffer, 0, length, position)
  if (bytesRead !== length) throw new Error('The zip is cut short')
  return buffer
}

async function entries(handle: FileHandle, size: number): Promise<Entry[]> {
  const tail = await readAt(handle, Math.max(0, size - END_SIZE - 0xffff), Math.min(size, END_SIZE + 0xffff))
  // The last end record whose comment runs to the end of the file, so a comment can't pass for one
  let end = -1
  for (let i = tail.length - END_SIZE; i >= 0 && end < 0; i--) {
    if (tail.readUInt32LE(i) === END && i + END_SIZE + tail.readUInt16LE(i + 20) === tail.length) end = i
  }
  if (end < 0) throw new Error("It isn't a zip file")
  const count = tail.readUInt16LE(end + 10), dirSize = tail.readUInt32LE(end + 12), dirOffset = tail.readUInt32LE(end + 16)
  if (tail.readUInt16LE(end + 4) || tail.readUInt16LE(end + 6)) throw new Error('Zips split across files are not read')
  if (count === 0xffff || dirSize === 0xffffffff || dirOffset === 0xffffffff) throw new Error('Zip64 files are not read')
  if (dirOffset + dirSize > size) throw new Error('The zip is cut short')
  const dir = await readAt(handle, dirOffset, dirSize)
  const out: Entry[] = []
  for (let k = 0, p = 0; k < count; k++) {
    if (p + 46 > dir.length || dir.readUInt32LE(p) !== CENTRAL) throw new Error("The zip's directory is damaged")
    const nameLength = dir.readUInt16LE(p + 28)
    // Names are compared as UTF-8: the ASCII paths model zips use read the same in any encoding
    out.push({
      name: dir.toString('utf8', p + 46, p + 46 + nameLength), flags: dir.readUInt16LE(p + 8), method: dir.readUInt16LE(p + 10),
      crc: dir.readUInt32LE(p + 16), packed: dir.readUInt32LE(p + 20), size: dir.readUInt32LE(p + 24), offset: dir.readUInt32LE(p + 42)
    })
    p += 46 + nameLength + dir.readUInt16LE(p + 30) + dir.readUInt16LE(p + 32)
  }
  return out
}

/**
 * Writes the entry `name` (its whole path in the zip) to `out`. It fails when the zip has no such
 * entry, packs it in a way this reader doesn't know, or holds bytes that don't match its size and
 * CRC-32; `out` may then hold part of it, for the caller to delete.
 */
export async function extractEntry(zip: string, name: string, out: string) {
  const handle = await fs.open(zip, 'r')
  let entry: Entry, start: number
  try {
    const { size } = await handle.stat()
    const found = (await entries(handle, size)).find((e) => e.name === name)
    if (!found) throw new Error(`The zip has no ${name}`)
    entry = found
    if (entry.flags & 1) throw new Error(`${name} is encrypted in the zip`)
    if (entry.method !== STORED && entry.method !== DEFLATED) throw new Error(`${name} is packed in a way this app can't read (method ${entry.method})`)
    // The local header's name and extra field come before the data, and its extra field may differ from the directory's
    const local = await readAt(handle, entry.offset, 30)
    if (local.readUInt32LE(0) !== LOCAL) throw new Error(`${name} is damaged in the zip`)
    start = entry.offset + 30 + local.readUInt16LE(26) + local.readUInt16LE(28)
    if (start + entry.packed > size) throw new Error('The zip is cut short')
  } finally {
    await handle.close()
  }
  let crc = 0, length = 0
  const damaged = () => new Error(`${name} is damaged in the zip: it doesn't match its size and CRC-32`)
  const check = new Transform({
    transform(chunk: Buffer, _encoding, done) {
      crc = crc32(chunk, crc)
      length += chunk.length
      // More than the directory says is damage too, and stops a bad entry from filling the disk
      done(length > entry.size ? damaged() : null, chunk)
    },
    flush(done) {
      done(length !== entry.size || crc !== entry.crc ? damaged() : null)
    }
  })
  const packed = entry.packed ? createReadStream(zip, { start, end: start + entry.packed - 1 }) : Readable.from([])
  const unpack = entry.method === DEFLATED ? [createInflateRaw()] : []
  await pipeline([packed, ...unpack, check, createWriteStream(out)]).catch((e: NodeJS.ErrnoException) => {
    // zlib's own errors say only "invalid block type" or the like
    throw e.code?.startsWith('Z_') ? new Error(`${name} is damaged in the zip: ${e.message}`) : e
  })
}
