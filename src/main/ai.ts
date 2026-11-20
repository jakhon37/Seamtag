import { createHash } from 'crypto'
import { createReadStream, createWriteStream, existsSync, promises as fs, readFileSync } from 'fs'
import { homedir } from 'os'
import path from 'path'
import { gunzipSync } from 'zlib'
import { app, net } from 'electron'
import { InferenceSession, Tensor } from 'onnxruntime-node'
import type { Grid } from '../shared/polygonize'
import { COCO_PERSON } from '../shared/skeleton'
import { Tokenizer } from './clip'
import { nms, roundHalfEven } from './detections'
import { rgbPixels } from './images'
import { pilResize } from './resize'
import { extractEntry } from './zip'

// labelme 7's AI models, run as osam runs them: the same files from the same addresses, kept
// in osam's cache so either app can use a download. SAM2 and EfficientSAM outline one object
// from clicks or a box; YOLO-World finds boxes for words. labelme's SAM and SAM 3 aren't here:
// their encoders need single blocks of memory larger than Electron's allocator gives in this
// process (between 512 and 768 MiB; SAM's global attention takes about 800 MB, SAM 3's 1.7 GB),
// and past that it stops the app. Beside them, RTMPose finds a person's keypoints in a box.

/**
 * A file a model needs, kept in osam's cache by its hash. One published only in a zip has the
 * zip's address as `url`, and in `zip` the zip's own hash and size and the file's whole path in it;
 * only the file is kept. `mirror` is another address for the same download, tried when the first fails.
 */
type Blob = { url: string; hash: string; size: number; zip?: { hash: string; size: number; entry: string }; mirror?: string }
type Kind = 'sam2' | 'efficientsam' | 'yoloworld' | 'rtmpose'
export type License = { name: string; url: string }
type ModelDef = {
  name: string
  title: string
  kind: Kind
  blobs: Record<string, Blob>
  license: License
  /** What it takes: clicks, a box, words */
  points: boolean
  box: boolean
  text: boolean
  /** labelme's name for it among the text prompt's models */
  textTitle?: string
  /** A pose model's keypoints, in the order it gives them; its one blob is `model` */
  keypoints?: string[]
  /** What to know before downloading it, beyond its licence */
  note?: string
}

const EFFICIENT = 'https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225'
const SAM2 = 'https://github.com/wkentaro/osam/releases/download/sam2.1'

const APACHE = (url: string): License => ({ name: 'Apache License 2.0', url })
const EFFICIENT_LICENSE = APACHE('https://github.com/wkentaro/efficient-sam/blob/6aebcba09318c4dfe2f9560f7a3f8c42d8b01657/LICENSE')
const SAM2_LICENSE = APACHE('https://github.com/facebookresearch/sam2/blob/3a7889d905e38ca043c9b5a571ec1635bab678ac/LICENSE')
const PROMPTABLE = { points: true, box: true, text: false }

// OpenMMLab's RTMPose for COCO's 17 person keypoints, exported to ONNX with MMDeploy, as rtmlib
// downloads it: each in a zip with the files it was exported with. rtmlib's mirror on Hugging Face
// (rtmlib/tools/file.py) holds the same zips.
const RTMPOSE = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk'
const RTMPOSE_MIRROR = 'https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk'
/** One of OpenMMLab's RTMPose zips, by its name: the ONNX file inside it, and the zip itself */
const rtmposeZip = (name: string, onnx: { hash: string; size: number }, zip: { hash: string; size: number }): Blob => ({
  url: `${RTMPOSE}/${name}.zip`, mirror: `${RTMPOSE_MIRROR}/${name}.zip`, ...onnx,
  zip: { ...zip, entry: `20230831/rtmpose_onnx/${name}/end2end.onnx` }
})
// The weights carry no licence of their own; they're published by MMPose, which is under Apache-2.0
const POSE = {
  points: false, box: false, text: false, keypoints: COCO_PERSON, license: APACHE('https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/LICENSE'),
  note: 'Its weights are published by OpenMMLab with MMPose. It was trained on a mix of public pose datasets (Body7), whose own terms vary.'
}

export const MODELS: ModelDef[] = [
  {
    name: 'efficientsam:10m', title: 'EfficientSam (speed)', kind: 'efficientsam', license: EFFICIENT_LICENSE, ...PROMPTABLE,
    blobs: {
      encoder: { url: `${EFFICIENT}/efficient_sam_vitt_encoder.onnx`, hash: 'sha256:7a73ee65aa2c37237c89b4b18e73082f757ffb173899609c5d97a2bbd4ebb02d', size: 24799761 },
      decoder: { url: `${EFFICIENT}/efficient_sam_vitt_decoder.onnx`, hash: 'sha256:e1afe46232c3bfa3470a6a81c7d3181836a94ea89528aff4e0f2d2c611989efd', size: 16565728 }
    }
  },
  {
    name: 'efficientsam:latest', title: 'EfficientSam (accuracy)', kind: 'efficientsam', license: EFFICIENT_LICENSE, ...PROMPTABLE,
    blobs: {
      encoder: { url: `${EFFICIENT}/efficient_sam_vits_encoder.onnx`, hash: 'sha256:4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896', size: 89558337 },
      decoder: { url: `${EFFICIENT}/efficient_sam_vits_decoder.onnx`, hash: 'sha256:4727baf23dacfb51d4c16795b2ac382c403505556d0284e84c6ff3d4e8e36f22', size: 16565728 }
    }
  },
  {
    name: 'sam2:small', title: 'Sam2 (speed)', kind: 'sam2', license: SAM2_LICENSE, ...PROMPTABLE,
    blobs: {
      encoder: { url: `${SAM2}/sam2.1_small_preprocess.onnx`, hash: 'sha256:06016c6dfb146ce10e4dadfdf49e88a05c8d1f97a6b7c57e150e60d2d46a72e7', size: 162703365 },
      decoder: { url: `${SAM2}/sam2.1_small.onnx`, hash: 'sha256:153aaef5047a3b95285d90cbb39dad6c7b5821bfd944dbf56483f3f735936941', size: 20642497 }
    }
  },
  {
    name: 'sam2:latest', title: 'Sam2 (balanced)', kind: 'sam2', license: SAM2_LICENSE, ...PROMPTABLE,
    blobs: {
      encoder: { url: `${SAM2}/sam2.1_base_plus_preprocess.onnx`, hash: 'sha256:ce95c44082b4532c25ae01e11da3c9337dab7b04341455c09ae599dc9ae5c438', size: 306084742 },
      decoder: { url: `${SAM2}/sam2.1_base_plus.onnx`, hash: 'sha256:2ad091af889b20ad2035503b4355cd8924fcf0e29fa6536924c48dc220ecdc56', size: 20642497 }
    }
  },
  {
    name: 'sam2:large', title: 'Sam2 (accuracy)', kind: 'sam2', license: SAM2_LICENSE, ...PROMPTABLE,
    blobs: {
      encoder: { url: `${SAM2}/sam2.1_large_preprocess.onnx`, hash: 'sha256:ab676f957528918496990f242163fd6b41a7222ae255862e846d9ab35115c12e', size: 889342060 },
      decoder: { url: `${SAM2}/sam2.1_large.onnx`, hash: 'sha256:a3ebc6b8e254bd4ca1346901b9472bc2fae9e827cfd67d67e162d0ae2b1ec9a0', size: 20642521 }
    }
  },
  {
    name: 'yoloworld:latest', title: 'YOLO-World', textTitle: 'YOLO-World (fast)', kind: 'yoloworld', points: false, box: false, text: true,
    license: { name: 'GNU General Public License v3.0', url: 'https://github.com/wkentaro/yolo-world-onnx/blob/959674c1f64095e78ee3d9bb246d01f2443caf35/LICENSE' },
    blobs: {
      textual: { url: 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx', hash: 'sha256:55c85d8cbb096023781c1d13c557eb95d26034c111bd001b7360fdb7399eec68', size: 254120034 },
      yolo: { url: 'https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx', hash: 'sha256:92660c6456766439a2670cf19a8a258ccd3588118622a15959f39e253731c05d', size: 418730806 }
    }
  },
  {
    name: 'rtmpose-m', title: 'RTMPose-m (speed)', kind: 'rtmpose', ...POSE,
    blobs: {
      model: rtmposeZip('rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504',
        { hash: 'sha256:5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c', size: 54330655 },
        { hash: 'sha256:f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741', size: 50799818 })
    }
  },
  {
    name: 'rtmpose-l', title: 'RTMPose-l (accuracy)', kind: 'rtmpose', ...POSE,
    blobs: {
      model: rtmposeZip('rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504',
        { hash: 'sha256:62425affc6b9edba8f8a65f55f5e278fec5d905d0c90578e0f8503bdf3954587', size: 111117190 },
        { hash: 'sha256:fed40dcfafdbadaebaa9681569f6594220313bd90f29d8ff039256076745f349', size: 103712404 })
    }
  }
]

// Bigger images are segmented on a smaller copy: the models work at 1024 or less anyway, and
// EfficientSAM takes the image as it is, which costs memory. A pose model reads only the part of
// the image a box needs, shrunk when that's bigger.
const MAX_SIDE: Record<Kind, number> = { sam2: 4096, efficientsam: 2048, yoloworld: 4096, rtmpose: 4096 }

const cacheDir = () => process.env.SEAMTAG_MODELS || path.join(homedir(), '.cache', 'osam', 'models', 'blobs')
// osam names a blob by its hash, "sha256-…" since Windows can't have ":" in a name.
// Older osam used the hash as it is, and those downloads are used too.
export function blobPath(b: Blob) {
  const current = path.join(cacheDir(), b.hash.replace('sha256:', 'sha256-'))
  const older = path.join(cacheDir(), b.hash)
  return process.platform !== 'win32' && !existsSync(current) && existsSync(older) ? older : current
}

export function modelDef(name: string) {
  const m = MODELS.find((x) => x.name === name)
  if (!m) throw new Error(`Unknown model ${name}`)
  return m
}

const blobsOf = (m: ModelDef) => Object.values(m.blobs)
/** What's downloaded for a blob: the file, or the zip it comes in */
const downloadSize = (b: Blob) => b.zip?.size ?? b.size
const sizeOf = (m: ModelDef) => blobsOf(m).reduce((s, b) => s + downloadSize(b), 0)
const isDownloaded = (m: ModelDef) => blobsOf(m).every((b) => existsSync(blobPath(b)))

export function listModels() {
  return MODELS.map((m) => ({
    name: m.name, title: m.title, textTitle: m.textTitle, size: sizeOf(m), downloaded: isDownloaded(m), downloading: downloading.has(m.name),
    license: m.license, points: m.points, box: m.box, text: m.text, pose: m.keypoints !== undefined, keypoints: m.keypoints, note: m.note
  }))
}

async function sha256Of(file: string) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  return `sha256:${hash.digest('hex')}`
}

/** Streams `url` into `file`; a download that doesn't match `hash` is never used */
async function fetchFile(url: string, file: string, hash: string, signal: AbortSignal, progress: (done: number) => void) {
  const response = await net.fetch(url, { signal })
  if (!response.ok || !response.body) throw new Error(`${path.basename(url)}: HTTP ${response.status}`)
  const digest = createHash('sha256')
  const out = createWriteStream(file)
  let done = 0
  try {
    const reader = response.body.getReader()
    for (;;) {
      const { done: end, value } = await reader.read()
      if (end) break
      digest.update(value)
      if (!out.write(value)) await new Promise<void>((r) => out.once('drain', () => r()))
      done += value.length
      progress(done)
    }
    await new Promise<void>((resolve, reject) => out.end((e?: Error | null) => (e ? reject(e) : resolve())))
  } catch (e) {
    out.destroy()
    throw e
  }
  if (`sha256:${digest.digest('hex')}` !== hash) throw new Error(`${path.basename(url)} didn't match its checksum`)
}

async function fetchBlob(blob: Blob, signal: AbortSignal, progress: (done: number) => void) {
  const target = blobPath(blob)
  if (existsSync(target)) return
  await fs.mkdir(path.dirname(target), { recursive: true })
  const tmp = `${target}.${process.pid}.part`
  // A zip is downloaded beside it, checked, and deleted once its file is out and checked too
  const zip = blob.zip && { ...blob.zip, file: `${tmp}.zip` }
  try {
    const from = (url: string) => fetchFile(url, zip?.file ?? tmp, zip?.hash ?? blob.hash, signal, progress)
    // The mirror, when the first address fails for any reason; that failure is the one told
    await from(blob.url).catch((e) => {
      if (!blob.mirror || signal.aborted) throw e
      return from(blob.mirror).catch(() => { throw e })
    })
    if (zip) {
      await extractEntry(zip.file, zip.entry, tmp)
      if (signal.aborted) throw new Error('Download cancelled')
      if ((await sha256Of(tmp)) !== blob.hash) throw new Error(`${path.basename(zip.entry)} in ${path.basename(blob.url)} didn't match its checksum`)
    }
    await fs.rename(tmp, target)
  } catch (e) {
    await fs.rm(tmp, { force: true })
    throw signal.aborted ? new Error('Download cancelled') : e
  } finally {
    if (zip) await fs.rm(zip.file, { force: true })
  }
}

const downloading = new Map<string, { job: Promise<void>; abort: AbortController }>()
export function download(name: string, progress: (p: { name: string; model: string; done: number; total: number }) => void) {
  const m = modelDef(name)
  if (!downloading.has(name)) {
    const total = sizeOf(m)
    const abort = new AbortController()
    const job = (async () => {
      let before = 0
      for (const blob of blobsOf(m)) {
        await fetchBlob(blob, abort.signal, (d) => progress({ name: m.title, model: name, done: before + d, total }))
        before += downloadSize(blob)
      }
      progress({ name: m.title, model: name, done: total, total })
    })().finally(() => downloading.delete(name))
    downloading.set(name, { job, abort })
  }
  return downloading.get(name)!.job
}

export function cancelDownload(name: string) {
  downloading.get(name)?.abort.abort()
}

/** Deletes a model's files from osam's cache, which labelme shares */
export async function removeModel(name: string) {
  const m = modelDef(name)
  if (downloading.has(name)) throw new Error(`${m.title} is still downloading`)
  if (loaded?.name === name) await releaseLoaded()
  if (posing?.name === name) await releasePose()
  for (const b of blobsOf(m)) {
    await fs.rm(path.join(cacheDir(), b.hash.replace('sha256:', 'sha256-')), { force: true })
    if (process.platform !== 'win32') await fs.rm(path.join(cacheDir(), b.hash), { force: true })
  }
}

// One model in memory at a time, each part loaded when first needed; the embeddings of the
// last three images, as labelme keeps
let loaded: { name: string; parts: Map<string, Promise<InferenceSession>> } | null = null
type Embedding = { key: string; width: number; height: number; full: [number, number]; tensors: Record<string, Tensor> }
const embeddings: Embedding[] = []

// No memory arena or memory pattern: each tensor has a block of its own, freed after the run
// (the largest, in Sam2 (balanced) and Sam2 (accuracy), is 512 MiB). With them, both closed
// the app on their second image: the pattern asks for all of a run's tensors in one block, 820
// and 896 MiB, and to hold it the arena, which grows in regions of up to 1 GiB, asked for 1 GiB.
// Electron's allocator rounds an aligned request that size up to 2 GiB, more than it gives in
// one block, and stops the app rather than fail the request.
const SESSION_OPTIONS: InferenceSession.SessionOptions = {
  executionProviders: ['cpu'], graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false
}

async function part(m: ModelDef, key: string): Promise<InferenceSession> {
  if (!isDownloaded(m)) throw new Error(`${m.title} isn't downloaded yet`)
  if (loaded?.name !== m.name) {
    await releaseLoaded()
    loaded = { name: m.name, parts: new Map() }
  }
  let session = loaded.parts.get(key)
  if (!session) {
    session = InferenceSession.create(blobPath(m.blobs[key]), SESSION_OPTIONS)
    loaded.parts.set(key, session)
    session.catch(() => loaded?.parts.delete(key))
  }
  return session
}

// A pose model has a session of its own beside that one model, so a pose found between two clicks
// of SAM2 leaves SAM2 loaded and the images it has encoded kept
let posing: { name: string; session: Promise<InferenceSession> } | null = null

/** A pose model, loaded when first needed: its session, its keypoints, and the longest side it reads a box's part of an image at */
export async function poseModel(name: string) {
  const m = modelDef(name)
  if (!m.keypoints) throw new Error(`${m.title} doesn't find poses`)
  if (!isDownloaded(m)) throw new Error(`${m.title} isn't downloaded yet`)
  let current = posing
  if (current?.name !== m.name) {
    // Taken at once, so a second call meanwhile shares it; the model it replaces is let go after
    const before = current?.session
    const session = InferenceSession.create(blobPath(m.blobs.model), SESSION_OPTIONS)
    current = posing = { name: m.name, session }
    session.catch(() => {
      if (posing?.session === session) posing = null
    })
    await (await before?.catch(() => null))?.release()
  }
  return { session: await current.session, keypoints: m.keypoints, maxSide: MAX_SIDE[m.kind] }
}

async function releasePose() {
  const session = posing?.session
  posing = null
  await (await session?.catch(() => null))?.release()
}

let tokenizer: Tokenizer | null = null
function clip() {
  if (!tokenizer) {
    const file = app.isPackaged
      ? path.join(process.resourcesPath, 'clip', 'bpe_simple_vocab_16e6.txt.gz')
      : path.join(app.getAppPath(), 'resources', 'clip', 'bpe_simple_vocab_16e6.txt.gz')
    tokenizer = new Tokenizer(gunzipSync(readFileSync(file)).toString('utf8'))
  }
  return tokenizer
}

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225]
type Pixels = { data: Uint8Array; width: number; height: number }

const inputSize = (s: InferenceSession) => ((s as any).inputMetadata?.[0]?.shape?.[2] as number | undefined) ?? 1024

async function encode(m: ModelDef, image: Pixels): Promise<Record<string, Tensor>> {
  const { width, height } = image
  if (m.kind === 'efficientsam') {
    const encoder = await part(m, 'encoder')
    const input = new Float32Array(3 * width * height)
    for (let i = 0; i < width * height; i++) for (let c = 0; c < 3; c++) input[c * width * height + i] = image.data[i * 3 + c] / 255
    const out = await encoder.run({ batched_images: new Tensor('float32', input, [1, 3, height, width]) })
    return { image_embeddings: out[encoder.outputNames[0]] }
  }
  // SAM2 takes the image squeezed to its input size, standardised as osam does
  const encoder = await part(m, 'encoder')
  const size = inputSize(encoder)
  const pixels = pilResize(image.data, width, height, size, size)
  const input = new Float32Array(3 * size * size)
  for (let i = 0; i < size * size; i++) for (let c = 0; c < 3; c++) input[c * size * size + i] = (pixels[i * 3 + c] / 255 - MEAN[c]) / STD[c]
  const out = await encoder.run({ input: new Tensor('float32', input, [1, 3, size, size]) })
  const [embed, feats1, feats2] = encoder.outputNames
  return { image_embeddings: out[embed], high_res_features1: out[feats1], high_res_features2: out[feats2] }
}

/** One result of a model: a box with inclusive pixel corners, the mask inside it, its score */
export type Annotation = { box: [number, number, number, number]; mask: Grid | null; score: number; text?: string }

/** imgviz's masks_to_bboxes and the crop inside, as osam returns a mask */
function annotate(mask: Uint8Array, width: number, height: number, score = 0): Annotation | null {
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
    }
  }
  if (x1 < 0) return null
  const w = x1 - x0 + 1, h = y1 - y0 + 1
  const data = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) data.set(mask.subarray((y0 + y) * width + x0, (y0 + y) * width + x0 + w), y * w)
  return { box: [x0, y0, x1, y1], mask: { data, width: w, height: h }, score }
}

/** Clicks (label 1 the object, 0 background) or a box (labels 2 and 3), in the image's pixels */
async function decodePrompt(m: ModelDef, e: Embedding, points: number[][], labels: number[]): Promise<Annotation | null> {
  const { width, height } = e
  const n = points.length
  const decoder = await part(m, 'decoder')
  let feeds: Record<string, Tensor>
  if (m.kind === 'efficientsam') {
    feeds = {
      image_embeddings: e.tensors.image_embeddings,
      batched_point_coords: new Tensor('float32', Float32Array.from(points.flat()), [1, 1, n, 2]),
      batched_point_labels: new Tensor('float32', Float32Array.from(labels), [1, 1, n]),
      orig_im_size: new Tensor('int64', BigInt64Array.from([BigInt(height), BigInt(width)]), [2])
    }
  } else {
    const size = inputSize(await part(m, 'encoder'))
    const sx = Math.fround(width / size), sy = Math.fround(height / size)
    const coords = points.map(([x, y]) => [Math.fround(x) / sx, Math.fround(y) / sy]).flat()
    feeds = {
      image_embeddings: e.tensors.image_embeddings,
      high_res_features1: e.tensors.high_res_features1,
      high_res_features2: e.tensors.high_res_features2,
      point_coords: new Tensor('float32', Float32Array.from(coords), [1, n, 2]),
      point_labels: new Tensor('float32', Float32Array.from(labels), [1, n]),
      mask_input: new Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
      has_mask_input: new Tensor('float32', Float32Array.from([0]), [1]),
      orig_im_size: new Tensor('int64', BigInt64Array.from([BigInt(height), BigInt(width)]), [2])
    }
  }
  const out = await decoder.run(feeds)
  const masks = out[decoder.outputNames[0]]
  const [mh, mw] = masks.dims.slice(-2)
  let pick = 0
  if (m.kind === 'sam2') {
    // SAM2 proposes several masks: the one it scores highest, as osam takes
    const scores = out[decoder.outputNames[1]].data as Float32Array
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[pick]) pick = i
  }
  const all = masks.data as Float32Array
  const mask = new Uint8Array(mw * mh)
  const offset = pick * mw * mh
  for (let i = 0; i < mask.length; i++) mask[i] = all[offset + i] > 0 ? 1 : 0
  return annotate(mask, mw, mh)
}

/** YOLO-World: boxes for words, as osam's _YoloWorld.generate finds them */
async function yoloWorld(m: ModelDef, image: Pixels, texts: string[]): Promise<Annotation[]> {
  const textual = await part(m, 'textual')
  const yolo = await part(m, 'yolo')
  const tokens = clip().tokenize([...texts, ' '], 77)
  const encoded = await textual.run({ input: new Tensor('int64', tokens, [texts.length + 1, 77]) })
  const features = Float32Array.from(encoded[textual.outputNames[0]].data as Float32Array)
  const dim = features.length / (texts.length + 1)
  for (let t = 0; t <= texts.length; t++) {
    let norm = 0
    for (let k = 0; k < dim; k++) norm += features[t * dim + k] ** 2
    norm = Math.fround(Math.sqrt(norm))
    for (let k = 0; k < dim; k++) features[t * dim + k] = Math.fround(features[t * dim + k] / norm)
  }
  // The image fitted into 640 square, centred on grey
  const size = 640
  const { width, height } = image
  const scale = size / Math.max(height, width)
  const rh = Math.trunc(height * scale), rw = Math.trunc(width * scale)
  const resized = pilResize(image.data, width, height, rw, rh)
  const ph = size - rh, pw = size - rw
  const top = Math.floor(ph / 2), left = Math.floor(pw / 2)
  const input = new Float32Array(3 * size * size).fill(114 / 255)
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      for (let c = 0; c < 3; c++) input[c * size * size + (y + top) * size + x + left] = resized[(y * rw + x) * 3 + c] / 255
    }
  }
  const out = await yolo.run({
    images: new Tensor('float32', input, [1, 3, size, size]),
    text_features: new Tensor('float32', features, [1, texts.length + 1, dim])
  })
  const scoreData = out.scores.data as Float32Array
  const boxData = out.boxes.data as Float32Array
  const classes = out.scores.dims[2]
  const count = out.scores.dims[1]
  const boxes = Array.from({ length: count }, (_, i) => Array.from(boxData.subarray(i * 4, i * 4 + 4)))
  const scores = Array.from({ length: count }, (_, i) => Array.from(scoreData.subarray(i * classes, (i + 1) * classes)))
  // osam asks for everything over 0.01, unsuppressed; the caller suppresses as the user chose
  return nms(boxes, scores, 1, 0.01, 1000)
    .filter((k) => k.label < texts.length)
    .map((k) => {
      const b = boxes[k.index]
      const f = Math.fround
      const fit = (v: number, pad: number, max: number) => roundHalfEven(Math.min(Math.max(f(f(v - pad) / f(size / Math.max(height, width))), 0), max))
      const box = [fit(b[0], left, width), fit(b[1], top, height), fit(b[2], left, width), fit(b[3], top, height)] as Annotation['box']
      return { box, mask: null, score: k.score, text: texts[k.label] }
    })
}

async function embedding(m: ModelDef, root: string, name: string): Promise<Embedding> {
  const key = `${m.name}\0${root}\0${name}`
  let found = embeddings.find((x) => x.key === key)
  if (!found) {
    const image = await rgbPixels(root, name, MAX_SIDE[m.kind])
    const tensors = await encode(m, image)
    found = { key, width: image.width, height: image.height, full: [image.fullWidth, image.fullHeight], tensors }
    embeddings.push(found)
    if (embeddings.length > 3) embeddings.shift()
  }
  return found
}

/** Results in the pixels of the copy the model saw, with that copy's size and the whole image's */
export type Generated = { annotations: Annotation[]; width: number; height: number; full: [number, number] }

/** Clicks or a box (points in the whole image's pixels) to the object they mark */
export async function generateFromPrompt(root: string, name: string, model: string, points: number[][], labels: number[]): Promise<Generated> {
  const m = modelDef(model)
  const box = labels.includes(2)
  if (box ? !m.box : !m.points) throw new Error(`${m.title} doesn't take ${box ? 'boxes' : 'clicks'}`)
  const e = await embedding(m, root, name)
  const sx = e.width / e.full[0], sy = e.height / e.full[1]
  const found = await decodePrompt(m, e, points.map(([x, y]) => [x * sx, y * sy]), labels)
  return { annotations: found ? [found] : [], width: e.width, height: e.height, full: e.full }
}

/** Words to every object they name, all at once */
export async function generateFromTexts(root: string, name: string, model: string, texts: string[]): Promise<Generated> {
  const m = modelDef(model)
  if (!m.text) throw new Error(`${m.title} doesn't take words`)
  const image = await rgbPixels(root, name, MAX_SIDE[m.kind])
  return { annotations: await yoloWorld(m, image, texts), width: image.width, height: image.height, full: [image.fullWidth, image.fullHeight] }
}

async function releaseLoaded() {
  embeddings.length = 0
  const parts = loaded?.parts
  loaded = null
  for (const s of parts?.values() ?? []) await (await s.catch(() => null))?.release()
}

/** Frees every model's memory */
export async function release() {
  await releaseLoaded()
  await releasePose()
}
