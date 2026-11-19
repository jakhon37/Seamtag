// labelme's files that the tests read: label files and photos from its examples, and two of its
// default configs. labelme (https://github.com/wkentaro/labelme) is GPL-3.0-only, as Seamtag is, but
// its files aren't kept in this repository. A test asks for one by its path in labelme's repository:
// the first time, it's fetched from GitHub at the commit its release tag points to (a tag can be
// moved, a commit can't), checked against its SHA-256 below, and kept in
// node_modules/.cache/seamtag-labelme, so later runs need no network. A kept copy is checked again
// each time a test run first uses it.
import { createHash } from 'crypto'
import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'fs'
import path from 'path'

/** labelme's release tags and their commits (git ls-remote https://github.com/wkentaro/labelme refs/tags/<tag>) */
const COMMITS = {
  'v5.8.1': '0e71e856aa59a0e61a5dd85033247519f9bf776b',
  'v5.11.4': '1a0daa1c90a082bb42b3b4df4f0e16e201838239',
  'v7.9.1': 'd179c2353dcf35a29809797b2702dba8d794b84e'
} as const
export type LabelmeTag = keyof typeof COMMITS

/** Each file the tests read, by tag and path in labelme's repository, with its SHA-256 */
export const LABELME_FILES: Record<LabelmeTag, Record<string, string>> = {
  'v5.8.1': {
    // config.test.ts: the default config labelme 5 copied to ~/.labelmerc
    'labelme/config/default_config.yaml': '5d0de71c1238f28d8ad9437cc0f7794a35699ed8d8083e74bff67b53016f460a'
  },
  'v5.11.4': {
    // labelfile.test.ts: every label file in the examples (the four in labelme's tests/data are links to some of these)
    'examples/bbox_detection/data_annotated/2011_000003.json': 'ac9a9f02ea5eff3c7751fc8ade43ceb1d095acaab75f6f3408d3fcd57c9d6fb1',
    'examples/bbox_detection/data_annotated/2011_000006.json': '3b2ccfe3e6ddebd0dec1ebba19a129b8093b613f2054da52025a8eff8704a4f6',
    'examples/bbox_detection/data_annotated/2011_000025.json': '3e823597258ff3a0057d9239f17de0ddbc3fb1558ccfcd720f2b1fc4e628f04c',
    'examples/classification/data_annotated/0001.json': '6978afa6e21f0fba910063118e67c07ca48dafa86b551f41fd52b2db532caf8c',
    'examples/classification/data_annotated/0002.json': '0d6e588e3f756b3f385b91109dc34acaa32a43d461835a4b55b31d733c8d853d',
    'examples/instance_segmentation/data_annotated/2011_000003.json': '0aa7c87bd4f11d3ab38d63fa9bc2a11640d77c0eca2a13968e852478f63021dc',
    'examples/instance_segmentation/data_annotated/2011_000006.json': '8f7f13768d9b25927f3dd19d317f33807b13d571d56ec801eea57798999ec9b8',
    'examples/instance_segmentation/data_annotated/2011_000025.json': 'df548de624182d99286bc3f5b0f18e287ee6d1fab00f53784b9deaf7c00870ab',
    'examples/primitives/primitives.json': '44ff5edccc4bcda337881dd5defbf528c1a96a68711e23e539bf1eb9dc18a388',
    'examples/semantic_segmentation/data_annotated/2011_000003.json': 'ac44e3d092191752af854153a62cf4b6964e0f7e7225d4282407b1b77eed179c',
    'examples/semantic_segmentation/data_annotated/2011_000006.json': 'ca6550c9ca63b3b469ec747467466ca406725fac4782974f8529d702f25c145d',
    'examples/semantic_segmentation/data_annotated/2011_000025.json': '45f4ebacdb2b254b1acabdd760cc7a91803357dabbe45d8a8baf0179742341dd',
    'examples/tutorial/apc2016_obj3.json': '554c4cf44f7577c35fd6ac471c5dc02671f03d2f82a087a466f93afacb47bd28',
    'examples/video_annotation/data_annotated/00000100.json': 'ad0e0b80353ee6c8e80980e92d450ff3f721793a4e2dcf3cf0a133f66bc0ea68',
    'examples/video_annotation/data_annotated/00000101.json': '6de3d5828c40c7119952b1f38eb5650daf1a6de3f24c69c71ae84142ed681d06',
    'examples/video_annotation/data_annotated/00000102.json': '88c7413440fb0bc930b01d7ba47fc45371f075890cc1752c8fb7033b4b17b0cb',
    'examples/video_annotation/data_annotated/00000103.json': 'f12fb0bb622d12ed2fe1f03a01f41929089a0f66494b51d8166fe5b4b8c87799',
    'examples/video_annotation/data_annotated/00000104.json': '1ae8ce8ea74602709f1732ef129b0db30a53ebb6d77009c08dab1dc12ab31b13',
    // export.test.ts: the instance segmentation example's photos, its label list and what labelme2coco made of it.
    // The first photo is also e2e/labelme7.spec.ts's for the text prompt.
    'examples/instance_segmentation/data_annotated/2011_000003.jpg': '94bc550055447e1fab3982571d7114eb4a02a868054f34cb15f95cadf3e48c9a',
    'examples/instance_segmentation/data_annotated/2011_000006.jpg': 'a0d04eb84ed3cd07282d61833d219c11fd6d98f53dc0135d4b3c720b26cfb244',
    'examples/instance_segmentation/data_annotated/2011_000025.jpg': '5dfbdc6981c971986773e226d827f44f7497cf5ca0d4fa1556fd7575aba301ce',
    'examples/instance_segmentation/labels.txt': 'ee35389f1e66d2751eed4d69868f33d672ad44091de79e006e1af8db64cbe2cc',
    'examples/instance_segmentation/data_dataset_coco/annotations.json': '935081d34679b92a74e0dc1acf04e5ac97b01cba7eefe26d5572e761ebf16ec2'
  },
  'v7.9.1': {
    // config.test.ts
    'labelme/_config/default_config.yaml': '4304a956a5d366582c59ae81e8b1f07ab3ac0426d73bac5d5cb9e38a8c1db696'
  }
}

const CACHE = path.resolve(__dirname, '../node_modules/.cache/seamtag-labelme')
const asked = new Map<string, Promise<string>>()
const sha256 = (data: Buffer) => createHash('sha256').update(data).digest('hex')

/** The path of a checked copy of a file in labelme's repository at a tag, fetched the first time it's asked for */
export function labelmeFile(tag: LabelmeTag, file: string): Promise<string> {
  const key = `${tag}/${file}`
  if (!asked.has(key)) asked.set(key, copy(tag, file))
  return asked.get(key)!
}

async function copy(tag: LabelmeTag, file: string) {
  const want = LABELME_FILES[tag][file]
  if (!want) throw new Error(`labelme ${tag}'s ${file} isn't listed in tests/labelme-files.ts, with its SHA-256`)
  const local = path.join(CACHE, COMMITS[tag], ...file.split('/'))
  const kept = () => {
    try {
      return sha256(readFileSync(local)) === want
    } catch {
      return false
    }
  }
  if (kept()) return local
  const url = `https://raw.githubusercontent.com/wkentaro/labelme/${COMMITS[tag]}/${file}`
  const data = await download(url, `labelme ${tag}'s ${file}`)
  const got = sha256(data)
  if (got !== want) throw new Error(`labelme ${tag}'s ${file}, fetched from ${url}, has SHA-256 ${got}, not ${want} as tests/labelme-files.ts lists`)
  mkdirSync(path.dirname(local), { recursive: true })
  // Written beside it, then renamed, as test files running at once may fetch the same file
  const part = `${local}.${process.pid}-${Math.random().toString(36).slice(2)}.part`
  writeFileSync(part, data)
  const drop = () => {
    try {
      rmSync(part, { force: true })
    } catch {}
  }
  for (let waited = 0; ; waited += 100) {
    try {
      renameSync(part, local)
      return local
    } catch (e) {
      // On Windows a file another test is reading can't be replaced, but then it's there already
      if (kept()) {
        drop()
        return local
      }
      // and a virus scanner or the search indexer can hold a file just written for a moment
      const busy = process.platform === 'win32' && ['EPERM', 'EACCES', 'EBUSY'].includes((e as NodeJS.ErrnoException).code ?? '')
      if (!busy || waited >= 5000) {
        drop()
        throw e
      }
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
  }
}

async function download(url: string, what: string) {
  let problem = ''
  let answered = false
  // Three tries end within Vitest's 60 s for a test, so it fails with this error rather than a timeout
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt) await new Promise((resolve) => setTimeout(resolve, 2000 * attempt))
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
      if (response.ok) return Buffer.from(await response.arrayBuffer())
      await response.body?.cancel()
      answered = true
      problem = `HTTP ${response.status} ${response.statusText}`.trim()
      // Only a busy or failing server is worth asking again
      if (response.status < 500 && response.status !== 429) break
    } catch (e) {
      const { message, cause } = e as Error & { cause?: { message?: string } }
      answered = false
      problem = cause?.message ? `${message} (${cause.message})` : message
    }
  }
  const offline = answered ? '' : ` The tests fetch labelme's files the first time they need them, so that run needs the network; ` +
    `later runs use the copies in ${CACHE}.`
  throw new Error(`Couldn't fetch ${what} from ${url}: ${problem}.${offline}`)
}
