// Dataset > Check labels, Statistics and renaming a label across the folder, in the main
// process: readDataset reads the folder, checks.ts and stats.ts do the rest. The window asks
// for what it shows (a page of each group of problems, the statistics' tables), so a folder of
// 100,000 images with 50,000 problems never crosses to it whole, and long passes give the
// window's own requests a turn between slices.
import { constants, promises as fs } from 'fs'
import path from 'path'
import {
  CHECKS, type CheckCode, checkDataset, type CheckOptions, type Clash, fixIssue, FIXES, type OrphanFile, renameLabel, type RenamePlan, renaming, tallyLabels
} from '../shared/checks'
import type { DatasetImage, Issue, MaskBits, Severity } from '../shared/dataset'
import { type LabelFormat, pyRound } from '../shared/formats'
import { isAi, SHAPE_TYPES, type Shape, type ShapeType } from '../shared/labelfile'
import { datasetStats, type DatasetStats, type LabelStats, statsCsv } from '../shared/stats'
import { findDuplicateImages, labelFileClashes, orphanLabels, readDataset, rewriteLabels, rewriteList } from './dataset'
import { imageSize } from './images'
import { decodeMask } from './masks'
import { nameKey } from './names'
import { folderOf, writeAtomic } from './project'

/** A pass the window can stop; one of each kind runs on a folder at a time */
export type Job = 'check' | 'stats' | 'rename'
/**
 * How far a pass has got: reading the labels, comparing image files, looking for label files
 * without images, checking, counting, or writing label files
 */
export type Progress = { phase: 'read' | 'compare' | 'orphans' | 'check' | 'count' | 'write'; done: number; total: number }
export type Stopped = { stopped: true }

export type CheckRequest = {
  /** The window's save format: an image with label files in two formats is read in this one */
  prefer: LabelFormat
  /** The label list: the config's labels, then the project's classes; null when there's none */
  labels: string[] | null
  /**
   * After saves in the window, which write label files and never images: the images saved since
   * the last check. Only their labels are read again; the rest of the folder, identical images
   * and label files without images are as the last check found them, unless the images changed.
   */
  saved?: string[]
}
export type CheckSummary = {
  /** When the check finished, in ms since 1970 */
  at: number
  images: number
  counts: Record<Severity, number>
  /** Images with an error or a warning, for the Files dock's Problems filter */
  problemImages: string[]
}

export type ProblemQuery = {
  severities: Severity[]
  /** Only this image's problems; null for the whole folder's */
  image: string | null
  /** How many rows of each group to send, by its code: 0 for a folded group, PAGE when left out */
  limits: Record<string, number>
}
export type ProblemRow = Issue & {
  /** The label of the shape it's about */
  label?: string
  /** A safe fix can be made to the image in the window */
  fix?: boolean
  /** For a label written another way: the spelling it looks like */
  to?: string
  /** For a label file without its image: the file, to show in its folder */
  file?: string
}
export type ProblemGroup = { code: string; severity: Severity; title: string; count: number; rows: ProblemRow[] }
/** What the Problems dock shows: the counts by severity, and the groups of the severities shown, worst first */
export type ProblemPage = { at: number; counts: Record<Severity, number>; groups: ProblemGroup[] }

export type RenamePlanView = {
  /** Images whose labels change, and the shapes renamed in them */
  images: number
  shapes: number
  /** The first few images that change */
  names: string[]
  /** The first few label files left alone, and why */
  skipped: { name: string; reason: string }[]
  skippedCount: number
}
export type RenameDone = {
  stopped: boolean
  shapes: number
  /** Images whose label files were rewritten */
  changed: string[]
  /** Label files left alone: unreadable, or shared by several images */
  skipped: number
  failed: { name: string; reason: string }[]
  /** Images whose labels have the old name still: their file is shared, failed, or wasn't reached before a stop */
  left: number
}

export type StatsRequest = {
  prefer: LabelFormat
  /** The label list, whose labels no shape uses are counted with 0 */
  labels: string[] | null
  /** The longer side images are scaled to for training */
  trainSize: number
  /** Read the folder again; otherwise count what was read last, as when only the training size changed */
  again: boolean
}
/** datasetStats with the first ten image sizes, and how many sizes more there are */
export type StatsView = DatasetStats & { moreSizes: number }

/** Rows of a group the dock gets until it asks for more */
export const PAGE = 200
// Images checked or counted between the window's turns: about 20 ms of work for usual labels
const SLICE = 200

const RANK: Record<Severity, number> = { error: 0, warning: 1, info: 2 }
const CODE_ORDER = new Map(Object.keys(CHECKS).map((code, k) => [code, k]))
const codeRank = (code: string) => CODE_ORDER.get(code) ?? CODE_ORDER.size
const reason = (e: unknown) => (e instanceof Error ? e.message : String(e))
const turn = () => new Promise<void>((go) => setImmediate(go))

const running = new Map<string, AbortController>()
const jobKey = (root: string, job: Job) => `${job}\0${root}`
/** Starts a pass, stopping the one of its kind already running on the folder */
function begin(root: string, job: Job) {
  const key = jobKey(root, job)
  running.get(key)?.abort()
  const control = new AbortController()
  running.set(key, control)
  return control.signal
}
function end(root: string, job: Job, signal: AbortSignal) {
  if (running.get(jobKey(root, job))?.signal === signal) running.delete(jobKey(root, job))
}
/** Stops a pass the window started */
export function stopJob(root: string, job: Job) {
  running.get(jobKey(root, job))?.abort()
}

/** Progress at most every 100 ms, and whenever a phase starts or ends */
function reporter(send: (p: Progress) => void) {
  let last = 0
  return (p: Progress, now = false) => {
    if (!now && p.done < p.total && Date.now() - last < 100) return
    last = Date.now()
    send(p)
  }
}

/** An image's own problems, with the read they were found in */
type Own = { image: DatasetImage; issues: Issue[] }
/**
 * What a check found: every problem, each image's own (by name, in the folder's order), those
 * across the folder, what the folder's were found with, whether any image was verified, which
 * each image's own hang on, and each image's place in the folder
 */
export type Found = { issues: Issue[]; own: Map<string, Own>; across: Issue[]; context: string; verifying: boolean; order: Map<string, number> }
type Checked = {
  at: number
  /** The save format it was read as */
  prefer: LabelFormat
  images: DatasetImage[]
  found: Found
  clashes: Clash[]
  duplicates: string[][]
  orphans: OrphanFile[]
  issues: Issue[]
  /** The file each problem about a label file without its image is about */
  files: Map<Issue, string>
}
/** An image of the check, as it was read */
const imageOf = (c: Checked, name: string) => c.found.own.get(name)?.image
// The last check of each open folder, and the last statistics' read
const checks = new Map<string, Checked>()
const counted = new Map<string, { images: DatasetImage[]; stats: DatasetStats }>()

/** Drops what was found and counted in a folder, and stops its passes, as when it's closed */
export function forgetChecks(root: string) {
  checks.delete(root)
  counted.delete(root)
  for (const job of ['check', 'stats', 'rename'] as const) stopJob(root, job)
}

// What checkDataset finds of an image that hangs on other images: how labels are spelt
// elsewhere and the label list, label files and image files other images share, and the
// image path a shared file names. The rest of an image's problems are its own, found from it
// alone, but for not-verified, which hangs on whether any image of the folder is verified.
const ACROSS = new Set<string>(['label-variants', 'unknown-label', 'shared-label-file', 'image-path-mismatch', 'duplicate-image'])

/**
 * An image as the checks across the folder see it: of no known size, with labels that are
 * shares of it as YOLO's are, so checkDataset reads its shapes' labels and measures nothing,
 * however big its outlines and masks are. What that hides, its size and not-verified, isn't
 * taken from this pass.
 */
const unmeasured = (image: DatasetImage): DatasetImage => ({ ...image, format: 'yolo', width: 0, height: 0 })
// In each slice when the folder verifies images: checkDataset says which aren't verified once one is
const VERIFIED: DatasetImage = { name: '\0verified', width: 0, height: 0, file: null, format: null, shapes: [], masks: [], flags: {}, verified: true }

/**
 * What the checks across the folder see of an image: its label file, verified mark and the
 * image its file names, and each shape's label and whether it's an AI proposal, in order
 */
const labelsOf = (i: DatasetImage) =>
  JSON.stringify([i.file, i.format, i.verified, i.labelImage ?? null, i.broken ?? null, i.shapes.map((s) => [s.label, isAi(s)])])

/** As checkDataset sorts problems: the worst first, then the folder's own, then by image and shape; ties in CHECKS' order */
function sorted(issues: Issue[], order: Map<string, number>) {
  // Each one's place worked out once, not at every comparison: a folder can have 100,000
  const keyed = issues.map((i) => ({ i, severity: RANK[i.severity], at: i.image === null ? -1 : order.get(i.image) ?? order.size, shape: i.shape ?? -1, code: codeRank(i.code) }))
  keyed.sort((a, b) => a.severity - b.severity || a.at - b.at || a.shape - b.shape || a.code - b.code)
  return keyed.map((k) => k.i)
}

/** The images named as the last check named them, in the same order */
function sameOrder(was: Map<string, Own>, images: DatasetImage[]) {
  if (was.size !== images.length) return false
  let k = 0
  for (const name of was.keys()) if (name !== images[k++].name) return false
  return true
}

// Counting labels is quicker than checking them: this many slices a turn
const TALLY_SLICES = 10

// Copies of image files are told of the same way whatever the labels say: once for each finding
// of them, which a check after a save keeps. In a folder of camera frames every image can be one.
const copiesTold = new WeakMap<string[][], Issue[]>()
function copyIssues(duplicates: string[][]) {
  let issues = copiesTold.get(duplicates)
  if (!issues) copiesTold.set(duplicates, (issues = checkDataset([], { labels: null, duplicates })))
  return issues
}

/**
 * What checkDataset finds that hangs on other images, as one call over the whole folder finds
 * it, but a slice at a time: the folder's labels are counted, then each slice is checked
 * against the count, then copies of image files and the folder's own problems are found once
 */
async function acrossOf(images: DatasetImage[], options: CheckOptions, signal?: AbortSignal): Promise<Issue[]> {
  const tally = tallyLabels([])
  for (let at = 0; at < images.length; at += SLICE * TALLY_SLICES) {
    signal?.throwIfAborted()
    tallyLabels(images.slice(at, at + SLICE * TALLY_SLICES), tally)
    await turn()
  }
  const across: Issue[] = []
  for (let at = 0; at < images.length; at += SLICE) {
    signal?.throwIfAborted()
    for (const issue of checkDataset(images.slice(at, at + SLICE).map(unmeasured), { labels: options.labels, clashes: options.clashes, tally })) {
      if (issue.image !== null && ACROSS.has(issue.code)) across.push(issue)
    }
    await turn()
  }
  signal?.throwIfAborted()
  for (const issue of copyIssues(options.duplicates ?? [])) across.push(issue)
  for (const issue of checkDataset([], { labels: options.labels, orphans: options.orphans, tally })) across.push(issue)
  return across
}

/**
 * checkDataset's problems, found in two passes that come to the same: one over the whole
 * folder, measuring nothing, for what hangs on other images, then one image by image. Both go
 * in slices, with a turn for the window between them. An image that's the same read as before
 * (readDataset gives the same object while its files are unchanged) keeps the problems found
 * in it then, so a check after a save measures only what was saved; and when no image's
 * labels changed (a shape moved, say), nor what the check is told, neither did the folder's.
 */
export async function issuesOf(images: DatasetImage[], options: CheckOptions, before?: Found,
  progress?: (done: number, total: number) => void, signal?: AbortSignal): Promise<Found> {
  const context = JSON.stringify([options.labels, options.clashes ?? [], options.duplicates ?? [], options.orphans ?? []])
  const verifying = images.some((i) => i.verified)
  const was = before?.own
  const same = !!was && before!.context === context && was.size === images.length && images.every((i) => {
    const read = was.get(i.name)?.image
    return read === i || (!!read && labelsOf(read) === labelsOf(i))
  })
  const order = was && sameOrder(was, images) ? before!.order : new Map(images.map((i, k) => [i.name, k]))
  const across = same ? before!.across : await acrossOf(images, options, signal)
  const issues = [...across]
  const own = new Map<string, Own>()
  // Once the first image is verified, or the last one isn't, every image says so again
  const kept = before?.verifying === verifying ? was : undefined
  for (let at = 0; at < images.length; at += SLICE) {
    signal?.throwIfAborted()
    const slice = images.slice(at, at + SLICE)
    const todo = slice.filter((i) => kept?.get(i.name)?.image !== i)
    const found = new Map(todo.map((i) => [i.name, [] as Issue[]]))
    if (todo.length) {
      for (const issue of checkDataset(verifying ? [...todo, VERIFIED] : todo, { labels: null })) {
        if (issue.image !== null && !ACROSS.has(issue.code)) found.get(issue.image)?.push(issue)
      }
    }
    for (const image of slice) {
      const mine = found.has(image.name) ? { image, issues: found.get(image.name)! } : kept!.get(image.name)!
      own.set(image.name, mine)
      for (const issue of mine.issues) issues.push(issue)
    }
    progress?.(Math.min(at + SLICE, images.length), images.length)
    await turn()
  }
  return { issues: sorted(issues, order), own, across, context, verifying, order }
}

/** The same image files as before, by name and byte size */
const sameImages = (a: DatasetImage[], b: DatasetImage[]) => a.length === b.length && a.every((i, k) => i.name === b[k].name && i.bytes === b[k].bytes)

/** Which label file without its image each problem is about: orphanLabels lists them in the order they're reported */
function strayFiles(issues: Issue[], orphans: OrphanFile[]) {
  const files = new Map<Issue, string>()
  let next = 0
  for (const issue of issues) {
    if (issue.code !== 'orphan-label-file') continue
    const names = (o?: OrphanFile) => !!o && issue.message.startsWith(`${o.file} `)
    const at = names(orphans[next]) ? next : orphans.findIndex(names)
    if (at < 0) continue
    files.set(issue, orphans[at].file)
    next = at + 1
  }
  return files
}

function summaryOf(c: Checked): CheckSummary {
  const counts = { error: 0, warning: 0, info: 0 }
  const problems = new Set<string>()
  for (const i of c.issues) {
    counts[i.severity]++
    if (i.image !== null && i.severity !== 'info') problems.add(i.image)
  }
  return { at: c.at, images: c.images.length, counts, problemImages: [...problems] }
}

/**
 * Dataset > Check labels: the folder read as the window's save format reads it, then
 * images that share a label file, identical image files, label files without their image,
 * and checkDataset's checks with the label list. Kept as the folder's last check. After saves
 * only the images saved are read again, so a folder of 100,000 isn't read once a save.
 */
export async function checkFolder(root: string, { prefer, labels, saved }: CheckRequest, send: (p: Progress) => void): Promise<CheckSummary | Stopped> {
  const signal = begin(root, 'check')
  const report = reporter(send)
  try {
    const before = checks.get(root)
    // Read as another format, the other images could have other files: those are read again too
    const quick = !!saved && before?.prefer === prefer
    const images = await readDataset(root, {
      prefer, signal, only: quick ? new Set(saved) : undefined, onProgress: (done, total) => report({ phase: 'read', done, total })
    })
    const same = quick && sameImages(before!.images, images)
    let duplicates = same ? before!.duplicates : null
    if (!duplicates) {
      // Counted in the files compared, which findDuplicateImages reports once it knows them
      report({ phase: 'compare', done: 0, total: 0 }, true)
      duplicates = await findDuplicateImages(root, images, { signal, onProgress: (done, total) => report({ phase: 'compare', done, total }) })
    }
    let orphans = same ? before!.orphans : null
    if (!orphans) {
      report({ phase: 'orphans', done: 0, total: 0 }, true)
      orphans = await orphanLabels(root, images, { signal, onProgress: (done, total) => report({ phase: 'orphans', done, total }) })
    }
    // Which images share a label file hangs on nothing but where each one's labels are read from
    const clashes = same && images.every((i, k) => i.file === before!.images[k].file && i.format === before!.images[k].format)
      ? before!.clashes : labelFileClashes(images)
    const found = await issuesOf(images, { labels, clashes, duplicates, orphans }, before?.found,
      (done, total) => report({ phase: 'check', done, total }), signal)
    // A folder closed as the check ended keeps none
    signal.throwIfAborted()
    const { issues } = found
    const result: Checked = { at: Date.now(), prefer, images, found, clashes, duplicates, orphans, issues, files: strayFiles(issues, orphans) }
    checks.set(root, result)
    return summaryOf(result)
  } catch (e) {
    if (signal.aborted) return { stopped: true }
    throw e
  } finally {
    end(root, 'check', signal)
  }
}

/** The spelling a label written another way looks like, as checkDataset words it */
const LOOKS_LIKE = /looks like “(.*?)” (?:from the label list|written another way)/

function rowOf(c: Checked, issue: Issue): ProblemRow {
  const image = issue.image === null ? undefined : imageOf(c, issue.image)
  const label = issue.shape === undefined ? undefined : image?.shapes[issue.shape]?.label
  const to = issue.code === 'label-variants' ? LOOKS_LIKE.exec(issue.message)?.[1] : undefined
  const file = c.files.get(issue)
  return {
    ...issue,
    ...(label === undefined ? {} : { label }),
    ...(image && FIXES[issue.code as CheckCode] && fixIssue(image, issue) ? { fix: true } : {}),
    ...(to === undefined ? {} : { to }),
    ...(file === undefined ? {} : { file })
  }
}

/** A page of the folder's last check for the Problems dock; null when it hasn't been checked */
export function problemPage(root: string, { severities, image, limits }: ProblemQuery): ProblemPage | null {
  const c = checks.get(root)
  if (!c) return null
  const counts = { error: 0, warning: 0, info: 0 }
  const shown = new Set(severities)
  const groups = new Map<string, ProblemGroup>()
  for (const issue of c.issues) {
    if (image !== null && issue.image !== image) continue
    counts[issue.severity]++
    if (!shown.has(issue.severity)) continue
    let group = groups.get(issue.code)
    if (!group) {
      group = { code: issue.code, severity: issue.severity, title: CHECKS[issue.code as CheckCode]?.title ?? issue.code, count: 0, rows: [] }
      groups.set(issue.code, group)
    }
    group.count++
    if (group.rows.length < Math.max(0, limits[issue.code] ?? PAGE)) group.rows.push(rowOf(c, issue))
  }
  const order = [...groups.values()].sort((a, b) => RANK[a.severity] - RANK[b.severity] || codeRank(a.code) - codeRank(b.code))
  return { at: c.at, counts, groups: order }
}

/** A label file without its image that the last check found, in full; null for any other file */
export function strayFile(root: string, file: string): string | null {
  if (!checks.get(root)?.orphans.some((o) => o.file === file)) return null
  return path.isAbsolute(file) ? file : path.join(root, ...file.split('/'))
}

const samePoints = (a: number[][], b: number[][]) => a.length === b.length && a.every((p, k) => p[0] === b[k][0] && p[1] === b[k][1])

/** Each mask's pixels: as the check read them where the shape is the one it read, otherwise decoded at its first corner */
function masksFor(shapes: Shape[], read: DatasetImage | undefined): Promise<(MaskBits | null)[]> {
  return Promise.all(shapes.map(async (s) => {
    if (s.shape_type !== 'mask' || !s.mask || !s.points.length) return null
    const k = read ? read.shapes.findIndex((r) => r.mask === s.mask && samePoints(r.points, s.points)) : -1
    if (k >= 0) return read!.masks[k] ?? null
    const m = await decodeMask(s.mask).catch(() => null)
    if (!m) return null
    const [x, y] = s.points[0].map(pyRound)
    return { x, y, width: m.width, height: m.height, data: m.data }
  }))
}

/**
 * A safe fix (checks.ts' FIXES) made to an image's shapes as the window has them now, which
 * may hold changes the check never saw: the image is checked again first, and the shapes come
 * back fixed only when the same problem is still there at the same shape. Null otherwise.
 */
export async function fixShapes(root: string, issue: Pick<Issue, 'code' | 'image' | 'shape'>, shapes: Shape[]): Promise<Shape[] | null> {
  const code = issue.code as CheckCode
  if (!FIXES[code] || !issue.image) return null
  // A name from the window: only an image of this folder
  if (!folderOf(root).images.includes(issue.image)) throw new Error(`${issue.image} isn't in the folder`)
  const c = checks.get(root)
  const read = c && imageOf(c, issue.image)
  if (read?.broken) return null
  const [width, height] = read && read.width > 0 ? [read.width, read.height] : await imageSize(root, issue.image).catch(() => [0, 0])
  const image: DatasetImage = {
    ...(read ?? { name: issue.image, file: null, format: null, flags: {}, verified: false }),
    width, height, shapes, masks: await masksFor(shapes, read)
  }
  const now = checkDataset([image], { labels: null }).find((i) => i.code === code && i.shape === issue.shape)
  return now ? fixIssue(image, now) : null
}

// What renaming a label changes doesn't hang on its new name: any other name plans it
const plannedFrom = (images: DatasetImage[], from: string) => renameLabel(images, from, `${from}\u0000`, labelFileClashes(images))

/** What renaming `from` across the folder would change, read as the window's save format reads it, before anything is written */
export async function renamePlan(root: string, from: string, prefer: LabelFormat, send: (p: Progress) => void): Promise<RenamePlanView | Stopped> {
  const signal = begin(root, 'rename')
  const report = reporter(send)
  try {
    const images = await readDataset(root, { prefer, signal, onProgress: (done, total) => report({ phase: 'read', done, total }) })
    const plan = plannedFrom(images, from)
    return {
      images: plan.images.length, shapes: plan.shapes, names: plan.images.slice(0, 8).map((i) => i.name),
      skipped: plan.skipped.slice(0, 8), skippedCount: plan.skipped.length
    }
  } catch (e) {
    if (signal.aborted) return { stopped: true }
    throw e
  } finally {
    end(root, 'rename', signal)
  }
}

/**
 * Renames a label in every label file of the folder that has it, each in its own format
 * through rewriteLabels, with the old file in the trash: the images of one CreateML list
 * together, through rewriteList, so the list is written once, not once an image. The rename
 * is made to what each file holds as it's written, so nothing saved since the read is undone.
 * A stop ends it between files, or in a list's own checks, before the list is written.
 * `allowed` is the label list when the config allows only its labels (validate_label: exact).
 * The new name is trimmed, as the window's label dialog trims a label.
 */
export async function renameRun(root: string, from: string, to: string, prefer: LabelFormat, allowed: string[] | null,
  send: (p: Progress) => void): Promise<RenameDone> {
  const name = to.trim()
  if (!name) throw new Error('A label needs a name, so nothing is renamed')
  if (allowed && !allowed.includes(name)) throw new Error(`“${name}” isn't in the label list, and only listed labels are allowed.`)
  const signal = begin(root, 'rename')
  const report = reporter(send)
  const done: RenameDone = { stopped: false, shapes: 0, changed: [], skipped: 0, failed: [], left: 0 }
  try {
    const images = await readDataset(root, { prefer, signal, onProgress: (n, total) => report({ phase: 'read', done: n, total }) })
    const plan = renameLabel(images, from, name, labelFileClashes(images))
    done.skipped = plan.skipped.length
    const byName = new Map(images.map((i) => [i.name, i]))
    // The images of one CreateML list go together, by its path; any other image's file is its own
    const batches = new Map<string, RenamePlan['images']>()
    for (const image of plan.images) {
      const read = byName.get(image.name)!
      const key = read.format === 'createml' && read.file ? nameKey(read.file) : image.name
      const batch = batches.get(key)
      if (batch) batch.push(image)
      else batches.set(key, [image])
    }
    let written = 0
    for (const batch of batches.values()) {
      if (signal.aborted) break
      const read = byName.get(batch[0].name)!
      // Why each image of the batch failed, or null; none at all when a stop came before its list was written
      const failures = read.format === 'createml'
        ? (await rewriteList(root, read.file!, batch.map((i) => ({ name: i.name, read: byName.get(i.name)! })), renaming(from, name), signal))
          ?.map((r) => (r instanceof Error ? r.message : null))
        : [await rewriteLabels(root, read.name, renaming(from, name), null, read).then(() => null, reason)]
      if (!failures) break
      batch.forEach((image, k) => {
        const failed = failures[k]
        if (failed === null) {
          done.changed.push(image.name)
          done.shapes += image.renamed
        } else done.failed.push({ name: image.name, reason: failed })
      })
      written += batch.length
      report({ phase: 'write', done: written, total: plan.images.length })
    }
    done.stopped = signal.aborted
    // A file that can't be read is left alone too, but what it holds isn't known
    const shared = plan.skipped.filter((s) => !byName.get(s.name)?.broken).length
    done.left = shared + plan.images.length - done.changed.length
    return done
  } catch (e) {
    if (signal.aborted) return { ...done, stopped: true }
    throw e
  } finally {
    end(root, 'rename', signal)
  }
}

/**
 * datasetStats of a folder from those of its slices, each counted without the label list:
 * the same counts in the same order, the list's unused labels last with 0
 */
export function together(parts: DatasetStats[], trainSize: number, labels: string[] | null = null): DatasetStats {
  const images = { total: 0, labelled: 0, empty: 0, never: 0, verified: 0, broken: 0, unreadable: 0, aiPending: 0 }
  const perLabel = new Map<string, LabelStats>()
  const perImage = new Map<number, number>()
  const sizes: DatasetStats['sizes'] = { trainSize, full: { small: 0, medium: 0, large: 0 }, train: { small: 0, medium: 0, large: 0 }, unsized: 0 }
  const imageSizes = new Map<string, { width: number; height: number; images: number }>()
  const used = new Set<ShapeType>()
  for (const p of parts) {
    for (const k of Object.keys(images) as (keyof typeof images)[]) images[k] += p.images[k]
    for (const l of p.labels) {
      let row = perLabel.get(l.label)
      if (!row) perLabel.set(l.label, (row = { label: l.label, objects: 0, images: 0, types: {} }))
      row.objects += l.objects
      row.images += l.images
      for (const [type, n] of Object.entries(l.types) as [ShapeType, number][]) row.types[type] = (row.types[type] ?? 0) + n
    }
    for (const { objects, images: n } of p.objectsPerImage) perImage.set(objects, (perImage.get(objects) ?? 0) + n)
    for (const range of ['small', 'medium', 'large'] as const) {
      sizes.full[range] += p.sizes.full[range]
      sizes.train[range] += p.sizes.train[range]
    }
    sizes.unsized += p.sizes.unsized
    for (const s of p.imageSizes) {
      const key = `${s.width}x${s.height}`
      const known = imageSizes.get(key)
      if (known) known.images += s.images
      else imageSizes.set(key, { ...s })
    }
    p.types.forEach((t) => used.add(t))
  }
  const rows = [...perLabel.values()].sort((a, b) => b.objects - a.objects || a.label.localeCompare(b.label))
  for (const label of labels ?? []) if (!perLabel.has(label)) rows.push({ label, objects: 0, images: 0, types: {} })
  return {
    images,
    labels: rows,
    objectsPerImage: [...perImage].sort((a, b) => a[0] - b[0]).map(([objects, n]) => ({ objects, images: n })),
    sizes,
    imageSizes: [...imageSizes.values()].sort((a, b) => b.images - a.images || a.width - b.width || a.height - b.height),
    types: SHAPE_TYPES.filter((t) => used.has(t))
  }
}

/**
 * Dataset > Statistics: the folder read as the window's save format reads it (or what was
 * read last, when only the training size changed), counted in slices with a turn for the
 * window between them. Kept for Save as CSV.
 */
export async function folderStats(root: string, { prefer, labels, trainSize, again }: StatsRequest, send: (p: Progress) => void): Promise<StatsView | Stopped> {
  const signal = begin(root, 'stats')
  const report = reporter(send)
  const size = Number.isFinite(trainSize) && trainSize >= 1 ? Math.round(trainSize) : 640
  try {
    const images = (!again && counted.get(root)?.images) ||
      await readDataset(root, { prefer, signal, onProgress: (done, total) => report({ phase: 'read', done, total }) })
    const parts: DatasetStats[] = []
    for (let at = 0; at < images.length; at += SLICE) {
      signal.throwIfAborted()
      parts.push(datasetStats(images.slice(at, at + SLICE), { trainSize: size }))
      report({ phase: 'count', done: Math.min(at + SLICE, images.length), total: images.length })
      await turn()
    }
    signal.throwIfAborted()
    const stats = together(parts, size, labels)
    counted.set(root, { images, stats })
    return { ...stats, imageSizes: stats.imageSizes.slice(0, 10), moreSizes: Math.max(0, stats.imageSizes.length - 10) }
  } catch (e) {
    if (signal.aborted) return { stopped: true }
    throw e
  } finally {
    end(root, 'stats', signal)
  }
}

/** The last statistics' per-label table as CSV; null when the folder hasn't been counted */
export function statsText(root: string): string | null {
  const c = counted.get(root)
  return c ? statsCsv(c.stats) : null
}

/**
 * Where Save as CSV suggests: beside the folder, as the keypoints file is, and named after it,
 * so it's never taken for a label file; in the folder itself when the one it's in can't be
 * written to, as when it's the top of a removable drive, which the system mounts in a folder of
 * its own (/media/<user>, /Volumes).
 */
export async function statsFileName(root: string) {
  const parent = path.dirname(root)
  const beside = await fs.access(parent, constants.W_OK).then(() => true, () => false)
  return path.join(beside ? parent : root, `${path.basename(root)}-statistics.csv`)
}

/**
 * Writes Save as CSV's file. The text goes to a file beside it first (writeAtomic), so a
 * failure is told with the name of the file that was chosen, not that one's.
 */
export async function saveStats(file: string, text: string) {
  try {
    await writeAtomic(file, text)
  } catch (e) {
    throw new Error(`${path.basename(file)} can't be written in ${path.dirname(file)} (${(e as NodeJS.ErrnoException).code ?? reason(e)})`)
  }
}
