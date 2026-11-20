import {
  Check, CheckCheck, Circle, Eraser, Eye, EyeOff, Hexagon, Maximize, MousePointer2, Paintbrush, Plus, Redo2, SkipForward,
  Slash, Sparkles, Spline, Square, Trash2, Undo2, Wand2, X, Dot, ScanLine, Lock, BadgeCheck, RectangleHorizontal, Type, Focus,
  PersonStanding, Accessibility, ChevronRight, ChevronDown
} from 'lucide-react'
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { AI_OUTPUTS, type AiOutput, type Config, formatKeys, matches, type ShortcutName, shortcutList } from '../../shared/config'
import type { KeypointSchema } from '../../shared/dataset'
import {
  boxOf, FORMAT_NAMES, holdsBoxesOnly, keepsDifficult, keepsVerified, LABEL_FORMATS, type LabelFormat, NEXT_FORMAT
} from '../../shared/formats'
import { accepted, isAi, newShape, scoreOf, type Shape, type ShapeType } from '../../shared/labelfile'
import { EMPTY_SUMMARY, summarize } from '../../shared/labels'
import {
  builtInKeypoints, type FolderKeypoints, isHidden, keypointOnlyNames, nextGroupId, type PoseGroup, posePoints, poseShapes, posesIn, sidesOf
} from '../../shared/skeleton'
import { HALO, labelColor, labelList, recolored, SIDE_COLORS, withAlpha, withClass } from './colors'
import { eraseShapes, paintShapes, strokeRegion } from './geometry'
import LabelDialog, { flagsFor, type LabelChoice, type LabelRequest } from './LabelDialog'
import { ContextMenu } from './Menu'
import Modal from './Modal'
import Models from './Models'
import {
  acceptTouched, aiCounts, aiPosePoints, askedNow, classForModel, guideIn, keptAsBoxes, layoutOf, MIN_POSE_POINTS, mirrorPose, poseIndex,
  type PoseLayout, type PoseSession, type PoseTarget, regroup, type ReviewItem, reviewItems, scoreOfPose, shapeRows, spoken, stepOf, swapSides,
  unposed, withHidden, withPoints, withSession
} from './posing'
import Problems, { counted, RenameDialog } from './Problems'
import {
  addPointsToEdge, bounds, boxCenter, canRemovePoint, CLOSED, contains, describe, EDITABLE_POINTS, edges, insertPoint, mergePolygons,
  moveCorner, moveShapes, nearestEdge, nearestVertex, outline, type Pt, reprojectCorners, rotatePoints, rotationHandles, squareFrom, toEdge, vertices
} from './shapes'
import { Stale } from './stale'
import Statistics from './Statistics'
import {
  type CheckSummary, type DrawnShape, fileUrl, type ImageItem, type LabelClass, type Loaded, type ModelInfo, type OpenImage, type OpenProject, plainError,
  type PrelabelModel, type ProblemRow, type ProjectInfo, type Progress, type ProposedShape, type RenameDone, type RunModel
} from './types'

export type Mode =
  | 'edit' | 'polygon' | 'rectangle' | 'oriented_rectangle' | 'circle' | 'line' | 'point' | 'linestrip' | 'pose' | 'ai_points' | 'ai_box' | 'ai_pose' | 'brush' | 'eraser'
export const CREATE_MODES: Mode[] = ['polygon', 'rectangle', 'oriented_rectangle', 'circle', 'line', 'point', 'linestrip', 'pose', 'ai_points', 'ai_box', 'ai_pose']
// LabelImg's formats keep boxes only: AI shapes become the box around them. A pose's points have no place there.
const BOX_MODES = new Set<Mode>(['edit', 'rectangle', 'ai_points', 'ai_box'])
const AI_MODES = new Set<Mode>(['ai_points', 'ai_box'])
// labelme 7's names for the AI tools, in its crosshair settings; the pose tools start with a box, so they follow the box tools'
const CROSSHAIR: Partial<Record<Mode, string>> = { ai_points: 'ai_points_to_shape', ai_box: 'ai_box_to_shape', pose: 'rectangle', ai_pose: 'ai_box_to_shape' }
export const AI_OUTPUT_NAMES: Record<AiOutput, string> = { polygon: 'Polygon', mask: 'Mask', rectangle: 'Rectangle', circle: 'Circle', oriented_rectangle: 'Rotated box' }

export type Settings = {
  autoSave: boolean
  storeData: boolean
  keepPrev: boolean
  keepPrevScale: boolean
  keepPrevBC: boolean
  fillDrawing: boolean
  fill: number
  fit: 'window' | 'width' | 'selection' | 'manual'
  aiModel: string
  /** What AI points and AI box make of what the model finds */
  aiOutput: AiOutput
  /** The model for the AI text prompt */
  aiTextModel: string
  /** The model for AI pose */
  poseModel: string
  /** The pre-labelling model last picked with Choose a model…, for folders with no runs and none chosen */
  prelabelModel: string
  /** The pre-labelling model chosen for each folder: from its runs in the menu, or with Choose a model… */
  prelabelModels: Record<string, string>
  docks: { files: boolean; labels: boolean; shapes: boolean; flags: boolean; problems: boolean }
  dark: boolean
  /** Where new labels are saved; it follows the label file of the image opened, as in LabelImg */
  format: LabelFormat
  /** LabelImg's Display Labels: each shape's label drawn on it */
  displayLabels: boolean
  /** LabelImg's Draw Squares: boxes are drawn and reshaped square */
  drawSquares: boolean
  /** labelme's display_label_popup: ask for each new shape's label; LabelImg's Single Class Mode is this off */
  askLabel: boolean
  /** LabelImg's Advanced Mode: a tool stays on. Off, it's LabelImg's beginner mode: one shape, then back to editing. */
  advanced: boolean
  /** LabelImg's Box Line Color: the colour a shape is drawn in until it's finished; null for its label's */
  drawColor: string | null
}

export type Command = { label: string; run: () => void; enabled?: () => boolean; checked?: () => boolean; shortcut?: ShortcutName; keys?: string }
export type Commands = Record<string, Command>
export type LabelerHandle = {
  commands: Commands
  /**
   * Before another image or folder opens: the open image's changes are saved, or asked about;
   * false when that's cancelled. With `stay`, as before the folder's label files are read, the
   * image stays open, and changes let go are taken back: it's read again as saved. With `folder`,
   * for another folder, a pre-label batch running here stops first, if the user agrees.
   */
  canLeave: (how?: { stay?: boolean; folder?: boolean }) => Promise<boolean>
  /** Opens an image, once its changes are saved or let go, and brings one of its shapes into view selected */
  showImage: (name: string, shape?: number) => Promise<void>
  /** Label files written outside the window: their undo history is dropped and the open one is read again */
  reloadImages: (names: string[]) => void
  /** Opens that image, as a second start asks, once what's on screen is saved, or asked about */
  show: (name: string) => void
  /** What here waits on the user, such as a dialog, which opening another image or folder would cut short, said for the user; '' when nothing does */
  busy: () => string
}

type Filter = 'all' | 'label' | 'review' | 'done' | 'problems'
type Props = {
  data: OpenProject
  config: Config
  settings: Settings
  setSettings: (fn: (s: Settings) => Settings) => void
  /** `shown`: the label list, whose colours a folder's first class list takes */
  setProject: (p: ProjectInfo, shown?: LabelClass[]) => void
  setImages: (fn: (images: ImageItem[]) => ImageItem[]) => void
  discovered: string[]
  /** A label no label file has any more, after a rename: it goes from those found in the files */
  forgetLabel: (label: string) => void
  /** The labels found in the files are found again, as main counts every image again once the folder's keypoints change */
  forgetFound: () => void
  mac: boolean
  /** `image`: the image on screen the message is about, by its folder and name, so it goes once another image is */
  onError: (m: string, image?: string) => void
  /** Another image is on screen, by its folder and name */
  onShown: (image: string) => void
  onOpenMenu: (x: number, y: number) => void
  onChange: () => void
  /** Where label files go: another folder, or back next to the images */
  onChangeOutput: () => void
  onNextToImages: () => void
}
type Drag =
  | { kind: 'pan'; x: number; y: number; press: number[]; button: number; moved: boolean }
  | { kind: 'vertex'; shape: number; vertex: number; press: number[]; active: boolean; start: Shape[] }
  | { kind: 'move'; shapes: number[]; x: number; y: number; press: number[]; active: boolean; start: Shape[] }
  | { kind: 'two'; press: number[] }
  | { kind: 'stroke'; points: number[][]; radius: number }
  | { kind: 'ghost'; shapes: number[]; x: number; y: number; press: number[]; active: boolean }
  | { kind: 'rotate'; shape: number; center: Pt; angle: number; press: number[]; active: boolean; start: Shape[] }
type Draft = { points: Pt[]; labels: number[]; preview: Shape[]; matched: Shape[] }
// `rotate`: one of a rotated box's handles, at the middle of its edges
type Hover = { shape: number; vertex: number; edge: number; rotate?: number } | null

const HISTORY_IMAGES = 30
// A page can't read the system's double-click time: this is the longest common default
const DOUBLE_CLICK_MS = 500
const CACHE_TILES = 400
const DRAG_START = 4

const FILTERS: { key: Filter; name: string; title: string; test: (i: ImageItem) => boolean }[] = [
  { key: 'all', name: 'All', title: 'Every image', test: () => true },
  { key: 'label', name: 'Label', title: 'No label file yet', test: (i) => !i.labelled && !i.broken },
  { key: 'review', name: 'Review', title: 'AI shapes to accept or reject', test: (i) => i.ai > 0 },
  { key: 'done', name: 'Done', title: 'Labelled, nothing left to review', test: (i) => i.labelled && !i.ai && !i.broken }
]
/** A filter's count, as 12k past 9,999, so the five fit side by side in the Files dock */
const few = (n: number) => (n < 10000 ? String(n) : n < 1e6 ? `${Math.floor(n / 1000)}k` : `${Math.floor(n / 1e5) / 10}M`)
const percent = (score: number) => `${Math.round(score * 100)}%`
const RELOADED = 'These labels changed in another app (such as Seamline), so they were reloaded.'
const COLOR_TITLES = { line: 'Shape line colour', fill: 'Shape fill colour', draw: 'Box line colour' }
const MODE_NAMES: Record<Mode, string> = {
  edit: 'Edit', polygon: 'Polygon', rectangle: 'Rectangle', oriented_rectangle: 'Rotated box', circle: 'Circle', line: 'Line', point: 'Point',
  linestrip: 'Line strip', pose: 'Pose', ai_points: 'AI points', ai_box: 'AI box', ai_pose: 'AI pose', brush: 'Brush', eraser: 'Eraser'
}
// Other poses, while one is selected or being placed
const DIM = 0.35
// A pose shorter than this on screen shows no keypoint numbers: they'd pile up on each other
const NUMBERED = 140
// A pose's keypoint dots grow with it on screen: their radius is its larger side over this, from 2.5 px up to point_size's
const DOT_SCALE = 45
// What the Pose tool's badge says, under the keypoint asked for
const PLACING_KEYS = 'Click: seen · Shift+click: hidden · X: skip · Enter: done'
// Pose every person takes YOLO-World's people from this score up, whatever the text prompt's: lower, it finds them in buses and shadows
const PERSON_SCORE = 0.25

/** The pre-labelling models chosen for folders, with this folder's choice: the last 50 folders' are kept */
function chooseFor(chosen: Record<string, string>, root: string, model: string) {
  const { [root]: _, ...rest } = chosen
  return Object.fromEntries([...Object.entries(rest), [root, model]].slice(-50))
}

/** A shape's own colour, [r, g, b, a] as labelme 3 and 4 kept it and LabelImg sets it; null for its label's */
export function ownColor(v: unknown): string | null {
  if (!Array.isArray(v) || v.length < 3 || !v.slice(0, 3).every((c) => typeof c === 'number' && Number.isFinite(c))) return null
  return '#' + v.slice(0, 3).map((c: number) => Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, '0')).join('')
}

// A labelme mask is a PNG of 0s and 1s: it becomes a tinted, see-through overlay
const maskCache = new Map<string, HTMLCanvasElement | null>()
function maskCanvas(mask: string, color: string, onReady: () => void) {
  const key = `${color}:${mask}`
  if (maskCache.has(key)) return maskCache.get(key)
  maskCache.set(key, null)
  const img = new Image()
  img.onload = () => {
    const c = document.createElement('canvas')
    c.width = img.naturalWidth
    c.height = img.naturalHeight
    const ctx = c.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const px = ctx.getImageData(0, 0, c.width, c.height)
    const v = parseInt(color.slice(1), 16)
    for (let i = 0; i < px.data.length; i += 4) {
      const on = px.data[i] > 0 || px.data[i + 1] > 0 || px.data[i + 2] > 0
      px.data[i] = (v >> 16) & 255; px.data[i + 1] = (v >> 8) & 255; px.data[i + 2] = v & 255
      px.data[i + 3] = on ? 255 : 0
    }
    ctx.putImageData(px, 0, 0)
    if (maskCache.size > 300) maskCache.delete(maskCache.keys().next().value!)
    maskCache.set(key, c)
    onReady()
  }
  img.src = `data:image/png;base64,${mask}`
  return null
}

const Labeler = forwardRef<LabelerHandle, Props>(function Labeler(props, ref) {
  const {
    data, config, settings, setSettings, setProject, setImages, discovered, forgetLabel, forgetFound, mac, onError, onShown, onOpenMenu, onChange, onChangeOutput,
    onNextToImages
  } = props
  const { project, images, root } = data
  const [index, setIndex] = useState(() => Math.max(0, data.select ? images.findIndex((i) => i.name === data.select) : 0))
  const [query, setQuery] = useState(config.file_search ?? '')
  const [filter, setFilter] = useState<Filter>('all')
  const [sort, setSort] = useState<'name' | 'certainty'>('name')
  const [mode, setModeState] = useState<Mode>('polygon')
  // The label new shapes get, by name, so it stays the same label when the list changes under it, as a
  // rename or another app changing the folder's classes changes it: the list's first until one is
  // picked, and null to be asked each time
  const [active, setActive] = useState<string | null | undefined>(undefined)
  const [shapes, setShapesState] = useState<Shape[]>([])
  const [selected, setSelectedState] = useState<number[]>([])
  const [imageFlags, setImageFlags] = useState<Record<string, boolean>>({})
  const [meta, setMeta] = useState<OpenImage | null>(null)
  const [readOnly, setReadOnly] = useState('')
  const [showShapes, setShowShapes] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [cursor, setCursor] = useState<number[] | null>(null)
  const [saved, setSaved] = useState<'saved' | 'saving' | 'unsaved'>('saved')
  const [loading, setLoading] = useState(false)
  const [busyAi, setBusyAi] = useState('')
  const [newClass, setNewClass] = useState('')
  const [brushSize, setBrushSize] = useState(config.brush.size)
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null)
  const [notice, setNotice] = useState('')
  const [dialog, setDialog] = useState<(LabelRequest & { resolve: (c: LabelChoice | null) => void }) | null>(null)
  const [pointsPrompt, setPointsPrompt] = useState<{ shape: number; edge: number; count: number } | null>(null)
  const [bc, setBc] = useState({ brightness: 1, contrast: 1 })
  const [verified, setVerifiedState] = useState(false)
  const [labelFilter, setLabelFilter] = useState('')
  const [colorPrompt, setColorPrompt] = useState<{ kind: 'line' | 'fill' | 'draw'; value: string } | null>(null)
  // LabelImg's right drag: where the copy was let go, waiting for Copy here or Move here
  const [ghostMenu, setGhostMenu] = useState<{ x: number; y: number } | null>(null)
  const [bcOpen, setBcOpen] = useState(false)
  const [models, setModels] = useState<ModelInfo[] | null>(null)
  const [modelsOpen, setModelsOpen] = useState(false)
  // labelme's AI Text Prompt: words to find, and how sure and how separate the finds must be
  const [aiPrompt, setAiPrompt] = useState('')
  const [aiScore, setAiScore] = useState(0.1)
  const [aiIou, setAiIou] = useState(0.5)
  const [download, setDownload] = useState<{ name: string; model: string; done: number; total: number } | null>(null)
  const [prelabelModel, setPrelabelModel] = useState<PrelabelModel | null>(null)
  // Models Seamline exported from this folder's training runs, newest first; null until read
  const [runModels, setRunModels] = useState<RunModel[] | null>(null)
  const [dragOver, setDragOver] = useState<number | null>(null)
  // Dataset > Check labels: the folder's last check, and one running (after saves, `quick`)
  const [check, setCheck] = useState<CheckSummary | null>(null)
  const [checking, setChecking] = useState<{ quick: boolean; progress: Progress | null } | null>(null)
  const [problemsFolded, setProblemsFolded] = useState(false)
  const [renaming, setRenaming] = useState<{ from: string; to: string } | null>(null)
  const [statsOpen, setStatsOpen] = useState(false)
  // Counts up when label files change outside the window's own saves, so Statistics reads them again
  const [labelsChanged, setLabelsChanged] = useState(0)
  // The folder's keypoints file, which poses are drawn and placed with: COCO's person until it's read
  const [keypoints, setKeypoints] = useState<FolderKeypoints>(() => ({ schema: builtInKeypoints(), file: null, problem: null }))
  // The poses opened or folded by hand in the Shapes list, by group_id. Others are folded, unless
  // some of their points are selected without their box.
  const [poseOpen, setPoseOpen] = useState<ReadonlyMap<number, boolean>>(() => new Map())
  // What the Pose tool asks for now, for the status bar; '' when no pose is being placed
  const [poseStep, setPoseStep] = useState('')
  // AI pose's score threshold: the config's, changed in its toolbar for the session
  const [poseScore, setPoseScore] = useState(config.ai.pose_score)
  const [, setTick] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const view = useRef({ scale: 1, x: 0, y: 0 })
  const shapesRef = useRef<Shape[]>([])
  const selectedRef = useRef<number[]>([])
  const flagsRef = useRef<Record<string, boolean>>({})
  const metaRef = useRef<OpenImage | null>(null)
  const readOnlyRef = useRef('')
  const loadedName = useRef('')
  const alive = useRef(true)
  const history = useRef(new Map<string, { undo: Shape[][]; redo: Shape[][] }>())
  const reviewOnLoad = useRef(false)
  // A shape to select once its image has loaded, for showImage
  const selectOnLoad = useRef<{ name: string; shape?: number } | null>(null)
  const reviewBack = useRef<string | null>(null)
  const undoOnLoad = useRef(false)
  const overCanvas = useRef(false)
  const draft = useRef<Draft | null>(null)
  const drag = useRef<Drag | null>(null)
  const hover = useRef<Hover>(null)
  const mouse = useRef<number[]>([0, 0])
  const screenMouse = useRef({ x: 0, y: 0 })
  const space = useRef(false)
  // When Space went down, until a pan: let go soon after, it's a tap, which runs a command set to Space
  const spaceTap = useRef<number | null>(null)
  const shift = useRef(false)
  const undo = useRef<Shape[][]>([])
  const redo = useRef<Shape[][]>([])
  const tiles = useRef(new Map<string, HTMLImageElement>())
  const frame = useRef(0)
  const saveTimer = useRef<ReturnType<typeof setTimeout>>()
  // `byHand`: Save, Save As or Save when asked, which writes an empty label file as labelme does; other saves are automatic
  const pending = useRef<{
    name: string; shapes: Shape[]; flags: Record<string, boolean>; size: [number, number]; format: LabelFormat; verified: boolean; from: LabelFormat | null
    byHand?: boolean
  } | null>(null)
  // The format of the image's label file, and the one it's being switched from until the switch is saved
  const fileFormat = useRef<LabelFormat | null>(null)
  const switchedFrom = useRef<LabelFormat | null>(null)
  const lastSave = useRef<Promise<boolean>>(Promise.resolve(true))
  // Counts up to open the image on screen again, in another format
  const [reload, setReload] = useState(0)
  const dirty = useRef(false)
  const saveTarget = useRef<string | null>(null)
  const clipboard = useRef<Shape[]>([])
  const prevShapes = useRef<Shape[]>([])
  const copyOnLoad = useRef(false)
  const bcByImage = useRef(new Map<string, { brightness: number; contrast: number }>())
  const aiSeq = useRef(0)
  const lastScale = useRef<number | null>(null)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const verifiedRef = useRef(false)
  // Changed at once when the format is switched, before the settings come back round
  const formatRef = useRef(settings.format)
  formatRef.current = settings.format
  const ctrl = useRef(false)
  const ghost = useRef<{ shapes: number[]; dx: number; dy: number } | null>(null)
  // Where the context menu was opened, for Add Point to Edge
  const menuPoint = useRef<Pt | null>(null)
  // A Fit to Selection put off until the mouse button is up and a double-click's time has passed
  const pendingFit = useRef(false)
  const fitTimer = useRef<ReturnType<typeof setTimeout>>()
  const checkRef = useRef(check)
  checkRef.current = check
  const checkingRef = useRef(false)
  const checkSeq = useRef(0)
  // Images saved since the last check, whose problems are out of date until the next one
  const stale = useRef(new Stale())
  const recheckTimer = useRef<ReturnType<typeof setTimeout>>()
  // Set each render: label files the window wrote, or saw written, put the last check out of date
  const touched = useRef<(names: string[]) => void>(() => {})
  // Waiting for an image to load, for a fix to another image than the one open
  const loadWaits = useRef<{ name: string; done: (ok: boolean) => void }[]>([])
  // The list and the image picked in it, for work that finishes after a render
  const listRef = useRef({ images, index })
  listRef.current = { images, index }
  // The folder being read again, and whether the window came back meanwhile, so it's read once more
  const refreshing = useRef({ running: false, again: false })
  // Pre-labelling under way, which writes into this folder as it goes, until it ends; and how far a batch is
  const prelabelling = useRef<Promise<unknown> | null>(null)
  const batchRef = useRef(batch)
  batchRef.current = batch
  // A pose being placed, one keypoint at a time, and the last one placed, for the guide the next is placed over
  const posing = useRef<PoseSession | null>(null)
  const lastLayout = useRef<PoseLayout | null>(null)
  const poseCache = useRef<{ shapes: Shape[]; schema: KeypointSchema; poses: PoseGroup[] } | null>(null)
  // Set each render: ends the pose being placed, keeping what's placed
  const endPoseRef = useRef<() => void>(() => {})

  // The keypoints file's names that are no class's, which validate_label: exact lets through as labels
  const keypointLabels = useMemo(() => keypointOnlyNames(keypoints.schema), [keypoints.schema])
  const poses = useMemo(() => posesIn(shapes, keypoints.schema), [shapes, keypoints.schema])
  // A pose's keypoints, by index, name keypoints, not classes: they stay out of the label list, its keys and
  // counts. A box or polygon named like a keypoint, or a point in no pose, is of a class as any shape is.
  const keypointAt = useMemo(() => posePoints(poses), [poses])
  // The label list as it was when a label was last added or recoloured here, which keeps its order
  const keptLabels = useRef<string[] | null>(null)
  // The label list: config labels, then a Seamline project's classes, then labels found in the files
  const classes = useMemo(() => labelList(config, project.classes, [...discovered, ...shapes.filter((_, i) => !keypointAt.has(i)).map((s) => s.label)], keptLabels.current),
    [config, project.classes, discovered, shapes, keypointAt])
  // A label gone from the list, as a class another app deleted, is picked again if it comes back; meanwhile new shapes ask for theirs
  const activeAt = classes.findIndex((c) => c.name === active)
  const activeClass = active === undefined ? 0 : activeAt < 0 ? null : activeAt
  // The list's first label until one is picked, held by its name, so a label another app puts ahead of it doesn't take its place
  useEffect(() => {
    if (active === undefined && classes.length) setActive(classes[0].name)
  }, [classes, active])
  const opts = useRef({ showShapes, settings, classes, mode, activeClass, brushSize, bc, keypoints: keypoints.schema })
  opts.current = { showShapes, settings, classes, mode, activeClass, brushSize, bc, keypoints: keypoints.schema }
  // labelme 7's allow_out_of_bounds_points: points may go past the image's edge
  const free = config.canvas.allow_out_of_bounds_points
  const imageSize = (): [number, number] | null => (free || !metaRef.current ? null : [metaRef.current.width, metaRef.current.height])

  const colorOf = useCallback((label: string) => {
    const list = opts.current.classes
    const at = list.findIndex((c) => c.name === label)
    return at >= 0 ? list[at].color : labelColor(label, list.length, config)
  }, [config])
  const currentLabel = () => (activeClass !== null ? classes[activeClass]?.name : undefined)
  /** The poses in a list of shapes, worked out again only when the list or the keypoints change */
  const posesOf = (list: Shape[]) => {
    const schema = opts.current.keypoints
    const c = poseCache.current
    if (c?.shapes === list && c.schema === schema) return c.poses
    const found = posesIn(list, schema)
    poseCache.current = { shapes: list, schema, poses: found }
    return found
  }

  const setSelected = (next: number[]) => {
    selectedRef.current = next
    setSelectedState(next)
  }
  const setMode = (m: Mode) => {
    if (readOnlyRef.current && m !== 'edit') return
    if (holdsBoxesOnly(formatRef.current) && !BOX_MODES.has(m)) {
      setNotice(`${FORMAT_NAMES[formatRef.current]} keeps boxes only: switch the save format to LabelMe JSON to draw other shapes.`)
      return
    }
    // A pose being placed keeps what's placed, as Escape leaves it
    endPoseRef.current()
    draft.current = null
    aiSeq.current++
    setModeState(m)
    invalidate()
  }

  const getTile = (url: string) => {
    const cache = tiles.current
    let img = cache.get(url)
    if (img) {
      cache.delete(url)
      cache.set(url, img)
      return img
    }
    img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => invalidate()
    img.onerror = () => {
      if (metaRef.current && !metaRef.current.pyramid && url === fileUrl(metaRef.current.file)) onError(`The window can't show ${loadedName.current}`)
    }
    img.src = url
    cache.set(url, img)
    if (cache.size > CACHE_TILES) cache.delete(cache.keys().next().value!)
    return img
  }

  const draw = useCallback(() => {
    frame.current = 0
    const canvas = canvasRef.current
    const m = metaRef.current
    if (!canvas || !canvas.clientWidth) return
    const dpr = devicePixelRatio
    const W = canvas.clientWidth, H = canvas.clientHeight
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
    }
    const ctx = canvas.getContext('2d', { alpha: false })!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--canvas').trim() || '#0d0e11'
    ctx.fillRect(0, 0, W, H)
    if (!m) return
    // LabelImg's Verify Image turns the background green
    if (verifiedRef.current) {
      ctx.fillStyle = 'rgba(184, 239, 38, 0.24)'
      ctx.fillRect(0, 0, W, H)
    }
    const { scale, x, y } = view.current
    const o = opts.current
    ctx.imageSmoothingEnabled = scale < 2
    if (o.bc.brightness !== 1 || o.bc.contrast !== 1) ctx.filter = `brightness(${o.bc.brightness}) contrast(${o.bc.contrast})`

    const put = (img: HTMLImageElement, ix: number, iy: number, iw: number, ih: number) => {
      if (!img.complete || !img.naturalWidth) return
      const x0 = Math.round(ix * scale + x), y0 = Math.round(iy * scale + y)
      ctx.drawImage(img, x0, y0, Math.round((ix + iw) * scale + x) - x0, Math.round((iy + ih) * scale + y) - y0)
    }
    if (!m.pyramid) {
      put(getTile(fileUrl(m.file)), 0, 0, m.width, m.height)
    } else {
      const { dir, maxLevel, tile } = m.pyramid
      const overview = Math.max(0, Math.ceil(Math.log2(Math.max(m.width, m.height) / tile)))
      const drawLevel = (level: number) => {
        const f = 2 ** (maxLevel - level)
        const c0 = Math.max(0, Math.floor(-x / scale / f / tile))
        const r0 = Math.max(0, Math.floor(-y / scale / f / tile))
        const c1 = Math.min(Math.ceil(m.width / f / tile) - 1, Math.floor((W - x) / scale / f / tile))
        const r1 = Math.min(Math.ceil(m.height / f / tile) - 1, Math.floor((H - y) / scale / f / tile))
        for (let r = r0; r <= r1; r++) {
          for (let c = c0; c <= c1; c++) {
            const img = getTile(fileUrl(`${dir}/${level}/${c}_${r}.jpeg`))
            if (img.naturalWidth) put(img, c * tile * f, r * tile * f, img.naturalWidth * f, img.naturalHeight * f)
          }
        }
      }
      // The whole-image tile fills gaps until sharper tiles arrive
      const coarsest = maxLevel - overview
      const want = Math.min(maxLevel, Math.max(coarsest, maxLevel + Math.ceil(Math.log2(scale * dpr))))
      drawLevel(coarsest)
      if (want !== coarsest) drawLevel(want)
    }
    ctx.filter = 'none'

    const S = (p: Pt): [number, number] => [p[0] * scale + x, p[1] * scale + y]
    const alpha = o.settings.fill
    const pointRadius = config.shape.point_size / 2
    const path = (s: Shape) => {
      ctx.beginPath()
      if (s.shape_type === 'circle' && s.points.length >= 2) {
        const [cx, cy] = S(s.points[0])
        ctx.arc(cx, cy, Math.hypot(s.points[1][0] - s.points[0][0], s.points[1][1] - s.points[0][1]) * scale, 0, Math.PI * 2)
        return
      }
      if (s.shape_type === 'point') {
        const [px, py] = S(s.points[0])
        ctx.arc(px, py, pointRadius + 1, 0, Math.PI * 2)
        return
      }
      const pts = outline(s)
      pts.forEach((p, k) => (k ? ctx.lineTo(...S(p)) : ctx.moveTo(...S(p))))
      if (CLOSED.has(s.shape_type) || s.shape_type === 'points') ctx.closePath()
    }
    const culling = config.performance.enable_viewport_culling
    const sel = new Set(selectedRef.current)
    const hv = hover.current
    // A pose being placed is drawn as it would be added
    const ps = posing.current
    const list = ps ? withSession(shapesRef.current, ps) : shapesRef.current
    const schema = o.keypoints
    const posed = posesOf(list)
    const poseOf = poseIndex(posed)
    // While a pose is selected or placed, the others dim
    const focused = new Set(posed.filter((p) => p.group === ps?.group || poseShapes(p).some((i) => sel.has(i))))
    const fadeOf = (p: PoseGroup | undefined) => (p && (focused.size || ps) && !focused.has(p) ? DIM : 1)
    if (o.showShapes) {
      list.forEach((s, i) => {
        // A pose's points are drawn with their pose, below
        if (s.hidden || (poseOf.has(i) && poseOf.get(i)!.box !== i)) return
        if (culling) {
          const [a, b, c, d] = bounds(s)
          const pad = 10 / scale
          if (c * scale + x < -pad || a * scale + x > W + pad || d * scale + y < -pad || b * scale + y > H + pad) return
        }
        const color = ownColor(s.other.line_color) ?? colorOf(s.label)
        const fillColor = ownColor(s.other.fill_color) ?? colorOf(s.label)
        const isSel = sel.has(i)
        const isHover = hv?.shape === i
        const fade = fadeOf(poseOf.get(i))
        // A pose's box: a rectangle marks the pose and isn't a region, so it isn't filled; selected with
        // its pose, its handles, which a long outline has by the dozen and which look like keypoints, stay
        // hidden but for the one under the pointer
        const boxOfPose = poseOf.get(i)?.box === i
        const whole = isSel && boxOfPose
        ctx.globalAlpha = fade
        if (s.shape_type === 'mask' && s.mask && s.points.length >= 2) {
          const c = maskCanvas(s.mask, fillColor, () => invalidate())
          if (c) {
            ctx.globalAlpha = fade * (isSel ? Math.min(1, alpha + 0.25) : Math.max(alpha, 0.35))
            const [x0, y0] = S([Math.min(s.points[0][0], s.points[1][0]), Math.min(s.points[0][1], s.points[1][1])])
            ctx.imageSmoothingEnabled = false
            ctx.drawImage(c, x0, y0, c.width * scale, c.height * scale)
            ctx.imageSmoothingEnabled = scale < 2
            ctx.globalAlpha = fade
          }
        }
        path(s)
        const closed = CLOSED.has(s.shape_type) || s.shape_type === 'point'
        if (closed && s.shape_type !== 'mask' && !(boxOfPose && s.shape_type === 'rectangle')) {
          ctx.globalAlpha = fade * (isSel ? Math.min(1, alpha + 0.2) : isHover ? Math.min(1, alpha + 0.08) : alpha)
          ctx.fillStyle = fillColor
          ctx.fill('evenodd')
          ctx.globalAlpha = fade
        }
        ctx.strokeStyle = isSel ? '#ffffff' : color
        ctx.lineWidth = isSel ? 2.5 : isHover ? 2 : 1.5
        ctx.setLineDash(isAi(s) ? [6, 4] : s.shape_type === 'mask' ? [3, 3] : [])
        if (s.shape_type !== 'point' || isSel) ctx.stroke()
        ctx.setLineDash([])
        if ((isSel && !whole) || (isHover && hv && hv.vertex >= 0)) {
          vertices(s).forEach((p, k) => {
            if (s.shape_type === 'mask') return
            const [px, py] = S(p)
            const lit = isHover && hv?.vertex === k
            if (whole && !lit) return
            ctx.beginPath()
            ctx.arc(px, py, lit ? 6 : 4.5, 0, Math.PI * 2)
            ctx.fillStyle = lit ? '#ffffff' : isSel ? color : '#ffffff'
            ctx.fill()
            ctx.strokeStyle = isSel ? '#ffffff' : color
            ctx.lineWidth = 2
            ctx.stroke()
          })
        }
        if (s.shape_type === 'oriented_rectangle' && s.points.length === 4) {
          // labelme's arrow along the first side, so the box's direction shows
          const [cx, cy] = S(boxCenter(s.points))
          const a = Math.atan2(s.points[1][1] - s.points[0][1], s.points[1][0] - s.points[0][0])
          const cos = Math.cos(a) * 8, sin = Math.sin(a) * 8
          ctx.beginPath()
          ;[[0.22, -0.5], [1, 0], [0.22, 0.5], [-1, 0]].forEach(([ax, ay], k) => {
            const px = cx + ax * cos - ay * sin, py = cy + ax * sin + ay * cos
            if (k) ctx.lineTo(px, py)
            else ctx.moveTo(px, py)
          })
          ctx.closePath()
          ctx.fillStyle = isSel ? '#ffffff' : color
          ctx.fill()
          if (isSel && !s.locked) {
            rotationHandles(s.points).forEach((q, r) => {
              const [hx, hy] = S(q)
              ctx.beginPath()
              ctx.rect(hx - 4, hy - 4, 8, 8)
              ctx.fillStyle = hv?.shape === i && hv.rotate === r ? '#ffffff' : color
              ctx.fill()
              ctx.strokeStyle = '#ffffff'
              ctx.lineWidth = 1.5
              ctx.stroke()
            })
          }
        }
        if (isHover && hv && hv.vertex < 0 && hv.edge >= 0 && (isSel || shift.current)) {
          const [a, b] = edges(s)[hv.edge] ?? []
          if (a !== undefined) {
            ctx.beginPath()
            ctx.moveTo(...S(s.points[a]))
            ctx.lineTo(...S(s.points[b]))
            ctx.strokeStyle = '#ffffff'
            ctx.lineWidth = 3
            ctx.stroke()
          }
        }
        ctx.globalAlpha = 1
      })
      // Poses: the keypoints file's lines between a pose's points, then the points, each in its
      // side's colour, whatever the labels' colours and the fill. Hidden points are rings, with
      // dashed lines to them. Points are sized to their pose on screen, whatever the zoom, so a
      // small person isn't buried under them.
      const biggest = pointRadius + 1, smallest = Math.min(2.5, biggest)
      const notes: { text: string; x: number; y: number; fade: number }[] = []
      const hoverPose = hv ? poseOf.get(hv.shape) : undefined
      for (const p of posed) {
        const kps = schema.classes[p.cls] ?? []
        const sides = sidesOf(schema, p.cls)
        const fade = fadeOf(p)
        const shown = (k: number) => {
          const i = k < 0 ? null : p.points[k]
          return i === null || list[i].hidden ? null : list[i]
        }
        // Its size on screen: its box's larger side, or its points' without a box
        let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity]
        for (const i of p.box !== null ? [p.box] : poseShapes(p)) {
          const [a, b, c, d] = bounds(list[i])
          x0 = Math.min(x0, a); y0 = Math.min(y0, b); x1 = Math.max(x1, c); y1 = Math.max(y1, d)
        }
        const size = Math.max(x1 - x0, y1 - y0) * scale
        const radius = Math.max(smallest, Math.min(biggest, size / DOT_SCALE))
        const line = Math.max(1.25, Math.min(2, radius * 0.4))
        ctx.globalAlpha = fade
        for (const [a, b] of schema.skeleton?.[p.cls] ?? []) {
          const ka = kps.indexOf(a), kb = kps.indexOf(b)
          const sa = shown(ka), sb = shown(kb)
          if (!sa || !sb) continue
          ctx.beginPath()
          ctx.moveTo(...S(sa.points[0]))
          ctx.lineTo(...S(sb.points[0]))
          ctx.setLineDash(isHidden(sa) || isHidden(sb) ? [5, 4] : [])
          ctx.strokeStyle = HALO
          ctx.lineWidth = line + 2
          ctx.stroke()
          ctx.strokeStyle = SIDE_COLORS[sides[ka] === sides[kb] ? sides[ka] : 'centre']
          ctx.lineWidth = line
          ctx.stroke()
        }
        ctx.setLineDash([])
        // A pose selected or being placed shows its points' numbers, when it's tall enough for them not to
        // pile up; names show on the point under the pointer, and with Display labels on the pose hovered
        // or selected. Points selected without their box are ringed.
        const numbered = focused.has(p) && size >= NUMBERED
        const named = o.settings.displayLabels && (focused.has(p) || p === hoverPose)
        const whole = p.box !== null && sel.has(p.box)
        const dot = (i: number, k: number | null) => {
          const s = list[i]
          if (s.hidden) return
          const side = sides[k ?? kps.indexOf(s.label)] ?? 'centre'
          const [px, py] = S(s.points[0])
          const lit = hv?.shape === i
          ctx.beginPath()
          ctx.arc(px, py, lit ? radius + 1.5 : radius, 0, Math.PI * 2)
          if (isHidden(s)) {
            ctx.strokeStyle = HALO
            ctx.lineWidth = 4.5
            ctx.stroke()
            ctx.strokeStyle = SIDE_COLORS[side]
            ctx.lineWidth = 2
            ctx.stroke()
          } else {
            ctx.fillStyle = SIDE_COLORS[side]
            ctx.fill()
            ctx.strokeStyle = HALO
            ctx.lineWidth = 1.5
            ctx.stroke()
          }
          if (sel.has(i) && !whole) {
            ctx.beginPath()
            ctx.arc(px, py, radius + 3.5, 0, Math.PI * 2)
            ctx.strokeStyle = '#ffffff'
            ctx.lineWidth = 2
            ctx.stroke()
          }
          const text = [numbered && k !== null ? String(k + 1) : '', lit || named ? spoken(s.label) : ''].filter(Boolean).join(' ')
          if (text) notes.push({ text, x: px + radius + 5, y: py, fade })
        }
        p.points.forEach((i, k) => i !== null && dot(i, k))
        for (const i of p.extra) dot(i, null)
        ctx.globalAlpha = 1
      }
      // The guide: the last pose placed here, scaled into the box of the one being placed, and
      // mirrored once the points placed say it faces the other way; its keypoint asked for now ringed
      const placing = ps ? posed.find((p) => p.group === ps.group) : undefined
      const placed = placing ? placing.points.map((i) => (i === null ? null : list[i].points[0])) : []
      const guide = ps?.box && lastLayout.current?.cls === ps.cls ? guideIn(lastLayout.current, boxOf(ps.box), placed, sidesOf(schema, ps.cls)) : null
      if (ps?.box && guide) {
        const kps = schema.classes[ps.cls] ?? []
        const sides = sidesOf(schema, ps.cls)
        const now = askedNow(ps)
        const [x0, y0, x1, y1] = boxOf(ps.box)
        const radius = Math.max(smallest, Math.min(biggest, (Math.max(x1 - x0, y1 - y0) * scale) / DOT_SCALE))
        ctx.globalAlpha = 0.3
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1.5
        for (const [a, b] of schema.skeleton?.[ps.cls] ?? []) {
          const ga = guide[kps.indexOf(a)], gb = guide[kps.indexOf(b)]
          if (!ga || !gb) continue
          ctx.beginPath()
          ctx.moveTo(...S(ga))
          ctx.lineTo(...S(gb))
          ctx.stroke()
        }
        guide.forEach((at, k) => {
          if (!at) return
          ctx.globalAlpha = k === now ? 0.9 : 0.4
          ctx.beginPath()
          ctx.arc(...S(at), k === now ? radius + 3 : radius - 1, 0, Math.PI * 2)
          ctx.strokeStyle = SIDE_COLORS[sides[k] ?? 'centre']
          ctx.lineWidth = 2
          ctx.stroke()
        })
        ctx.globalAlpha = 1
      }
      if (notes.length) {
        ctx.font = '600 11px system-ui, sans-serif'
        ctx.textBaseline = 'middle'
        ctx.lineJoin = 'round'
        for (const n of notes) {
          ctx.globalAlpha = n.fade
          ctx.strokeStyle = HALO
          ctx.lineWidth = 3
          ctx.strokeText(n.text, n.x, n.y)
          ctx.fillStyle = '#ffffff'
          ctx.fillText(n.text, n.x, n.y)
        }
        ctx.globalAlpha = 1
        ctx.textBaseline = 'alphabetic'
        ctx.lineJoin = 'miter'
      }
      const g = ghost.current
      if (g) {
        ctx.setLineDash([5, 4])
        ctx.lineWidth = 2
        for (const i of g.shapes) {
          const s = shapesRef.current[i]
          if (!s) continue
          path({ ...s, points: s.points.map(([px, py]) => [px + g.dx, py + g.dy]) })
          ctx.globalAlpha = 0.25
          ctx.fillStyle = colorOf(s.label)
          if (CLOSED.has(s.shape_type)) ctx.fill('evenodd')
          ctx.globalAlpha = 1
          ctx.strokeStyle = '#ffffff'
          ctx.stroke()
        }
        ctx.setLineDash([])
      }
      // LabelImg's Display Labels: each label at its shape's top left, inside when there's no room above;
      // a pose's points have their names beside them
      if (o.settings.displayLabels) {
        ctx.font = '600 12px system-ui, sans-serif'
        ctx.textBaseline = 'middle'
        list.forEach((s, i) => {
          if (s.hidden || !s.label || (poseOf.has(i) && poseOf.get(i)!.box !== i)) return
          const text = s.group_id !== null ? `${s.label} (${s.group_id})` : s.label
          const [x0, y0] = bounds(s)
          const [lx, top] = S([x0, y0])
          const w = ctx.measureText(text).width + 8, h = 17
          const ly = top - h < 0 ? top : top - h
          if (lx > W || ly > H || lx + w < 0 || ly + h < 0) return
          const color = colorOf(s.label)
          const v = parseInt(color.slice(1), 16)
          ctx.globalAlpha = 0.85
          ctx.fillStyle = color
          ctx.fillRect(lx, ly, w, h)
          ctx.globalAlpha = 1
          ctx.fillStyle = ((v >> 16) & 255) * 0.299 + ((v >> 8) & 255) * 0.587 + (v & 255) * 0.114 > 150 ? '#000000' : '#ffffff'
          ctx.fillText(text, lx + 4, ly + h / 2 + 0.5)
        })
        ctx.textBaseline = 'alphabetic'
      }
    }

    const current = o.activeClass !== null ? o.classes[o.activeClass]?.color ?? '#ffffff' : '#ffffff'
    const chosen = selectedRef.current.length === 1 ? shapesRef.current[selectedRef.current[0]] : undefined
    const brushColor = chosen && !chosen.hidden ? colorOf(chosen.label) : current
    const d = drag.current
    const erasing = o.mode === 'eraser'
    if (d?.kind === 'stroke') {
      ctx.beginPath()
      d.points.forEach((p, k) => (k ? ctx.lineTo(...S(p)) : ctx.moveTo(...S(p))))
      if (d.points.length === 1) ctx.lineTo(...S(d.points[0]))
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = d.radius * 2 * scale
      ctx.globalAlpha = 0.55
      ctx.strokeStyle = erasing ? '#ffffff' : brushColor
      ctx.stroke()
      ctx.globalAlpha = 1
      ctx.lineCap = 'butt'
      ctx.lineJoin = 'miter'
    }
    if ((erasing || o.mode === 'brush') && overCanvas.current) {
      ctx.beginPath()
      ctx.arc(...S(mouse.current), o.brushSize / 2, 0, Math.PI * 2)
      ctx.strokeStyle = erasing ? '#ffffff' : brushColor
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    // What's being drawn, and the line or box to the pointer
    const dr = draft.current
    const live = livePoint()
    // LabelImg's Box Line Color is the colour of what's being drawn
    const pen = o.settings.drawColor ?? current
    if (dr && live) {
      const m2 = o.mode
      ctx.strokeStyle = pen
      ctx.fillStyle = withAlpha(pen, 0.25)
      ctx.lineWidth = 2
      if (m2 === 'ai_points') {
        for (const found of dr.matched) {
          if (!shapesRef.current.includes(found)) continue
          path(found)
          ctx.strokeStyle = '#ffd166'
          ctx.lineWidth = 3
          ctx.stroke()
          ctx.strokeStyle = pen
          ctx.lineWidth = 2
        }
        for (const shape of dr.preview) {
          path(shape)
          if (shape.shape_type === 'mask' && shape.mask) {
            const c = maskCanvas(shape.mask, current, () => invalidate())
            if (c) {
              ctx.globalAlpha = 0.45
              const [x0, y0] = S(shape.points[0])
              ctx.imageSmoothingEnabled = false
              ctx.drawImage(c, x0, y0, c.width * scale, c.height * scale)
              ctx.globalAlpha = 1
            }
          } else if (CLOSED.has(shape.shape_type)) ctx.fill('evenodd')
          ctx.setLineDash([5, 4])
          ctx.stroke()
          ctx.setLineDash([])
        }
        dr.points.forEach((p, k) => {
          ctx.beginPath()
          ctx.arc(...S(p), 5, 0, Math.PI * 2)
          ctx.fillStyle = dr.labels[k] ? '#3ecf8e' : '#ff6b6b'
          ctx.fill()
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 1.5
          ctx.stroke()
        })
      } else {
        const shape = draftShape(live)
        if (shape) {
          path(shape)
          if (CLOSED.has(shape.shape_type) && (o.settings.fillDrawing || shape.shape_type !== 'polygon')) {
            ctx.fillStyle = withAlpha(pen, 0.25)
            ctx.fill('evenodd')
          }
          ctx.strokeStyle = pen
          ctx.lineWidth = 2
          ctx.stroke()
        }
        dr.points.forEach((p, k) => {
          ctx.beginPath()
          ctx.arc(...S(p), k === 0 && m2 === 'polygon' ? 6 : 3.5, 0, Math.PI * 2)
          ctx.fillStyle = k === 0 ? pen : '#ffffff'
          ctx.fill()
        })
      }
    }
    // labelme's crosshair, per drawing mode
    const ch = config.canvas.crosshair as Record<string, boolean>
    if (overCanvas.current && ch[CROSSHAIR[o.mode] ?? o.mode] && !d) {
      const [mx, my] = S(mouse.current)
      ctx.strokeStyle = 'rgba(255,255,255,0.45)'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(0, my); ctx.lineTo(W, my); ctx.moveTo(mx, 0); ctx.lineTo(mx, H)
      ctx.stroke()
      ctx.setLineDash([])
    }
    // The Pose tool's badge by the pointer: the keypoint asked for, a ring with Shift for a hidden
    // one, and under it the keys, which the status bar has no room for. Beside a small person's box,
    // at the pointer's height, so it doesn't cover the rest of them.
    const asked = ps ? askedNow(ps) : null
    if (ps && asked !== null && overCanvas.current) {
      const kps = schema.classes[ps.cls] ?? []
      const text = `${asked + 1}/${kps.length} ${spoken(kps[asked] ?? '')}${shift.current ? ' · hidden' : ''}`
      ctx.textBaseline = 'middle'
      ctx.font = '500 10.5px system-ui, sans-serif'
      const keysWidth = ctx.measureText(PLACING_KEYS).width
      ctx.font = '600 12px system-ui, sans-serif'
      const w = Math.max(ctx.measureText(text).width + 27, keysWidth + 16), h = 36
      const [mx, my] = S(mouse.current)
      let bx = mx + 16 + w > W ? mx - 16 - w : mx + 16
      let by = my + 16 + h > H ? my - 16 - h : my + 16
      if (ps.box) {
        const [x0, y0, x1, y1] = boxOf(ps.box)
        const [left, top] = S([x0, y0]), [right, bottom] = S([x1, y1])
        const side = right + 10 + w <= W ? right + 10 : left - 10 - w >= 0 ? left - 10 - w : null
        if (right - left < 240 && side !== null && my > top - h && my < bottom + h) {
          bx = side
          by = Math.max(4, Math.min(H - h - 4, my - h / 2))
        }
      }
      ctx.fillStyle = 'rgba(16, 18, 22, 0.88)'
      ctx.beginPath()
      ctx.roundRect(bx, by, w, h, 5)
      ctx.fill()
      const color = SIDE_COLORS[sidesOf(schema, ps.cls)[asked] ?? 'centre']
      ctx.beginPath()
      ctx.arc(bx + 12, by + 12, 4, 0, Math.PI * 2)
      if (shift.current) {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.stroke()
      } else {
        ctx.fillStyle = color
        ctx.fill()
      }
      ctx.fillStyle = '#ffffff'
      ctx.fillText(text, bx + 21, by + 12.5)
      ctx.font = '500 10.5px system-ui, sans-serif'
      ctx.fillStyle = 'rgba(255, 255, 255, 0.68)'
      ctx.fillText(PLACING_KEYS, bx + 8, by + 27)
      ctx.textBaseline = 'alphabetic'
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorOf, config])

  const invalidate = useCallback(() => {
    if (!frame.current) frame.current = requestAnimationFrame(draw)
  }, [draw])

  /** Boxes come out square with Shift held, as in labelme, or Ctrl, or Draw Squares on, as in LabelImg */
  const squareNow = () => shift.current || ctrl.current || settingsRef.current.drawSquares
  const squareKeys = (e: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean }) =>
    e.shiftKey || e.ctrlKey || e.metaKey || settingsRef.current.drawSquares

  /** The pointer as a point for the shape being drawn: on the image, as labelme keeps it */
  const livePoint = (): Pt | null => {
    const m = metaRef.current
    const dr = draft.current
    if (!m || !dr) return null
    // A rotated box's third corner is fitted on the image by reprojectCorners itself
    if (free || (opts.current.mode === 'oriented_rectangle' && dr.points.length === 4)) return mouse.current
    const last = dr.points[dr.points.length - 1]
    return last ? toEdge(last, mouse.current, m.width, m.height) : mouse.current
  }
  const draftShape = (p: Pt): Shape | null => {
    const dr = draft.current
    if (!dr || !dr.points.length) return null
    const md = opts.current.mode
    const first = dr.points[0]
    // Drawn open until it's closed
    if (md === 'polygon' || md === 'linestrip') return newShape('', 'linestrip', [...dr.points, p])
    if (md === 'rectangle') return newShape('', 'rectangle', [first, squareNow() ? squareFrom(first, p) : p])
    if (md === 'ai_box' || md === 'pose' || md === 'ai_pose') return newShape('', 'rectangle', [first, p])
    if (md === 'oriented_rectangle') {
      // Two corners make one side; then the pointer sets how wide it is
      if (dr.points.length === 1) return newShape('', 'line', [first, p])
      return newShape('', 'oriented_rectangle', reprojectCorners(dr.points, 2, p, imageSize()))
    }
    if (md === 'circle') return newShape('', 'circle', [first, p])
    if (md === 'line') return newShape('', 'line', [first, p])
    return null
  }

  // Zoom, fit and scroll
  const fitScale = useCallback((how: 'window' | 'width') => {
    const canvas = canvasRef.current, m = metaRef.current
    if (!canvas || !m) return 1
    const pad = 24
    const sw = (canvas.clientWidth - pad * 2) / m.width
    return how === 'width' ? sw : Math.min(sw, (canvas.clientHeight - pad * 2) / m.height)
  }, [])
  const setScale = useCallback((scale: number, at?: { x: number; y: number }) => {
    const canvas = canvasRef.current, m = metaRef.current
    if (!canvas || !m) return
    const max = config.max_zoom / 100
    const next = Math.max(0.005, Math.min(max, scale))
    const px = at?.x ?? canvas.clientWidth / 2, py = at?.y ?? canvas.clientHeight / 2
    const v = view.current
    view.current = { scale: next, x: px - ((px - v.x) / v.scale) * next, y: py - ((py - v.y) / v.scale) * next }
    lastScale.current = next
    setZoom(next)
    invalidate()
  }, [config.max_zoom, invalidate])
  const fit = useCallback((how: 'window' | 'width' = 'window') => {
    const canvas = canvasRef.current, m = metaRef.current
    if (!canvas || !m) return
    const scale = fitScale(how)
    view.current = {
      scale,
      x: (canvas.clientWidth - m.width * scale) / 2,
      y: how === 'width' ? 24 : (canvas.clientHeight - m.height * scale) / 2
    }
    setZoom(scale)
    invalidate()
  }, [fitScale, invalidate])
  const setFit = (f: Settings['fit']) => setSettings((s) => ({ ...s, fit: f }))
  const zoomBy = (factor: number, at?: { x: number; y: number }) => {
    setFit('manual')
    setScale(view.current.scale * factor, at)
  }

  /**
   * Another app, such as Seamline, changed the labels of the image on screen since the window read
   * them: they're read again, and what was being done here is dropped, as the note says
   */
  const reloadChanged = useCallback((name: string, note = '') => {
    const { images: list, index: at } = listRef.current
    if (list[at]?.name !== name) return
    clearTimeout(saveTimer.current)
    pending.current = null
    dirty.current = false
    setSaved('saved')
    setReload((n) => n + 1)
    if (note) onError(`${RELOADED} ${note}`, `${root}\n${name}`)
    else setNotice(RELOADED)
  }, [onError, root])
  /** Nothing here that reading the labels again would lose: no unsaved change, and nothing being drawn, dragged or asked */
  const idle = useCallback(() => !dirty.current && !drag.current && !draft.current && !posing.current && !document.querySelector('.modal-back, .context'), [])

  // Saving: a moment after each change when saving is automatic, or when asked
  const doSave = useCallback(async (job: NonNullable<typeof pending.current>, target?: string) => {
    setSaved('saving')
    try {
      const result = await window.api.saveLabels(root, job.name, {
        shapes: job.shapes.map(({ hidden: _h, ...s }) => s), flags: job.flags, size: job.size,
        storeData: settingsRef.current.storeData, target, format: job.format, verified: job.verified,
        classes: opts.current.classes.map((c) => c.name), from: job.from ?? undefined, auto: !job.byHand
      })
      if ('changed' in result) {
        // Nothing was written over them
        const { images: list, index: at } = listRef.current
        if (list[at]?.name === job.name) reloadChanged(job.name, "Your last change wasn't saved.")
        else onError(`The labels of ${job.name} changed in another app (such as Seamline), so your last change to them wasn't saved.`)
        return false
      }
      const { item, replaced, kept, format } = result
      if (!target) setImages((all) => all.map((i) => (i.name === job.name ? { ...i, ...item } : i)))
      if (!target) touched.current([job.name])
      if (!target && loadedName.current === job.name) {
        fileFormat.current = format
        if (switchedFrom.current === job.from) switchedFrom.current = null
      }
      const file = (p: string) => p.split(/[\\/]/).pop()
      if (replaced) setNotice((n) => `${n ? `${n} ` : ''}Its old ${file(replaced)} went to the trash.`)
      if (kept && format) onError(`Saved as ${FORMAT_NAMES[format]}, but ${kept}`)
      if (pending.current === null && loadedName.current === job.name) dirty.current = false
      setSaved(dirty.current ? 'unsaved' : 'saved')
      return true
    } catch (e) {
      onError(`Save failed: ${plainError(e)}`)
      setSaved('unsaved')
      return false
    }
  }, [root, setImages, onError, reloadChanged])

  const flush = useCallback(async () => {
    clearTimeout(saveTimer.current)
    const job = pending.current
    if (!job) return true
    pending.current = null
    const done = doSave(job, saveTarget.current ?? undefined)
    lastSave.current = done
    return done
  }, [doSave])

  const queueSave = () => {
    const m = metaRef.current
    const name = loadedName.current
    if (!m || !name || readOnlyRef.current) return
    pending.current = {
      name, shapes: shapesRef.current, flags: flagsRef.current, size: [m.width, m.height], format: formatRef.current, verified: verifiedRef.current,
      from: switchedFrom.current
    }
    dirty.current = true
    setSaved('unsaved')
    onChange()
    clearTimeout(saveTimer.current)
    if (settingsRef.current.autoSave) saveTimer.current = setTimeout(flush, 400)
  }

  const apply = (next: Shape[]) => {
    shapesRef.current = next
    setShapesState(next)
    invalidate()
    const name = loadedName.current
    if (!name || readOnlyRef.current) return
    setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(next, true, opts.current.keypoints) } : i)))
    queueSave()
  }
  /** Shapes as the save format keeps them: boxes in LabelImg's formats, which leave points out */
  const forFormat = (list: Shape[]) => (holdsBoxesOnly(formatRef.current) ? keptAsBoxes(list).shapes : list)
  /** What a LabelImg format did to shapes, said: '' when it kept them as they were */
  const boxedNote = (list: Shape[]) => {
    if (!holdsBoxesOnly(formatRef.current)) return ''
    const { boxed, left } = keptAsBoxes(list)
    const parts = [boxed ? counted(boxed, 'shape became a box', 'shapes became boxes') : '', left ? `${counted(left, 'point was', 'points were')} left out` : '']
    return boxed || left ? `${FORMAT_NAMES[formatRef.current]} keeps boxes only: ${parts.filter(Boolean).join(' and ')}.` : ''
  }
  const setShapes = (next: Shape[], record = true) => {
    if (readOnlyRef.current) return
    if (record) {
      undo.current.push(shapesRef.current)
      if (undo.current.length > config.canvas.num_backups) undo.current.shift()
      redo.current = []
    }
    apply(next)
  }
  const setFlagsValue = (next: Record<string, boolean>) => {
    if (readOnlyRef.current) return
    flagsRef.current = next
    setImageFlags(next)
    queueSave()
  }

  /** Before another image or folder opens: save, or ask when saving isn't automatic */
  const canLeave = useCallback(async ({ stay = false }: { stay?: boolean } = {}) => {
    // A pose being placed is kept as far as it's placed, then saved with the rest
    endPoseRef.current()
    if (!dirty.current) return true
    if (settingsRef.current.autoSave) return flush()
    const name = loadedName.current
    const answer = await window.api.ask(`Save the changes to ${name}?`, "They're lost if you don't save them.", ['Save', "Don't save", 'Cancel'])
    if (answer === 0) {
      if (pending.current) pending.current.byHand = true
      return flush()
    }
    if (answer === 1) {
      pending.current = null
      dirty.current = false
      setSaved('saved')
      // Staying on the image, the changes let go would be saved with the next: it's read again as saved
      if (stay && loadedName.current === name) {
        history.current.delete(name)
        undo.current = []
        redo.current = []
        setReload((n) => n + 1)
      }
      return true
    }
    return false
  }, [flush])
  /** Before another folder opens: a pre-label batch, which writes into this one, stops if the user agrees, and after the image it's on */
  const canLeaveFolder = useCallback(async () => {
    const run = batchRef.current
    if (run && (await window.api.ask('Stop pre-labelling?', `${run.done} of ${run.total} images are done and keep their labels. The rest aren't pre-labelled.`, ['Stop', 'Cancel'])) !== 0) return false
    if (!(await canLeave())) return false
    if (prelabelling.current) {
      window.api.prelabelStop()
      await prelabelling.current.catch(() => {})
    }
    return true
  }, [canLeave])
  /** What here waits on the user, which opening another image or folder would cut short: '' when nothing does */
  const busy = useCallback(() => {
    if (window.api.asking() || document.querySelector('.modal-back, .context')) return 'A dialog or menu is open here: close it, then try again.'
    if (draft.current || posing.current || (drag.current && drag.current.kind !== 'pan')) return 'A shape is being drawn here: finish it, then try again.'
    return ''
  }, [])

  // Load the current image; older requests are dropped if you move on quickly
  useEffect(() => {
    const name = images[index]?.name
    if (!name) return
    let stale = false
    // The same image again, as when its labels changed elsewhere: the view stays where it is
    const again = name === loadedName.current
    if (!again) setLoading(true)
    const rememberHistory = () => {
      const last = loadedName.current
      if (!last) return
      history.current.delete(last)
      history.current.set(last, { undo: undo.current, redo: redo.current })
      if (history.current.size > HISTORY_IMAGES) history.current.delete(history.current.keys().next().value!)
      bcByImage.current.set(last, opts.current.bc)
    }
    ;(async () => {
      await flush()
      // Never an image's own shapes, read again: changes let go would come back
      const copy = !again && (settingsRef.current.keepPrev || copyOnLoad.current) && loadedName.current !== name
      copyOnLoad.current = false
      try {
        // Labels last: a big image can take a while, and pre-labelling may write meanwhile
        const m = await window.api.openImage(root, name)
        if (stale) return
        let doc = null
        let found: Loaded | null = null
        let broken = ''
        try {
          found = await window.api.loadLabels(root, name, formatRef.current, [m.width, m.height], true)
          doc = found.doc
        } catch (e) {
          broken = plainError(e)
        }
        if (stale) return
        // Read again, as when a save was refused for another app's change: the lists count these labels, not the change dropped
        if (again && !broken) {
          const item = { ...summarize(doc?.shapes ?? [], !!found?.format, opts.current.keypoints), broken: undefined, verified: found?.verified || undefined }
          setImages((all) => all.map((i) => (i.name === name ? { ...i, ...item } : i)))
        }
        rememberHistory()
        // Undo would bring back what was there before another app changed it
        if (found?.changed) history.current.delete(name)
        const kept = history.current.get(name)
        const fromReview = reviewOnLoad.current
        const undoHere = undoOnLoad.current
        reviewOnLoad.current = false
        undoOnLoad.current = false
        reviewBack.current = fromReview ? loadedName.current : null
        const previous = loadedName.current ? shapesRef.current : []
        // labelme fills in the flags a label's patterns list, keeping the file's values
        let loaded = (doc?.shapes ?? []).map((s) => ({ ...s, flags: { ...flagsFor(s.label, config.label_flags), ...s.flags } }))
        const flags: Record<string, boolean> = {}
        ;(config.flags ?? []).forEach((f) => (flags[f] = false))
        Object.assign(flags, doc?.flags ?? {})
        const sameView = again && metaRef.current?.width === m.width && metaRef.current?.height === m.height
        metaRef.current = m
        loadedName.current = name
        readOnlyRef.current = broken
        setReadOnly(broken)
        saveTarget.current = null
        undo.current = kept?.undo ?? []
        redo.current = kept?.redo ?? []
        draft.current = null
        setPosing(null)
        drag.current = null
        hover.current = null
        flagsRef.current = flags
        setImageFlags(flags)
        verifiedRef.current = !!found?.verified
        setVerifiedState(verifiedRef.current)
        fileFormat.current = found?.format ?? null
        switchedFrom.current = null
        setLabelFilter('')
        // As in LabelImg, the save format follows the label file opened
        if (found?.format && found.format !== formatRef.current) {
          const format = found.format
          formatRef.current = format
          setSettings((s) => ({ ...s, format }))
        }
        let copied = false
        if (copy && !doc && !broken && previous.length) {
          // labelme's "keep previous annotation": an image with no labels starts from the last one's
          loaded = forFormat(previous.map((s) => ({ ...s, hidden: undefined })))
          copied = true
        }
        prevShapes.current = previous
        shapesRef.current = loaded
        setMeta(m)
        setShapesState(loaded)
        setSelected([])
        if (!again) setPoseOpen(new Map())
        // Until a pose is placed here, the guide is the last pose of the first image that has one
        if (!lastLayout.current) {
          lastLayout.current = posesOf(loaded).map((p) => layoutOf(loaded, p)).filter((l) => l !== null).pop() ?? null
        }
        dirty.current = false
        pending.current = null
        setSaved('saved')
        // What was said of the last image's labels isn't about these
        if (!again) {
          setNotice((n) => (n === RELOADED ? '' : n))
          onShown(`${root}\n${name}`)
        }
        if (!settingsRef.current.keepPrevBC) setBc(bcByImage.current.get(name) ?? { brightness: 1, contrast: 1 })
        if (!sameView && settingsRef.current.keepPrevScale && lastScale.current !== null) {
          const canvas = canvasRef.current!
          const s = lastScale.current
          view.current = { scale: s, x: (canvas.clientWidth - m.width * s) / 2, y: (canvas.clientHeight - m.height * s) / 2 }
          setZoom(s)
        } else if (!sameView) fit(settingsRef.current.fit === 'width' ? 'width' : 'window')
        if (copied) queueSave()
        if (broken) onError(`${name}: its label file can't be read, so it's shown without labels and won't be changed. ${broken}`)
        if (undoHere) stepHistory(undo, redo)
        const first = reviewSteps(shapesRef.current)[0]
        if ((fromReview || undoHere) && first) selectStep(first, shapesRef.current)
        const pick = selectOnLoad.current
        selectOnLoad.current = null
        if (pick?.name === name && pick.shape !== undefined && shapesRef.current[pick.shape]) selectForReview(pick.shape, shapesRef.current)
        invalidate()
        for (const w of loadWaits.current.splice(0)) w.done(w.name === name)
      } catch (e) {
        if (stale) return
        for (const w of loadWaits.current.splice(0)) w.done(false)
        onError(`Could not open ${name}: ${plainError(e)}`)
        rememberHistory()
        reviewOnLoad.current = false
        undoOnLoad.current = false
        reviewBack.current = null
        metaRef.current = null
        loadedName.current = ''
        shapesRef.current = []
        undo.current = []
        redo.current = []
        drag.current = null
        draft.current = null
        setPosing(null)
        setMeta(null)
        setShapesState([])
        setSelected([])
        invalidate()
      } finally {
        if (!stale) setLoading(false)
      }
    })()
    return () => { stale = true }
    // By name: images another app adds or removes move it in the list, which doesn't load it again
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images[index]?.name, root, reload])

  useEffect(invalidate, [showShapes, settings.fill, settings.fillDrawing, settings.displayLabels, verified, selected, mode, activeClass, brushSize, classes, bc, keypoints, invalidate])
  // LabelImg's formats keep boxes: a tool for other shapes gives way to the rectangle
  useEffect(() => {
    if (holdsBoxesOnly(settings.format) && !BOX_MODES.has(mode)) setModeState('rectangle')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.format])
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 5000)
    return () => clearTimeout(timer)
  }, [notice])
  useEffect(() => {
    // A keypoint's own row when its pose is open, else its pose's
    ;(document.querySelector('.shapes .shape.on') ?? document.querySelector('.shapes .pose-row.on'))?.scrollIntoView({ block: 'nearest' })
  }, [selected])
  useEffect(() => {
    document.querySelector('.file.on')?.scrollIntoView({ block: 'nearest' })
  }, [index])
  useEffect(() => {
    const canvas = canvasRef.current!
    const observer = new ResizeObserver(() => {
      const f = settingsRef.current.fit
      if (f === 'selection') fitSelection()
      else if (f !== 'manual' && metaRef.current) fit(f)
      else invalidate()
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [invalidate, fit])
  useEffect(() => () => void flush(), [flush])
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])
  useEffect(() => window.api.onPrelabel(({ root: at, name, shapes: found, changed, done, total }) => {
    if (at !== root) return
    setBatch(done >= total ? null : { done, total })
    // Left alone, as another app changed it since the window read it: read again, unless something here isn't saved yet
    if (changed && name === loadedName.current && idle()) reloadChanged(name)
    if (!found) return
    history.current.delete(name)
    touched.current([name])
    if (name === loadedName.current) {
      // Anything being edited here wins: it saves over this result
      if (pending.current || drag.current || shapesRef.current.some((s) => !isAi(s))) return
      shapesRef.current = found
      undo.current = []
      redo.current = []
      setShapesState(found)
      setSelected([])
      // Proposals are kept in a LabelMe file, and the format follows the file
      fileFormat.current = 'labelme'
      switchedFrom.current = null
      if (formatRef.current !== 'labelme') {
        formatRef.current = 'labelme'
        setSettings((s) => ({ ...s, format: 'labelme' }))
      }
      invalidate()
    }
    setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(found, true, opts.current.keypoints) } : i)))
  }), [invalidate, root, setImages, idle, reloadChanged])
  useEffect(() => {
    window.api.prelabelModels(root).then((list) => alive.current && setRunModels(list)).catch(() => alive.current && setRunModels([]))
  }, [root])
  // The folder's keypoints file, read as the folder opens and again as the window comes back, as
  // another app may change it; what's wrong with it is said when that changes
  const keypointsRef = useRef(keypoints)
  // Read once: main counted the folder with its file from the start, so the first read changes nothing it found
  const keypointsRead = useRef(false)
  const takeKeypoints = (k: FolderKeypoints) => {
    const was = keypointsRef.current
    keypointsRef.current = k
    if (JSON.stringify(was) !== JSON.stringify(k)) setKeypoints(k)
    // Other points are a pose's now, or no longer: main counts again, and the labels it found before may name keypoints
    if (keypointsRead.current && JSON.stringify(was.schema) !== JSON.stringify(k.schema)) forgetFound()
    keypointsRead.current = true
    if (k.problem && k.problem !== was.problem) onError(k.problem)
  }
  const readKeypoints = useRef(async () => {})
  readKeypoints.current = async () => {
    try {
      const k = await window.api.keypointsGet(root)
      if (alive.current) takeKeypoints(k)
    } catch (e) {
      if (alive.current) onError(`The folder's keypoints file can't be read: ${plainError(e)}`)
    }
  }
  useEffect(() => {
    readKeypoints.current()
  }, [root])
  /** Dataset > Keypoints: the folder's keypoints file made from COCO's person, a file chosen, or the folder's own poses */
  const changeKeypoints = async (how: 'coco' | 'choose' | 'make') => {
    // A pose being placed is placed with the keypoints it started with
    endPose()
    // Made from the label files, with the open image's changes saved first
    if (how === 'make' && !(await canLeave({ stay: true }))) return
    try {
      const k = how === 'coco' ? await window.api.keypointsSet(root, builtInKeypoints())
        : how === 'choose' ? await window.api.keypointsChoose(root)
          : await window.api.keypointsMake(root, formatRef.current)
      if (!k || !alive.current) return
      // This image's poses of a class the new file hasn't are no poses now: said, as it can't be seen at a glance
      const was = posesOf(shapesRef.current)
      takeKeypoints(k)
      const gone = [...new Set(was.map((p) => p.cls))].filter((cls) => !Object.hasOwn(k.schema.classes, cls)).map((cls) =>
        `The keypoints file has no ${cls} now: this image's ${counted(was.filter((p) => p.cls === cls).length, 'pose shows', 'poses show')} as plain points.`)
      setNotice([`Keypoints: ${Object.entries(k.schema.classes).map(([cls, names]) => `${cls}, ${counted(names.length, 'point')}`).join('; ')}.`, ...gone].join(' '))
    } catch (e) {
      onError(plainError(e))
    }
  }
  // The folder as another app, such as Seamline, left it: models it exported, images added or
  // removed, label files rewritten. One read at a time; the window coming back meanwhile reads it once more.
  const refreshFolder = useCallback(async () => {
    const state = refreshing.current
    if (state.running) {
      state.again = true
      return
    }
    state.running = true
    try {
      do {
        state.again = false
        window.api.prelabelModels(root).then((list) => alive.current && setRunModels(list)).catch(() => {})
        const { names, added } = await window.api.refreshProject(root)
        if (!alive.current) return
        const { images: list, index: at } = listRef.current
        const showing = list[at]?.name
        const present = new Set(names)
        let next = names
        // The image on screen stays, even when another app removed it, until the window moves on: it's still shown, and saved
        if (showing !== undefined && !present.has(showing)) {
          const before = list.slice(0, at).map((i) => i.name).filter((n) => present.has(n)).pop()
          const k = before === undefined ? 0 : names.indexOf(before) + 1
          next = [...names.slice(0, k), showing, ...names.slice(k)]
        }
        if (next.length !== list.length || next.some((n, i) => list[i].name !== n)) {
          const fresh = new Map(added.map((i) => [i.name, i]))
          setImages((all) => {
            const byName = new Map(all.map((i) => [i.name, i]))
            return next.map((n) => byName.get(n) ?? fresh.get(n) ?? { ...EMPTY_SUMMARY, name: n })
          })
          setIndex(showing === undefined ? 0 : next.indexOf(showing))
        }
        // Counts follow for the label files that changed
        await window.api.summarize(root)
      } while (state.again && alive.current)
    } catch {
      // Tried again the next time the window comes back
    } finally {
      state.running = false
    }
  }, [root, setImages])
  useEffect(() => window.api.onFocus(async () => {
    // The image on screen first: its few label files are read at once, where the whole folder can take seconds
    const name = loadedName.current
    if (name && (await window.api.labelsChanged(root, name).catch(() => false)) && alive.current && loadedName.current === name) {
      if (idle()) reloadChanged(name)
      // Saving by hand, changes can wait a long time: told now, rather than when saving drops them
      else if (dirty.current && !settingsRef.current.autoSave) {
        onError('These labels changed in another app (such as Seamline). Saving will reload them, and the changes made here since will be lost: copy any shapes you want to keep first.', `${root}\n${name}`)
      }
    }
    if (alive.current) {
      refreshFolder()
      readKeypoints.current()
    }
  }), [root, idle, reloadChanged, onError, refreshFolder])
  useEffect(() => window.api.onAiDownload((p) => setDownload(p.done >= p.total ? null : p)), [])
  useEffect(() => window.api.onCheck(({ root: at, ...p }) => {
    if (at === root && checkingRef.current) setChecking((c) => c && { ...c, progress: p })
  }), [root])
  useEffect(() => () => clearTimeout(recheckTimer.current), [])
  // Fit to Selection follows the selection, as labelme 7 does: once the mouse button is up and a
  // double-click's time has passed, since a press that selects a shape may become a drag of it, or
  // the first click of a double-click, each measured in the view it started in
  useEffect(() => {
    if (settings.fit !== 'selection') return
    if (drag.current || fitTimer.current) pendingFit.current = true
    else fitSelection()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, settings.fit])
  useEffect(() => () => clearTimeout(fitTimer.current), [])
  useEffect(() => {
    window.api.aiModels().then(setModels).catch(() => setModels(null))
  }, [])
  // A model kept from before, or named in a config, that this version doesn't offer: SAM2 instead
  useEffect(() => {
    if (models && !models.some((m) => m.name === settings.aiModel && (m.points || m.box))) setSettings((s) => ({ ...s, aiModel: 'sam2:latest' }))
  }, [models, settings.aiModel, setSettings])
  // The pre-labelling model chosen for this folder, or in a folder with no runs, the one last picked as a file
  const chosenModel = settings.prelabelModels[root] || (runModels && !runModels.length ? settings.prelabelModel : '')
  useEffect(() => {
    if (!chosenModel) return setPrelabelModel(null)
    let stale = false
    window.api.prelabelInfo(chosenModel).then((m) => !stale && setPrelabelModel(m)).catch(() => !stale && setPrelabelModel(null))
    return () => { stale = true }
  }, [chosenModel])
  useEffect(() => {
    const broken = images.filter((i) => i.broken)
    if (!broken.length) return
    const list = broken.slice(0, 3).map((i) => i.name).join(', ') + (broken.length > 3 ? ', …' : '')
    onError(`${broken.length} ${broken.length === 1 ? 'label file' : 'label files'} can't be read and won't be changed: ${list}`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images.filter((i) => i.broken).length])

  const toImage = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const { scale, x, y } = view.current
    return [(e.clientX - rect.left - x) / scale, (e.clientY - rect.top - y) / scale]
  }
  const toCanvas = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }
  const clampPoint = (p: Pt) => {
    if (free) return [p[0], p[1]]
    const m = metaRef.current!
    return [Math.max(0, Math.min(m.width, p[0])), Math.max(0, Math.min(m.height, p[1]))]
  }
  const onImage = (p: Pt) => {
    if (free) return true
    const m = metaRef.current!
    return p[0] >= 0 && p[1] >= 0 && p[0] <= m.width && p[1] <= m.height
  }
  const tolerance = () => config.epsilon / view.current.scale

  const onWheel = (e: React.WheelEvent) => {
    const at = toCanvas(e)
    const zoomWheel = config.canvas.wheel !== 'scroll' || e.ctrlKey || e.metaKey
    if (zoomWheel) {
      zoomBy(Math.exp(-e.deltaY * 0.0022), at)
      return
    }
    // labelme: the wheel scrolls, Shift scrolls sideways
    const v = view.current
    view.current = e.shiftKey ? { ...v, x: v.x - e.deltaY } : { ...v, x: v.x - e.deltaX, y: v.y - e.deltaY }
    invalidate()
  }

  /** labelme asks for a label when no class is picked, or always with display_label_popup */
  const askLabel = (request: LabelRequest) =>
    new Promise<LabelChoice | null>((resolve) => setDialog({ ...request, resolve }))
  /** New shapes from a tool: labelled with the class picked, or as the label dialog says; `retry` when it's cancelled */
  const addShapes = async (made: Shape[], retry?: () => void) => {
    if (!metaRef.current || !made.length) return
    const picked = currentLabel()
    let label = picked
    let flags = picked ? flagsFor(picked, config.label_flags) : {}
    let group_id: number | null = null
    let description = ''
    if (settingsRef.current.askLabel || !picked) {
      const choice = await askLabel({ text: picked ?? '', flags: null, group_id: null, description: null, at: screenMouse.current })
      if (!choice) {
        // labelme goes back to drawing, so a slip on the last click isn't the end of the shape
        retry?.()
        invalidate()
        return
      }
      label = choice.text
      flags = choice.flags
      group_id = choice.group_id
      description = choice.description
    }
    if (!label) return
    // An AI tool's polygons of one object share a group ID, unless the dialog gave one
    const list = forFormat(made.map((s) => ({ ...s, label: label!, flags: { ...flags }, group_id: group_id ?? s.group_id, description: description || s.description })))
    const next = [...shapesRef.current, ...list]
    setShapes(next)
    setSelected(list.map((_, k) => next.length - list.length + k))
    if (classes.some((c) => c.name === label)) setActive(label)
    // LabelImg's beginner mode: one shape, then back to editing
    if (!settingsRef.current.advanced) setMode('edit')
  }
  const finishShape = (type: ShapeType, points: Pt[], extra: Partial<Shape> = {}) =>
    addShapes([newShape('', type, points, extra)], type === 'polygon' || type === 'linestrip'
      ? () => { draft.current = { points, labels: [], preview: [], matched: [] } }
      : undefined)
  const finishDraft = () => {
    const dr = draft.current
    const md = opts.current.mode
    if (!dr) return
    if (md === 'ai_points') return finishAi()
    draft.current = null
    if (md === 'polygon' && dr.points.length >= 3) finishShape('polygon', dr.points)
    else if (md === 'linestrip' && dr.points.length >= 2) finishShape('linestrip', dr.points)
    invalidate()
  }
  /** A box made with a two-corner tool: the shape, or what AI box, AI pose or the Pose tool make of it */
  const boxDone = (md: Mode, a: Pt, b: Pt) => {
    if (md === 'ai_box') runBox(a, b)
    else if (md === 'ai_pose') runPose(a, b)
    else if (md === 'pose') startPose(a, b)
    else finishShape(md as ShapeType, [a, b])
  }

  // The Pose tool: a box, dragged or one the image has, then each keypoint of its class in turn
  /** The pose being placed, with what it asks for now in the status bar */
  const setPosing = (s: PoseSession | null) => {
    posing.current = s
    const k = s ? askedNow(s) : null
    const names = s ? opts.current.keypoints.classes[s.cls] ?? [] : []
    setPoseStep(s && k !== null ? `${k + 1}/${names.length} ${spoken(names[k] ?? '')}` : '')
    invalidate()
  }
  /** The class a pose drawn now takes: the class picked, if it has keypoints, else the keypoints file's first */
  const poseClass = () => {
    const schema = opts.current.keypoints
    const picked = currentLabel()
    return picked !== undefined && Object.hasOwn(schema.classes, picked) ? picked : Object.keys(schema.classes)[0] ?? null
  }
  /** With validate_label: exact, a pose's box takes only a listed label, as any shape does */
  const listedLabel = (label: string) => {
    if (config.validate_label !== 'exact' || (config.labels ?? []).includes(label)) return true
    onError(`“${label}” isn't in the label list, and only listed labels are allowed.`)
    return false
  }
  const startPose = (a: Pt, b: Pt) => {
    const cls = poseClass()
    if (!cls) return setNotice('The keypoints file names no class to pose.')
    if (!listedLabel(cls)) return
    const box = newShape(cls, 'rectangle', [[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], {
      group_id: nextGroupId(shapesRef.current), flags: flagsFor(cls, config.label_flags)
    })
    setSelected([])
    setPosing({ cls, group: box.group_id!, box, fresh: true, asked: opts.current.keypoints.classes[cls].map((_, k) => k), done: [] })
  }
  /** The topmost of the image's shapes under a point that could be a pose's box: closed, of a class with keypoints; -1 when there's none */
  const poseBoxAt = (p: Pt) => {
    const list = shapesRef.current
    const schema = opts.current.keypoints
    const tol = tolerance()
    for (let i = list.length - 1; i >= 0; i--) {
      const s = list[i]
      if (!s.hidden && CLOSED.has(s.shape_type) && Object.hasOwn(schema.classes, s.label) && contains(s, p, tol)) return i
    }
    return -1
  }
  /** Carries on the pose of a box the image has: its keypoints not placed yet are asked for */
  const continuePose = (i: number) => {
    const list = shapesRef.current
    const box = list[i]
    const names = opts.current.keypoints.classes[box.label] ?? []
    const pose = box.group_id === null ? undefined : posesOf(list).find((p) => p.group === box.group_id)
    const missing = names.map((_, k) => k).filter((k) => !pose || pose.points[k] === null)
    if (!missing.length) {
      setSelected(pose ? poseShapes(pose) : [i])
      return setNotice('This pose has every keypoint.')
    }
    setSelected([])
    setPosing({ cls: box.label, group: box.group_id ?? nextGroupId(list), box, fresh: false, asked: missing, done: [] })
  }
  /** The next click places one keypoint a pose is missing */
  const placeMissing = (pose: PoseGroup, k: number) => {
    const list = shapesRef.current
    setPosing({ cls: pose.cls, group: pose.group, box: pose.box === null ? null : list[pose.box], fresh: false, asked: [k], done: [] })
  }
  /** The keypoint asked for, seen or hidden, or null to skip it */
  const answerKeypoint = (point: Shape | null) => {
    const s = posing.current
    if (!s) return
    const next = { ...s, done: [...s.done, point] }
    if (askedNow(next) === null) endPose(next)
    else setPosing(next)
  }
  const placeKeypoint = (at: Pt, hidden: boolean) => {
    const s = posing.current
    const k = s ? askedNow(s) : null
    const name = s && k !== null ? opts.current.keypoints.classes[s.cls]?.[k] : undefined
    if (!s || name === undefined) return
    answerKeypoint(newShape(name, 'point', [[at[0], at[1]]], {
      group_id: s.group, flags: { ...flagsFor(name, config.label_flags), ...(hidden ? { difficult: true } : {}) }
    }))
  }
  /** Back one keypoint; before the first, a box drawn for the pose goes too */
  const backKeypoint = () => {
    const s = posing.current
    if (!s) return false
    setPosing(s.done.length ? { ...s, done: s.done.slice(0, -1) } : null)
    return true
  }
  /** Escape: the pose ends, keeping what's placed; a box drawn for it with no keypoint placed goes, as Backspace takes it */
  const stopPose = () => {
    const s = posing.current
    if (s?.fresh && !s.done.some(Boolean)) setPosing(null)
    else endPose()
  }
  /** Ends the pose being placed, keeping what's placed, as one step to undo, and selects it */
  const endPose = (s = posing.current) => {
    setPosing(null)
    if (!s) return
    const list = shapesRef.current
    const next = withSession(list, s)
    if (next === list) return
    // Points added by hand to an AI pose are an edit of it, which accepts it, so a rejection can't leave them alone
    setShapes(s.fresh ? next : acceptEdited(next, [next.length - 1], next))
    const pose = posesOf(next).find((p) => p.group === s.group)
    // One keypoint placed for N stays selected, so N goes on from it
    setSelected(!pose || (!s.fresh && s.asked.length === 1) ? [next.length - 1] : poseShapes(pose))
    const layout = pose && layoutOf(next, pose)
    if (layout) lastLayout.current = layout
    // LabelImg's beginner mode: one shape, then back to editing
    if (s.fresh && !settingsRef.current.advanced) setMode('edit')
  }
  endPoseRef.current = () => endPose()

  // labelme 7's AI tools: AI points (each click refines the object), AI box, and the text prompt
  const toShape = (p: ProposedShape) =>
    newShape(p.label ?? '', p.shape_type, p.points, { mask: p.mask ?? null, description: p.description ?? '', group_id: p.group_id ?? null })
  const drawnShapes = (): DrawnShape[] => shapesRef.current.map((s) => ({ shape_type: s.shape_type, points: s.points, label: s.label, mask: s.mask }))
  const groupsInUse = () => shapesRef.current.flatMap((s) => (s.group_id === null ? [] : [s.group_id]))
  // LabelImg's formats keep boxes, so AI tools make boxes there
  const aiOutput = () => (holdsBoxesOnly(formatRef.current) ? 'rectangle' : settingsRef.current.aiOutput)
  const megabytes = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`)

  /** A model that isn't downloaded yet is downloaded first, when the user agrees */
  const ensureModel = async (name: string) => {
    const info = models?.find((x) => x.name === name)
    if (!info) return false
    if (info.downloaded) return true
    const ok = await window.api.ask(`Download ${info.title}?`,
      `It's ${megabytes(info.size)} under the ${info.license.name}, downloaded once and kept in ~/.cache/osam, which labelme uses too.${info.note ? ` ${info.note}` : ''}`,
      ['Download', 'Cancel'])
    if (ok !== 0) return false
    try {
      setBusyAi('Downloading the model…')
      await window.api.aiDownload(name)
      return true
    } catch (e) {
      if (!/cancelled/.test(plainError(e))) onError(`The model couldn't be downloaded: ${plainError(e)}`)
      return false
    } finally {
      // Cancelled or failed, a download sends no last progress of its own
      setDownload(null)
      setBusyAi('')
      setModels(await window.api.aiModels())
    }
  }
  /** The click and box tools' model, if it takes what the tool gives it */
  const assistModel = (what: 'points' | 'box') => {
    const model = settingsRef.current.aiModel
    const info = models?.find((x) => x.name === model)
    if (info && !info[what]) {
      setNotice(`${info.title} takes ${info.points ? 'clicks' : 'boxes'} only: pick another model, or use AI ${info.points ? 'points' : 'box'}.`)
      return null
    }
    return model
  }
  const assistRequest = (model: string, points: Pt[], labels: number[]) => ({
    root, name: loadedName.current, model, points, labels, output: aiOutput(), detail: config.mask_polygonization.detail,
    drawn: config.ai.suppress_existing_shape_matches ? drawnShapes() : null, groups: groupsInUse(), outside: free
  })
  const runAi = async () => {
    const dr = draft.current
    const name = loadedName.current
    if (!dr || !metaRef.current || !name) return
    const model = assistModel('points')
    if (!model || !(await ensureModel(model))) {
      draft.current = null
      invalidate()
      return
    }
    const seq = ++aiSeq.current
    setBusyAi('Segmenting…')
    try {
      const before = shapesRef.current
      const result = await window.api.aiAssist(assistRequest(model, dr.points, dr.labels))
      if (seq !== aiSeq.current || draft.current !== dr || loadedName.current !== name) return
      dr.preview = result.shapes.map(toShape)
      dr.matched = result.matched.map((i) => before[i]).filter(Boolean)
      if (!result.shapes.length) setNotice(result.matched.length ? 'That object is drawn already.' : 'The model found nothing there. Try another point.')
      invalidate()
    } catch (e) {
      if (seq === aiSeq.current) onError(`AI segmentation failed: ${plainError(e)}`)
    } finally {
      if (seq === aiSeq.current) setBusyAi('')
    }
  }
  const finishAi = async () => {
    const dr = draft.current
    if (!dr) return
    // Wait for the last click's result
    while (busyAiRef.current && draft.current === dr) await new Promise((r) => setTimeout(r, 30))
    if (draft.current !== dr) return
    draft.current = null
    // Nothing new: the shapes it found again are picked out instead
    if (!dr.preview.length) setSelected(dr.matched.map((m) => shapesRef.current.indexOf(m)).filter((i) => i >= 0))
    else await addShapes(dr.preview)
    invalidate()
  }
  const runBox = async (a: Pt, b: Pt) => {
    const name = loadedName.current
    const model = assistModel('box')
    if (!model || !name) return
    // Escape or another tool while the model downloads drops the box
    const seq = ++aiSeq.current
    if (!(await ensureModel(model)) || seq !== aiSeq.current) return
    const box = [[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]]
    setBusyAi('Segmenting…')
    try {
      const before = shapesRef.current
      const result = await window.api.aiAssist(assistRequest(model, box, [2, 3]))
      if (seq !== aiSeq.current || loadedName.current !== name) return
      setBusyAi('')
      if (result.shapes.length) await addShapes(result.shapes.map(toShape))
      else if (result.matched.length) {
        setSelected(result.matched.map((i) => shapesRef.current.indexOf(before[i])).filter((i) => i >= 0))
        setNotice('That object is drawn already.')
      } else setNotice('The model found nothing in that box.')
    } catch (e) {
      if (seq === aiSeq.current) onError(`AI segmentation failed: ${plainError(e)}`)
    } finally {
      if (seq === aiSeq.current) setBusyAi('')
    }
  }
  /** labelme's AI Text Prompt: every object the words name, labelled with the word */
  const runText = async () => {
    const name = loadedName.current
    if (!name || readOnlyRef.current) return
    const texts = aiPrompt.split(',').map((t) => t.trim()).filter(Boolean)
    if (!texts.length) return setNotice('Type what to find, such as dog,cat,bird.')
    const model = settingsRef.current.aiTextModel
    const seq = ++aiSeq.current
    if (!(await ensureModel(model)) || seq !== aiSeq.current) return
    setBusyAi('Finding…')
    try {
      const result = await window.api.aiText({
        root, name, model, texts, output: aiOutput(), detail: config.mask_polygonization.detail, drawn: drawnShapes(), groups: groupsInUse(),
        outside: free, iou: aiIou, score: aiScore
      })
      if (seq !== aiSeq.current || loadedName.current !== name) return
      if (result.notice) return setNotice(result.notice)
      if (!result.shapes.length) return setNotice(`Nothing new found for ${texts.join(', ')}.`)
      const found = forFormat(result.shapes.map((p) => ({ ...toShape(p), flags: flagsFor(p.label ?? '', config.label_flags) })))
      const next = [...shapesRef.current, ...found]
      setShapes(next)
      setSelected(found.map((_, k) => next.length - found.length + k))
      setNotice(`Found ${found.length} ${found.length === 1 ? 'shape' : 'shapes'}.`)
    } catch (e) {
      if (seq === aiSeq.current) onError(`The text prompt failed: ${plainError(e)}`)
    } finally {
      if (seq === aiSeq.current) setBusyAi('')
    }
  }

  // AI pose: a person's keypoints in a box, from a pose model such as RTMPose
  const poseModels = (models ?? []).filter((m) => m.pose)
  // The model picked, else the config's, else the first there is
  const poseModel = (poseModels.find((m) => m.name === settings.poseModel) ?? poseModels.find((m) => m.name === config.ai.pose_model) ?? poseModels[0])?.name ?? ''
  /** AI pose's model, and the class of the keypoints file its keypoints go to; null, said why, when there's none */
  const poseTarget = () => {
    const info = poseModels.find((m) => m.name === poseModel)
    if (!info) {
      setNotice('There is no AI pose model.')
      return null
    }
    const cls = info.keypoints?.length ? classForModel(opts.current.keypoints, info.keypoints) : poseClass()
    if (!cls) {
      setNotice(`No class in the keypoints file has ${info.title}'s keypoints.`)
      return null
    }
    return { model: info.name, cls }
  }
  /**
   * AI pose on people: the image's own, whose shapes their points join, and new boxes, which come
   * with theirs as AI shapes; each pose under one group_id, all of them one step to undo. A pose
   * with fewer than MIN_POSE_POINTS keypoints kept is left out.
   */
  const poseInBoxes = async (target: { model: string; cls: string }, own: (PoseTarget & { members: Shape[] })[], found: Shape[], seq: number, name: string) => {
    const boxes = [...own.map((o) => o.box), ...found.map((s) => boxOf(s))]
    setBusyAi('Finding poses…')
    try {
      const answer = await window.api.aiPose({ root, name, model: target.model, boxes })
      const m = metaRef.current
      if (seq !== aiSeq.current || loadedName.current !== name || !m) return
      const before = shapesRef.current
      let next = before
      const added: Shape[] = []
      const chosen: number[] = []
      // New groups follow the image's, and a pose being placed meanwhile, whose group_id its shapes don't have yet
      let group = nextGroupId([...before, { group_id: posing.current?.group ?? null }])
      let made = 0, few = 0
      boxes.forEach((_, k) => {
        const mine = own[k]
        // A person whose shapes were taken away meanwhile gets nothing
        const at = mine ? mine.members.map((s) => next.indexOf(s)) : []
        if (at.some((i) => i < 0)) return
        const g = mine?.group ?? group
        const { points, score } = aiPosePoints(opts.current.keypoints, target.cls, answer.names, answer.poses[k] ?? [], {
          size: [m.width, m.height], threshold: poseScore, group: g, flagsOf: (label) => flagsFor(label, config.label_flags)
        })
        if (points.length < MIN_POSE_POINTS) return void few++
        made++
        if (g === group) group++
        if (mine) {
          // A shape in no group takes the pose's
          if (mine.group === null) {
            if (next === before) next = [...before]
            next[at[0]] = { ...next[at[0]], group_id: g }
          }
          chosen.push(...at)
        } else {
          const box = found[k - own.length]
          added.push({ ...box, group_id: g, flags: { ...box.flags, ai: true }, other: { ...box.other, score } })
        }
        added.push(...points)
      })
      const short = `fewer than ${MIN_POSE_POINTS} keypoints scored ${percent(poseScore)} or more`
      if (!made) {
        if (few) setNotice(few === 1 ? `The model found ${short} in that box, so it made no pose.` : `The model found ${short} in each of the ${few} boxes, so it made no pose.`)
        return
      }
      setShapes([...next, ...added])
      setSelected([...chosen, ...added.map((_, k) => next.length + k)])
      setNotice(`Found ${counted(made, 'pose')}.${few ? ` ${counted(few, 'box', 'boxes')} had ${short}, so ${few === 1 ? 'it was' : 'they were'} left out.` : ''}`)
    } catch (e) {
      if (seq === aiSeq.current) onError(`AI pose failed: ${plainError(e)}`)
    } finally {
      if (seq === aiSeq.current) setBusyAi('')
    }
  }
  /** AI pose in a box dragged: the box and its keypoints, as AI shapes */
  const runPose = async (a: Pt, b: Pt) => {
    const name = loadedName.current
    const target = poseTarget()
    if (!name || !target || !listedLabel(target.cls)) return
    // Escape or another tool while the model downloads drops the box
    const seq = ++aiSeq.current
    if (!(await ensureModel(target.model)) || seq !== aiSeq.current) return
    const box = newShape(target.cls, 'rectangle', [[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], {
      flags: flagsFor(target.cls, config.label_flags)
    })
    await poseInBoxes(target, [], [box], seq, name)
  }
  /**
   * AI pose on the image's people of the pose's class that have no keypoints yet: those with the
   * shapes given, by index, or else every one. A group's shapes of the class are one person, posed
   * in the box around them. With no shape of the class at all, YOLO-World finds the people first.
   */
  const poseBoxes = async (which?: number[]) => {
    // A pose being placed is kept first, with its group_id
    endPose()
    const name = loadedName.current
    const target = poseTarget()
    if (!name || readOnlyRef.current || !target) return
    const list = shapesRef.current
    const own = unposed(list, posesOf(list), target.cls, which).map((t) => ({ ...t, members: t.shapes.map((i) => list[i]) }))
    if (which && !own.length) return setNotice(`Click a ${target.cls} box that has no keypoints yet, or drag a box.`)
    // The people drawn are the people here: YOLO-World would find them again, or the ones they were left without on purpose
    if (!own.length && list.some((s) => s.label === target.cls && CLOSED.has(s.shape_type))) return setNotice(`Every ${target.cls} here has keypoints already.`)
    if (!own.length && !listedLabel(target.cls)) return
    const seq = ++aiSeq.current
    if (!(await ensureModel(target.model)) || seq !== aiSeq.current) return
    let found: Shape[] = []
    if (!own.length) {
      const textModel = settingsRef.current.aiTextModel
      if (!(await ensureModel(textModel)) || seq !== aiSeq.current) return
      setBusyAi('Finding people…')
      try {
        const result = await window.api.aiText({
          root, name, model: textModel, texts: ['person'], output: 'rectangle', detail: config.mask_polygonization.detail, drawn: drawnShapes(),
          groups: groupsInUse(), outside: free, iou: aiIou, score: Math.max(aiScore, PERSON_SCORE)
        })
        if (seq !== aiSeq.current || loadedName.current !== name) return
        if (result.notice) return setNotice(result.notice)
        found = result.shapes.map((p) => newShape(target.cls, 'rectangle', p.points, { description: p.description ?? '', flags: flagsFor(target.cls, config.label_flags) }))
      } catch (e) {
        if (seq === aiSeq.current) onError(`Finding people failed: ${plainError(e)}`)
        return
      } finally {
        if (seq === aiSeq.current) setBusyAi('')
      }
      if (!found.length) return setNotice('YOLO-World found nobody here.')
    }
    await poseInBoxes(target, own, found, seq, name)
  }
  const busyAiRef = useRef('')
  busyAiRef.current = busyAi

  const hitShape = (p: Pt) => {
    const list = shapesRef.current
    const tol = tolerance()
    for (let i = list.length - 1; i >= 0; i--) {
      if (!list[i].hidden && contains(list[i], p, tol)) return i
    }
    return -1
  }
  /** What a click on a shape selects: a pose's box selects the whole pose */
  const pickOf = (i: number) => {
    const pose = posesOf(shapesRef.current).find((p) => p.box === i)
    return pose ? poseShapes(pose) : [i]
  }
  const findHover = (p: Pt): Hover => {
    const list = shapesRef.current
    const tol = tolerance()
    // A selected pose's keypoints first, the nearest: its region's vertices and edges, many along a
    // person's outline, would otherwise take a drag meant for a shoulder
    let nearest = -1, best = tol
    for (const pose of posesAmong(selectedRef.current)) {
      for (const i of poseShapes(pose)) {
        const s = list[i]
        if (i === pose.box || s.hidden || s.locked) continue
        const d = Math.hypot(s.points[0][0] - p[0], s.points[0][1] - p[1])
        if (d <= best) { nearest = i; best = d }
      }
    }
    if (nearest >= 0) return { shape: nearest, vertex: 0, edge: -1 }
    // Vertices of the selection first, then of whatever is on top, as labelme does
    const order = [...selectedRef.current, ...list.map((_, i) => list.length - 1 - i).filter((i) => !selectedRef.current.includes(i))]
    for (const i of order) {
      const s = list[i]
      // A lock keeps a shape's geometry
      if (!s || s.hidden || s.locked || s.shape_type === 'mask') continue
      const v = nearestVertex(s, p, tol)
      if (v >= 0) return { shape: i, vertex: v, edge: -1 }
      // A selected rotated box turns by the handles at the middle of its edges
      if (s.shape_type === 'oriented_rectangle' && s.points.length === 4 && selectedRef.current.includes(i)) {
        const r = rotationHandles(s.points).findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) <= tol)
        if (r >= 0) return { shape: i, vertex: -1, edge: -1, rotate: r }
      }
    }
    for (const i of order) {
      const s = list[i]
      if (!s || s.hidden || s.locked) continue
      const e = nearestEdge(s, p, tol)
      if (e >= 0) return { shape: i, vertex: -1, edge: e }
    }
    const inside = hitShape(p)
    return inside >= 0 ? { shape: inside, vertex: -1, edge: -1 } : null
  }

  const onPointerDown = (e: React.PointerEvent) => {
    const m = metaRef.current
    if (!m || loadedName.current !== images[index]?.name) return
    canvasRef.current!.setPointerCapture(e.pointerId)
    canvasRef.current!.focus()
    const p = toImage(e)
    const press = [e.clientX, e.clientY]
    if (e.button === 2 && mode === 'edit' && !space.current && !readOnlyRef.current) {
      // LabelImg's right drag: a copy follows the pointer, then Copy here or Move here. Elsewhere it pans.
      const hit = hitShape(p)
      if (hit >= 0) {
        const sel = withPoints(selectedRef.current.includes(hit) ? selectedRef.current : pickOf(hit), posesOf(shapesRef.current))
        if (!selectedRef.current.includes(hit)) setSelected(pickOf(hit))
        drag.current = { kind: 'ghost', shapes: sel, x: p[0], y: p[1], press, active: false }
        return
      }
    }
    if (e.button === 1 || e.button === 2 || space.current) {
      if (space.current) spaceTap.current = null
      drag.current = { kind: 'pan', x: e.clientX, y: e.clientY, press, button: e.button, moved: false }
      return
    }
    if (e.button !== 0) return
    const md = mode
    const ro = !!readOnlyRef.current
    // A pose being placed: a click places the keypoint asked for, Shift+click a hidden one
    if (!ro && posing.current) {
      if (onImage(p)) placeKeypoint(p, e.shiftKey)
      return
    }
    if (!ro && (md === 'polygon' || md === 'linestrip')) {
      const dr = draft.current
      if (dr) {
        const q = livePoint()!
        const first = dr.points[0]
        const closeTo = config.epsilon / view.current.scale
        if (md === 'polygon' && dr.points.length >= 3 && Math.hypot(q[0] - first[0], q[1] - first[1]) < closeTo) return finishDraft()
        dr.points.push(q)
        if (md === 'linestrip' && (e.ctrlKey || e.metaKey)) return finishDraft()
        invalidate()
      } else if (onImage(p)) {
        draft.current = { points: [p], labels: [], preview: [], matched: [] }
        invalidate()
      }
      return
    }
    if (!ro && (md === 'rectangle' || md === 'circle' || md === 'line' || md === 'ai_box' || md === 'pose' || md === 'ai_pose')) {
      const dr = draft.current
      if (dr) {
        const q = livePoint()!
        draft.current = null
        const second = md === 'rectangle' && squareKeys(e) ? squareFrom(dr.points[0], q) : q
        if (Math.hypot(second[0] - dr.points[0][0], second[1] - dr.points[0][1]) > 0.5) boxDone(md, dr.points[0], second)
        invalidate()
      } else if (onImage(p)) {
        draft.current = { points: [p], labels: [], preview: [], matched: [] }
        drag.current = { kind: 'two', press }
        invalidate()
      }
      return
    }
    if (!ro && md === 'point') {
      if (onImage(p)) finishShape('point', [p])
      return
    }
    if (!ro && md === 'oriented_rectangle') {
      // labelme 7's rotated box: a corner, the next corner along one side, then the width
      const dr = draft.current
      if (!dr) {
        if (onImage(p)) draft.current = { points: [p], labels: [], preview: [], matched: [] }
      } else if (dr.points.length === 1) {
        const q = livePoint()!
        if (Math.hypot(q[0] - dr.points[0][0], q[1] - dr.points[0][1]) > 0.5) dr.points = [dr.points[0], q, [q[0], q[1]], [dr.points[0][0], dr.points[0][1]]]
      } else {
        const corners = reprojectCorners(dr.points, 2, livePoint()!, imageSize())
        draft.current = null
        const area = Math.abs(corners.reduce((a, q, k) => a + q[0] * corners[(k + 1) % 4][1] - corners[(k + 1) % 4][0] * q[1], 0)) / 2
        if (area > 0.5) finishShape('oriented_rectangle', corners)
      }
      invalidate()
      return
    }
    if (!ro && md === 'ai_points') {
      if (!onImage(p)) return
      const dr = draft.current ?? (draft.current = { points: [], labels: [], preview: [], matched: [] })
      dr.points.push(p)
      dr.labels.push(e.shiftKey ? 0 : 1)
      invalidate()
      runAi().then(() => {
        if (e.ctrlKey || e.metaKey) finishAi()
      })
      return
    }
    if (!ro && (md === 'brush' || md === 'eraser')) {
      drag.current = { kind: 'stroke', points: [p], radius: brushSize / 2 / view.current.scale }
      invalidate()
      return
    }
    // Edit
    const list = shapesRef.current
    const h = findHover(p)
    const multi = e.ctrlKey || e.metaKey
    if (h && !ro) {
      const s = list[h.shape]
      if (h.rotate !== undefined) {
        const center = boxCenter(s.points)
        const handle = rotationHandles(s.points)[h.rotate]
        drag.current = { kind: 'rotate', shape: h.shape, center, angle: Math.atan2(handle[1] - center[1], handle[0] - center[0]), press, active: false, start: list }
        return
      }
      if (h.vertex >= 0 && e.altKey) {
        // Alt+click, or labelme's Alt+Shift+click, takes a point away
        if (canRemovePoint(s)) setShapes(acceptEdited(list.map((q, i) => (i === h.shape ? { ...accepted(q), points: q.points.filter((_, k) => k !== h.vertex) } : q)), [h.shape], list))
        hover.current = null
        return
      }
      if (h.vertex < 0 && h.edge >= 0 && e.altKey && EDITABLE_POINTS.has(s.shape_type)) {
        // labelme's Alt+click on an edge adds a point there, and it can be dragged at once
        const point = clampPoint(p)
        const start = list
        const next = list.map((q, i) => (i === h.shape ? insertPoint(q, h.edge, point) : q))
        // Recorded for undo when the button comes up, like any other drag
        shapesRef.current = next
        setShapesState(next)
        setSelected([h.shape])
        drag.current = { kind: 'vertex', shape: h.shape, vertex: edges(s)[h.edge][0] + 1, press, active: true, start }
        invalidate()
        return
      }
      if (h.vertex >= 0) {
        if (!selectedRef.current.includes(h.shape)) setSelected([h.shape])
        drag.current = { kind: 'vertex', shape: h.shape, vertex: h.vertex, press, active: false, start: list }
        return
      }
    }
    const hit = hitShape(p)
    if (hit >= 0) {
      const sel = selectedRef.current
      const group = pickOf(hit)
      if (multi) {
        setSelected(group.every((i) => sel.includes(i)) ? sel.filter((i) => !group.includes(i)) : [...sel, ...group.filter((i) => !sel.includes(i))])
        return
      }
      const moving = withPoints(sel.includes(hit) ? sel : group, posesOf(list)).filter((i) => !list[i].locked)
      if (!sel.includes(hit)) setSelected(group)
      if (!ro && moving.length) drag.current = { kind: 'move', shapes: moving, x: p[0], y: p[1], press, active: false, start: list }
      return
    }
    if (!multi) setSelected([])
    drag.current = { kind: 'pan', x: e.clientX, y: e.clientY, press, button: 0, moved: false }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e)
    mouse.current = p
    screenMouse.current = { x: e.clientX, y: e.clientY }
    shift.current = e.shiftKey
    ctrl.current = e.ctrlKey || e.metaKey
    setCursor(p)
    overCanvas.current = true
    const d = drag.current
    if (!d) {
      if (mode === 'edit' && metaRef.current) {
        const h = findHover(p)
        const before = hover.current
        if (h?.shape !== before?.shape || h?.vertex !== before?.vertex || h?.edge !== before?.edge) {
          hover.current = h
          const c = canvasRef.current!
          c.style.cursor = h?.rotate !== undefined ? 'grab' : h?.vertex !== undefined && h.vertex >= 0 ? 'move' : h?.edge !== undefined && h.edge >= 0 && e.altKey ? 'copy' : h ? 'pointer' : ''
          invalidate()
        }
      }
      if (draft.current || posing.current || mode === 'brush' || mode === 'eraser' || (config.canvas.crosshair as Record<string, boolean>)[CROSSHAIR[mode] ?? mode]) invalidate()
      return
    }
    if ((d.kind === 'vertex' || d.kind === 'move' || d.kind === 'pan' || d.kind === 'ghost' || d.kind === 'rotate') && !('active' in d ? d.active : d.moved)) {
      // A slip of the hand during a click isn't a drag
      if (Math.hypot(e.clientX - d.press[0], e.clientY - d.press[1]) < DRAG_START) return
      if (d.kind === 'pan') d.moved = true
      else d.active = true
    }
    const m = metaRef.current
    if (d.kind === 'pan') {
      view.current.x += e.clientX - d.x
      view.current.y += e.clientY - d.y
      d.x = e.clientX
      d.y = e.clientY
      if (settingsRef.current.fit !== 'manual') setFit('manual')
    } else if (d.kind === 'stroke') {
      const last = d.points[d.points.length - 1]
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= d.radius * 0.4) d.points.push(p)
    } else if (d.kind === 'rotate') {
      const s = d.start[d.shape]
      const turn = Math.atan2(p[1] - d.center[1], p[0] - d.center[0]) - d.angle
      shapesRef.current = shapesRef.current.map((x, i) => (i === d.shape ? { ...x, points: rotatePoints(s.points, d.center, turn) } : x))
    } else if (d.kind === 'vertex' && m) {
      const s = d.start[d.shape]
      let q = clampPoint(p)
      if (s.shape_type === 'oriented_rectangle' && s.points.length === 4) {
        shapesRef.current = shapesRef.current.map((x, i) => (i === d.shape ? { ...x, points: reprojectCorners(s.points, d.vertex, p, imageSize()) } : x))
      } else if (s.shape_type === 'rectangle' && s.points.length === 2) {
        // Any of the four corners moves, the one across stays; square as when drawing it
        if (squareKeys(e)) q = squareFrom(vertices(s)[(d.vertex + 2) % 4], q)
        shapesRef.current = shapesRef.current.map((x, i) => (i === d.shape ? moveCorner(s, d.vertex, q) : x))
      } else {
        shapesRef.current = shapesRef.current.map((x, i) =>
          i === d.shape ? { ...x, points: x.points.map((pt, k) => (k === d.vertex ? q : pt)) } : x
        )
      }
    } else if (d.kind === 'ghost' && m) {
      // The copy stays inside the image, as a move would
      const list = shapesRef.current
      const moved = moveShapes(list, d.shapes, p[0] - d.x, p[1] - d.y, m.width, m.height)
      const first = d.shapes[0]
      ghost.current = { shapes: d.shapes, dx: moved[first].points[0][0] - list[first].points[0][0], dy: moved[first].points[0][1] - list[first].points[0][1] }
    } else if (d.kind === 'move' && m) {
      const dx = p[0] - d.x, dy = p[1] - d.y
      const before = shapesRef.current
      const next = moveShapes(before, d.shapes, dx, dy, m.width, m.height)
      if (next !== before) {
        // Only the part of the move that fits is applied, so the grip point follows that
        const moved = next[d.shapes[0]].points[0][0] - before[d.shapes[0]].points[0][0]
        const movedY = next[d.shapes[0]].points[0][1] - before[d.shapes[0]].points[0][1]
        d.x += moved
        d.y += movedY
        shapesRef.current = next
      }
    }
    invalidate()
  }

  const onPointerUp = (e: React.PointerEvent) => {
    releasePointer(e)
    if (!pendingFit.current) return
    clearTimeout(fitTimer.current)
    fitTimer.current = setTimeout(() => {
      fitTimer.current = undefined
      // Pressed again meanwhile: that release starts the wait over
      if (drag.current || !pendingFit.current) return
      pendingFit.current = false
      if (settingsRef.current.fit === 'selection') fitSelection()
    }, DOUBLE_CLICK_MS)
  }
  const releasePointer = (e: React.PointerEvent) => {
    const d = drag.current
    drag.current = null
    if (!d) return
    if (d.kind === 'stroke') {
      commitStroke(d.points, d.radius)
      invalidate()
    } else if (d.kind === 'two') {
      // A drag draws the shape in one go; a click waits for the second click, as labelme does
      const dr = draft.current
      if (dr && Math.hypot(e.clientX - d.press[0], e.clientY - d.press[1]) >= DRAG_START) {
        const q = livePoint()!
        const second = mode === 'rectangle' && squareKeys(e) ? squareFrom(dr.points[0], q) : q
        draft.current = null
        if (Math.hypot(second[0] - dr.points[0][0], second[1] - dr.points[0][1]) > 0.5) boxDone(mode, dr.points[0], second)
      } else if (dr && (mode === 'pose' || mode === 'ai_pose')) {
        // A click on a box of a class with keypoints carries on its pose, or has AI pose it
        const at = poseBoxAt(dr.points[0])
        if (at >= 0) {
          draft.current = null
          if (mode === 'pose') continuePose(at)
          else poseBoxes([at])
        }
      }
      invalidate()
    } else if (d.kind === 'vertex' || d.kind === 'move' || d.kind === 'rotate') {
      // A click only selects: looking at an AI shape must not accept it
      if (!d.active) {
        shapesRef.current = d.start
        return
      }
      const moved = new Set(d.kind === 'move' ? d.shapes : [d.shape])
      undo.current.push(d.start)
      if (undo.current.length > config.canvas.num_backups) undo.current.shift()
      redo.current = []
      apply(acceptEdited(shapesRef.current.map((s, i) => (moved.has(i) && isAi(s) ? accepted(s) : s)), moved, d.start))
    } else if (d.kind === 'pan' && d.button === 2 && !d.moved) {
      menuPoint.current = toImage(e)
      onOpenMenu(e.clientX, e.clientY)
    } else if (d.kind === 'ghost') {
      if (!d.active || !ghost.current) {
        ghost.current = null
        menuPoint.current = toImage(e)
        onOpenMenu(e.clientX, e.clientY)
      } else setGhostMenu({ x: e.clientX, y: e.clientY })
      invalidate()
    }
  }

  const commitStroke = (points: number[][], radius: number) => {
    const m = metaRef.current!
    const list = shapesRef.current
    try {
      const region = strokeRegion(points, radius, 0.5 / view.current.scale, [m.width, m.height])
      if (!region.length) return
      const only = selectedRef.current.length === 1 ? selectedRef.current[0] : null
      if (mode === 'eraser') {
        const result = eraseShapes(list, region, only)
        if (result.shapes !== list) {
          setShapes(result.shapes)
          setSelected(result.selected === null ? [] : [result.selected])
        }
        return
      }
      // Painting with a shape selected continues that shape's class
      const label = (only !== null && list[only]?.label) || currentLabel()
      if (!label) return setNotice('Pick a class in the Labels list to paint with.')
      if (classes.some((c) => c.name === label)) setActive(label)
      setSelected([])
      setShapes(paintShapes(list, region, label))
    } catch (err) {
      onError(`That stroke couldn't be applied, so the shapes are unchanged: ${plainError(err)}`)
    }
  }

  const onDoubleClick = (e: React.MouseEvent) => {
    if (readOnlyRef.current) return
    if (mode === 'polygon' || mode === 'linestrip') {
      const dr = draft.current
      if (!dr || config.canvas.double_click !== 'close') return
      // The second click of the double-click added a point already
      dr.points = dr.points.slice(0, -1)
      finishDraft()
      return
    }
    if (mode === 'ai_points') {
      if (config.canvas.double_click === 'close') finishAi()
      return
    }
    if (mode !== 'edit') return
    const p = toImage(e)
    const h = findHover(p)
    if (h && h.vertex < 0 && h.edge >= 0) {
      setShapes(acceptEdited(shapesRef.current.map((q, i) => (i === h.shape ? insertPoint(q, h.edge, clampPoint(p)) : q)), [h.shape]))
      setSelected([h.shape])
      return
    }
    const hit = hitShape(p)
    if (hit >= 0) {
      setSelected([hit])
      editLabel([hit])
    }
  }

  const stepHistory = (from: typeof undo, to: typeof undo) => {
    if (readOnlyRef.current) return
    const prev = from.current.pop()
    if (!prev) {
      // Right after a review jump, the last thing done was on the image before
      const back = from === undo && reviewBack.current ? images.findIndex((i) => i.name === reviewBack.current) : -1
      if (back >= 0) {
        undoOnLoad.current = true
        setIndex(back)
      }
      return
    }
    to.current.push(shapesRef.current)
    setSelected([])
    apply(prev)
  }
  const removeSelected = () => {
    // A pose's box goes with its points
    const which = withPoints(selectedRef.current, posesOf(shapesRef.current))
    const sel = new Set(which.filter((i) => !shapesRef.current[i]?.locked))
    if (sel.size < which.length) setNotice('Locked shapes stay: unlock them first.')
    if (!sel.size) return
    // Some of an AI pose's points taken out, the rest is accepted, as after any edit
    setShapes(acceptEdited(shapesRef.current, sel).filter((_, i) => !sel.has(i)))
    setSelected([])
  }
  const relabel = (which: number[], label: string) => {
    if (!which.length) return
    setShapes(acceptEdited(shapesRef.current.map((s, k) => (which.includes(k) ? { ...accepted(s), label, flags: { ...flagsFor(label, config.label_flags, s.flags), ...accepted(s).flags } } : s)), which))
  }
  const chooseClass = (i: number) => {
    setActive(classes[i].name)
    // A pose's points keep their keypoints' names
    const points = new Set(posesOf(shapesRef.current).flatMap((p) => poseShapes(p).filter((k) => k !== p.box)))
    relabel(selectedRef.current.filter((k) => !points.has(k)), classes[i].name)
  }

  /** labelme's Edit Label: fields that differ across the selection can't be changed */
  const editLabel = async (which = selectedRef.current) => {
    if (!which.length || readOnlyRef.current) return
    const list = which.map((i) => shapesRef.current[i])
    const [first] = list
    const same = <K extends keyof Shape>(k: K) => list.every((s) => JSON.stringify(s[k]) === JSON.stringify(first[k]))
    const locked = { text: !same('label'), flags: !same('flags'), group_id: !same('group_id'), description: !same('description') }
    const choice = await askLabel({
      text: locked.text ? '' : first.label,
      flags: locked.flags ? null : accepted(first).flags,
      group_id: locked.group_id ? null : first.group_id,
      description: locked.description ? null : first.description,
      locked, title: 'Edit label', at: screenMouse.current
    })
    if (!choice) return
    setShapes(acceptEdited(shapesRef.current.map((s, i) => {
      if (!which.includes(i)) return s
      const a = accepted(s)
      return {
        ...a,
        label: locked.text ? a.label : choice.text,
        flags: locked.flags ? a.flags : choice.flags,
        group_id: locked.group_id ? a.group_id : choice.group_id,
        description: locked.description ? a.description : choice.description
      }
    }), which))
  }

  const problemImages = useMemo(() => new Set(check?.problemImages ?? []), [check])
  const visible = useMemo(() => {
    const test = filter === 'problems' ? (i: ImageItem) => problemImages.has(i.name) : FILTERS.find((f) => f.key === filter)!.test
    let match: (name: string) => boolean
    try {
      // labelme's file search is a regular expression; one that isn't valid filters as plain text
      const re = new RegExp(query, 'i')
      match = (n) => re.test(n)
    } catch {
      match = (n) => n.toLowerCase().includes(query.toLowerCase())
    }
    const list = images.map((image, i) => ({ image, i })).filter(({ image }) => (!query || match(image.name)) && test(image))
    if (sort === 'certainty') {
      const rank = (i: ImageItem) => (i.ai ? i.score ?? 1 : 2)
      list.sort((a, b) => rank(a.image) - rank(b.image) || a.i - b.i)
    }
    return list
  }, [images, query, filter, sort, problemImages])

  // An image asked for from outside, as when Seamtag is started again on it: one another app added
  // since the list was read opens once the list is read again, unless another is opened first
  const askedFor = useRef<string | null>(null)
  const goTo = useCallback(async (i: number, copy = false) => {
    askedFor.current = null
    if (i === index || i < 0 || i >= images.length) return
    if (!(await canLeave())) return
    copyOnLoad.current = copy
    setIndex(i)
  }, [index, images.length, canLeave])
  const showImage = async (name: string, shape?: number) => {
    const i = images.findIndex((x) => x.name === name)
    if (i < 0) return
    if (i === index) {
      if (loadedName.current !== name) selectOnLoad.current = { name, shape }
      else if (shape !== undefined && shapesRef.current[shape]) selectForReview(shape, shapesRef.current)
      return
    }
    if (!(await canLeave())) return
    selectOnLoad.current = { name, shape }
    copyOnLoad.current = false
    setIndex(i)
  }
  const reloadImages = (names: string[]) => {
    markStale(names)
    for (const name of names) history.current.delete(name)
    if (!names.includes(loadedName.current)) return
    undo.current = []
    redo.current = []
    setReload((n) => n + 1)
  }
  const show = (name: string) => {
    const i = images.findIndex((x) => x.name === name)
    // A dialog, a menu or a shape being drawn would be cut short, or a label given to the wrong image: it says so instead
    const waiting = i === index ? '' : busy()
    if (waiting) return onError(`${name} wasn't opened. ${waiting}`)
    if (i >= 0) return void goTo(i)
    askedFor.current = name
    refreshFolder()
  }
  useEffect(() => {
    const name = askedFor.current
    const i = name === null ? -1 : images.findIndex((x) => x.name === name)
    if (i < 0) return
    askedFor.current = null
    // A dialog, a menu or a shape drawn while the folder was read again isn't cut short either
    const waiting = i === index ? '' : busy()
    if (waiting) return onError(`${name} wasn't opened. ${waiting}`)
    goTo(i)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images, goTo])
  const go = (delta: number, copy = false) => {
    const at = visible.findIndex((v) => v.i === index)
    const next = visible[Math.max(0, Math.min(visible.length - 1, (at < 0 ? 0 : at) + delta))]
    if (next) goTo(next.i, copy)
  }

  // Bring shapes into view: closer in if they're off screen or too small to judge
  const focusShapes = (which: Shape[]) => {
    const canvas = canvasRef.current
    if (!canvas || !which.length) return
    const all = which.map(bounds)
    const x0 = Math.min(...all.map((b) => b[0])), y0 = Math.min(...all.map((b) => b[1]))
    const x1 = Math.max(...all.map((b) => b[2])), y1 = Math.max(...all.map((b) => b[3]))
    const W = canvas.clientWidth, H = canvas.clientHeight
    const { scale, x, y } = view.current
    const seen = x0 * scale + x >= 0 && y0 * scale + y >= 0 && x1 * scale + x <= W && y1 * scale + y <= H
    const large = Math.max(x1 - x0, y1 - y0) * scale >= 80
    if (seen && large) return
    const fits = (x1 - x0) * scale <= W && (y1 - y0) * scale <= H
    const next = large && fits ? scale : Math.min(8, W / ((x1 - x0 || 1) * 3), H / ((y1 - y0 || 1) * 3))
    view.current = { scale: next, x: W / 2 - ((x0 + x1) / 2) * next, y: H / 2 - ((y0 + y1) / 2) * next }
    setFit('manual')
    setZoom(next)
    invalidate()
  }
  /** A shape selected and in view; one of a pose is framed with the whole pose, as a keypoint alone says nothing of where it belongs */
  const selectForReview = (i: number, list: Shape[]) => {
    setSelected([i])
    setMode('edit')
    const pose = posesOf(list).find((p) => poseShapes(p).includes(i))
    focusShapes(pose ? poseShapes(pose).map((k) => list[k]) : [list[i]])
  }
  const nextReviewImage = () => {
    const at = visible.findIndex((v) => v.i === index)
    const order = [...visible.slice(at + 1), ...visible.slice(0, Math.max(at, 0))]
    const next = order.find((v) => v.i !== index && v.image.ai > 0)
    setSelected([])
    if (!next) return setNotice('Nothing left to review in this list.')
    reviewOnLoad.current = true
    goTo(next.i)
  }
  /** Review's steps on these shapes, least certain first, an AI pose as one */
  const reviewSteps = (list: Shape[]) => reviewItems(list, posesOf(list))
  /** A step of review selected and in view: an AI pose with all its shapes */
  const selectStep = (step: ReviewItem, list: Shape[]) => {
    if (!step.pose) return selectForReview(step.shapes[0], list)
    setSelected(poseShapes(step.pose))
    setMode('edit')
    focusShapes(poseShapes(step.pose).map((k) => list[k]))
  }
  const selectedStep = (list: Shape[]) => stepOf(reviewSteps(list), selectedRef.current)
  const reviewNext = (list: Shape[]) => {
    const first = reviewSteps(list)[0]
    if (first) selectStep(first, list)
    else nextReviewImage()
  }
  // Only once this image's shapes are in: keys can arrive faster than images load
  const reviewReady = () => !drag.current && loadedName.current === images[index]?.name && !readOnlyRef.current
  // Accepting or rejecting any shape of an AI pose accepts or rejects all its AI shapes
  const acceptSelected = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    const step = selectedStep(list)
    if (!step) return reviewNext(list)
    const next = list.map((x, i) => (step.shapes.includes(i) ? accepted(x) : x))
    setShapes(next)
    reviewNext(next)
  }
  const rejectSelected = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    const step = selectedStep(list)
    if (!step) return reviewNext(list)
    const next = list.filter((_, i) => !step.shapes.includes(i))
    setShapes(next)
    reviewNext(next)
  }
  const skipShape = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    const steps = reviewSteps(list)
    if (!steps.length) return nextReviewImage()
    const now = selectedStep(list)
    selectStep(steps[(steps.findIndex((step) => step.shapes[0] === now?.shapes[0]) + 1) % steps.length], list)
  }
  const acceptAll = () => {
    if (!reviewReady()) return
    if (shapesRef.current.some(isAi)) setShapes(shapesRef.current.map((s) => (isAi(s) ? accepted(s) : s)))
    nextReviewImage()
  }

  const copySelected = () => {
    // A pose's box is copied with its points
    const sel = withPoints(selectedRef.current, posesOf(shapesRef.current))
    if (!sel.length) return
    clipboard.current = sel.map((i) => ({ ...shapesRef.current[i], hidden: undefined }))
    setTick((t) => t + 1)
  }
  const paste = () => {
    if (!clipboard.current.length || readOnlyRef.current || !metaRef.current) return
    // A pose being placed is kept first, so the copies are numbered after its group_id
    endPose()
    // Pasted onto a smaller image, shapes are kept inside it; a pose pasted is a pose of its own
    const copies = regroup(clipboard.current, shapesRef.current, opts.current.keypoints).map((s) => ({ ...s, points: s.points.map(clampPoint) }))
    const fitIn = forFormat(copies)
    const note = boxedNote(copies)
    if (note) setNotice(note)
    if (!fitIn.length) return
    const next = [...shapesRef.current, ...fitIn]
    setShapes(next)
    setSelected(fitIn.map((_, k) => next.length - fitIn.length + k))
  }
  const duplicate = () => {
    copySelected()
    paste()
  }
  const merge = async () => {
    const list = shapesRef.current
    const which = selectedRef.current.filter((i) => !list[i]?.locked)
    if (which.length >= 2 && which.every((i) => list[i].shape_type === 'mask' && list[i].mask)) {
      // labelme 7's Merge Shapes for masks: one label, every pixel kept, in the first one's place
      if (new Set(which.map((i) => list[i].label)).size !== 1) return setNotice('Masks merge when they share one label.')
      const merged = await window.api.mergeMasks(which.map((i) => ({ points: list[i].points, mask: list[i].mask! })))
      if (shapesRef.current !== list) return
      const shape = { ...accepted(list[which[0]]), points: merged.points, mask: merged.mask }
      const next = list.flatMap((s, i) => (i === which[0] ? [shape] : which.includes(i) ? [] : [s]))
      setShapes(next)
      setSelected([next.indexOf(shape)])
      return
    }
    const result = mergePolygons(list, which)
    if (!result) return setNotice('Select two or more polygons or boxes, or masks of one label, to merge.')
    setShapes(result.shapes)
    setSelected(result.selected)
  }
  /** labelme 7's Add Point to Edge: a point where the menu was opened, on the selected shape's nearest edge */
  const addPointToEdge = () => {
    const i = selectedRef.current[0]
    const s = shapesRef.current[i]
    if (!s || s.locked || !EDITABLE_POINTS.has(s.shape_type)) return
    const p = menuPoint.current ?? mouse.current
    const edge = nearestEdge(s, p, Infinity)
    if (edge < 0) return
    const [a, b] = edges(s)[edge].map((k) => s.points[k])
    const dx = b[0] - a[0], dy = b[1] - a[1]
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)))
    setShapes(acceptEdited(shapesRef.current.map((q, k) => (k === i ? insertPoint(q, edge, [a[0] + t * dx, a[1] + t * dy]) : q)), [i]))
  }
  /** labelme 7's Ctrl+arrow: the nearest shape that way, within 90°, a sideways step costing double */
  const selectNeighbour = (vx: number, vy: number) => {
    const list = shapesRef.current
    const center = (b: number[]) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    const candidates = list.map((s, i) => ({ s, i })).filter(({ s, i }) => !s.hidden && !selectedRef.current.includes(i))
    if (!selectedRef.current.length) {
      if (!candidates.length) return
      const [first] = [...candidates].sort((a, b) => {
        const ca = center(bounds(a.s)), cb = center(bounds(b.s))
        return Math.abs(ca[0]) + Math.abs(ca[1]) - (Math.abs(cb[0]) + Math.abs(cb[1]))
      })
      return setSelected([first.i])
    }
    const chosen = selectedRef.current.map((i) => bounds(list[i]))
    const origin = center([Math.min(...chosen.map((b) => b[0])), Math.min(...chosen.map((b) => b[1])), Math.max(...chosen.map((b) => b[2])), Math.max(...chosen.map((b) => b[3]))])
    let best = -1, bestCost = Infinity
    for (const { s, i } of candidates) {
      const c = center(bounds(s))
      const dx = c[0] - origin[0], dy = c[1] - origin[1]
      const along = dx * vx + dy * vy, sideways = Math.abs(dx * vy - dy * vx)
      if (along > 0 && sideways <= along && along + 2 * sideways < bestCost) { bestCost = along + 2 * sideways; best = i }
    }
    if (best >= 0) setSelected([best])
  }
  /** labelme 7's Fit to Selection: the view fits the selected shapes, with a margin */
  const fitSelection = () => {
    const canvas = canvasRef.current
    const chosen = selectedRef.current.map((i) => shapesRef.current[i]).filter(Boolean)
    if (!canvas || !metaRef.current || !chosen.length) return
    const all = chosen.map(bounds)
    const x0 = Math.min(...all.map((b) => b[0])), y0 = Math.min(...all.map((b) => b[1]))
    const x1 = Math.max(...all.map((b) => b[2])), y1 = Math.max(...all.map((b) => b[3]))
    const pad = 48
    const scale = Math.min((canvas.clientWidth - pad * 2) / Math.max(x1 - x0, 1), (canvas.clientHeight - pad * 2) / Math.max(y1 - y0, 1), config.max_zoom / 100)
    view.current = { scale, x: canvas.clientWidth / 2 - ((x0 + x1) / 2) * scale, y: canvas.clientHeight / 2 - ((y0 + y1) / 2) * scale }
    lastScale.current = scale
    setZoom(scale)
    invalidate()
  }
  const removeHoveredPoint = () => {
    const h = hover.current
    const list = shapesRef.current
    // A pose's keypoint under the pointer goes, as a polygon's point does: it's not labelled then
    const k = keypointUnder()
    if (k !== null && !list[k].locked) {
      setShapes(acceptEdited(list, [k]).filter((_, i) => i !== k))
      setSelected(selectedRef.current.filter((i) => i !== k).map((i) => (i > k ? i - 1 : i)))
      hover.current = null
      return true
    }
    if (!h || h.vertex < 0 || !canRemovePoint(list[h.shape])) return false
    setShapes(acceptEdited(list.map((q, i) => (i === h.shape ? { ...accepted(q), points: q.points.filter((_, k) => k !== h.vertex) } : q)), [h.shape]))
    hover.current = null
    return true
  }
  const addMultiplePoints = () => {
    const h = hover.current
    if (!h || h.edge < 0) return setNotice('Point at a polygon edge first, then Ctrl+M.')
    setPointsPrompt({ shape: h.shape, edge: h.edge, count: config.default_num_points_to_add })
  }
  const nudge = (dx: number, dy: number) => {
    const m = metaRef.current
    const which = withPoints(selectedRef.current, posesOf(shapesRef.current)).filter((i) => !shapesRef.current[i]?.locked)
    if (!m || !which.length) return
    const next = moveShapes(shapesRef.current, which, dx, dy, m.width, m.height)
    if (next !== shapesRef.current) setShapes(acceptEdited(next, which))
  }
  const undoLastPoint = () => {
    // Placing a pose, it's back one keypoint
    if (backKeypoint()) return true
    const dr = draft.current
    if (!dr) return false
    if (mode === 'oriented_rectangle' && dr.points.length === 4) dr.points = [dr.points[0]]
    else {
      dr.points.pop()
      dr.labels.pop()
    }
    if (!dr.points.length) draft.current = null
    else if (mode === 'ai_points') runAi()
    invalidate()
    return true
  }

  // The pose keys: hidden or seen, left and right, and stepping through a pose's keypoints
  /** The poses with shapes among those given */
  const posesAmong = (which: number[]) => {
    const among = new Set(which)
    return posesOf(shapesRef.current).filter((p) => poseShapes(p).some((i) => among.has(i)))
  }
  /** Shapes after an edit to some, by index, with any AI pose among them accepted whole; `before` has the poses as they were */
  const acceptEdited = (next: Shape[], touched: Iterable<number>, before = shapesRef.current) => acceptTouched(next, posesOf(before), touched)
  /** A pose's keypoint under the pointer, by index; null when there's none */
  const keypointUnder = () => {
    const h = hover.current
    if (!h || !overCanvas.current) return null
    const pose = posesOf(shapesRef.current).find((p) => p.box !== h.shape && poseShapes(p).includes(h.shape))
    return pose ? h.shape : null
  }
  /**
   * Q: the keypoint under the pointer, or else those selected, hidden, or seen when they all are.
   * A whole pose selected isn't a choice of its points: only the one under the pointer changes.
   */
  const toggleHidden = () => {
    const list = shapesRef.current
    const posed = posesOf(list)
    const points = new Set(posed.flatMap((p) => poseShapes(p).filter((i) => i !== p.box)))
    const whole = posed.some((p) => p.box !== null && selectedRef.current.includes(p.box))
    const under = keypointUnder()
    const which = under !== null ? [under] : whole ? [] : selectedRef.current.filter((i) => points.has(i))
    if (!which.length) return setNotice(whole ? 'Point at a keypoint to hide or show it.' : 'Point at a keypoint, or select some, first.')
    const hide = !which.every((i) => isHidden(list[i]))
    setShapes(acceptEdited(list.map((s, i) => (which.includes(i) ? withHidden(s, hide, flagsFor(s.label, config.label_flags)) : s)), which, list))
  }
  /** M: left and right swapped on the poses selected; with only some of a pose's points selected, on their pairs */
  const swapSelectedSides = () => {
    const sel = selectedRef.current
    const touched = posesAmong(sel)
    if (!touched.length) return setNotice('Select a pose first.')
    const schema = opts.current.keypoints
    if (touched.every((p) => !schema.flip?.[p.cls]?.length)) return setNotice(`The keypoints file has no left and right pairs for ${touched[0].cls}.`)
    const chosen = new Set(sel)
    let next = shapesRef.current
    for (const p of touched) next = swapSides(next, p, schema, p.box !== null && chosen.has(p.box) ? undefined : chosen)
    if (next !== shapesRef.current) setShapes(acceptEdited(next, touched.flatMap(poseShapes)))
  }
  /** Shift+M: the poses selected mirrored inside their boxes */
  const mirrorSelected = () => {
    const touched = posesAmong(selectedRef.current)
    if (!touched.length) return setNotice('Select a pose first.')
    let next = shapesRef.current
    for (const p of touched) next = mirrorPose(next, p)
    if (next !== shapesRef.current) setShapes(acceptEdited(next, touched.flatMap(poseShapes)))
  }
  /** N and Shift+N: the selected pose's next or previous keypoint selected, or, when it's missing, placed with the next click */
  const stepKeypoint = (step: 1 | -1) => {
    const sel = selectedRef.current
    const asked = posing.current
    // A keypoint waiting to be placed, from N before, counts as where N is
    const pose = asked && !asked.fresh && asked.asked.length === 1 ? posesOf(shapesRef.current).find((p) => p.group === asked.group) : posesAmong(sel)[0]
    if (!pose) return setNotice('Select a pose first.')
    const n = pose.points.length
    const from = asked?.group === pose.group && asked.asked.length === 1 ? asked.asked[0] : sel.length === 1 ? pose.points.indexOf(sel[0]) : -1
    const k = from < 0 ? (step > 0 ? 0 : n - 1) : (from + step + n) % n
    setPosing(null)
    const at = pose.points[k]
    if (at !== null) setSelected([at])
    else placeMissing(pose, k)
  }
  const deleteLabelFile = async () => {
    const name = loadedName.current
    if (!name) return
    const file = await window.api.labelPath(root, name, formatRef.current)
    const answer = await window.api.ask(`Delete the label file of ${name}?`, `${file} goes to the trash.`, ['Delete', 'Cancel'])
    if (answer !== 0) return
    clearTimeout(saveTimer.current)
    pending.current = null
    dirty.current = false
    try {
      const gone = await window.api.trashLabels(root, name, formatRef.current)
      if (gone && typeof gone === 'object') return reloadChanged(name, 'Nothing was deleted.')
      history.current.delete(name)
      markStale([name])
      fileFormat.current = null
      switchedFrom.current = null
      setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize([], false), broken: undefined, verified: undefined } : i)))
      verifiedRef.current = false
      setVerifiedState(false)
      shapesRef.current = []
      setShapesState([])
      setSelected([])
      readOnlyRef.current = ''
      setReadOnly('')
      undo.current = []
      redo.current = []
      invalidate()
    } catch (e) {
      onError(`It couldn't be deleted: ${plainError(e)}`)
    }
  }
  const saveAs = async () => {
    const name = loadedName.current
    if (!name || !metaRef.current) return
    const target = await window.api.pickSave('Save labels as', await window.api.labelPath(root, name, formatRef.current), formatRef.current)
    if (!target) return
    saveTarget.current = target
    pending.current = {
      name, shapes: shapesRef.current, flags: flagsRef.current, size: [metaRef.current.width, metaRef.current.height], format: formatRef.current,
      verified: verifiedRef.current, from: null, byHand: true
    }
    await flush()
  }
  const saveNow = async () => {
    const name = loadedName.current
    if (!name || !metaRef.current || readOnlyRef.current) return
    pending.current = {
      name, shapes: shapesRef.current, flags: flagsRef.current, size: [metaRef.current.width, metaRef.current.height], format: formatRef.current,
      verified: verifiedRef.current, from: switchedFrom.current, byHand: true
    }
    await flush()
  }

  // Pre-labelling with a segmentation model, as in Seamline
  const toPrelabel = images.filter((i) => !i.broken && !i.count && !(i.labelled && !i.ai))
  const pickModel = async () => {
    try {
      const model = await window.api.prelabelPick(root)
      // For this folder, and for folders with no runs and no model of their own
      if (model) setSettings((s) => ({ ...s, prelabelModel: model.path, prelabelModels: chooseFor(s.prelabelModels, root, model.path) }))
      return model
    } catch (e) {
      onError(plainError(e))
      return null
    }
  }
  /**
   * The model chosen for this folder. With none, the newest one its runs exported, as the folder is
   * now; in a folder with no runs, the model last picked as a file; or else one picked now.
   */
  const modelToUse = async () => {
    const { prelabelModels: chosen, prelabelModel: picked } = settingsRef.current
    const here = chosen[root]
    const runs = here ? null : await window.api.prelabelModels(root).catch(() => runModels ?? [])
    if (runs && alive.current) setRunModels(runs)
    const newest = runs?.[0]
    const file = here || newest?.path || picked
    if (!file) return pickModel()
    if (prelabelModel?.path === file) return prelabelModel
    try {
      return await window.api.prelabelInfo(file)
    } catch (e) {
      // A model chosen before that's gone: another one, as a file
      if (!newest || file !== newest.path) return pickModel()
      onError(`${newest.title} can't be used: ${plainError(e)}`)
      return null
    }
  }
  // A model another folder's run exported says which folder
  const modelTitle = (m: PrelabelModel) => (m.project && m.project !== root ? `${m.name} · from ${m.project.split(/[\\/]/).pop()}` : m.name)
  // What pre-labelling would use, for the buttons' tips
  const prelabelName = chosenModel ? (prelabelModel?.path === chosenModel ? modelTitle(prelabelModel) : null) : runModels?.[0]?.title ?? null
  const modelClasses = (model: PrelabelModel) => (model.classes.length ? model.classes : classes.map((c) => c.name))
  const prelabelOne = async () => {
    const model = await modelToUse()
    if (!model) return
    const name = loadedName.current
    if (!name || readOnlyRef.current) return
    setBusyAi('Predicting…')
    try {
      await flush()
      const run = (prelabelling.current = window.api.prelabelRun(root, model.path, [name], modelClasses(model)))
      const result = await run
      if (result.error) onError(`Pre-label failed: ${result.error}`)
      else if (result.failed.length) onError(`Pre-label failed: ${result.failed[0]}`)
      // In Seamline's words: nothing was written, as an empty label file would read as cleared by someone
      else if (result.nothing.includes(name)) setNotice(`The model found nothing on ${name}, so its labels stay as they were.`)
    } catch (e) {
      onError(`Pre-label failed: ${plainError(e)}`)
    } finally {
      prelabelling.current = null
      setBusyAi('')
      setBatch(null)
    }
  }
  const prelabelAll = async () => {
    const model = await modelToUse()
    if (!model) return
    await flush()
    setBatch({ done: 0, total: toPrelabel.length })
    try {
      const run = (prelabelling.current = window.api.prelabelRun(root, model.path, toPrelabel.map((i) => i.name), modelClasses(model), true))
      const result = await run
      setNotice(`${result.stopped ? 'Stopped after' : 'Pre-labelled'} ${result.done} of ${result.total} images.`)
      if (result.error) onError(`Pre-labelling stopped early: ${result.error}`)
      else if (result.failed.length) onError(`${result.failed.length} ${result.failed.length === 1 ? 'image' : 'images'} failed. First: ${result.failed[0]}`)
    } catch (e) {
      onError(`Pre-label failed: ${plainError(e)}`)
    } finally {
      prelabelling.current = null
      setBatch(null)
    }
  }

  // Dataset > Check labels. The check runs in the main process, which keeps it; the Problems
  // dock pages through it. A save puts an image's problems out of date, and a second after the
  // last save the folder is checked again: only the images saved are read and measured again.
  const listedLabels = useMemo(() => {
    const list = [...new Set([...(config.labels ?? []), ...project.classes.map((c) => c.name)])]
    return list.length ? list : null
  }, [config.labels, project.classes])
  // The label list as the check takes it: a change to it is checked again, as a save is
  const labelKey = (listedLabels ?? []).join('\n')
  const labelKeyRef = useRef(labelKey)
  labelKeyRef.current = labelKey
  const runCheck = async (quick: boolean) => {
    clearTimeout(recheckTimer.current)
    // One after saves waits for a check that's running: that one looks again once it's done. A
    // window closed meanwhile, with its folder, checks nothing.
    if ((quick && checkingRef.current) || !alive.current) return
    const seq = ++checkSeq.current
    const { names, upTo } = stale.current.begin()
    // The list this check is given, which a later render may have changed already
    const listed = labelKey
    checkingRef.current = true
    setChecking({ quick, progress: null })
    let more = false
    try {
      const result = await window.api.checkRun(root, { prefer: formatRef.current, labels: listedLabels, ...(quick ? { saved: names } : {}) })
      if (seq !== checkSeq.current || !alive.current) return
      if ('stopped' in result) {
        if (!quick) setNotice(checkRef.current ? 'Check stopped: the problems shown are from the check before.' : 'Check stopped.')
        return
      }
      // Saves made while it ran are checked next, and a label list changed meanwhile
      more = stale.current.settle(upTo) || labelKeyRef.current !== listed
      checkRef.current = result
      setCheck(result)
    } catch (e) {
      if (seq === checkSeq.current && alive.current) onError(`The labels couldn't be checked: ${plainError(e)}`)
    } finally {
      if (seq === checkSeq.current) {
        checkingRef.current = false
        setChecking(null)
        if (more && alive.current) recheckTimer.current = setTimeout(() => runCheckRef.current(true), 1000)
      }
    }
  }
  const runCheckRef = useRef(runCheck)
  runCheckRef.current = runCheck
  // A label added to the list, or taken off it, changes which labels the check calls unknown: no file is read again
  useEffect(() => {
    if (!checkRef.current || checkingRef.current) return
    clearTimeout(recheckTimer.current)
    recheckTimer.current = setTimeout(() => runCheckRef.current(true), 1000)
  }, [labelKey])
  /** Label files changed: their images' problems are out of date until the check that follows a second later */
  const markStale = (names: string[]) => {
    // Before the first check there's nothing to put out of date, unless it's running; nor once the window has closed
    if ((!checkRef.current && !checkingRef.current) || !names.length || !alive.current) return
    stale.current.mark(names)
    setTick((t) => t + 1)
    clearTimeout(recheckTimer.current)
    if (!checkingRef.current) recheckTimer.current = setTimeout(() => runCheckRef.current(true), 1000)
  }
  touched.current = markStale
  const checkLabels = async () => {
    if (!(await canLeave({ stay: true }))) return
    setSettings((s) => ({ ...s, docks: { ...s.docks, problems: true } }))
    setProblemsFolded(false)
    runCheck(false)
  }
  const whenLoaded = (name: string) => new Promise<boolean>((done) => loadWaits.current.push({ name, done }))
  /**
   * A safe fix, made to the image's shapes as they are in the window, as an edit Ctrl+Z
   * undoes. Another image than the one open opens first. The main process checks the shapes
   * again and fixes only a problem that's still there.
   */
  const fixProblem = async (row: ProblemRow) => {
    const name = row.image
    if (!name || row.shape === undefined) return
    if (loadedName.current !== name) {
      const i = images.findIndex((x) => x.name === name)
      // Still loading: it can be fixed once it's open
      if (i < 0 || i === index || !(await canLeave())) return
      const loaded = whenLoaded(name)
      selectOnLoad.current = { name, shape: row.shape }
      copyOnLoad.current = false
      setIndex(i)
      if (!(await loaded)) return
    }
    if (readOnlyRef.current) return
    const before = shapesRef.current
    let next: Shape[] | null
    try {
      next = await window.api.checkFix(root, { code: row.code, image: name, shape: row.shape }, before)
    } catch (e) {
      return onError(`It couldn't be fixed: ${plainError(e)}`)
    }
    // Changed meanwhile: the next check says what's left
    if (loadedName.current !== name || shapesRef.current !== before) return
    if (!next) {
      // Changes the check hasn't seen: the saved file is checked once they're in it
      if (dirty.current) return setNotice("That problem isn't in the image as it is now. It's checked again once the image is saved.")
      setNotice("That problem isn't in the image as it is now, so it's checked again.")
      return markStale([name])
    }
    const removed = before.length - next.length
    // The save that follows puts its problems out of date; until then they're shown as not saved
    setShapes(next)
    setSelected(row.code === 'duplicate-shape' || row.code === 'keypoint-repeated' || !next[row.shape] ? [] : [row.shape])
    const undoKeys = formatKeys(config.shortcuts.undo, mac)
    const what = row.code === 'duplicate-shape' ? `Deleted ${counted(removed, 'exact copy', 'exact copies')} of a shape.`
      : row.code === 'repeated-points' ? `Took the repeated points out of shape ${row.shape + 1}; its outline is as it was.`
        : row.code === 'keypoint-repeated' ? `Deleted shape ${row.shape + 1}, a second ${row.label ?? 'keypoint'} in its pose; the first stays.`
        : `Clipped shape ${row.shape + 1} to the image.`
    const then = settingsRef.current.autoSave ? '' : " It's checked again once saved."
    setNotice(`${what}${undoKeys ? ` ${undoKeys} undoes it.` : ''}${then}`)
  }
  /** Rename a label across the folder: the open image's changes are saved first, so the plan reads them */
  const openRename = async (from: string, to = '') => {
    if (!(await canLeave({ stay: true }))) return
    setRenaming({ from, to })
  }
  const renamed = (from: string, done: RenameDone & { to: string }) => {
    // The folder closed while it ran: what's said and changed here would be about another one
    if (!alive.current) return
    const { to } = done
    setRenaming(null)
    // The project's label list follows once no label file has the old name: it gives way to the new, keeping its colour
    const has = (n: string) => project.classes.some((c) => c.name === n)
    if (has(from) && !done.left && done.changed.length) {
      setProject({ ...project, classes: has(to) ? project.classes.filter((c) => c.name !== from) : project.classes.map((c) => (c.name === from ? { ...c, name: to } : c)) })
    }
    // So do the labels found in the files, and the label new shapes get. The config's labels stay listed.
    if (!done.left && done.changed.length) {
      forgetLabel(from)
      // The label list keeps its order: the new name takes the old one's place
      const kept = keptLabels.current
      if (kept) keptLabels.current = kept.includes(to) ? kept.filter((l) => l !== from) : kept.map((l) => (l === from ? to : l))
      setActive((a) => ((a === undefined ? classes[0]?.name : a) === from ? to : a))
    }
    if (done.changed.length) reloadImages(done.changed)
    setLabelsChanged((n) => n + 1)
    const told = [`${done.stopped ? 'Stopped after renaming' : 'Renamed'} ${counted(done.shapes, 'shape')} in ${counted(done.changed.length, 'image')} from “${from}” to “${to}”.`]
    // Why files were left alone: one that can't be read is left whatever it holds, and isn't to be taken for one with the label
    const { unreadable, shared, file } = done.skipped
    if (unreadable) {
      told.push(file ? `${file} can't be read, so it was left alone.` : `${counted(unreadable, "label file that can't be read was", "label files that can't be read were")} left alone.`)
    }
    if (shared) {
      told.push(file ? `${file} holds the labels of several images, so it was left alone.`
        : `${counted(shared, 'label file that several images share was', 'label files that several images share were')} left alone.`)
    }
    if (has(from) && done.left) told.push(`“${from}” stays in the label list while ${counted(done.left, 'image has', 'images have')} it.`)
    setNotice(told.join(' '))
    if (done.failed.length) {
      const [first] = done.failed
      onError(`${counted(done.failed.length, "label file couldn't", "label files couldn't")} be renamed. ${first.name}: ${first.reason}`)
    }
  }
  const openStats = async () => {
    if (!(await canLeave({ stay: true }))) return
    setStatsOpen(true)
  }

  const addClass = () => {
    const name = newClass.trim()
    if (!name || classes.some((c) => c.name === name)) return
    if (config.validate_label === 'exact' && !config.labels?.includes(name)) return onError(`“${name}” isn't in the label list, and only listed labels are allowed.`)
    const color = labelColor(name, classes.length, config)
    // The folder's classes and this label: the list's other labels aren't classes, and stay in their places
    keptLabels.current = [...classes.map((c) => c.name), name]
    setProject({ ...project, classes: withClass(project.classes, name, color) }, classes)
    setActive(name)
    setNewClass('')
  }
  const recolor = (name: string, color: string) => {
    keptLabels.current = classes.map((c) => c.name)
    setProject({ ...project, classes: recolored(project.classes, name, color) }, classes)
  }

  // LabelImg's Verify Image: a mark that the image has been checked, saved at once
  const verifyImage = async () => {
    if (!loadedName.current || !metaRef.current || readOnlyRef.current) return
    if (!keepsVerified(formatRef.current)) return setNotice('YOLO files have no place to keep that an image is verified.')
    verifiedRef.current = !verifiedRef.current
    setVerifiedState(verifiedRef.current)
    invalidate()
    await saveNow()
  }
  /** LabelImg's difficult flag, on the selected shapes. Labels whose label_flags list it keep it, unticked. */
  const toggleDifficult = () => {
    const which = selectedRef.current
    if (!which.length || readOnlyRef.current) return
    const on = !which.every((i) => shapesRef.current[i]?.flags.difficult === true)
    setShapes(acceptEdited(shapesRef.current.map((s, i) => {
      if (!which.includes(i)) return s
      const { difficult: _d, ...rest } = s.flags
      const listed = 'difficult' in flagsFor(s.label, config.label_flags)
      return { ...s, flags: on ? { ...rest, difficult: true } : listed ? { ...rest, difficult: false } : rest }
    }), which))
  }
  // LabelImg's format button: the image on screen is saved in the new format, and so are new label files
  const setFormat = async (format: LabelFormat) => {
    if (format === formatRef.current) return
    if (images[index]?.embedded) return setNotice('This image lives inside its LabelMe file, so its labels stay LabelMe JSON.')
    const name = loadedName.current
    if (holdsBoxesOnly(format) && shapesRef.current.some(isAi)) return setNotice(`Accept or reject the AI shapes first: ${FORMAT_NAMES[format]} can't keep them apart.`)
    // A switch saved a moment ago decides which file this one replaces, so it lands first
    await lastSave.current
    if (loadedName.current !== name || format === formatRef.current) return
    if (name && metaRef.current && !readOnlyRef.current && await window.api.hasLabelsIn(root, name, format)) {
      // The image has labels in that format already: they open, and nothing is converted or written over
      if (loadedName.current !== name || !(await canLeave())) return
      const was = fileFormat.current
      formatRef.current = format
      setSettings((s) => ({ ...s, format }))
      undo.current = []
      redo.current = []
      setReload((n) => n + 1)
      setNotice(`${name} has ${FORMAT_NAMES[format]} labels already, so they open as they are${was ? `, and its ${FORMAT_NAMES[was]} file stays` : ''}.`)
      return
    }
    const list = shapesRef.current
    formatRef.current = format
    setSettings((s) => ({ ...s, format }))
    if (!metaRef.current || readOnlyRef.current) return
    // The file in the old format goes once the new one is written
    if (fileFormat.current && fileFormat.current !== format) switchedFrom.current = fileFormat.current
    else if (fileFormat.current === format) switchedFrom.current = null
    const note = boxedNote(list)
    if (note) {
      setShapes(forFormat(list))
      setSelected([])
      setNotice(note)
    } else if (images[index]?.labelled || list.length) queueSave()
  }
  /** Shapes from elsewhere take the place of this image's, clipped to it, as one step to undo */
  const replaceShapes = (from: Shape[]) => {
    // A pose being placed ends first: carried on over shapes from elsewhere, its group_id could be one of theirs
    endPose()
    const fitIn = from.map((s) => ({ ...s, points: s.points.map(clampPoint), hidden: undefined }))
    const next = forFormat(fitIn)
    setShapes(next)
    setSelected(next.map((_, k) => k))
    const note = boxedNote(fitIn)
    if (note) setNotice(note)
  }
  // LabelImg's Copy previous Bounding Boxes: the image before this one in the list lends its shapes
  const copyPrevious = async () => {
    const name = loadedName.current
    const at = visible.findIndex((v) => v.i === index)
    if (!name || !metaRef.current || readOnlyRef.current || at < 1) return
    const prev = visible[at - 1].image.name
    try {
      const { doc } = await window.api.loadLabels(root, prev, formatRef.current)
      if (loadedName.current !== name) return
      if (!doc?.shapes.length) return setNotice(`${prev} has no shapes to copy.`)
      replaceShapes(doc.shapes)
    } catch (e) {
      onError(`The labels of ${prev} can't be read: ${plainError(e)}`)
    }
  }
  // LabelImg's Open Annotation: a label file picked by hand, read onto this image
  const openAnnotation = async () => {
    const name = loadedName.current, m = metaRef.current
    if (!name || !m || readOnlyRef.current) return
    const file = await window.api.pickFile('Open annotation', [{ name: 'Label files', extensions: ['xml', 'txt', 'json'] }])
    if (!file || loadedName.current !== name) return
    try {
      const { doc } = await window.api.readLabelFile(root, name, file, [m.width, m.height])
      if (loadedName.current !== name) return
      if (!doc.shapes.length) return setNotice(`${file.split(/[\\/]/).pop()} has no shapes for ${name}.`)
      replaceShapes(doc.shapes)
    } catch (e) {
      onError(`That file can't be read: ${plainError(e)}`)
    }
  }
  // LabelImg's Delete Image, to the trash with its label files; the next image takes its place
  const deleteImage = async () => {
    const name = loadedName.current
    if (!name || images[index]?.name !== name) return
    const answer = await window.api.ask(`Move ${name} to the trash?`, 'Its label files go with it.', ['Move to trash', 'Cancel'])
    if (answer !== 0 || loadedName.current !== name) return
    clearTimeout(saveTimer.current)
    pending.current = null
    dirty.current = false
    // Nothing more is saved for it, even an edit made while it goes
    loadedName.current = ''
    try {
      const gone = await window.api.trashImage(root, name)
      if ('changed' in gone) {
        loadedName.current = name
        return reloadChanged(name, "The image wasn't moved to the trash.")
      }
      if (gone.kept) onError(`${name} went to the trash, but ${gone.kept}`)
    } catch (e) {
      loadedName.current = name
      return onError(`It couldn't be moved to the trash: ${plainError(e)}`)
    }
    history.current.delete(name)
    markStale([name])
    const left = images.length - 1
    setImages((all) => all.filter((i) => i.name !== name))
    setIndex((i) => Math.max(0, Math.min(i, left - 1)))
    if (!left) {
      metaRef.current = null
      shapesRef.current = []
      undo.current = []
      redo.current = []
      setMeta(null)
      setShapesState([])
      setSelected([])
      invalidate()
    }
  }
  // LabelImg's label filter above its box list: only shapes of one label show
  const filterLabel = (label: string) => {
    setLabelFilter(label)
    // A pose's points show with its class
    const posed = new Set(posesOf(shapesRef.current).filter((p) => p.cls === label).flatMap(poseShapes))
    const next = shapesRef.current.map((s, i) => ({ ...s, hidden: label && s.label !== label && !posed.has(i) ? true : undefined }))
    shapesRef.current = next
    setShapesState(next)
    invalidate()
  }

  /**
   * Copy here or Move here, at the end of a right drag; null puts the copy away. The menu
   * closes before a choice runs, so closing puts it away a moment later, when nothing was chosen.
   */
  const ghostDone = (how: 'copy' | 'move' | null) => {
    const g = ghost.current
    ghost.current = null
    setGhostMenu(null)
    invalidate()
    if (!g || !how || readOnlyRef.current) return
    const shifted = (s: Shape) => ({ ...s, points: s.points.map(([x, y]) => [x + g.dx, y + g.dy]) })
    if (how === 'copy') {
      // A pose being placed is kept first, so the copies are numbered after its group_id; it goes at the end, so the copied indexes stay
      endPose()
      const list = shapesRef.current
      const copies = regroup(g.shapes.map((i) => shifted({ ...list[i], hidden: undefined })), list, opts.current.keypoints)
      setShapes([...list, ...copies])
      setSelected(copies.map((_, k) => list.length + k))
      return
    }
    const list = shapesRef.current
    const moving = g.shapes.filter((i) => !list[i].locked)
    if (moving.length < g.shapes.length) setNotice('Locked shapes stay: unlock them first.')
    setShapes(acceptEdited(list.map((s, i) => (moving.includes(i) ? accepted(shifted(s)) : s)), moving, list))
  }
  // LabelImg's colours: the selected shapes' own line or fill colour, or the colour shapes are drawn in
  const pickColor = (kind: 'line' | 'fill' | 'draw') => {
    const first = shapesRef.current[selectedRef.current[0]]
    const value = kind === 'draw'
      ? settings.drawColor ?? (activeClass !== null ? classes[activeClass]?.color : undefined) ?? '#00ff00'
      : ownColor(first?.other[kind === 'line' ? 'line_color' : 'fill_color']) ?? colorOf(first?.label ?? '')
    setColorPrompt({ kind, value })
  }
  /** Saved in LabelMe files as labelme 3 and 4 kept them: [r, g, b, a]. null goes back to the label's colour. */
  const setColor = (kind: 'line' | 'fill' | 'draw', value: string | null) => {
    setColorPrompt(null)
    if (kind === 'draw') return setSettings((s) => ({ ...s, drawColor: value }))
    const which = selectedRef.current
    const key = kind === 'line' ? 'line_color' : 'fill_color'
    const rgba = value ? [1, 3, 5].map((k) => parseInt(value.slice(k, k + 2), 16)).concat(255) : null
    setShapes(shapesRef.current.map((s, i) => {
      if (!which.includes(i)) return s
      const { [key]: _old, ...other } = s.other
      return { ...s, other: rgba ? { ...other, [key]: rgba } : other }
    }))
  }

  const has = () => !!metaRef.current && !readOnlyRef.current
  const sel = () => selectedRef.current.length > 0
  const boxesOnly = holdsBoxesOnly(settings.format)
  /** Tools for shapes other than boxes are off in LabelImg's formats */
  const draws = (m: Mode) => () => has() && (!boxesOnly || BOX_MODES.has(m))
  const commands: Commands = {
    open_next: { label: 'Next image', run: () => go(1), shortcut: 'open_next', enabled: () => images.length > 1 },
    open_prev: { label: 'Previous image', run: () => go(-1), shortcut: 'open_prev', enabled: () => images.length > 1 },
    save: { label: 'Save', run: saveNow, shortcut: 'save', enabled: has },
    save_as: { label: 'Save as…', run: saveAs, shortcut: 'save_as', enabled: has },
    delete_file: { label: 'Delete label file', run: deleteLabelFile, shortcut: 'delete_file', enabled: () => !!images[index]?.labelled },
    auto_save: { label: 'Save automatically', run: () => setSettings((s) => ({ ...s, autoSave: !s.autoSave })), checked: () => settings.autoSave },
    with_image_data: { label: 'Save with image data', run: () => setSettings((s) => ({ ...s, storeData: !s.storeData })), checked: () => settings.storeData },
    reveal: { label: 'Show in folder', run: () => loadedName.current && window.api.reveal(root, loadedName.current), enabled: () => !!loadedName.current },

    create_polygon: { label: 'Create polygons', run: () => setMode('polygon'), shortcut: 'create_polygon', checked: () => mode === 'polygon', enabled: draws('polygon') },
    create_rectangle: { label: 'Create rectangle', run: () => setMode('rectangle'), shortcut: 'create_rectangle', checked: () => mode === 'rectangle', enabled: has },
    create_oriented_rectangle: {
      label: 'Create rotated box', run: () => setMode('oriented_rectangle'), shortcut: 'create_oriented_rectangle', checked: () => mode === 'oriented_rectangle', enabled: draws('oriented_rectangle')
    },
    create_circle: { label: 'Create circle', run: () => setMode('circle'), shortcut: 'create_circle', checked: () => mode === 'circle', enabled: draws('circle') },
    create_line: { label: 'Create line', run: () => setMode('line'), shortcut: 'create_line', checked: () => mode === 'line', enabled: draws('line') },
    create_point: { label: 'Create point', run: () => setMode('point'), shortcut: 'create_point', checked: () => mode === 'point', enabled: draws('point') },
    create_pose: { label: 'Create pose', run: () => setMode('pose'), shortcut: 'create_pose', checked: () => mode === 'pose', enabled: draws('pose') },
    create_linestrip: { label: 'Create line strip', run: () => setMode('linestrip'), shortcut: 'create_linestrip', checked: () => mode === 'linestrip', enabled: draws('linestrip') },
    create_ai_points: { label: 'AI points', run: () => setMode('ai_points'), shortcut: 'create_ai_points_to_shape', checked: () => mode === 'ai_points', enabled: () => has() && !!models },
    create_ai_box: { label: 'AI box', run: () => setMode('ai_box'), shortcut: 'create_ai_box_to_shape', checked: () => mode === 'ai_box', enabled: () => has() && !!models },
    create_ai_pose: {
      label: 'AI pose', run: () => setMode('ai_pose'), shortcut: 'create_ai_pose', checked: () => mode === 'ai_pose', enabled: () => draws('ai_pose')() && poseModels.length > 0
    },
    create_brush: { label: 'Brush', run: () => setMode('brush'), shortcut: 'create_brush', checked: () => mode === 'brush', enabled: draws('brush') },
    create_eraser: { label: 'Eraser', run: () => setMode('eraser'), shortcut: 'create_eraser', checked: () => mode === 'eraser', enabled: draws('eraser') },
    draw_squares: { label: 'Draw squares', run: () => setSettings((s) => ({ ...s, drawSquares: !s.drawSquares })), shortcut: 'draw_squares', checked: () => settings.drawSquares },
    edit_shape: { label: 'Edit polygons', run: () => setMode('edit'), shortcut: 'edit_shape', checked: () => mode === 'edit' },
    edit_label: { label: 'Edit label…', run: () => editLabel(), shortcut: 'edit_label', enabled: () => has() && sel() },
    duplicate_shape: { label: 'Duplicate', run: duplicate, shortcut: 'duplicate_shape', enabled: () => has() && sel() },
    copy_shape: { label: 'Copy', run: copySelected, shortcut: 'copy_shape', enabled: sel },
    paste_shape: { label: 'Paste', run: paste, shortcut: 'paste_shape', enabled: () => has() && clipboard.current.length > 0 },
    delete_shape: { label: 'Delete', run: removeSelected, shortcut: 'delete_shape', enabled: () => has() && sel() },
    merge_shapes: { label: 'Merge shapes', run: merge, shortcut: 'merge_shapes', enabled: () => has() && selectedRef.current.length >= 2 && !boxesOnly },
    undo: {
      label: 'Undo', run: () => (undoLastPoint() || stepHistory(undo, redo)), shortcut: 'undo',
      enabled: () => has() && (undo.current.length > 0 || !!draft.current || !!posing.current || !!reviewBack.current)
    },
    // Off while a pose is placed: what it brings back could hold the group_id the pose has taken
    redo: { label: 'Redo', run: () => !posing.current && stepHistory(redo, undo), shortcut: 'redo', enabled: () => has() && redo.current.length > 0 && !posing.current },
    remove_selected_point: { label: 'Remove point', run: removeHoveredPoint, shortcut: 'remove_selected_point', enabled: has },
    add_multiple_points: { label: 'Add points to edge…', run: addMultiplePoints, shortcut: 'add_multiple_points', enabled: has },
    add_point_to_edge: { label: 'Add point to edge', run: addPointToEdge, enabled: () => has() && selectedRef.current.length === 1 && EDITABLE_POINTS.has(shapesRef.current[selectedRef.current[0]]?.shape_type) },
    select_shape_up: { label: 'Select the shape above', run: () => selectNeighbour(0, -1), shortcut: 'select_shape_up', enabled: () => shapesRef.current.length > 0 },
    select_shape_down: { label: 'Select the shape below', run: () => selectNeighbour(0, 1), shortcut: 'select_shape_down', enabled: () => shapesRef.current.length > 0 },
    select_shape_left: { label: 'Select the shape to the left', run: () => selectNeighbour(-1, 0), shortcut: 'select_shape_left', enabled: () => shapesRef.current.length > 0 },
    select_shape_right: { label: 'Select the shape to the right', run: () => selectNeighbour(1, 0), shortcut: 'select_shape_right', enabled: () => shapesRef.current.length > 0 },
    toggle_lock: {
      label: 'Lock shapes', run: () => {
        const which = selectedRef.current
        const lock = !which.every((i) => shapesRef.current[i]?.locked)
        setShapes(shapesRef.current.map((x, i) => (which.includes(i) ? { ...x, locked: lock || undefined } : x)))
      },
      shortcut: 'lock_shape', enabled: () => has() && sel(), checked: () => sel() && selectedRef.current.every((i) => shapesRef.current[i]?.locked)
    },
    toggle_keypoint_hidden: { label: 'Hidden keypoint', run: toggleHidden, shortcut: 'toggle_keypoint_hidden', enabled: () => has() && !boxesOnly },
    swap_keypoint_sides: { label: 'Swap left and right', run: swapSelectedSides, shortcut: 'swap_keypoint_sides', enabled: () => has() && sel() && !boxesOnly },
    mirror_pose: { label: 'Mirror pose', run: mirrorSelected, shortcut: 'mirror_pose', enabled: () => has() && sel() && !boxesOnly },
    next_keypoint: { label: 'Next keypoint', run: () => stepKeypoint(1), shortcut: 'next_keypoint', enabled: () => has() && !boxesOnly },
    prev_keypoint: { label: 'Previous keypoint', run: () => stepKeypoint(-1), shortcut: 'prev_keypoint', enabled: () => has() && !boxesOnly },
    toggle_difficult: {
      label: 'Difficult', run: toggleDifficult, shortcut: 'toggle_difficult', enabled: () => has() && sel() && keepsDifficult(settings.format),
      checked: () => sel() && selectedRef.current.every((i) => shapesRef.current[i]?.flags.difficult === true)
    },
    verify_image: { label: 'Verify image', run: verifyImage, shortcut: 'verify_image', checked: () => verified, enabled: () => has() && keepsVerified(settings.format) },
    toggle_keep_prev_mode: { label: 'Keep previous annotation', run: () => setSettings((s) => ({ ...s, keepPrev: !s.keepPrev })), shortcut: 'toggle_keep_prev_mode', checked: () => settings.keepPrev },
    copy_previous_shapes: { label: "Copy previous image's shapes", run: copyPrevious, shortcut: 'copy_previous_shapes', enabled: () => has() && visible.findIndex((v) => v.i === index) > 0 },
    single_class_mode: { label: 'Single class mode', run: () => setSettings((s) => ({ ...s, askLabel: !s.askLabel })), shortcut: 'single_class_mode', checked: () => !settings.askLabel },
    open_annotation: { label: 'Open annotation…', run: openAnnotation, shortcut: 'open_annotation', enabled: has },
    delete_image: { label: 'Delete image', run: deleteImage, shortcut: 'delete_image', enabled: () => !!images[index] && !images[index].embedded },
    change_format: { label: 'Next save format', run: () => setFormat(NEXT_FORMAT[settings.format]), shortcut: 'change_format', enabled: () => !images[index]?.embedded },
    display_labels: { label: 'Display labels', run: () => setSettings((s) => ({ ...s, displayLabels: !s.displayLabels })), shortcut: 'display_labels', checked: () => settings.displayLabels },
    advanced_mode: { label: 'Advanced mode', run: () => setSettings((s) => ({ ...s, advanced: !s.advanced })), shortcut: 'advanced_mode', checked: () => settings.advanced },
    box_line_color: { label: 'Box line colour…', run: () => pickColor('draw'), shortcut: 'box_line_color' },
    shape_line_color: { label: 'Shape line colour…', run: () => pickColor('line'), shortcut: 'shape_line_color', enabled: () => has() && sel() },
    shape_fill_color: { label: 'Shape fill colour…', run: () => pickColor('fill'), shortcut: 'shape_fill_color', enabled: () => has() && sel() },

    fill_drawing: { label: 'Fill polygon while drawing', run: () => setSettings((s) => ({ ...s, fillDrawing: !s.fillDrawing })), checked: () => settings.fillDrawing },
    hide_all_shapes: { label: 'Hide shapes', run: () => setShowShapes(false), shortcut: 'hide_all_shapes' },
    show_all_shapes: { label: 'Show shapes', run: () => setShowShapes(true), shortcut: 'show_all_shapes' },
    toggle_all_shapes: { label: 'Show shapes', run: () => setShowShapes((v) => !v), shortcut: 'toggle_all_shapes', checked: () => showShapes },
    zoom_in: { label: 'Zoom in', run: () => zoomBy(config.zoom_increment), shortcut: 'zoom_in', enabled: () => !!metaRef.current },
    zoom_out: { label: 'Zoom out', run: () => zoomBy(1 / config.zoom_increment), shortcut: 'zoom_out', enabled: () => !!metaRef.current },
    zoom_to_original: { label: 'Original size', run: () => { setFit('manual'); setScale(1) }, shortcut: 'zoom_to_original', enabled: () => !!metaRef.current },
    fit_window: { label: 'Fit window', run: () => { setFit('window'); fit('window') }, shortcut: 'fit_window', checked: () => settings.fit === 'window' },
    fit_selection: { label: 'Fit to selection', run: () => { setFit('selection'); fitSelection() }, shortcut: 'fit_selection', checked: () => settings.fit === 'selection' },
    fit_width: { label: 'Fit width', run: () => { setFit('width'); fit('width') }, shortcut: 'fit_width', checked: () => settings.fit === 'width' },
    keep_prev_scale: { label: 'Keep previous scale', run: () => setSettings((s) => ({ ...s, keepPrevScale: !s.keepPrevScale })), checked: () => settings.keepPrevScale },
    brightness_contrast: { label: 'Brightness and contrast…', run: () => setBcOpen(true), enabled: () => !!metaRef.current },
    keep_prev_brightness_contrast: { label: 'Keep previous brightness and contrast', run: () => setSettings((s) => ({ ...s, keepPrevBC: !s.keepPrevBC })), checked: () => settings.keepPrevBC },

    review_accept: { label: 'Accept AI shape', run: acceptSelected, shortcut: 'review_accept' },
    review_reject: { label: 'Reject AI shape', run: rejectSelected, shortcut: 'review_reject' },
    review_skip: { label: 'Skip AI shape', run: skipShape, shortcut: 'review_skip' },
    review_accept_all: { label: 'Accept all here, then next', run: acceptAll, shortcut: 'review_accept_all' },
    order_certainty: { label: 'Least certain first', run: () => setSort((s) => (s === 'certainty' ? 'name' : 'certainty')), checked: () => sort === 'certainty' },
    prelabel_pick: { label: 'Choose a model…', run: pickModel },
    check_labels: { label: 'Check labels', run: checkLabels, enabled: () => images.length > 0 },
    keypoints_coco: { label: "Use COCO's person", run: () => changeKeypoints('coco') },
    keypoints_choose: { label: 'Choose a keypoints file…', run: () => changeKeypoints('choose') },
    keypoints_make: { label: "Make one from the folder's poses", run: () => changeKeypoints('make'), enabled: () => images.length > 0 },
    dataset_stats: { label: 'Statistics…', run: openStats, enabled: () => images.length > 0 },
    ai_models: { label: 'AI models…', run: () => setModelsOpen(true), enabled: () => !!models },
    prelabel_one: { label: 'Pre-label this image', run: prelabelOne, enabled: () => has() && !batch && !busyAi && !boxesOnly },
    prelabel_all: { label: `Pre-label ${toPrelabel.length} images`, run: prelabelAll, enabled: () => !batch && toPrelabel.length > 0 && !boxesOnly },
    pose_everyone: { label: 'Pose every person', run: () => poseBoxes(), enabled: () => has() && poseModels.length > 0 && !busyAi && !boxesOnly },
    brush_smaller: { label: 'Smaller brush', run: () => setBrushSize((s) => Math.max(config.brush.min, s - 4)), shortcut: 'brush_smaller' },
    brush_larger: { label: 'Larger brush', run: () => setBrushSize((s) => Math.min(config.brush.max, s + 4)), shortcut: 'brush_larger' }
  }
  for (const m of models ?? []) {
    if (m.points || m.box) commands[`ai_model:${m.name}`] = { label: m.title, run: () => setSettings((s) => ({ ...s, aiModel: m.name })), checked: () => settings.aiModel === m.name }
    if (m.text) commands[`ai_text_model:${m.name}`] = { label: m.textTitle ?? m.title, run: () => setSettings((s) => ({ ...s, aiTextModel: m.name })), checked: () => settings.aiTextModel === m.name }
    if (m.pose) commands[`pose_model:${m.name}`] = { label: m.title, run: () => setSettings((s) => ({ ...s, poseModel: m.name })), checked: () => poseModel === m.name }
  }
  for (const o of AI_OUTPUTS) {
    commands[`ai_output:${o}`] = { label: AI_OUTPUT_NAMES[o], run: () => setSettings((s) => ({ ...s, aiOutput: o })), checked: () => settings.aiOutput === o, enabled: () => !boxesOnly }
  }
  // The models this folder's runs exported, newest first, and the one chosen if it's another. Newest
  // run, ticked until a model is chosen for the folder, takes whichever run is newest when pre-labelling
  // starts, and choosing it again forgets the folder's choice; a run chosen is ticked, and stays.
  const choices: { path: string; title: string }[] = [...(runModels ?? [])]
  if (prelabelModel?.path === chosenModel && !choices.some((m) => m.path === chosenModel)) choices.push({ path: prelabelModel.path, title: modelTitle(prelabelModel) })
  if (runModels?.length) {
    commands['prelabel_model:newest'] = {
      label: 'Newest run', checked: () => !chosenModel,
      run: () => setSettings((s) => ({ ...s, prelabelModels: Object.fromEntries(Object.entries(s.prelabelModels).filter(([k]) => k !== root)) }))
    }
  }
  for (const m of choices) {
    commands[`prelabel_model:${m.path}`] = {
      label: m.title, run: () => setSettings((s) => ({ ...s, prelabelModels: chooseFor(s.prelabelModels, root, m.path) })), checked: () => chosenModel === m.path
    }
  }
  for (const f of LABEL_FORMATS) {
    commands[`format:${f}`] = { label: FORMAT_NAMES[f], run: () => setFormat(f), checked: () => settings.format === f, enabled: () => !images[index]?.embedded }
  }
  const commandsRef = useRef(commands)
  commandsRef.current = commands
  const handleRef = useRef({ showImage, reloadImages, show })
  handleRef.current = { showImage, reloadImages, show }
  useImperativeHandle(ref, () => ({
    get commands() { return commandsRef.current },
    canLeave: (how = {}) => (how.folder ? canLeaveFolder() : canLeave(how)),
    showImage: (name, shape) => handleRef.current.showImage(name, shape),
    reloadImages: (names) => handleRef.current.reloadImages(names),
    show: (name) => handleRef.current.show(name),
    busy
  }), [canLeave, canLeaveFolder, busy])

  // Keys: the configurable shortcuts, then the fixed ones (classes, Escape, arrows)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (dialog || pointsPrompt || bcOpen || colorPrompt || ghostMenu || document.querySelector('.modal-back, .context')) {
        // Keys let go meanwhile still count, so Space or Shift isn't left held down
        if (e.type === 'keyup') {
          if (e.code === 'Space') { space.current = false; spaceTap.current = null }
          if (e.key === 'Shift') shift.current = false
          if (e.key === 'Control' || e.key === 'Meta') ctrl.current = false
        }
        return
      }
      if (target.closest?.('.menubar, .menu')) return
      if (target.tagName === 'SELECT' || target.tagName === 'TEXTAREA' || (target.tagName === 'INPUT' && (target as HTMLInputElement).type !== 'range' && (target as HTMLInputElement).type !== 'checkbox')) return
      // Tab moves the focus through the docks' controls: only from the canvas, or with nothing focused, does it skip an AI shape
      if (e.key === 'Tab' && target !== document.body && target !== canvasRef.current) return
      // Enter and Space press the Problems dock's buttons and tick its box, as they would anywhere, rather than finish a shape or verify the image.
      // A click on a row leaves the focus here, so Space held down still pans under a drag, which takes the focus to the canvas: the button isn't pressed
      if ((e.key === 'Enter' || e.code === 'Space') && target.closest?.('section.problems') && (target.tagName === 'BUTTON' || target.tagName === 'INPUT')) {
        if (e.code === 'Space') { space.current = e.type === 'keydown'; spaceTap.current = null }
        return
      }
      if (e.code === 'Space') {
        e.preventDefault()
        if (e.type === 'keydown') {
          if (!e.repeat && !space.current) spaceTap.current = Date.now()
          space.current = true
          return
        }
        space.current = false
        const pressed = spaceTap.current
        spaceTap.current = null
        // Held a while, it was meant for panning
        if (pressed === null || Date.now() - pressed > 500 || drag.current || dialog || pointsPrompt || bcOpen || colorPrompt || ghostMenu) return
        // labelme's Space finishes a shape being drawn, and a pose being placed
        if (posing.current) return endPose()
        const drawing = opts.current.mode
        if (draft.current && (drawing === 'polygon' || drawing === 'linestrip' || drawing === 'ai_points')) {
          if (drawing === 'ai_points') finishAi()
          else if (draft.current.points.length >= (drawing === 'polygon' ? 3 : 2)) finishDraft()
          return
        }
        // Space+drag pans; a tap runs the command set to Space, such as LabelImg's Verify Image
        for (const cmd of Object.values(commandsRef.current)) {
          if (cmd.shortcut && shortcutList(config.shortcuts[cmd.shortcut]).some((k) => matches(k, e, mac))) {
            if (!cmd.enabled || cmd.enabled()) cmd.run()
            return
          }
        }
        return
      }
      if (e.key === 'Shift') { shift.current = e.type === 'keydown'; invalidate() }
      if (e.key === 'Control' || e.key === 'Meta') { ctrl.current = e.type === 'keydown'; invalidate() }
      if (e.type !== 'keydown') return
      // Nothing may change the image, tool or shapes under a drag in progress
      if (drag.current && e.key !== 'Escape') return
      const all = commandsRef.current
      // Ctrl+Z takes back the last point while drawing, as in labelme, and the last keypoint placed
      if ((draft.current || posing.current) && shortcutList(config.shortcuts.undo_last_point).some((k) => matches(k, e, mac))) {
        e.preventDefault()
        undoLastPoint()
        return
      }
      if (e.key === 'Backspace' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        if (undoLastPoint() || removeHoveredPoint()) return
        removeSelected()
        return
      }
      // Placing a pose, X skips the keypoint asked for, and Enter or Escape ends the pose, keeping what's placed
      if (posing.current && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.key === 'x' || e.key === 'X') {
          e.preventDefault()
          return answerKeypoint(null)
        }
        if (e.key === 'Enter' || e.key === 'Escape') {
          e.preventDefault()
          return e.key === 'Enter' ? endPose() : stopPose()
        }
      }
      // Ctrl+Shift with next or previous copies this image's shapes over, as in labelme
      const copy = (e.ctrlKey || e.metaKey) && e.shiftKey
      for (const [id, cmd] of Object.entries(all)) {
        if (!cmd.shortcut || id === 'remove_selected_point' && e.key === 'Backspace') continue
        if (shortcutList(config.shortcuts[cmd.shortcut]).some((k) => matches(k, e, mac))) {
          e.preventDefault()
          if (e.repeat && (id.startsWith('review') || id.startsWith('create') || id === 'save')) return
          if (cmd.enabled && !cmd.enabled()) return
          if (id === 'open_next') return go(1, copy)
          if (id === 'open_prev') return go(-1, copy)
          cmd.run()
          return
        }
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const k = e.key
      if (k === 'Enter') {
        e.preventDefault()
        finishDraft()
      } else if (k === 'Escape') {
        draft.current = null
        aiSeq.current++
        setBusyAi('')
        if (drag.current?.kind === 'stroke' || drag.current?.kind === 'two') drag.current = null
        setSelected([])
        invalidate()
      } else if (k.startsWith('Arrow') && selectedRef.current.length && !readOnlyRef.current) {
        e.preventDefault()
        const step = e.shiftKey ? 10 : 1
        nudge(k === 'ArrowLeft' ? -step : k === 'ArrowRight' ? step : 0, k === 'ArrowUp' ? -step : k === 'ArrowDown' ? step : 0)
      } else if (/^[1-9]$/.test(k) && classes[+k - 1]) {
        chooseClass(+k - 1)
      }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKey) }
  })

  // LabelImg's box size in the status bar: the box being drawn, moved, pointed at or selected
  const boxSize = (() => {
    const list = shapesRef.current
    const d = drag.current
    const live = livePoint()
    let s: Shape | null | undefined = null
    if (draft.current && live && (mode === 'rectangle' || mode === 'circle')) s = draftShape(live)
    else if (d?.kind === 'vertex' || d?.kind === 'move') s = list[d.kind === 'vertex' ? d.shape : d.shapes[0]]
    else if (hover.current && mode === 'edit') s = list[hover.current.shape]
    else if (selected.length === 1) s = list[selected[0]]
    if (!s?.points.length || s.shape_type === 'point') return ''
    const [x0, y0, x1, y1] = boxOf(s)
    return `${Math.round(x1 - x0)} × ${Math.round(y1 - y0)}`
  })()
  // A pose's keypoints aren't shapes of a class: a pose shows, and counts, with its class
  const counts = useMemo(() => {
    const out: Record<string, number> = {}
    shapes.forEach((s, i) => !keypointAt.has(i) && (out[s.label] = (out[s.label] ?? 0) + 1))
    return out
  }, [shapes, keypointAt])
  const shapeLabels = Object.keys(counts).sort((a, b) => a.localeCompare(b))
  const aiCount = shapes.filter(isAi).length
  // Review counts an AI pose as one, however many shapes it has
  const aiLeft = useMemo(() => aiCounts(shapes, poses), [shapes, poses])
  const totals = useMemo(() => {
    const out: Record<string, number> = {}
    images.forEach((i) => Object.entries(i.classes).forEach(([k, v]) => (out[k] = (out[k] ?? 0) + v)))
    return out
  }, [images])
  const largest = Math.max(1, ...classes.map((c) => totals[c.name] ?? 0))
  const tally = (test: (i: ImageItem) => boolean) => images.filter(test).length
  const problemCount = check ? tally((i) => problemImages.has(i.name)) : 0
  const doneShare = images.length ? tally(FILTERS[3].test) / images.length : 0
  const reviewShare = images.length ? tally(FILTERS[2].test) / images.length : 0
  // The step of review selected, an AI shape or an AI pose, and how many there are
  const steps = useMemo(() => reviewItems(shapes, poses), [shapes, poses])
  const current = stepOf(steps, selected)
  const flagKeys = Object.keys(imageFlags)
  const tool = (m: Mode, icon: React.ReactNode, title: string, cmd: string) => {
    const keys = commands[cmd]?.shortcut ? formatKeys(config.shortcuts[commands[cmd].shortcut!], mac) : ''
    // Greyed out in LabelImg's formats: the tip says why
    const why = boxesOnly && !BOX_MODES.has(m) ? `: ${FORMAT_NAMES[settings.format]} keeps boxes only. Save as LabelMe JSON to use it.` : ''
    return (
      <button className={mode === m ? 'on' : ''} onClick={() => setMode(m)} disabled={(!!readOnly && m !== 'edit') || (boxesOnly && !BOX_MODES.has(m))}
        title={`${title}${keys ? ` (${keys})` : ''}${why}`} aria-label={title}>{icon}</button>
    )
  }
  const modeHint: Record<Mode, string> = {
    edit: 'Click to select, Ctrl+click for more · drag to move · Alt+click an edge adds a point, a point removes it',
    polygon: 'Click to add points · click the first point, Enter or double-click to close',
    rectangle: 'Drag, or click two corners · Shift or Ctrl for a square',
    oriented_rectangle: 'Click two corners along one side, then one more for its width',
    circle: 'Click the centre, then a point on the edge',
    line: 'Click both ends',
    point: 'Click to place a point',
    linestrip: 'Click to add points · Ctrl+click, Enter or double-click to finish',
    pose: "Drag a box, or click a box to go on with its pose, then click each keypoint",
    ai_points: 'Click the object · Shift+click marks background · Enter, Space or Ctrl+click to finish',
    ai_box: 'Drag a box around the object',
    ai_pose: "Drag a box around a person, or click a person's box",
    brush: 'Paint the selected class · [ and ] change the size',
    eraser: 'Erase from the selected shape, or anything it touches'
  }

  // The Shapes list: each pose one row, opened by hand or while some of its points are selected without its box
  const opened = new Set(poses.filter((p) => (p.box === null || !selected.includes(p.box)) && poseShapes(p).some((i) => i !== p.box && selected.includes(i))).map((p) => p.group))
  const rows = shapeRows(shapes.length, poses, (group) => poseOpen.get(group) ?? opened.has(group))
  const waiting = poseStep ? posing.current : null
  /** Shapes shown or hidden from the Shapes list: undone like any change; hidden shapes are never saved, so there's nothing to save */
  const showOnly = (which: number[], show: boolean) => {
    const next = shapesRef.current.map((q, k) => (which.includes(k) ? { ...q, hidden: !show || undefined } : q))
    undo.current.push(shapesRef.current)
    if (undo.current.length > config.canvas.num_backups) undo.current.shift()
    redo.current = []
    shapesRef.current = next
    setShapesState(next)
    invalidate()
  }
  /** A click on a row: alone, with Ctrl added or taken away, with Shift the rows between by index */
  const pickRows = (e: React.MouseEvent, which: number[]) => {
    // A pose being placed ends first, at the end of the list, so the row clicked is what's selected, not the pose ended
    endPose()
    const at = Math.min(...which)
    if (e.ctrlKey || e.metaKey) setSelected(which.every((i) => selected.includes(i)) ? selected.filter((k) => !which.includes(k)) : [...selected, ...which.filter((k) => !selected.includes(k))])
    else if (e.shiftKey && selected.length) {
      const a = Math.min(selected[selected.length - 1], at), b = Math.max(selected[selected.length - 1], at)
      const range = Array.from({ length: b - a + 1 }, (_, k) => a + k)
      setSelected([...range, ...which.filter((k) => !range.includes(k))])
    } else setSelected(which)
    if (mode !== 'edit') setMode('edit')
  }
  const shapeRow = (i: number, pose?: PoseGroup, keypoint?: number) => {
    const s = shapes[i]
    const point = pose && i !== pose.box
    const side = point ? sidesOf(keypoints.schema, pose.cls)[keypoint ?? (keypoints.schema.classes[pose.cls] ?? []).indexOf(s.label)] ?? 'centre' : null
    return (
      <div key={i} role="option" aria-selected={selected.includes(i)} draggable={!readOnly}
        className={`shape ${pose ? 'in-pose' : ''} ${selected.includes(i) ? 'on' : ''} ${dragOver === i ? 'drop' : ''} ${s.hidden ? 'off' : ''}`}
        onClick={(e) => pickRows(e, [i])}
        onDoubleClick={() => { setSelected([i]); editLabel([i]) }}
        onDragStart={(e) => e.dataTransfer.setData('text/x-shape', String(i))}
        onDragOver={(e) => { if (e.dataTransfer.types.includes('text/x-shape')) { e.preventDefault(); setDragOver(i) } }}
        onDragLeave={() => setDragOver(null)}
        onDrop={(e) => {
          setDragOver(null)
          const from = Number(e.dataTransfer.getData('text/x-shape'))
          if (Number.isNaN(from) || from === i) return
          // Order matters: later shapes are drawn, and trained, on top
          const next = [...shapesRef.current]
          const [moved] = next.splice(from, 1)
          next.splice(i, 0, moved)
          setShapes(next)
          setSelected([i])
        }}>
        <input type="checkbox" checked={!s.hidden} aria-label={`Show ${s.label}`} onClick={(e) => e.stopPropagation()}
          onChange={(e) => showOnly(selectedRef.current.includes(i) ? selectedRef.current : [i], e.target.checked)} />
        <i style={{ background: side ? SIDE_COLORS[side] : colorOf(s.label) }} />
        <span className="name">{s.label}{s.group_id !== null && !point ? ` (${s.group_id})` : ''}
          {Object.entries(s.flags).some(([k, v]) => v && k !== 'ai' && k !== 'difficult') && (
            <span className="muted"> [{Object.entries(s.flags).filter(([k, v]) => v && k !== 'ai' && k !== 'difficult').map(([k]) => k).join(', ')}]</span>
          )}
        </span>
        {s.locked && <Lock size={12} className="muted" aria-label="Locked" />}
        {/* A keypoint marked difficult is hidden: COCO's visibility 1 */}
        {point ? isHidden(s) && <span className="tag difficult">hidden</span> : s.flags.difficult === true && <span className="tag difficult">difficult</span>}
        {isAi(s) && <span className="tag">AI{scoreOf(s) !== undefined ? ` ${percent(scoreOf(s)!)}` : ''}</span>}
        <span className="muted">{keypoint !== undefined ? keypoint + 1 : describe(s)}</span>
      </div>
    )
  }
  const poseRow = (p: PoseGroup, open: boolean) => {
    const members = poseShapes(p)
    const placed = p.points.filter((i) => i !== null).length
    const ai = members.some((i) => isAi(shapes[i]))
    const score = ai ? scoreOfPose(shapes, p) : undefined
    const hidden = members.every((i) => shapes[i].hidden)
    return (
      <div key={`pose-${p.group}`} role="option" aria-selected={members.every((i) => selected.includes(i))} aria-expanded={open}
        className={`pose-row ${members.some((i) => selected.includes(i)) ? 'on' : ''} ${hidden ? 'off' : ''}`}
        onClick={(e) => pickRows(e, members)} onDoubleClick={() => { if (p.box !== null) { setSelected(members); editLabel([p.box]) } }}>
        <button className="fold" aria-label={open ? 'Fold' : 'Open'} title={open ? 'Fold this pose' : 'List its keypoints'}
          onClick={(e) => { e.stopPropagation(); setPoseOpen((was) => new Map(was).set(p.group, !open)) }}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        <input type="checkbox" checked={!hidden} aria-label={`Show ${p.cls} (${p.group})`} onClick={(e) => e.stopPropagation()}
          onChange={(e) => showOnly(members, e.target.checked)} />
        <i style={{ background: colorOf(p.cls) }} />
        {/* A long label gives way, never the count */}
        <span className="name">{p.cls} ({p.group})</span>
        <span className="pose-count"> · {placed}/{p.points.length}</span>
        {ai && <span className="tag">AI{score !== undefined ? ` ${percent(score)}` : ''}</span>}
      </div>
    )
  }
  const missingRow = (p: PoseGroup, k: number) => (
    <div key={`missing-${p.group}-${k}`} role="option" aria-selected={false}
      className={`pose-missing ${waiting?.group === p.group && waiting.asked.length === 1 && waiting.asked[0] === k ? 'on' : ''}`}
      title="Not placed: click here, then on the image" onClick={() => { if (mode !== 'edit') setMode('edit'); placeMissing(p, k) }}>
      <i style={{ borderColor: SIDE_COLORS[sidesOf(keypoints.schema, p.cls)[k] ?? 'centre'] }} />
      <span className="name">{(keypoints.schema.classes[p.cls] ?? [])[k]}</span>
      <span className="muted">{k + 1}</span>
    </div>
  )

  return (
    <div className="labeler" onDragOver={(e) => e.preventDefault()}>
      {settings.docks.files && (
        <aside className="panel left" aria-label="Files">
          <input className="search" placeholder={`Search ${images.length} images`} value={query} aria-label="Search files"
            onChange={(e) => setQuery(e.target.value)} />
          <div className="progress" data-testid="progress">
            <i className="done" style={{ width: `${doneShare * 100}%` }} />
            <i className="review" style={{ width: `${reviewShare * 100}%` }} />
          </div>
          <div className="seg small filters">
            {FILTERS.map((f) => {
              const n = tally(f.test)
              return (
                <button key={f.key} className={filter === f.key ? 'on' : ''} onClick={() => setFilter(f.key)} title={`${f.title}${n > 9999 ? `: ${n.toLocaleString('en-GB')}` : ''}`}>
                  <b>{few(n)}</b><span>{f.name}</span>
                </button>
              )
            })}
            <button className={filter === 'problems' ? 'on' : ''} onClick={() => setFilter('problems')} disabled={!check}
              title={check ? `An error or a warning in the last check${problemCount > 9999 ? `: ${problemCount.toLocaleString('en-GB')}` : ''}`
                : 'Check the labels first: Dataset > Check labels'}>
              <b>{check ? few(problemCount) : '–'}</b><span>Problems</span>
            </button>
          </div>
          <label className="sort">Order
            <select value={sort} onChange={(e) => { setSort(e.target.value as typeof sort); e.target.blur() }}>
              <option value="name">Name</option>
              <option value="certainty">Least certain first</option>
            </select>
          </label>
          <div className="files" role="listbox" aria-label="Images">
            {visible.map(({ image, i }) => (
              <button key={image.name} role="option" aria-selected={i === index} className={`file ${i === index ? 'on' : ''}`} onClick={() => goTo(i)} title={image.broken ? `${image.name}: ${image.broken}` : image.name}>
                <i className={image.broken ? 'dot broken' : image.ai ? 'dot ai' : image.labelled ? 'dot done' : 'dot'}
                  title={image.broken ? "Its label file can't be read" : image.ai ? 'AI shapes to review' : image.labelled ? 'Labelled' : 'Not labelled yet'} />
                <span className="name">{image.name}</span>
                {image.verified && <BadgeCheck size={13} className="verified" aria-label="Verified" />}
                {image.ai > 0 && image.score !== null && <span className="certainty">{percent(image.score)}</span>}
                {image.labelled && !image.ai && <span className="muted">{image.count}</span>}
              </button>
            ))}
            {!visible.length && <p className="muted pad">No images match.</p>}
          </div>
          {batch ? (
            <button className="ghost batch" onClick={() => window.api.prelabelStop()}>
              <i style={{ width: `${(batch.done / Math.max(1, batch.total)) * 100}%` }} />
              <X size={15} /> Stop · {batch.done} / {batch.total}
            </button>
          ) : (
            <button className="ghost batch" onClick={prelabelAll} disabled={!toPrelabel.length || !!busyAi || boxesOnly}
              title={prelabelName ? `Run ${prelabelName} on every image with no label file or only AI shapes` : 'Choose a segmentation model (.onnx), then pre-label'}>
              <Sparkles size={15} /> Pre-label {toPrelabel.length} {toPrelabel.length === 1 ? 'image' : 'images'}
            </button>
          )}
          <div className="annotation-folder" data-testid="annotation-folder">
            <span title={data.outputDir ?? undefined}>
              {images.filter((x) => x.labelled).length} of {images.length} annotated · {data.outputDir ? `in ${data.outputDir.split(/[\\/]/).filter(Boolean).pop()}` : 'next to images'}
            </span>
            <button className="link" onClick={onChangeOutput}>Change…</button>
            {data.outputDir && <button className="link" onClick={onNextToImages}>Next to images</button>}
          </div>
        </aside>
      )}

      <main className="stage">
        <div className="toolbar" role="toolbar" aria-label="Tools">
          <div className="seg">
            {tool('edit', <MousePointer2 size={16} />, 'Edit', 'edit_shape')}
            {tool('polygon', <Hexagon size={16} />, 'Polygon', 'create_polygon')}
            {tool('rectangle', <Square size={16} />, 'Rectangle', 'create_rectangle')}
            {tool('oriented_rectangle', <RectangleHorizontal size={16} className="turned" />, 'Rotated box', 'create_oriented_rectangle')}
            {tool('circle', <Circle size={16} />, 'Circle', 'create_circle')}
            {tool('line', <Slash size={16} />, 'Line', 'create_line')}
            {tool('linestrip', <Spline size={16} />, 'Line strip', 'create_linestrip')}
            {tool('point', <Dot size={16} />, 'Point', 'create_point')}
            {tool('pose', <PersonStanding size={16} />, 'Pose', 'create_pose')}
          </div>
          <div className="seg">
            {tool('brush', <Paintbrush size={16} />, 'Brush: paint a class, joining shapes it touches', 'create_brush')}
            {tool('eraser', <Eraser size={16} />, 'Eraser: cut from the selected shape, or anything it touches', 'create_eraser')}
          </div>
          {models && (
            <div className="seg">
              {tool('ai_points', <Wand2 size={16} />, 'AI points', 'create_ai_points')}
              {tool('ai_box', <ScanLine size={16} />, 'AI box', 'create_ai_box')}
              {poseModels.length > 0 && tool('ai_pose', <Accessibility size={16} />, 'AI pose', 'create_ai_pose')}
            </div>
          )}
          {AI_MODES.has(mode) && models && (
            <>
              <label className="slider">Model
                <select value={settings.aiModel} aria-label="AI model" onChange={(e) => { setSettings((s) => ({ ...s, aiModel: e.target.value })); e.target.blur() }}>
                  {models.filter((m) => (mode === 'ai_points' ? m.points : m.box)).map((m) => <option key={m.name} value={m.name}>{m.title}{m.downloaded ? '' : ' (download)'}</option>)}
                </select>
              </label>
              <label className="slider">Makes
                <select value={boxesOnly ? 'rectangle' : settings.aiOutput} disabled={boxesOnly} aria-label="AI output"
                  title={boxesOnly ? `${FORMAT_NAMES[settings.format]} keeps boxes` : 'What the AI makes of what it finds'}
                  onChange={(e) => { setSettings((s) => ({ ...s, aiOutput: e.target.value as AiOutput })); e.target.blur() }}>
                  {AI_OUTPUTS.map((o) => <option key={o} value={o}>{AI_OUTPUT_NAMES[o]}</option>)}
                </select>
              </label>
            </>
          )}
          {mode === 'ai_pose' && poseModels.length > 0 && (
            <>
              <label className="slider">Model
                <select value={poseModel} aria-label="AI pose model" onChange={(e) => { setSettings((s) => ({ ...s, poseModel: e.target.value })); e.target.blur() }}>
                  {poseModels.map((m) => <option key={m.name} value={m.name}>{m.title}{m.downloaded ? '' : ' (download)'}</option>)}
                </select>
              </label>
              <label className="slider" title="Keypoints scored below this are left out">Score
                <input type="number" min={0} max={1} step={0.05} value={poseScore} aria-label="Keypoint score threshold"
                  onChange={(e) => setPoseScore(Math.max(0, Math.min(1, +e.target.value)))} />
              </label>
              <button className="ghost" onClick={() => poseBoxes()} disabled={!!busyAi || !!readOnly}
                title="Each person on this image with no keypoints yet, a group's shapes as one; on an image with no one drawn, the people YOLO-World finds">Pose every person</button>
            </>
          )}
          <button className="icon" onClick={() => commands.undo.run()} title="Undo (Ctrl+Z)" aria-label="Undo"><Undo2 size={16} /></button>
          <button className="icon" onClick={() => commands.redo.run()} title="Redo (Ctrl+Shift+Z)" aria-label="Redo"><Redo2 size={16} /></button>
          <button className="icon" onClick={() => commands.fit_window.run()} title="Fit window (0)" aria-label="Fit window"><Maximize size={16} /></button>
          <button className={`icon${settings.fit === 'selection' ? ' on' : ''}`} onClick={() => commands.fit_selection.run()} disabled={!selected.length && settings.fit !== 'selection'}
            title="Fit to selection (Ctrl+Shift+F)" aria-label="Fit to selection"><Focus size={16} /></button>
          <button className="icon" onClick={() => setShowShapes(!showShapes)} title="Show or hide shapes (F)" aria-label="Show or hide shapes">
            {showShapes ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <label className="slider">Fill
            <input type="range" min={0} max={0.8} step={0.05} value={settings.fill} aria-label="Fill"
              onChange={(e) => setSettings((s) => ({ ...s, fill: +e.target.value }))} />
          </label>
          {(mode === 'brush' || mode === 'eraser') && (
            <label className="slider" title="[ and ] change it">Size
              <input type="range" min={config.brush.min} max={config.brush.max} step={4} value={brushSize} aria-label="Brush size" onChange={(e) => setBrushSize(+e.target.value)} />
            </label>
          )}
          <div className="spacer" />
          <span className="zoom" title="Zoom">{Math.round(zoom * 1000) / 10}%</span>
          <button className="ghost" onClick={prelabelOne} disabled={!!busyAi || !!batch || !!readOnly || boxesOnly}
            title={prelabelName ? `Pre-label with ${prelabelName}` : 'Choose a segmentation model (.onnx), then pre-label this image'}>
            <Sparkles size={15} /> {busyAi === 'Predicting…' ? 'Predicting…' : 'Pre-label'}
          </button>
        </div>
        {AI_MODES.has(mode) && models && (
          <form className="toolbar prompt" aria-label="AI text prompt" onSubmit={(e) => { e.preventDefault(); runText() }}>
            <Type size={15} aria-hidden />
            <input type="text" value={aiPrompt} placeholder="Find by words, such as dog,cat,bird" aria-label="Words to find" onChange={(e) => setAiPrompt(e.target.value)} />
            <select value={settings.aiTextModel} aria-label="Text prompt model" onChange={(e) => { setSettings((s) => ({ ...s, aiTextModel: e.target.value })); e.target.blur() }}>
              {models.filter((m) => m.text).map((m) => <option key={m.name} value={m.name}>{m.textTitle ?? m.title}{m.downloaded ? '' : ' (download)'}</option>)}
            </select>
            <label className="slider" title="Only finds scored at least this">Score
              <input type="number" min={0} max={1} step={0.05} value={aiScore} aria-label="Score threshold" onChange={(e) => setAiScore(Math.max(0, Math.min(1, +e.target.value)))} />
            </label>
            <label className="slider" title="Finds overlapping more than this are one object">IoU
              <input type="number" min={0} max={1} step={0.05} value={aiIou} aria-label="IoU threshold" onChange={(e) => setAiIou(Math.max(0, Math.min(1, +e.target.value)))} />
            </label>
            <button className="ghost" disabled={!!busyAi || !!readOnly}>Find</button>
          </form>
        )}
        <div className="canvas-wrap">
          <canvas ref={canvasRef} data-testid="canvas" className={`tool-${mode}`} tabIndex={0} onWheel={onWheel}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
            onPointerLeave={() => {
              overCanvas.current = false
              // Nothing is under the pointer off the canvas: a keypoint's name shown on hover goes
              if (!drag.current) hover.current = null
              invalidate()
            }}
            onDoubleClick={onDoubleClick} onContextMenu={(e) => e.preventDefault()} />
          {loading && <div className="loading">Preparing image…</div>}
          {(notice || busyAi || download) && (
            <div className="loading notice">
              {download ? `Downloading ${download.name}: ${Math.round((download.done / Math.max(1, download.total)) * 100)}%` : busyAi || notice}
            </div>
          )}
          {readOnly && <div className="readonly" role="alert">This image's label file can't be read, so it's shown without labels and won't be changed.</div>}
          {aiCount > 0 && (
            <div className="review-bar" data-testid="review-bar" onMouseDown={(e) => e.preventDefault()}>
              <span>
                {aiLeft.poses > 0 && <><b>{aiLeft.poses}</b> AI {aiLeft.poses === 1 ? 'pose' : 'poses'}</>}
                {aiLeft.poses > 0 && aiLeft.shapes > 0 && ' and '}
                {aiLeft.shapes > 0 && <><b>{aiLeft.shapes}</b> AI {aiLeft.shapes === 1 ? 'shape' : 'shapes'}</>}
                {current && current.score !== undefined && <> · this one <b className="certainty">{percent(current.score)}</b></>}
              </span>
              <button className="ghost" onClick={acceptSelected}><Check size={15} /> {current ? 'Accept' : 'Review'} <kbd>W</kbd></button>
              <button className="ghost" onClick={rejectSelected} disabled={!current}><X size={15} /> Reject <kbd>S</kbd></button>
              <button className="ghost" onClick={skipShape} disabled={steps.length < 2 && !!current}><SkipForward size={15} /> Skip <kbd>Tab</kbd></button>
              <button className="ghost" onClick={acceptAll}><CheckCheck size={15} /> Accept all <kbd>⇧W</kbd></button>
            </div>
          )}
          {!images.length && <div className="loading">This folder has no images.</div>}
        </div>
        <footer className="status">
          <span>{images[index]?.name}</span>
          {images.length > 0 && <span className="muted" data-testid="counter">{index + 1} / {images.length}</span>}
          {meta && <span>{meta.width} × {meta.height}</span>}
          {cursor && meta && <span>{Math.round(cursor[0])}, {Math.round(cursor[1])}</span>}
          {boxSize && <span data-testid="box-size">box {boxSize}</span>}
          <span className="muted">{MODE_NAMES[mode]}</span>
          {meta && (
            <button className={`verify ${verified ? 'verified' : 'muted'}`} onClick={verifyImage} disabled={!!readOnly || !keepsVerified(settings.format)}
              title={keepsVerified(settings.format) ? 'Verify image: mark it as checked (Space)' : "YOLO files can't keep a verified mark"}
              aria-label="Verify image" aria-pressed={verified}>
              <BadgeCheck size={13} /> {verified ? 'Verified' : 'Verify'}
            </button>
          )}
          <div className="spacer" />
          <span data-testid="save-state">{readOnly ? 'Read only' : saved === 'saved' ? 'Saved' : saved === 'saving' ? 'Saving…' : settings.autoSave ? 'Saving…' : 'Not saved'}</span>
          {poseStep && <span data-testid="pose-step">{poseStep}</span>}
          <span className="muted hint">{poseStep ? 'Click to place · Shift+click: hidden · X skips · Backspace goes back · Enter ends' : modeHint[mode]}</span>
        </footer>
      </main>

      {(settings.docks.labels || settings.docks.flags || settings.docks.shapes || settings.docks.problems) && (
        <aside className="panel right">
          {settings.docks.labels && (
            <section aria-label="Labels">
              <h3>Labels <span title="Shapes on this image · accepted shapes in the whole folder">here · all</span></h3>
              <div className="classes">
                {classes.map((c, i) => (
                  <div key={c.name} className={`class ${i === activeClass ? 'on' : ''}`} role="button" tabIndex={0}
                    onClick={() => (i === activeClass && !selected.length ? setActive(null) : chooseClass(i))}
                    title={i === activeClass ? 'New shapes get this label. Click again to be asked each time.' : 'Label new shapes with this'}>
                    <label className="swatch" style={{ background: c.color }} onClick={(e) => e.stopPropagation()} title="Change colour">
                      <input type="color" value={c.color} onChange={(e) => recolor(c.name, e.target.value)} aria-label={`Colour of ${c.name}`} />
                    </label>
                    <span className="name">{c.name}</span>
                    <span className="muted">{counts[c.name] ?? 0} · {totals[c.name] ?? 0}</span>{i < 9 && <kbd>{i + 1}</kbd>}
                    <em style={{ width: `${((totals[c.name] ?? 0) / largest) * 100}%`, background: c.color }} />
                  </div>
                ))}
                {!classes.length && <p className="muted pad">No labels yet. Add one, or draw a shape and name it.</p>}
              </div>
              <div className="add">
                <input placeholder="New label" value={newClass} aria-label="New label" onChange={(e) => setNewClass(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addClass()} />
                <button className="icon" onClick={addClass} title="Add label" aria-label="Add label"><Plus size={16} /></button>
              </div>
            </section>
          )}
          {settings.docks.flags && flagKeys.length > 0 && (
            <section aria-label="Flags">
              <h3>Flags <span className="muted">this image</span></h3>
              <div className="flag-list">
                {flagKeys.map((k) => (
                  <label key={k} className="check">
                    <input type="checkbox" checked={imageFlags[k]} disabled={!!readOnly} onChange={(e) => setFlagsValue({ ...flagsRef.current, [k]: e.target.checked })} /> {k}
                  </label>
                ))}
              </div>
            </section>
          )}
          {settings.docks.shapes && (
            <section className="grow" aria-label="Shapes">
              <h3>Shapes
                {shapeLabels.length > 1 ? (
                  <select className="label-filter" value={labelFilter} aria-label="Show shapes of"
                    onChange={(e) => { filterLabel(e.target.value); e.target.blur() }}>
                    <option value="">All {shapes.length}</option>
                    {shapeLabels.map((l) => <option key={l} value={l}>{l} {counts[l]}</option>)}
                  </select>
                ) : <span className="muted">{shapes.length}</span>}
              </h3>
              <div className="shapes" role="listbox" aria-multiselectable>
                {rows.map((row) => {
                  if (row.kind === 'pose') return poseRow(row.pose, row.open)
                  if (row.kind === 'missing') return missingRow(row.pose, row.keypoint)
                  return shapeRow(row.index, row.pose, row.keypoint)
                })}
                {!shapes.length && <p className="muted pad">Nothing yet. Click on the image to start a polygon.</p>}
              </div>
              {selected.length > 0 && !readOnly && (
                <div className="row selection">
                  <label className="check" title={keepsDifficult(settings.format) ? 'Hard to recognise without context, as Pascal VOC marks it' : `${FORMAT_NAMES[settings.format]} doesn't keep it`}>
                    <input type="checkbox" checked={!!commands.toggle_difficult.checked?.()} disabled={!keepsDifficult(settings.format)} onChange={toggleDifficult} /> Difficult
                  </label>
                  <button className="ghost danger" onClick={removeSelected}><Trash2 size={15} /> Delete {selected.length > 1 ? `${selected.length} shapes` : 'shape'}</button>
                </div>
              )}
            </section>
          )}
          {settings.docks.problems && (
            <Problems root={root} current={images[index]?.name ?? null} summary={check} running={checking} stale={stale.current}
              unsaved={saved !== 'saved' && !readOnly && loadedName.current ? loadedName.current : null}
              folded={problemsFolded} onFold={() => setProblemsFolded((f) => !f)} onCheck={checkLabels} onStop={() => window.api.checkStop(root)}
              onClose={() => setSettings((s) => ({ ...s, docks: { ...s.docks, problems: false } }))}
              onShow={(row) => row.image && showImage(row.image, row.shape)} onFix={fixProblem}
              onRename={(row) => row.label !== undefined && openRename(row.label, row.to ?? '')}
              onReveal={(file) => window.api.checkReveal(root, file)} />
          )}
        </aside>
      )}

      {dialog && (
        <LabelDialog request={dialog} config={config} labels={classes.map((c) => c.name)} keypoints={keypointLabels}
          onDone={(choice) => {
            const { resolve } = dialog
            // A keypoint's name isn't a label of the list: it's checked against the keypoints file
            if (choice && config.validate_label === 'exact' && !dialog.locked?.text && !(config.labels ?? []).includes(choice.text) && !keypointLabels.has(choice.text)) {
              onError(`Invalid label “${choice.text}”: only labels from the list are allowed.`)
              choice = null
            }
            setDialog(null)
            resolve(choice)
            canvasRef.current?.focus()
          }} />
      )}
      {pointsPrompt && (
        <Modal label="Add points to edge" className="small" onClose={() => setPointsPrompt(null)} onSubmit={() => {
          const n = Math.max(1, Math.min(100, Math.round(pointsPrompt.count)))
          const s = shapesRef.current[pointsPrompt.shape]
          if (s) setShapes(acceptEdited(shapesRef.current.map((q, i) => (i === pointsPrompt.shape ? addPointsToEdge(q, pointsPrompt.edge, n) : q)), [pointsPrompt.shape]))
          setPointsPrompt(null)
        }}>
          <label>Points to add along this edge
            <input type="number" min={1} max={100} autoFocus value={pointsPrompt.count} onChange={(e) => setPointsPrompt({ ...pointsPrompt, count: +e.target.value })} />
          </label>
          <div className="actions">
            <button type="button" className="ghost" onClick={() => setPointsPrompt(null)}>Cancel</button>
            <button className="primary">Add</button>
          </div>
        </Modal>
      )}
      {modelsOpen && models && (
        <Models models={models} progress={download} onClose={() => setModelsOpen(false)} onLink={(url) => window.api.openLink(url)}
          onDownload={async (name) => { await ensureModel(name) }}
          onCancel={(name) => window.api.aiCancel(name)}
          onRemove={async (name) => {
            const m = models.find((x) => x.name === name)!
            const ok = await window.api.ask(`Delete ${m.title}?`, 'Its files are removed from ~/.cache/osam, which labelme uses too. It can be downloaded again.', ['Delete', 'Cancel'])
            if (ok !== 0) return
            try {
              await window.api.aiRemove(name)
            } catch (e) {
              onError(`${m.title} couldn't be deleted: ${plainError(e)}`)
            }
            setModels(await window.api.aiModels())
          }} />
      )}
      {bcOpen && (
        <Modal label="Brightness and contrast" className="small" onClose={() => setBcOpen(false)}>
          {(['brightness', 'contrast'] as const).map((k) => (
            <label key={k} className="slider wide">{k[0].toUpperCase() + k.slice(1)}
              <input type="range" min={0} max={3} step={0.02} value={bc[k]} aria-label={k} onChange={(e) => setBc({ ...bc, [k]: +e.target.value })} />
              <span className="muted">{Math.round(bc[k] * 100)}%</span>
            </label>
          ))}
          <div className="actions">
            <button className="ghost" onClick={() => setBc({ brightness: 1, contrast: 1 })}>Reset</button>
            <button className="primary" onClick={() => setBcOpen(false)}>Done</button>
          </div>
        </Modal>
      )}
      {colorPrompt && (
        <Modal label={COLOR_TITLES[colorPrompt.kind]} className="small" onClose={() => setColorPrompt(null)} onSubmit={() => setColor(colorPrompt.kind, colorPrompt.value)}>
          <label className="slider wide">{COLOR_TITLES[colorPrompt.kind]}
            <input type="color" value={colorPrompt.value} aria-label="Colour" autoFocus onChange={(e) => setColorPrompt({ ...colorPrompt, value: e.target.value })} />
          </label>
          <div className="actions">
            <button type="button" className="ghost" onClick={() => setColor(colorPrompt.kind, null)}>Label's colour</button>
            <button type="button" className="ghost" onClick={() => setColorPrompt(null)}>Cancel</button>
            <button className="primary">OK</button>
          </div>
        </Modal>
      )}
      {statsOpen && (
        <Statistics root={root} prefer={settings.format} labels={listedLabels} colorOf={colorOf} version={labelsChanged}
          onRename={(label) => openRename(label)} onError={onError} onClose={() => setStatsOpen(false)} />
      )}
      {renaming && (
        <RenameDialog root={root} from={renaming.from} to={renaming.to} prefer={settings.format} labels={classes.map((c) => c.name)} totals={totals}
          allowed={config.validate_label === 'exact' ? [...(config.labels ?? []), ...keypointLabels] : null} canLeave={() => canLeave({ stay: true })}
          onClose={(done) => (done ? renamed(renaming.from, done) : setRenaming(null))} />
      )}
      {ghostMenu && (
        <ContextMenu at={ghostMenu} items={['copy_here', 'move_here']} config={config} mac={mac}
          commands={{ copy_here: { label: 'Copy here', run: () => ghostDone('copy') }, move_here: { label: 'Move here', run: () => ghostDone('move') } }}
          onDone={() => setTimeout(() => ghost.current && ghostDone(null))} />
      )}
    </div>
  )
})

export default Labeler
