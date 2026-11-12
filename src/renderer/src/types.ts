import type { ExportOptions, ExportProgress, ExportResult, KeypointsSummary, Preflight } from '../../main/export'
import type { AiOutput, Config } from '../../shared/config'
import type { Issue } from '../../shared/dataset'
import type { LabelFormat } from '../../shared/formats'
import type { LabelDoc, Shape } from '../../shared/labelfile'
import type { Summary } from '../../shared/labels'
import type {
  CheckRequest, CheckSummary, ProblemPage, ProblemQuery, ProblemRow, Progress, RenameDone, RenamePlanView, StatsRequest, StatsView, Stopped
} from '../../main/checking'
import type { ImportPolicy, ImportPreview, ImportProgress, ImportReport, ImportRequest, IssueGroup } from '../../main/import'

export type { Config, LabelDoc, LabelFormat, Shape }
// Dataset > Check labels, Statistics and renames: one definition, in the main process
export type { CheckRequest, CheckSummary, ProblemPage, ProblemQuery, ProblemRow, Progress, RenameDone, RenamePlanView, StatsRequest, StatsView, Stopped }
export type { ImportPolicy, ImportPreview, ImportProgress, ImportReport, ImportRequest, IssueGroup }

export type LabelClass = { name: string; color: string }
export type ProjectInfo = { name: string; classes: LabelClass[] }
export type ImageItem = Summary & { name: string; broken?: string; embedded?: boolean; verified?: boolean }
export type OpenProject = {
  root: string
  outputDir: string | null
  project: ProjectInfo
  hasProjectFile: boolean
  images: ImageItem[]
  select: string | null
}
export type OpenImage = {
  width: number; height: number; file: string; ms: number
  pyramid: { dir: string; maxLevel: number; tile: number } | null
}
export type Start = {
  config: Config | null
  configError: string
  configFile: string | null
  warnings: string[]
  version: string
  platform: string
  filename: string | null
  output: string | null
  recent: string[]
}
export type SaveRequest = {
  shapes: Shape[]
  flags: Record<string, boolean>
  size: [number, number]
  storeData: boolean
  target?: string
  format?: LabelFormat
  verified?: boolean
  classes?: string[]
  from?: LabelFormat
}
export type Loaded = { file: string; format: LabelFormat | null; verified: boolean; doc: LabelDoc | null }
export type ModelInfo = {
  name: string; title: string; textTitle?: string; size: number; downloaded: boolean; downloading: boolean
  license: { name: string; url: string }
  /** What it takes: clicks, a box, words */
  points: boolean; box: boolean; text: boolean
}
/** A shape the AI proposes; a mask shape's pixels as labelme's base64 PNG */
export type ProposedShape = { label?: string; shape_type: AiOutput; points: number[][]; mask?: string; description?: string; group_id?: number | null }
export type DrawnShape = { shape_type: string; points: number[][]; label?: string; mask?: string | null }
export type AssistRequest = {
  root: string; name: string; model: string; points: number[][]; labels: number[]; output: AiOutput; detail: number
  drawn: DrawnShape[] | null; groups: number[]; outside: boolean
}
export type TextRequest = Omit<AssistRequest, 'points' | 'labels' | 'drawn'> & { texts: string[]; drawn: DrawnShape[]; iou: number; score: number }
/** New shapes, and the indexes of drawn shapes the AI found again; `notice` when nothing could be made */
export type Proposal = { shapes: ProposedShape[]; matched: number[]; notice?: string }
export type PrelabelModel = { path: string; name: string; classes: string[]; inputSize: number[]; tile: number }

declare global {
  interface Window {
    api: {
      start(): Promise<Start>
      pickFolder(title?: string): Promise<string | null>
      pickFile(title: string, filters: { name: string; extensions: string[] }[]): Promise<string | null>
      pickSave(title: string, defaultPath: string, format?: LabelFormat): Promise<string | null>
      ask(message: string, detail: string, buttons: string[]): Promise<number>
      openProject(target: string, output: string | null): Promise<OpenProject>
      closeProject(root: string): Promise<void>
      saveProject(root: string, project: ProjectInfo): Promise<void>
      openImage(root: string, name: string): Promise<OpenImage>
      loadLabels(root: string, name: string, prefer?: LabelFormat, size?: [number, number]): Promise<Loaded>
      saveLabels(root: string, name: string, req: SaveRequest): Promise<{ file: string; item: ImageItem; format: LabelFormat; replaced?: string; kept?: string }>
      labelPath(root: string, name: string, prefer: LabelFormat): Promise<string>
      trashLabels(root: string, name: string, prefer: LabelFormat): Promise<string | null>
      readLabelFile(root: string, name: string, file: string, size?: [number, number]): Promise<Loaded & { doc: LabelDoc; format: LabelFormat }>
      trashImage(root: string, name: string): Promise<{ image: string; kept?: string }>
      hasLabelsIn(root: string, name: string, format: LabelFormat): Promise<boolean>
      resetSettings(): Promise<void>
      reveal(root: string, name: string): Promise<void>
      openConfig(): Promise<void>
      openLink(url: string): Promise<void>
      pathForFile(file: File): string
      onSummaries(fn: (p: { root: string; items: ImageItem[]; labels: string[] }) => void): () => void
      onFlush(fn: () => void): () => void
      flushed(ok: boolean): void
      onError(fn: (message: string) => void): () => void
      aiModels(): Promise<ModelInfo[]>
      aiDownload(name: string): Promise<void>
      onAiDownload(fn: (p: { name: string; model: string; done: number; total: number }) => void): () => void
      aiCancel(name: string): Promise<void>
      aiRemove(name: string): Promise<void>
      aiAssist(req: AssistRequest): Promise<Proposal>
      aiText(req: TextRequest): Promise<Proposal>
      mergeMasks(shapes: { points: number[][]; mask: string }[]): Promise<{ points: number[][]; mask: string }>
      prelabelPick(): Promise<PrelabelModel | null>
      prelabelInfo(path: string): Promise<PrelabelModel>
      prelabelRun(root: string, model: string, names: string[], classes: string[]): Promise<{ done: number; total: number; stopped: boolean; failed: string[]; error?: string }>
      prelabelStop(): Promise<void>
      onPrelabel(fn: (p: { root: string; name: string; shapes: Shape[] | null; done: number; total: number }) => void): () => void
      checkRun(root: string, req: CheckRequest): Promise<CheckSummary | Stopped>
      checkStop(root: string): Promise<void>
      checkView(root: string, query: ProblemQuery): Promise<ProblemPage | null>
      checkFix(root: string, issue: Pick<ProblemRow, 'code' | 'image' | 'shape'>, shapes: Shape[]): Promise<Shape[] | null>
      checkReveal(root: string, file: string): Promise<void>
      onCheck(fn: (p: Progress & { root: string }) => void): () => void
      renamePlan(root: string, from: string, prefer: LabelFormat): Promise<RenamePlanView | Stopped>
      renameRun(root: string, from: string, to: string, prefer: LabelFormat): Promise<RenameDone>
      renameStop(root: string): Promise<void>
      onRename(fn: (p: Progress & { root: string }) => void): () => void
      statsRead(root: string, req: StatsRequest): Promise<StatsView | Stopped>
      statsStop(root: string): Promise<void>
      statsSave(root: string): Promise<string | null>
      onStats(fn: (p: Progress & { root: string }) => void): () => void
      exportCheck(root: string, options: ExportOptions, read: string): Promise<Preflight>
      exportIssues(id: number, key: string, from: number): Promise<Issue[]>
      exportFolder(root: string): Promise<{ path: string; problem: string | null } | null>
      exportKeypoints(): Promise<KeypointsSummary | null>
      exportMakeKeypoints(root: string, prefer: LabelFormat): Promise<KeypointsSummary | null>
      exportRun(root: string, options: ExportOptions, out: string): Promise<ExportResult>
      exportStop(): Promise<void>
      exportClose(): Promise<void>
      exportReveal(out: string): Promise<void>
      onExportProgress(fn: (p: ExportProgress) => void): () => void
      /** File > Import dataset: reads a dataset into the open folder without writing; null when stopped */
      importRead(root: string, req: ImportRequest): Promise<ImportPreview | null>
      importRows(list: 'issues' | 'skipped', code: string, offset: number): Promise<string[]>
      importWrite(policy: ImportPolicy): Promise<ImportReport>
      importStop(): Promise<void>
      importClose(): Promise<void>
      onImportProgress(fn: (p: ImportProgress) => void): () => void
    }
  }
}

export const fileUrl = (p: string) => 'myseg://f/' + encodeURIComponent(p)
export const plainError = (e: unknown) =>
  String((e as Error)?.message ?? e).replace(/^Error invoking remote method '[^']+': (\w*Error: )?/, '')
