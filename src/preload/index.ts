import { contextBridge, ipcRenderer as ipc, webUtils } from 'electron'

const on = (channel: string) => (fn: (payload: any) => void) => {
  const listener = (_e: unknown, payload: any) => fn(payload)
  ipc.on(channel, listener)
  return () => void ipc.off(channel, listener)
}

// A dialog that opens while a mouse button is down keeps the button's release from the window, and
// on X11 the window then keeps its hold on the pointer: nothing on the screen takes a click until
// the dialog is closed from the keyboard. The AI tools ask to download a model on a press, so
// dialogs wait until no button is down, as every pointer event tells.
let pressed = false
const waiting: (() => void)[] = []
for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) {
  window.addEventListener(type, (e) => {
    pressed = (e as PointerEvent).buttons !== 0
    if (!pressed) for (const go of waiting.splice(0)) go()
  }, true)
}
// The dialogs asking now, so nothing opened from outside slips in under one
let asking = 0
const dialog = (channel: string, ...args: unknown[]) => {
  asking++
  return (pressed ? new Promise<void>((go) => waiting.push(go)) : Promise.resolve()).then(() => ipc.invoke(channel, ...args))
    .finally(() => { asking-- })
}

contextBridge.exposeInMainWorld('api', {
  start: () => ipc.invoke('app:start'),
  pickFolder: (title?: string) => dialog('dialog:folder', title),
  pickFile: (title: string, filters: unknown) => dialog('dialog:file', title, filters),
  pickSave: (title: string, defaultPath: string, format?: string) => dialog('dialog:save', title, defaultPath, format),
  ask: (message: string, detail: string, buttons: string[]) => dialog('dialog:ask', message, detail, buttons),
  asking: () => asking > 0,
  openProject: (target: string, output: string | null, started?: boolean, again?: boolean) => ipc.invoke('project:open', target, output, started, again),
  closeProject: (root: string) => ipc.invoke('project:close', root),
  saveProject: (root: string, project: unknown, shown?: unknown) => ipc.invoke('project:save', root, project, shown),
  reloadProject: (root: string) => ipc.invoke('project:reload', root),
  refreshProject: (root: string) => ipc.invoke('project:refresh', root),
  summarize: (root: string) => ipc.invoke('project:summarize', root),
  openImage: (root: string, name: string) => ipc.invoke('image:open', root, name),
  loadLabels: (root: string, name: string, prefer?: string, size?: number[], edit?: boolean) => ipc.invoke('labels:load', root, name, prefer, size, edit),
  saveLabels: (root: string, name: string, req: unknown) => ipc.invoke('labels:save', root, name, req),
  labelsChanged: (root: string, name: string) => ipc.invoke('labels:changed', root, name),
  labelPath: (root: string, name: string, prefer: string) => ipc.invoke('labels:path', root, name, prefer),
  trashLabels: (root: string, name: string, prefer: string) => ipc.invoke('labels:trash', root, name, prefer),
  readLabelFile: (root: string, name: string, file: string, size?: number[]) => ipc.invoke('labels:read', root, name, file, size),
  trashImage: (root: string, name: string) => ipc.invoke('image:trash', root, name),
  hasLabelsIn: (root: string, name: string, format: string) => ipc.invoke('labels:has', root, name, format),
  resetSettings: () => ipc.invoke('app:reset'),
  reveal: (root: string, name: string) => ipc.invoke('file:reveal', root, name),
  openConfig: () => ipc.invoke('config:open'),
  openLink: (url: string) => ipc.invoke('app:link', url),
  legal: (kind: string) => ipc.invoke('app:legal', kind),
  pathForFile: (file: File) => webUtils.getPathForFile(file),
  onSummaries: on('project:summaries'),
  onFocus: on('app:focus'),
  onOpen: on('app:open'),
  onFlush: on('app:flush'),
  flushed: (ok: boolean) => ipc.send('app:flushed', ok),
  onError: on('app:error'),
  aiModels: () => ipc.invoke('ai:models'),
  aiDownload: (name: string) => ipc.invoke('ai:download', name),
  onAiDownload: on('ai:download'),
  aiCancel: (name: string) => ipc.invoke('ai:cancel', name),
  aiRemove: (name: string) => ipc.invoke('ai:remove', name),
  aiAssist: (req: unknown) => ipc.invoke('ai:assist', req),
  aiPose: (req: unknown) => ipc.invoke('ai:pose', req),
  aiText: (req: unknown) => ipc.invoke('ai:text', req),
  mergeMasks: (shapes: unknown) => ipc.invoke('shapes:merge-masks', shapes),
  prelabelPick: (root?: string) => dialog('prelabel:pick', root),
  prelabelInfo: (path: string) => ipc.invoke('prelabel:info', path),
  prelabelModels: (root: string) => ipc.invoke('prelabel:models', root),
  prelabelRun: (root: string, model: string, names: string[], classes: string[], batch?: boolean) =>
    ipc.invoke('prelabel:run', root, model, names, classes, batch),
  prelabelStop: () => ipc.invoke('prelabel:stop'),
  onPrelabel: on('prelabel:progress'),
  checkRun: (root: string, req: unknown) => ipc.invoke('check:run', root, req),
  checkStop: (root: string) => ipc.invoke('check:stop', root),
  checkView: (root: string, query: unknown) => ipc.invoke('check:view', root, query),
  checkFix: (root: string, issue: unknown, shapes: unknown) => ipc.invoke('check:fix', root, issue, shapes),
  checkReveal: (root: string, file: string) => ipc.invoke('check:reveal', root, file),
  onCheck: on('check:progress'),
  renamePlan: (root: string, from: string, prefer: string) => ipc.invoke('rename:plan', root, from, prefer),
  renameRun: (root: string, from: string, to: string, prefer: string) => ipc.invoke('rename:run', root, from, to, prefer),
  renameStop: (root: string) => ipc.invoke('rename:stop', root),
  onRename: on('rename:progress'),
  statsRead: (root: string, req: unknown) => ipc.invoke('stats:read', root, req),
  statsStop: (root: string) => ipc.invoke('stats:stop', root),
  statsSave: (root: string) => dialog('stats:save', root),
  onStats: on('stats:progress'),
  exportCheck: (root: string, options: unknown, read: string) => ipc.invoke('export:check', root, options, read),
  exportIssues: (id: number, key: string, from: number) => ipc.invoke('export:issues', id, key, from),
  exportFolder: (root: string) => dialog('export:folder', root),
  exportKeypoints: () => dialog('export:keypoints'),
  exportMakeKeypoints: (root: string, prefer: string) => dialog('export:make-keypoints', root, prefer),
  keypointsGet: (root: string) => ipc.invoke('keypoints:get', root),
  keypointsSet: (root: string, schema: unknown) => ipc.invoke('keypoints:set', root, schema),
  keypointsChoose: (root: string) => dialog('keypoints:choose', root),
  keypointsMake: (root: string, prefer: string) => ipc.invoke('keypoints:make', root, prefer),
  exportRun: (root: string, options: unknown, out: string) => ipc.invoke('export:run', root, options, out),
  exportStop: () => ipc.invoke('export:stop'),
  exportClose: () => ipc.invoke('export:close'),
  exportReveal: (out: string) => ipc.invoke('export:reveal', out),
  onExportProgress: on('export:progress'),
  importRead: (root: string, req: unknown) => ipc.invoke('import:read', root, req),
  importRows: (list: string, code: string, offset: number) => ipc.invoke('import:rows', list, code, offset),
  importWrite: (policy: string) => ipc.invoke('import:write', policy),
  importStop: () => ipc.invoke('import:stop'),
  importClose: () => ipc.invoke('import:close'),
  onImportProgress: on('import:progress')
})
