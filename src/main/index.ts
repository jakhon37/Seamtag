import { app, BrowserWindow, dialog, ipcMain, Menu, net, protocol, shell } from 'electron'
import { existsSync, promises as fs, readFileSync, writeFileSync } from 'fs'
import path from 'path'
import { pathToFileURL } from 'url'
import { cancelDownload, download, listModels, release, removeModel } from './ai'
import {
  checkFolder, fixShapes, folderStats, forgetChecks, problemPage, renamePlan, renameRun, statsFileName, statsText, stopJob, strayFile
} from './checking'
import { forgetDataset } from './dataset'
import {
  type ExportOptions, type ExportProgress, folderProblem, forgetChecks, keypointsFromFolder, moreIssues, preflight, readKeypoints, runExport, saveKeypoints
} from './export'
import { mergeMasks } from './masks'
import { assist, type AssistRequest, fromTexts, MaskOutputUnavailable, type TextRequest } from './assist'
import { type Args, loadConfig, type LoadedConfig, parseArgs, USAGE, USER_CONFIG } from './config'
import { FORMAT_EXT, FORMAT_NAMES, type LabelFormat } from '../shared/formats'
import { cacheRoot, imageData, imageDepth, imageSize, openImage, setCacheDir } from './images'
import { closeImport, type ImportPolicy, type ImportRequest, importRows, readImport, stopImport, writeImport } from './import'
import { modelInfo, prelabel, stopPrelabel } from './prelabel'
import {
  closeProject, hasLabelsIn, imagePath, labelsFile, loadLabels, openProject, type ProjectInfo, readLabelFile, rootsOpen, saveLabels,
  saveProjectFile, type SaveRequest, setTrash, summarizeAll, trashImage, trashLabels
} from './project'

const NAME = 'Seamtag'
app.setName(NAME)
// macOS and Windows put the profile where the system says, whatever HOME is, so tests name their own
if (process.env.SEAMTAG_USER_DATA) app.setPath('userData', process.env.SEAMTAG_USER_DATA)

// In development Electron's own arguments come first: `electron . image.jpg`
const argv = process.argv.slice(app.isPackaged ? 1 : 2)
let args: Args
try {
  args = parseArgs(argv)
} catch (e: any) {
  process.stderr.write(`${e.message}\n\n${USAGE}`)
  process.exit(2)
}
if (args.help) {
  process.stdout.write(USAGE)
  process.exit(0)
}
if (args.version) {
  process.stdout.write(`${NAME} ${app.getVersion()}\n`)
  process.exit(0)
}

protocol.registerSchemesAsPrivileged([
  { scheme: 'myseg', privileges: { standard: true, supportFetchAPI: true, stream: true, secure: true, corsEnabled: true } }
])

let win: BrowserWindow | null = null
let loaded: LoadedConfig
let configError = ''

type State = { recent: string[]; bounds?: { width: number; height: number; x?: number; y?: number }; maximized?: boolean }
const statePath = () => path.join(app.getPath('userData'), 'state.json')
function readState(): State {
  try {
    return { recent: [], ...JSON.parse(readFileSync(statePath(), 'utf8')) }
  } catch {
    return { recent: [] }
  }
}
function writeState(state: State) {
  try {
    writeFileSync(statePath(), JSON.stringify(state, null, 2))
  } catch {
    // Not worth stopping for
  }
}
function addRecent(target: string) {
  const state = readState()
  state.recent = [target, ...state.recent.filter((r) => r !== target)].slice(0, 10)
  writeState(state)
}

function send(channel: string, payload: unknown) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload)
}

async function open(target: string, output: string | null) {
  const project = await openProject(target, output)
  addRecent(target)
  win?.setTitle(`${path.basename(project.root)} - ${NAME}`)
  // Counts and broken files arrive after the list, so a big folder opens at once
  summarizeAll(project.root, (batch) => send('project:summaries', batch), loaded?.config.label_format).catch((e) =>
    send('app:error', `Some label files couldn't be read: ${e.message}`)
  )
  return project
}

function registerIpc() {
  ipcMain.handle('app:start', () => ({
    config: loaded?.config ?? null,
    configError,
    configFile: loaded?.file ?? null,
    warnings: loaded?.warnings ?? [],
    version: app.getVersion(),
    platform: process.platform,
    filename: args.filename ? path.resolve(args.filename) : null,
    output: args.output ? path.resolve(args.output) : null,
    recent: readState().recent.filter((r) => existsSync(r))
  }))
  ipcMain.handle('dialog:folder', async (_e, title?: string) => {
    const r = await dialog.showOpenDialog(win!, { title: title ?? 'Open folder', properties: ['openDirectory', 'createDirectory'] })
    return r.canceled ? null : r.filePaths[0]
  })
  ipcMain.handle('dialog:file', async (_e, title: string, filters: Electron.FileFilter[]) => {
    const r = await dialog.showOpenDialog(win!, { title, properties: ['openFile'], filters })
    return r.canceled ? null : r.filePaths[0]
  })
  ipcMain.handle('dialog:save', async (_e, title: string, defaultPath: string, format: LabelFormat = 'labelme') => {
    const filters = [{ name: `${FORMAT_NAMES[format]} files`, extensions: [FORMAT_EXT[format].slice(1)] }]
    const r = await dialog.showSaveDialog(win!, { title, defaultPath, filters })
    return r.canceled ? null : r.filePath
  })
  ipcMain.handle('dialog:ask', async (_e, message: string, detail: string, buttons: string[]) => {
    const r = await dialog.showMessageBox(win!, { type: 'question', message, detail, buttons, defaultId: 0, cancelId: buttons.length - 1, noLink: true })
    return r.response
  })
  ipcMain.handle('project:open', (_e, target: string, output: string | null) => open(target, output))
  ipcMain.handle('project:close', (_e, root: string) => {
    // What the dataset reader keeps of the folder goes with it
    forgetDataset(root)
    forgetChecks(root)
    return closeProject(root)
  })
  ipcMain.handle('project:save', (_e, root: string, project: ProjectInfo) => saveProjectFile(root, project))
  ipcMain.handle('image:open', (_e, root: string, name: string) => openImage(root, name))
  // YOLO's boxes are shares of the image, so reading them needs its size: the window's, or the file's
  const sizeOf = (root: string, name: string, size?: [number, number]) => async () => size ?? imageSize(root, name)
  ipcMain.handle('labels:load', (_e, root: string, name: string, prefer?: LabelFormat, size?: [number, number]) =>
    loadLabels(root, name, prefer, sizeOf(root, name, size)))
  // The image's data and depth are read in the save's turn, so saves of one image keep their order
  ipcMain.handle('labels:save', (_e, root: string, name: string, req: SaveRequest & { storeData?: boolean }) =>
    saveLabels(root, name, { ...req, imageData: null }, {
      imageData: req.storeData ? () => imageData(root, name) : undefined,
      depth: req.format === 'voc' ? () => imageDepth(root, name) : undefined
    }))
  ipcMain.handle('labels:has', (_e, root: string, name: string, format: LabelFormat) => hasLabelsIn(root, name, format))
  ipcMain.handle('labels:path', (_e, root: string, name: string, prefer: LabelFormat) => labelsFile(root, name, prefer))
  // labelme deletes the file. It goes to the trash here, so a slip can be undone.
  ipcMain.handle('labels:trash', (_e, root: string, name: string, prefer: LabelFormat) => trashLabels(root, name, prefer))
  ipcMain.handle('labels:read', (_e, root: string, name: string, file: string, size?: [number, number]) =>
    readLabelFile(root, name, file, sizeOf(root, name, size)))
  ipcMain.handle('image:trash', (_e, root: string, name: string) => trashImage(root, name))
  ipcMain.handle('file:reveal', (_e, root: string, name: string) => shell.showItemInFolder(imagePath(root, name)))
  ipcMain.handle('config:open', async () => {
    const file = loaded?.file ?? USER_CONFIG
    if (!existsSync(file)) await fs.writeFile(file, '')
    const error = await shell.openPath(file)
    if (error) throw new Error(error)
  })
  // LabelImg's Reset All: the window's settings, layout and recent files are forgotten, then it starts again
  ipcMain.handle('app:reset', async () => {
    await fs.rm(statePath(), { force: true })
    await win?.webContents.session.clearStorageData({ storages: ['localstorage'] })
    app.relaunch()
    app.exit(0)
  })
  ipcMain.handle('app:link', (_e, url: string) => {
    if (/^https:\/\//.test(url)) return shell.openExternal(url)
  })
  ipcMain.handle('ai:models', () => listModels())
  ipcMain.handle('ai:download', (_e, name: string) => download(name, (p) => send('ai:download', p)))
  ipcMain.handle('ai:cancel', (_e, name: string) => cancelDownload(name))
  ipcMain.handle('ai:remove', (_e, name: string) => removeModel(name))
  ipcMain.handle('ai:assist', (_e, req: AssistRequest) => assist(req))
  ipcMain.handle('shapes:merge-masks', (_e, shapes: { points: number[][]; mask: string }[]) => mergeMasks(shapes))
  // A model that finds boxes only, asked for masks, is the user's choice to change, not a failure
  ipcMain.handle('ai:text', (_e, req: TextRequest) => fromTexts(req).catch((e) => {
    if (e instanceof MaskOutputUnavailable) return { shapes: [], matched: [], notice: e.message }
    throw e
  }))
  ipcMain.handle('prelabel:pick', async () => {
    const r = await dialog.showOpenDialog(win!, { title: 'Choose a segmentation model', properties: ['openFile'], filters: [{ name: 'ONNX models', extensions: ['onnx'] }] })
    return r.canceled ? null : modelInfo(r.filePaths[0])
  })
  ipcMain.handle('prelabel:info', (_e, file: string) => modelInfo(file))
  ipcMain.handle('prelabel:run', (_e, root: string, model: string, names: string[], classes: string[]) =>
    prelabel(root, model, names, classes, loaded?.config.prelabel.min_area ?? 20, (p) => send('prelabel:progress', p)))
  ipcMain.handle('prelabel:stop', () => stopPrelabel())
  // Dataset > Check labels, Statistics and renaming a label across the folder (checking.ts)
  ipcMain.handle('check:run', (_e, root: string, req) => checkFolder(root, req, (p) => send('check:progress', { root, ...p })))
  ipcMain.handle('check:stop', (_e, root: string) => stopJob(root, 'check'))
  ipcMain.handle('check:view', (_e, root: string, query) => problemPage(root, query))
  ipcMain.handle('check:fix', (_e, root: string, issue, shapes) => fixShapes(root, issue, shapes))
  // Only a label file the last check found without its image
  ipcMain.handle('check:reveal', (_e, root: string, file: string) => {
    const full = strayFile(root, file)
    if (full) shell.showItemInFolder(full)
  })
  ipcMain.handle('rename:plan', (_e, root: string, from: string, prefer: LabelFormat) =>
    renamePlan(root, from, prefer, (p) => send('rename:progress', { root, ...p })))
  ipcMain.handle('rename:run', async (_e, root: string, from: string, to: string, prefer: LabelFormat) => {
    const config = loaded?.config
    const done = await renameRun(root, from, to, prefer, config?.validate_label === 'exact' ? config.labels : null, (p) => send('rename:progress', { root, ...p }))
    // The image list counts the changed images again and takes in the new label; the rename stands either way
    if (done.changed.length) {
      await summarizeAll(root, (b) => send('project:summaries', b), config?.label_format, done.changed).catch((e) =>
        send('app:error', `Some renamed label files couldn't be counted again: ${e.message}`))
    }
    return done
  })
  ipcMain.handle('rename:stop', (_e, root: string) => stopJob(root, 'rename'))
  ipcMain.handle('stats:read', (_e, root: string, req) => folderStats(root, req, (p) => send('stats:progress', { root, ...p })))
  ipcMain.handle('stats:stop', (_e, root: string) => stopJob(root, 'stats'))
  ipcMain.handle('stats:save', async (_e, root: string) => {
    const text = statsText(root)
    if (text === null) throw new Error('Count the folder first')
    const r = await dialog.showSaveDialog(win!, { title: 'Save statistics as CSV', defaultPath: statsFileName(root), filters: [{ name: 'CSV files', extensions: ['csv'] }] })
    if (r.canceled || !r.filePath) return null
    await fs.writeFile(r.filePath, text)
    return r.filePath
  })

  // File > Export dataset…: one job at a time, a check or an export, and a new one stops the last.
  // The folder is checked again before anything is written to it, and only keypoints files main's
  // own dialogs gave are read.
  let exporting: AbortController | null = null
  let exported: string | null = null
  const keypointFiles = new Set<string>()
  const exportJob = () => {
    exporting?.abort()
    return { signal: (exporting = new AbortController()).signal, onProgress: (p: ExportProgress) => send('export:progress', p), version: app.getVersion() }
  }
  const exportOptions = (options: ExportOptions) => {
    if (options.keypoints && !keypointFiles.has(options.keypoints)) throw new Error('Choose the keypoints file again')
    return { ...options, labels: options.labels ?? [] }
  }
  const keypointsFilter = [{ name: 'Keypoints files', extensions: ['yaml', 'yml'] }]
  // A check given the id of the last check's read uses that read again; the window gives a new one when it opens and after an export fails
  ipcMain.handle('export:check', (_e, root: string, options: ExportOptions, read: unknown) =>
    preflight(root, exportOptions(options), { ...exportJob(), read: typeof read === 'string' ? read : undefined }))
  ipcMain.handle('export:issues', (_e, id: number, key: string, from: number) => moreIssues(id, key, from))
  ipcMain.handle('export:folder', async (_e, root: string) => {
    const r = await dialog.showOpenDialog(win!, { title: 'Export to a new or empty folder', defaultPath: path.dirname(root), properties: ['openDirectory', 'createDirectory'] })
    if (r.canceled || !r.filePaths[0]) return null
    return { path: r.filePaths[0], problem: await folderProblem(root, r.filePaths[0]) }
  })
  ipcMain.handle('export:keypoints', async () => {
    const r = await dialog.showOpenDialog(win!, { title: 'Choose a keypoints file', properties: ['openFile'], filters: keypointsFilter })
    if (r.canceled || !r.filePaths[0]) return null
    const { summary } = await readKeypoints(r.filePaths[0])
    keypointFiles.add(summary.file)
    return summary
  })
  ipcMain.handle('export:make-keypoints', async (_e, root: string, prefer: LabelFormat) => {
    const schema = await keypointsFromFolder(root, prefer)
    if (!schema) throw new Error('The folder has no posed objects to make a keypoints file from: give a box and its points one group_id')
    // GTK's save dialog asks before replacing a file only when told to: a keypoints file edited by hand is worth the question
    const r = await dialog.showSaveDialog(win!, {
      title: 'Save the keypoints file', defaultPath: path.join(path.dirname(root), `${path.basename(root)}-keypoints.yaml`), filters: keypointsFilter,
      properties: ['showOverwriteConfirmation']
    })
    if (r.canceled || !r.filePath) return null
    const summary = await saveKeypoints(r.filePath, schema)
    keypointFiles.add(summary.file)
    return summary
  })
  ipcMain.handle('export:run', async (_e, root: string, options: ExportOptions, out: string) => {
    const result = await runExport(root, exportOptions(options), out, exportJob())
    if (!result.stopped) exported = result.out
    return result
  })
  ipcMain.handle('export:stop', () => exporting?.abort())
  // The window closed: what its checks kept goes with it
  ipcMain.handle('export:close', () => {
    exporting?.abort()
    forgetChecks()
  })
  ipcMain.handle('export:reveal', (_e, out: string) => {
    if (out === exported) shell.showItemInFolder(out)
  })

  // File > Import dataset: reading writes nothing; writing goes through saveLabels, then the image list counts the images again
  ipcMain.handle('import:read', (_e, root: string, req: ImportRequest) => readImport(root, req, (p) => send('import:progress', p)))
  ipcMain.handle('import:rows', (_e, list: 'issues' | 'skipped', code: string, offset: number) => importRows(list, code, offset))
  ipcMain.handle('import:write', (_e, policy: ImportPolicy) =>
    writeImport(policy, (p) => send('import:progress', p), (batch) => send('project:summaries', batch)))
  ipcMain.handle('import:stop', () => stopImport())
  ipcMain.handle('import:close', () => closeImport())
}

function buildMenu() {
  // The window has its own menu bar. A Mac still needs the app menu, and Edit for text fields.
  if (process.platform !== 'darwin') return Menu.setApplicationMenu(null)
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { role: 'appMenu' },
    { role: 'editMenu' },
    { role: 'windowMenu' }
  ]))
}

function createWindow() {
  const state = readState()
  win = new BrowserWindow({
    width: state.bounds?.width ?? 1500, height: state.bounds?.height ?? 940, x: state.bounds?.x, y: state.bounds?.y,
    minWidth: 960, minHeight: 600, backgroundColor: '#14161a', title: NAME, show: false,
    icon: path.join(__dirname, '../../resources/icon.png'),
    webPreferences: { preload: path.join(__dirname, '../preload/index.js'), sandbox: false }
  })
  if (state.maximized) win.maximize()
  win.once('ready-to-show', () => win?.show())
  // Saves wait a moment after each change: the window flushes them, or asks about unsaved
  // work when saving isn't automatic, before it closes
  let closing = false
  win.on('close', (e) => {
    if (closing || !win) return
    e.preventDefault()
    const finish = (ok: boolean) => {
      if (!ok || closing || !win) return
      closing = true
      const s = readState()
      s.maximized = win.isMaximized()
      if (!s.maximized) s.bounds = win.getBounds()
      writeState(s)
      win.close()
    }
    ipcMain.once('app:flushed', (_e, ok: boolean) => finish(ok))
    send('app:flush', null)
  })
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else win.loadFile(path.join(__dirname, '../renderer/index.html'))
}

app.whenReady().then(async () => {
  setCacheDir(path.join(app.getPath('userData'), 'cache'))
  if (args.resetConfig) {
    // labelme forgets its Qt settings; here that's the layout, recent files and app settings
    await fs.rm(statePath(), { force: true })
    await fs.rm(path.join(app.getPath('userData'), 'Local Storage'), { recursive: true, force: true })
    process.stdout.write('Settings were reset.\n')
    app.exit(0)
    return
  }
  try {
    loaded = loadConfig(args)
  } catch (e: any) {
    configError = e.message
  }
  setTrash((file) => shell.trashItem(file))
  protocol.handle('myseg', async (request) => {
    const file = path.resolve(decodeURIComponent(new URL(request.url).pathname.slice(1)))
    const allowed = [...rootsOpen(), cacheRoot()].some((dir) => dir && file.startsWith(dir + path.sep))
    if (!allowed) return new Response('Forbidden', { status: 403 })
    // Shared with the page's own origin, so drawing an image leaves the canvas readable
    const response = await net.fetch(pathToFileURL(file).toString())
    const headers = new Headers(response.headers)
    headers.set('Access-Control-Allow-Origin', '*')
    return new Response(response.body, { status: response.status, headers })
  })
  registerIpc()
  buildMenu()
  createWindow()
})

app.on('window-all-closed', () => {
  stopPrelabel()
  release().finally(() => app.quit())
})
