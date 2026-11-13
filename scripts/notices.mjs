// Writes THIRD-PARTY-NOTICES.txt at the top of the repository: everything the installers carry
// that isn't Seamtag's own code, each part with its licence in full. It reads the node_modules of
// the computer that builds, so each system's installer lists what it holds:
//   - the packages Vite builds into out/ from the devDependencies (BUILT_IN below),
//   - the packages app.asar carries: the dependencies, and theirs, as installed,
//   - libvips and the libraries built into it, from sharp's @img packages,
//   - ONNX Runtime, and Electron.
// Texts node_modules lacks are in resources/notices, whose README says where each came from.
// Code from other projects inside Seamtag's own source files is in NOTICE.md instead.
//
// `npm run build`, `dist` and `dist:dir` run it. electron-builder runs it again before it packs
// (beforePack in electron-builder.yml), so a build that calls electron-builder itself, as the
// release workflow does, has it too, and stops there if a file extraResources names is missing.

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { builtinModules } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const OUT = path.join(ROOT, 'THIRD-PARTY-NOTICES.txt')
const NOTICES = path.join(ROOT, 'resources', 'notices')

// What Vite builds into out/ from the devDependencies: React, the icons, the Inter font and
// Clipper in the window, d3-contour and Clipper in the main process. Every package src/ imports
// must be a dependency or here, and each of these packages' own dependencies here or in LEFT_OUT.
const BUILT_IN = ['react', 'react-dom', 'scheduler', 'lucide-react', '@fontsource-variable/inter', 'clipper-lib', 'd3-contour', 'd3-array', 'internmap']
// React's build-time envify transform, which nothing in the bundles calls
const LEFT_OUT = ['loose-envify', 'js-tokens']
const ABOUT = { '@fontsource-variable/inter': 'the Inter font' }

const text = (file) => readFileSync(file, 'utf8').replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').trimEnd() + '\n'
const json = (file) => JSON.parse(readFileSync(file, 'utf8'))
const line = (c) => c.repeat(80)
const heading = (title, ...facts) => [line('='), title, ...facts.filter(Boolean), line('-'), '', ''].join('\n')

function packageAt(dir) {
  const pkg = json(path.join(dir, 'package.json'))
  let url = (typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url) ?? ''
  url = url.replace(/^git\+/, '').replace(/^git:\/\//, 'https://').replace(/^(ssh:\/\/)?git@github\.com[:/]/, 'https://github.com/')
    .replace(/^github:/, 'https://github.com/').replace(/\.git$/, '')
  if (/^[\w.-]+\/[\w.-]+$/.test(url)) url = `https://github.com/${url}`
  const licences = readdirSync(dir).filter((f) => /^(licen[cs]e|copying|notice)([-.][\w-]+)?(\.(md|markdown|txt))?$/i.test(f)).sort()
  return { dir, name: pkg.name, version: pkg.version, licence: pkg.license ?? 'none named', url: url || pkg.homepage || '', licences }
}
const title = (p) => `${p.name} ${p.version}${ABOUT[p.name] ? ` (${ABOUT[p.name]})` : ''}`

/** The dependencies app.asar carries, and theirs: the lockfile's packages that aren't for development, as installed here */
function appPackages() {
  const lock = json(path.join(ROOT, 'package-lock.json'))
  return Object.entries(lock.packages)
    .filter(([key, entry]) => key && !entry.dev && !entry.devOptional && existsSync(path.join(ROOT, key, 'package.json')))
    .map(([key]) => packageAt(path.join(ROOT, key)))
}

/** Stops when src/ imports a package that's neither a dependency nor in BUILT_IN, or one in BUILT_IN needs another */
function checkBuiltIn() {
  const known = new Set([...Object.keys(json(path.join(ROOT, 'package.json')).dependencies ?? {}), ...BUILT_IN, 'electron'])
  const unknown = new Set()
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, e.name)
      if (e.isDirectory()) walk(file)
      else if (/\.(ts|tsx|css)$/.test(e.name)) {
        for (const [, type, spec] of readFileSync(file, 'utf8').matchAll(/\b(?:import|export)\s+(type\s+)?(?:[\w*{}\s,$]*?from\s*)?['"]([^'"]+)['"]/g)) {
          const name = spec.split('/').slice(0, spec.startsWith('@') ? 2 : 1).join('/')
          if (!type && !spec.startsWith('.') && !spec.startsWith('node:') && !builtinModules.includes(name) && !known.has(name)) unknown.add(name)
        }
      }
    }
  }
  walk(path.join(ROOT, 'src'))
  for (const name of BUILT_IN) {
    for (const dep of Object.keys(json(path.join(ROOT, 'node_modules', name, 'package.json')).dependencies ?? {})) {
      if (!BUILT_IN.includes(dep) && !LEFT_OUT.includes(dep)) unknown.add(dep)
    }
  }
  if (unknown.size) {
    const them = unknown.size === 1 ? 'it' : 'them'
    throw new Error(`Vite builds ${[...unknown].join(', ')} into the app: add ${them} to BUILT_IN in scripts/notices.mjs, or to LEFT_OUT if the bundles leave ${them} out`)
  }
}

/** A package's own licence files, or what stands in for them where it has none */
function licenceOf(p) {
  const own = p.licences.map((f) => (p.licences.length > 1 ? `${f}:\n\n` : '') + text(path.join(p.dir, f))).join('\n')
  if (p.name === 'clipper-lib') {
    // It names its licences in clipper.js's opening comments, which the bundles leave out
    const head = /^(\s*\/\*[\s\S]*?\*\/)+/.exec(readFileSync(path.join(p.dir, 'clipper.js'), 'utf8'))[0].trim()
    return [own, `clipper.js begins:\n\n${head}\n`, `The Boost Software License 1.0, which it names:\n\n${text(path.join(NOTICES, 'clipper-lib', 'BSL-1.0.txt'))}`,
      `The licence of Tom Wu's JSBN, part of which it holds:\n\n${text(path.join(NOTICES, 'clipper-lib', 'jsbn-LICENSE'))}`].filter(Boolean).join('\n')
  }
  if (/^onnxruntime-/.test(p.name)) return `${own}${own ? '\n' : ''}Part of ONNX Runtime: its licence and notices are under ONNX Runtime below.\n`
  if (/LGPL/.test(p.licence)) return `${own}${own ? '\n' : ''}It holds libvips: the licences are under libvips below.\n`
  if (!own) throw new Error(`${p.name} ${p.version} carries no licence file: give scripts/notices.mjs its licence text`)
  return own
}

/** libvips and the libraries built into it, from the @img package sharp loads it from */
function libvips(packages) {
  return packages.filter((p) => /LGPL/.test(p.licence)).map((p) => {
    const readme = existsSync(path.join(p.dir, 'README.md')) ? text(path.join(p.dir, 'README.md')) : ''
    const licensing = readme.split(/^(?=## )/m).find((s) => /^## Licen[cs]ing\b/.test(s))
    if (!licensing) throw new Error(`${p.name}'s README has no Licensing section, the list of the libraries in libvips and their licences`)
    const versions = existsSync(path.join(p.dir, 'versions.json')) ? json(path.join(p.dir, 'versions.json')) : {}
    const lib = existsSync(path.join(p.dir, 'lib')) ? readdirSync(path.join(p.dir, 'lib')).filter((f) => /vips/.test(f)).sort() : []
    const vips = versions.vips ? `libvips ${versions.vips}` : 'libvips'
    const built = /^@img\/sharp-libvips-/.test(p.name) ? `https://github.com/lovell/sharp-libvips/tree/v${p.version}` : 'https://github.com/lovell/sharp-libvips'
    return {
      listed: `  ${vips} and the libraries built into it, in ${p.name} ${p.version}: ${p.licence}, and the licences listed there`,
      body: [
        heading(`${vips} and the libraries built into it`, `In ${p.name} ${p.version}. Licence: ${p.licence}`) +
          `sharp loads libvips as a shared library${lib.length ? ` (${lib.join(', ')})` : ''}, from node_modules/${p.name}/lib in app.asar.unpacked, beside app.asar.`,
        ...(Object.keys(versions).length ? ['', `Versions: ${Object.entries(versions).map(([k, v]) => `${k} ${v}`).join(', ')}.`] : []),
        '',
        `Sources: libvips at ${versions.vips ? `https://github.com/libvips/libvips/tree/v${versions.vips}` : 'https://github.com/libvips/libvips'}. ${built} builds it and the libraries listed below, and says where each one's source comes from.`,
        '',
        `${p.name}'s README says:`,
        '',
        licensing.trimEnd(),
        '',
        'This software is based in part on the work of the Independent JPEG Group.',
        'This software is based in part on the work of the FreeType Team (https://freetype.org).',
        '',
        'The GNU Lesser General Public License version 3 follows. It adds to the GNU General Public License version 3, which is in LICENSE, beside this file.',
        '',
        text(path.join(NOTICES, 'LGPL-3.0.txt'))
      ].join('\n')
    }
  })
}

function onnxRuntime(packages) {
  const ort = packages.find((p) => p.name === 'onnxruntime-node')
  if (!ort) return []
  const dir = path.join(NOTICES, `onnxruntime-${ort.version}`)
  if (!existsSync(path.join(dir, 'ThirdPartyNotices.txt')) || !existsSync(path.join(dir, 'LICENSE'))) {
    throw new Error(`resources/notices/onnxruntime-${ort.version} needs LICENSE and ThirdPartyNotices.txt from https://github.com/microsoft/onnxruntime at tag v${ort.version} (see resources/notices/README.md)`)
  }
  return [{
    listed: `  ONNX Runtime ${ort.version}, in onnxruntime-node and onnxruntime-common: MIT, and the licences in its notices`,
    body: [
      heading(`ONNX Runtime ${ort.version}`, 'In onnxruntime-node and onnxruntime-common. Licence: MIT', 'https://github.com/microsoft/onnxruntime') +
        `The packages carry no licence file. ONNX Runtime's licence, and its notices for the libraries built into it, follow, from https://github.com/microsoft/onnxruntime at tag v${ort.version}.`,
      '',
      text(path.join(dir, 'LICENSE')),
      text(path.join(dir, 'ThirdPartyNotices.txt'))
    ].join('\n')
  }]
}

function electron() {
  const p = packageAt(path.join(ROOT, 'node_modules', 'electron'))
  return {
    listed: `  Electron ${p.version}: ${p.licence}, and Chromium's components under theirs`,
    body: [
      heading(`Electron ${p.version}, and Chromium`, `Licence: ${p.licence}`, p.url) +
        'Seamtag runs on Electron, which builds in Chromium and the components Chromium uses, FFmpeg among them. Their licences are in LICENSES.chromium.html, and Electron\'s in LICENSE.electron.txt: in the folder that holds the seamtag program on Windows and Linux, and beside this file on macOS. Electron\'s licence:',
      '',
      text(path.join(p.dir, 'LICENSE'))
    ].join('\n')
  }
}

export function notices() {
  checkBuiltIn()
  const byName = (a, b) => a.name.localeCompare(b.name) || a.dir.localeCompare(b.dir)
  const builtIn = BUILT_IN.map((name) => packageAt(path.join(ROOT, 'node_modules', name))).sort(byName)
  // A package installed twice, at one version, as semver is for sharp and global-agent, is listed once
  const app = [...new Map(appPackages().sort(byName).map((p) => [`${p.name}@${p.version}`, p])).values()]
  const libraries = [...libvips(app), ...onnxRuntime(app), electron()]
  const listed = (p) => `  ${title(p)}: ${p.licence}`
  const entry = (p) => heading(title(p), `Licence: ${p.licence}`, p.url) + licenceOf(p)
  return [
    `Seamtag ${json(path.join(ROOT, 'package.json')).version}: third-party notices`,
    '',
    'Seamtag is free software under the GNU General Public License version 3 only, which is in LICENSE, beside this file. NOTICE.md, also beside it, says which of Seamtag\'s own source files hold code from other projects, and under which licences.',
    '',
    'This file lists the other software the app carries, each part under its own licence, given in full below.',
    '',
    `Built into the app's code (${builtIn.length}):`,
    ...builtIn.map(listed),
    '',
    `In the app's node_modules (${app.length}):`,
    ...app.map(listed),
    '',
    'Libraries:',
    ...libraries.map((l) => l.listed),
    '',
    ...builtIn.map(entry),
    ...app.map(entry),
    ...libraries.map((l) => l.body)
  ].join('\n')
}

export function write() {
  const out = notices()
  writeFileSync(OUT, out)
  console.log(`THIRD-PARTY-NOTICES.txt: ${Math.round(Buffer.byteLength(out) / 1024)} KB`)
}

/**
 * electron-builder's beforePack: the notices again, from this system's packages. It also stops the
 * build when a file extraResources names is missing, as NOTICE.md or, on macOS, Electron's licences
 * could be: electron-builder would only warn, and leave it out of the installer.
 */
export default async function beforePack({ packager }) {
  write()
  const asList = (v) => (v == null ? [] : Array.isArray(v) ? v : [v])
  const missing = [...asList(packager.config.extraResources), ...asList(packager.platformSpecificBuildOptions.extraResources)]
    .flatMap((r) => (typeof r === 'object' && r.from && !r.from.includes('${') ? [r.from] : []))
    .filter((from) => !existsSync(path.resolve(packager.projectDir, from)))
  if (missing.length) throw new Error(`The installers need ${missing.join(' and ')}, which ${missing.length === 1 ? "isn't" : "aren't"} there`)
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) write()
