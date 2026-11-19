// File names compared as the file system compares them. Windows and macOS ignore case, so a.PNG is
// found by the name a.png. macOS also ignores Unicode form: a name written in NFC, as Linux and
// Windows write it, finds a file whose name is in NFD, as macOS apps often save it (café, 사과).
export type Folding = 'exact' | 'case' | 'case-and-form'
const SYSTEM: Folding = process.platform === 'darwin' ? 'case-and-form' : process.platform === 'win32' ? 'case' : 'exact'
let folding: Folding = SYSTEM

/** How names are compared, or as this system compares them when left out: for tests */
export function setFolding(how: Folding = SYSTEM) {
  folding = how
}

/** A name or a path as the file system compares it: two that give one key are one file */
export function nameKey(name: string) {
  if (folding === 'exact') return name
  return (folding === 'case-and-form' ? name.normalize('NFC') : name).toLowerCase()
}
