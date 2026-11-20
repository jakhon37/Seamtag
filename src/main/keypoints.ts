// An open folder's keypoints file: .myseg/keypoints.yaml, X-AnyLabeling's pose file with
// Seamtag's skeleton and flip pairs. The editor draws poses with it and Export starts from it.
// COCO's person is used until the folder has one, and when its file can't be read.
import { promises as fs } from 'fs'
import type { KeypointSchema } from '../shared/dataset'
import { readPoseYaml, writePoseYaml } from '../shared/pose'
import { builtInKeypoints, type FolderKeypoints, withKnownLines } from '../shared/skeleton'
import { meta, writeAtomic } from './project'

export const keypointsFile = (root: string) => meta(root, 'keypoints.yaml')

/** The folder's keypoints, or COCO's person with what was wrong with the folder's file */
export async function folderKeypoints(root: string): Promise<FolderKeypoints> {
  const file = keypointsFile(root)
  let text: string
  try {
    text = await fs.readFile(file, 'utf8')
  } catch (e) {
    const missing = (e as NodeJS.ErrnoException).code === 'ENOENT'
    return { schema: builtInKeypoints(), file: null, problem: missing ? null : `The keypoints file ${file} can't be read (${(e as Error).message}), so COCO's person is used` }
  }
  try {
    // A file written by hand or by X-AnyLabeling may lack COCO's lines and flip pairs for COCO's person
    return { schema: withKnownLines(readPoseYaml(text).schema), file, problem: null }
  } catch (e) {
    return { schema: builtInKeypoints(), file: null, problem: `${(e as Error).message.replace(/^The keypoints file /, `The keypoints file ${file} `)}, so COCO's person is used` }
  }
}

/**
 * Writes the folder's keypoints file. The schema goes through the file's own reader first, so
 * what's saved is what the next read gets, and a schema the reader refuses is never written.
 */
export async function saveFolderKeypoints(root: string, schema: KeypointSchema): Promise<FolderKeypoints> {
  const text = writePoseYaml(schema)
  const read = readPoseYaml(text).schema
  await fs.mkdir(meta(root), { recursive: true })
  await writeAtomic(keypointsFile(root), text)
  return { schema: read, file: keypointsFile(root), problem: null }
}
