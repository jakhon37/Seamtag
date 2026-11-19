# Contributing

Thanks for helping. Seamtag is an Electron app in TypeScript; it needs Node 22.

```bash
npm ci
npm run dev          # the app, reloading as you edit
npm run typecheck
npm test             # unit tests (Vitest)
npm run test:e2e     # builds, then drives the real app (Playwright)
npm run dist         # the installer for this system, in release/
```

On Linux the end-to-end tests run without a window (`--ozone-platform=headless`); set `E2E_SHOW=1` to see them. Each launch gets its own home folder and profile (`SEAMTAG_USER_DATA`), so tests never touch your settings. The SAM test needs `SEAMTAG_MODELS` set to a folder holding `efficientsam:10m`, as osam keeps it.

The tests fetch the labelme files they read (`tests/labelme-files.ts`) from GitHub the first time, so that run needs the network; later runs use the copies in `node_modules/.cache/seamtag-labelme`.

## Where things are

| Path | |
|---|---|
| `src/shared/labelfile.ts` | LabelMe JSON: reading, checking and writing, byte for byte as labelme writes it |
| `src/shared/formats.ts` | Pascal VOC, YOLO and CreateML, byte for byte as LabelImg writes them |
| `src/shared/config.ts`, `src/main/config.ts` | Config keys (labelme's, so a `~/.labelmerc` works), the command line, shortcuts |
| `src/main/project.ts`, `images.ts` | Folders, label files and the save queue; images and tile pyramids |
| `src/main/ai.ts`, `masks.ts`, `resize.ts` | AI models through onnxruntime, masks to shapes, resizing images for the models as Pillow does, and for pre-labelling as OpenCV does |
| `src/main/prelabel.ts`, `outline.ts`, `outliner.ts`, `runs.ts` | Pre-labelling with an ONNX segmentation model, its polygons outlined in a worker thread; the models Seamline exported from a folder's runs |
| `src/renderer/src/Labeler.tsx` | The canvas and everything on it |
| `src/renderer/src/shapes.ts`, `geometry.ts` | Shape geometry; brush and eraser |
| `tests/unit`, `e2e`, `tests/parity` | Unit tests, end-to-end tests, and labelme, LabelImg, pycocotools and Ultralytics reading this app's files |

## Changes

- Keep label files exactly as labelme and LabelImg read them. A change to `labelfile.ts` or `formats.ts` needs a test in `tests/unit/labelfile.test.ts` or `formats.test.ts`, and the CI jobs that load the files with labelme and LabelImg must still pass.
- Anything that saves needs an end-to-end test that reads the file afterwards.
- Code ported or translated from another project keeps that project's licence notice, and is listed in [NOTICE.md](NOTICE.md) with its licence.
- Match the code around you; run `npm run typecheck` and the tests before sending a pull request.
