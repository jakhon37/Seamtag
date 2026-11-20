# Changelog

All notable changes to Seamtag are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Keypoints and poses: the Pose tool, skeletons on the canvas, AI pose with RTMPose, and checks and statistics for poses.

### Added
- The Pose tool (`K`): a box, then each keypoint as it's asked for; Shift+click for a hidden one, `X` to skip one, `Backspace` to go back, `Enter` to finish. A faint guide shows where the last pose's keypoints went.
- Poses drawn as skeletons: lines between the keypoints, left, right and middle in their own colours, hidden points hollow, at the same size at any zoom. With a pose selected, the others dim and its keypoints show their numbers.
- A pose selected by its box moves, nudges, copies, pastes, duplicates and deletes as one; a copy takes a new group ID, after a pose being placed. `Q` hides or shows a keypoint, `M` swaps left and right, `Shift+M` mirrors a pose in its box, `N` and `Shift+N` step through its keypoints.
- The Shapes list folds each pose into one row with its count of keypoints, opened to its keypoints in order, missing ones included.
- AI pose (`Shift+K`) and Pose every person: RTMPose-m or RTMPose-l finds a person's 17 keypoints in a box, or in each person of the image without keypoints, a group's shapes of the class as one (found with YOLO-World, from a score of 0.25, on an image with none). Points scored below `ai.pose_score` are left out, as are points past the photo's edge, and so is a pose with fewer than 5 points. A pose is reviewed as one, by its mean score, and accepted or rejected as one; an edit to any of it accepts it. The models are downloaded from OpenMMLab when you agree, with Hugging Face's copy if OpenMMLab can't be reached.
- The folder's keypoints file, `.myseg/keypoints.yaml`, set under Dataset → Keypoints… from a file or made from the folder's poses; COCO's 17-point person until there is one.
- Check labels finds keypoints outside their box, repeated (with a Fix) or not their class's, keypoints with no group or no box, and a likely left-right swap.
- Statistics counts each keypoint seen, hidden and not labelled, in the window and the CSV.

### Changed
- A pose's keypoints aren't shapes of a class: the Labels list, its keys and counts and the image list's counts leave them out, and `validate_label: exact` takes the keypoints file's names. A box or a point named like a keypoint outside a pose is of a class as before.
- Pascal VOC, YOLO and CreateML leave points out, where they made them boxes with no size: a pose keeps its box, and the window says how many points were left out.
- Export dataset's pose formats start from the folder's keypoints file, or COCO's person, so no file need be chosen.
- AI pose keeps its own model in memory beside SAM2's, so SAM2 doesn't encode the image again after a pose.

## [3.3.0] - 2026-11-19

Datasets: a check of the whole folder's labels, statistics, and COCO and Ultralytics YOLO export and import. Seamline and Seamtag on one folder: the models Seamline exports are offered for pre-labelling, and Seamtag never writes over labels or classes another app changed. Seamtag's licence and notices, in About and in the installers.

### Added
- Dataset → Check labels: 23 checks over every label file in the folder, listed as errors, warnings and notes in a Problems panel under the shapes (View → Problems). A problem opens its image with the shape selected, and This image only narrows the list to the open image. Repeated points, exact copies and boxes past the image's edge have a Fix, undone like any edit. An image is checked again as it's saved.
- Renaming a label in every label file of the folder, from the Problems panel's rows for a label written several ways or not in the label list, or from any label in Statistics: the window lists the images it changes first, the old files go to the trash, and the folder's label list follows once no file has the old name (labels from the config stay).
- A Problems filter in the image list: the images with an error or a warning in the last check.
- Dataset → Statistics: images by state; objects and images by label (shapes sharing a label and group ID are one object, as the exports count them), with each label's shapes by type; objects per image; object sizes as COCO ranges them at full size and at a training size; and image sizes. Save as CSV writes the labels table, which Excel reads with any label.
- File → Export dataset (also under Dataset): COCO instances and keypoints, and Ultralytics YOLO detect, segment, pose and rotated boxes with `data.yaml`. Whole, split by ratio with a seed, or split as the folders are; images copied, hard-linked or, for COCO, left in place. Everything is checked before anything is written, and any error stops the export, listed first with what fixes it: AI proposals not yet accepted (unless Leave unaccepted AI proposals out is ticked), label files that give another image size, label files or images that can't be read, images sharing a label file or written to one path, and a missing keypoints file for COCO with keypoints or YOLO pose. The dataset is written into a hidden folder and moved into place at the end. Shapes labelled `__ignore__` or `_background_` are left out: they mark regions to ignore and the background, not objects. Make a keypoints file writes X-AnyLabeling's pose file from the folder's posed objects.
- File → Import dataset (also under Dataset): a COCO file (instances or person_keypoints, as pycocotools, CVAT, labelme2coco and Seamtag write them), or a YOLO dataset in Ultralytics', Roboflow's, CVAT's or X-AnyLabeling's layout, read into each image's own label file as LabelMe JSON. Images are matched by path, or by a file name only one image on each side has; any other is listed and left out, never guessed. Images that already have labels are skipped, added to, or replaced; replaced files, and Pascal VOC, YOLO or CreateML files that are added to, go to the trash. COCO files are read in a worker with a memory limit of its own, and datasets from elsewhere are kept within bounds, so a file too big or malformed is refused with a reason instead of taking the app down.
- CI: pycocotools 2.0.11 and Ultralytics 8.4.211's own label check read what the app exports, from the unit tests and from the Export dataset window.
- NOTICE.md: Seamtag's copyright and licence, and the code from other projects it includes, each with its licence.
- Licence and Third-party notices buttons in Help → About Seamtag: the first shows LICENSE, the second NOTICE.md and THIRD-PARTY-NOTICES.txt.
- The installers carry LICENSE, NOTICE.md and THIRD-PARTY-NOTICES.txt in the app's `resources/legal` folder; on macOS, Electron's and Chromium's licences are there too. THIRD-PARTY-NOTICES.txt is made at each build and gives the full licence texts of the packages in the app, libvips and the libraries in it, ONNX Runtime and Electron.
- AI → Model for pre-labelling lists the models Seamline exported from the open folder's training runs with Export for Seamtag, newest first, each named by its run, architecture, encoder and best IoU. Newest run, at the top, takes the newest run the folder has when pre-labelling starts; it's ticked until a run or a model is chosen for the folder, and choosing it goes back to it. A model chosen in the menu or with Choose a model… is kept for that folder; a folder with no runs takes the model last picked with Choose a model…, and a run another folder exported says which folder. The list is read again when another folder opens and when the window comes back into focus.
- When the window comes back into focus, the image on screen is reloaded at once if another app changed its labels and nothing in Seamtag is unsaved. Then the image list picks up images another app added or removed and label files it changed, and the label list picks up classes it added, deleted or recoloured in `.myseg/project.json`. Saving by hand, unsaved changes to an image whose labels changed get a warning at once.

### Changed
- Windows over the page (About, Keyboard shortcuts, AI models, Add points to edge, Brightness and contrast, the line and fill colour windows, and the new ones) keep the keyboard inside them: Tab goes round them, Escape closes them, and the page behind takes no clicks, keys or dropped files until they close.
- Tab skips to the next AI proposal when the image has the focus; in the panels it moves the focus, as in other apps.
- A YOLO label file holding Ultralytics' outline, rotated box or keypoint rows says so when it opens, and points to File → Import dataset.
- Closing the window while an import, a rename or an export is under way stops it and waits, ten seconds at most: an import or a rename finishes the image it's on, and an export removes what it wrote.
- Help → About Seamtag gives the copyright and the licence, the GNU GPL version 3 only with no warranty, and says Seamtag includes code from other projects under their own licences. On macOS, the app menu's About panel gives the same, and where the notices are.
- The start screen names the four formats labels are saved in, and says folders labelled in labelme or LabelImg open as they are.
- The question before a model download, the Save format tooltip and `--help`'s line on the class file and save folder are reworded.
- labelme's example files and two of its default configs are no longer kept in the repository; the tests fetch the ones they read from labelme's repository, checked by hash.
- Seamtag keeps one window. Started again while it's open, as Seamline's Label in Seamtag starts it, it brings that window forward and opens the image or folder there, where 3.2.0 opened a second window whose settings and saves crossed the first's. Unsaved work is saved first, or asked about when saving by hand, and pre-labelling stops for another folder only if you agree. Nothing opens while a dialog or menu is open or a shape is being drawn, or when the image or folder isn't there: Seamtag says why instead. Options other than the image, folder and `--output` apply only when Seamtag starts.
- A model Seamline exported goes by its run in the pre-label buttons' tips, rather than as model.onnx.
- Started on an image in a subfolder of a Seamline project, one with `.myseg/project.json` in a folder above, as Seamline's Label in Seamtag starts it, Seamtag opens the whole project on that image, as Seamline lists it, rather than the subfolder. That goes for a start while Seamtag is open on another folder too. Open recent then lists the project, which opens on that image, and so does the last folder, which a start with no file opens, after such a start while Seamtag was open. File → Open and a dropped image open the image's own folder, as in labelme, and are remembered that way.
- AI → Choose a model… opens in the folder's `.myseg/runs`, where Seamline exports its runs' models, and shows hidden files.
- Switching to a format the image already has labels in opens them without moving the view.
- Pre-labelling measures a shape's area as Seamline does, through the centres of its outer pixels, and drops shapes under 25 rather than 20 (`prelabel.min_area`). A part inside another part's hole is left out, as the other's shape covers it. Parts of a class that touch only at a corner each get a shape of their own, where Seamline makes one shape of them.

### Fixed
- Saving a label file on Windows no longer fails when an antivirus scanner, the search indexer or OneDrive holds it for a moment: the rename is tried again for a few seconds.
- A label file another app changed after Seamtag read it, such as Seamline saving the same image, is no longer written over or trashed: saving, switching formats, pre-labelling the image on screen, Delete label file and Delete image leave it as it is, and the window reloads it and says so. The image list and the label list count the labels reloaded at once, and what the window says goes once another image is on screen. A change is found by what the file holds, so it counts even when a coarse file clock gives the file the same size and time. Undo doesn't reach back past such a reload.
- Adding or recolouring a label no longer drops classes another app, such as Seamline, added to `.myseg/project.json` meanwhile, or brings back ones it deleted: Seamtag's change is merged with the file as it is, and the file's other keys are kept.
- Adding a label, or giving one a colour, wrote every label in the label list into `.myseg/project.json` as a class, those from the config and from label files too, and Seamline trains each class there as an output of its model. A class Seamline had deleted came back that way while label files still used it. Only that one label goes in now, and the label list still shows every label where it was. In a folder with no class list yet, the list starts as Seamline starts one, with the labels of the folder's shapes that have an area, so Seamline still trains on them.
- Opening another folder, closing the folder or changing the output folder while pre-labelling ran left it running for the folder before: its remaining images failed, or were saved in the new output folder. Seamtag now asks whether to stop pre-labelling, then waits for the image it's on.
- A byte order mark at the start of `classes.txt`, which some editors write, became part of YOLO's first class name, so a box drawn with that label was saved as a new class of the same name. The mark is now left out, as Seamline leaves it out; LabelImg keeps it. Classes a save adds go at the end of `classes.txt`, which otherwise stays as it was, line ends too, so no other class's number moves.
- Seamtag started by an app that runs in development, such as Seamline started with `npm run dev`, showed that app's development page instead of its own window. An installed Seamtag now always opens its own.
- Pre-labelled shapes covered a column and a row of pixels more than the model found, at their right and bottom, once filled as labelme and Seamline fill them for training, and small ones came out skewed. They now run through the centres of the outer pixels, as Seamline's own pre-labelling does.
- The image list's counts follow a `classes.txt` another app rewrote within one tick of a coarse file clock.
- The label picked for new shapes is kept by its name. When labels came into the list or left it, as when another app adds or deletes a class, new shapes took whichever label then stood in its place. A label picked that has gone from the list isn't used: new shapes ask for theirs.
- Pre-labelling a big image, or one that a model early in training makes noisy, held up everything else Seamtag does for seconds or minutes at a time, saves included. Shapes are now outlined in a thread of their own, so Seamtag keeps saving and answering, a second start too. Opening a big folder no longer asks after each image's label files one by one, which is slow on a network share, but reads each folder's list of files once.
- Choosing another pre-labelling model while pre-labelling ran made the image in progress fail. The run keeps its model, and the one chosen is used from the next run on.
- Pre-labelling resized each tile to the model's input with libvips, which smooths as it shrinks, and its results back in a way of its own, so a model Seamline exported found other pixels than in Seamline: on real photos, other shapes, other counts and other scores. Both resizes now follow Seamline's, with OpenCV's bilinear and nearest ported to the bit, so the two apps propose the same shapes with the same scores, parts that touch only at a corner aside. Each app draws a shape's outline its own way, so the points differ a little.
- Pre-labelling wrote a label file with no shapes for an image the model found nothing on, so the image counted as labelled and done, and Pre-label N images left it out, though no one had looked at it. Nothing is written now, as in Seamline: the image stays to label, and an image with an older model's proposals keeps them. Pre-label on one image says so, in Seamline's words, such as "The model found nothing on a.png, so its labels stay as they were." A label file emptied while pre-labelling runs, as by rejecting every proposal, stays empty.
- Saving automatically, a shape drawn on an image with no label file and then undone or deleted left a label file with no shapes. That reads as an image someone cleared, so it counted as done and Pre-label N images left it out, in Seamline too. No file is written now for an image that has none while it has no shapes and no flags, and the file saved meanwhile goes again, unless it was saved by hand or another app changed it. Saving by hand (Save, Ctrl+S, or Save when asked) still writes the empty file, as labelme does; labelme's automatic save writes it too.
- The README's example of a pre-labelling model's JSON gave ImageNet's mean and spread, and the code said Seamline trains with them. Seamline's runs take pixels scaled to 0-1, as the model.json it exports says; a model without one still gets ImageNet's. Models must give raw scores, as Seamline's export does, since Seamtag applies the softmax itself.
- Sam2 (balanced) and Sam2 (accuracy) no longer close the app when AI points or AI box segments a second image. ONNX Runtime asked for memory in one block of 1 GiB, more than Electron gives at once, and Electron closed the app rather than refuse it. The AI models now take memory tensor by tensor and give it back after each image, so SAM2 also holds about 1 GB less between images.

## [3.2.0] - 2026-11-10

AI points, AI box and the text prompt, drawing rotated boxes, locking shapes, and new options and settings, with two fixes: the mouse no longer locks on Linux (X11) when an AI tool asks to download its model, and SAM, which closed the app when chosen, is gone.

### Added
- AI points (clicks) and AI box (a box), each making a polygon, mask, rectangle, circle or rotated box of what the model finds. Polygons follow every part of the mask, grouped by one group ID, as `mask_polygonization.detail` sets; repeats of one object are dropped, and with `ai.suppress_existing_shape_matches` so are objects already drawn, whose shapes are selected instead.
- AI text prompt: words such as `dog,cat` find and box everything they name with YOLO-World, each box labelled with its word, with its score in the description, and with score and overlap thresholds to set. Boxes over shapes of that label already drawn aren't made again. SAM 3 isn't offered there, as its image encoder needs larger blocks of memory than Electron's allocator gives.
- AI → AI models: every model with what it takes, its size and licence; download, cancel or delete it. The files stay in osam's cache, which labelme shares.
- Rotated boxes (`oriented_rectangle` shapes) are drawn with three clicks, reshaped from a corner with the opposite one fixed, and turned by handles on their sides; an arrow shows the first side.
- Lock shapes (`Ctrl+Shift+L`), Fit to selection (`Ctrl+Shift+F`), selecting the next shape that way with `Ctrl`+arrow, Add point to edge on the right-click menu, and merging masks of one label.
- `Space` finishes a polygon, line strip or AI points shape being drawn.
- The command line takes labelme 7's spellings (`--with-image-data`, `--no-auto-save`, `--no-sort-labels`, `--label-flags`, `--validate-label`) beside labelme 5's.
- Config keys as labelme 7 names them: `with_image_data`, `shape_color` with its `auto`, `uniform` and `by_label` modes, `shape.show_labels`, `mask_polygonization.detail`, `ai.text_model`, `ai.suppress_existing_shape_matches`, `canvas.allow_out_of_bounds_points` and the new crosshair and shortcut names. Files written for labelme 5 are migrated as labelme 7 migrates them.
- A footer under the image list: how many images are annotated, and whether their label files are next to them or in an output folder, with Change… and Next to images.

### Changed
- AI polygon and AI mask are AI points with a polygon or mask chosen; their shortcut keys carry over.
- Hiding and showing shapes is undone like any change, and applies to every selected shape.
- The shape list shows each shape's flags after its label, and labels drawn on shapes show their group ID.
- `Ctrl+Shift+L` locks shapes and `Ctrl+Shift+F` fits the selection; the label panel and Fit width have no key, and docs/config.md has the lines that give them LabelImg's. Keep previous annotation has no key.

### Fixed
- The question to download a model, asked on the first click of an AI tool, no longer locks the mouse on Linux (X11), where no window could be clicked until the question was closed from the keyboard. Questions and file dialogs now open once the mouse button is up.
- SAM, in all three sizes, is no longer offered for AI points and AI box: inside Electron its image encoder asks for a block of memory (about 800 MB) larger than Electron's allocator gives, and choosing it closed the app. A config or an earlier choice that names it gets SAM2 (balanced).
- The image an AI model sees is resized exactly as Pillow 13 resizes it, which runs the vertical pass first only when the height shrinks more than twice as much as the width. Some sizes came out one level apart, which changed YOLO-World's scores.

## [3.1.0] - 2026-11-10

Pascal VOC, YOLO and CreateML files, box tools, and a mark of Seamtag's own.

### Added
- Pascal VOC, YOLO with `classes.txt`, and CreateML, read and written byte for byte as LabelImg writes them; CreateML files keep the verified mark, as LabelImg's master branch writes it. An image keeps the format of its label file; the top bar and File → Save format switch it, moving the old file to the trash, or opening the labels the image already has in the new format. The `label_format` key sets the format of new label files.
- Verify Image on a tap of `Space`, the difficult flag, Draw Squares and squares with Ctrl held, all four corners of a box, the box's size in the status bar, labels drawn on shapes, Copy previous image's shapes, Open Annotation, Delete Image (to the trash), a label filter over the shape list, Single Class Mode, Advanced Mode, box and shape colours, right drag to copy or move, and Reset Settings.
- LabelImg's keys where Seamtag had none: `Ctrl+R`, `Ctrl+E`, `Ctrl+J`, `Ctrl+H`, `Ctrl+A`, `Ctrl+L`, `Ctrl+Shift+O`, `Ctrl+Shift+R`, `Ctrl+Shift+P` and `Ctrl+Shift+L`. docs/config.md lists the lines that switch the rest to LabelImg's.
- LabelImg's command line: an image folder, a file of classes, and a folder for the labels.

### Changed
- A mark of its own: a seam, with cloth in the app's blue and a label sewn in. The installers carry a small version where icons are drawn at 32 px and under.
- A right click on a shape selects it before the menu opens, and a right drag that starts on a shape carries a copy of it instead of panning.
- In an output folder, images of the same name in different folders each keep their own label file, instead of sharing one straight in the folder.

## [3.0.0] - 2026-11-10

Renamed Seamtag, and rewritten as an Electron app in place of the Qt app.

### Added
- Polygons, rectangles, circles, lines, line strips, points and masks, and their editing: the label dialog with completion, flags per label pattern, group ID, description and validation; image flags; the shape list with visibility and order; copy and paste across images; merge; adding points along an edge; keep previous annotation; brightness and contrast; output folders; image data in label files; label files opened on their own; images in subfolders.
- labelme's command line and config keys. Settings live in `~/.seamtagrc`; `--config ~/.labelmerc` reads a labelme config as it is.
- AI polygon and AI mask with eight SAM-family models (the ONNX files labelme uses), through onnxruntime, sharing osam's model cache. On the test photo the masks equal osam's, pixel for pixel.
- Pre-labelling with any ONNX segmentation model, with scores, and review of the proposals: `W` accept, `S` reject, `Tab` skip, least certain first.
- Brush and eraser.
- Rotated boxes and locked shapes, saved as labelme 7 saves them.
- Installers for Windows, macOS (Apple silicon and Intel) and Linux (AppImage, .deb).

### Changed
- Label files are written as labelme writes them, byte for byte, and saved through a temporary file. A file that can't be read is shown read-only and never written over.
- Deleting a label file moves it to the trash.
- Large images open from tile pyramids instead of being downsampled.

### Removed
- The Qt app and its Python package. labelme's export scripts in `examples/` still work on these files with `pip install labelme`.

## [2.x] - the Qt app

### Added
- **Performance Optimizations** for large images and many polygons:
  - **Smart Image Loading**: Use QImageReader for 3-5x faster loading (10s → 2-3s)
  - **LRU Image Cache**: Instant navigation to recently viewed images (100x faster)
  - **Viewport Culling**: Only render visible polygons (5-10x faster with 100+ polygons)
  - **Adaptive Image Downsampling**: Auto-downsample large images (4x less memory, 2-5x faster)
  - Configurable performance settings in config
  - Smooth performance even with 16000×8000 images and 500+ polygons
- **Enhanced Zoom Capabilities**: Increased maximum zoom from 1000% to 5000% (50x magnification) for tiny defect annotation
  - Configurable `max_zoom` setting (default: 5000%)
  - Configurable `zoom_increment` for smoother control (default: 1.1)
  - Better support for microscopic defect inspection
- **Multiple Point Addition Feature (`Ctrl+M`)**: Add multiple evenly-spaced points along polygon edges for refined annotation
- **Polygon Merge Feature (`Ctrl+Shift+M`)**: Merge multiple polygons with intelligent handling of connected/disconnected regions
- **Dark Mode**: Eye-friendly dark theme inspired by VS Code (toggle via View menu)
- **Improved Default Settings**: Better defaults for production use
  - `auto_save` now `true` by default (never lose work)
  - `store_data` now `false` by default (smaller JSON files)
  - `keep_prev_brightness_contrast` now `true` (consistent visuals)
  - `canvas.fill_drawing` now `false` (see through shapes while drawing)
  - `canvas.num_backups` increased to 20 (more undo history)
- **Better Keyboard Shortcuts**: More intuitive shortcuts that don't conflict
  - `Ctrl+Shift+C` for Create Polygon (was `Ctrl+N`)
  - `Ctrl+Shift+X` for Create Rectangle (was `Ctrl+R`)
  - `Ctrl+Shift+V` for Edit Polygon (was `Ctrl+J`)
- Pre-commit hooks for automated code quality checks
- `.editorconfig` for consistent coding style across editors
- `.gitattributes` for consistent line endings
- Comprehensive test suite with 38 tests (100% passing)
- Production-level code quality (all linting checks pass)

### Changed
- Refactored polygon editing logic for better maintainability
- Improved error messages and user feedback
- Updated tests to reflect new default settings
- Consolidated documentation structure

### Fixed
- Test suite now fully working with all dependencies properly configured
- All code style and linting issues resolved
- Line length issues in message dialogs fixed

## [0.0.1] - 2026-02-25

### Added
- Initial fork (see [NOTICE.md](NOTICE.md))
- Basic dark mode implementation
- File reorganization

---

## Contributing

For the development workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).
