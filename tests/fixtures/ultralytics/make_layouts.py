"""Writes small YOLO dataset folders laid out as tools write them, for tests/unit/ultralytics.test.ts:
Seamtag must find every image, its label file and the task, as Ultralytics would.

    uv run --no-project --python 3.12 --with pillow==12.0.0 python -I make_layouts.py

Each folder under layouts/ is made up here, with 16 x 16 grey images:
- roboflow: a Roboflow YOLOv8 download (train/, valid/ and test/, each with images/ and labels/;
  data.yaml naming ../train/images, which Ultralytics reads by dropping the ../), outlines
- cvat: a CVAT "Ultralytics YOLO" export (data.yaml with path: . and list files whose lines start
  with data/, as CVAT's datumaro writes them), boxes
- ultralytics: Ultralytics' own layout (images/train and labels/train, a nested folder, data.yaml
  with no path), keypoints with kpt_shape and kpt_names as coco8-pose.yaml has them
- bare: images/ and labels/ with a classes.txt and no data.yaml, rotated boxes
- flat: X-AnyLabeling's export, each .txt beside its image and a classes.txt, boxes
"""
import shutil
from pathlib import Path

from PIL import Image

here = Path(__file__).parent / "layouts"
shutil.rmtree(here, ignore_errors=True)


def write(path, text):
    path = here / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def image(path):
    path = here / path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), 128).save(path)


# Roboflow: outlines
write("roboflow/data.yaml", """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['head', 'helmet', 'person']

roboflow:
  workspace: seamtag
  project: hard-hats
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/seamtag/hard-hats/dataset/1
""")
write("roboflow/README.roboflow.txt", "Exported from Roboflow as YOLOv8, made up for Seamtag's tests\n")
for split, name, rows in [
    ("train", "a", "0 0.1 0.1 0.4 0.1 0.4 0.5\n2 0.5 0.5 0.9 0.5 0.9 0.9 0.5 0.9\n"),
    ("train", "b", "1 0.2 0.2 0.6 0.2 0.4 0.7\n"),
    ("valid", "c", "0 0.3 0.3 0.6 0.3 0.6 0.6 0.45 0.7 0.3 0.6\n"),
    ("test", "d", None),
]:
    image(f"roboflow/{split}/images/{name}.jpg")
    if rows is not None:
        write(f"roboflow/{split}/labels/{name}.txt", rows)

# CVAT: boxes, list files with data/ paths
write("cvat/data.yaml", "path: .\ntrain: train.txt\nval: val.txt\nnames:\n  0: car\n  1: person\n")
write("cvat/train.txt", "data/images/train/frame_000000.png\ndata/images/train/frame_000001.png\n")
write("cvat/val.txt", "data/images/val/frame_000002.png\n")
for split, name, rows in [
    ("train", "frame_000000", "0 0.500000 0.500000 0.250000 0.250000\n"),
    ("train", "frame_000001", "1 0.250000 0.250000 0.125000 0.250000\n0 0.750000 0.750000 0.250000 0.125000\n"),
    ("val", "frame_000002", None),
]:
    image(f"cvat/images/{split}/{name}.png")
    if rows is not None:
        write(f"cvat/labels/{split}/{name}.txt", rows)

# Ultralytics' own layout: keypoints, a nested folder
write("ultralytics/data.yaml", """train: images/train
val: images/val

kpt_shape: [3, 3]
flip_idx: [0, 2, 1]

names:
  0: person

kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
""")
for split, name, rows in [
    ("train", "a", "0 0.5 0.5 0.5 0.5 0.5 0.4 2 0.45 0.35 2 0.55 0.35 1\n"),
    ("train", "sub/b", "0 0.3 0.3 0.2 0.2 0.3 0.25 2 0 0 0 0 0 0\n"),
    ("val", "c", ""),
]:
    image(f"ultralytics/images/{split}/{name}.png")
    write(f"ultralytics/labels/{split}/{name}.txt", rows)

# No data.yaml: rotated boxes, classes.txt in labels/
write("bare/labels/classes.txt", "plane\nship\n")
for name, rows in [
    ("a", "0 0.2 0.1 0.5 0.3 0.4 0.45 0.1 0.25\n"),
    ("b", "1 0.6 0.6 0.9 0.7 0.85 0.85 0.55 0.75\n0 0.1 0.1 0.3 0.2 0.25 0.3 0.05 0.2\n"),
]:
    image(f"bare/images/{name}.jpg")
    write(f"bare/labels/{name}.txt", rows)

# X-AnyLabeling: labels beside the images
write("flat/classes.txt", "cat\ndog\n")
for name, rows in [("x", "0 0.500000 0.500000 0.500000 0.500000\n"), ("y", "1 0.250000 0.250000 0.250000 0.250000\n")]:
    image(f"flat/{name}.jpg")
    write(f"flat/{name}.txt", rows)

print("\n".join(sorted(str(p.relative_to(here)) for p in here.rglob("*") if p.is_file())))
