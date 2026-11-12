"""Reads YOLO datasets Seamtag wrote with Ultralytics 8.4.211's own code, without torch: data.yaml as its
YAML reader and check_class_names take it, every image found and paired with its label file by
img2label_paths, and each pair through verify_image_label with the real check_image. Also reads the
names with PyYAML, as X-AnyLabeling, CVAT and older Ultralytics do.

    uv run --no-project --python 3.12 --with numpy==2.3.4 --with pyyaml==6.0.3 --with pillow==12.0.0 python -I tests/parity/ultralytics_reads.py DIR...

Each DIR is a dataset with DIR.expected.json beside it: its task, names, kpt_shape, flip_idx and the
rows each label file holds, which tests/unit/ultralytics.test.ts writes when YOLO_PARITY_DIR is set
(e2e/export.spec.ts writes one for each dataset the app exports). Fails if Ultralytics finds an image
or label corrupt, drops a repeated row, reads a row other than as written, or would stop on the
dataset as a whole. For an image whose entry lists `joined` objects (a class and its outlines' parts
in pixels), the label file must hold the row Ultralytics' own merge_multi_segment makes of the parts,
as convert_coco joins a COCO object's polygons, written as Seamtag writes rows.

Ultralytics is AGPL-3.0, so its files are fetched at commit d2b7876 (v8.4.211, whose annotated tag
is 483bc1b), checked against their SHA-256 and never kept. check_det_dataset, get_img_files and
verify_labels need much of Ultralytics around them, so the few rules of theirs used here are written
again below from what they do.
"""
import ast
import glob
import hashlib
import json
import os
import re
import sys
import types
import urllib.request
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from PIL import Image, ImageOps

COMMIT = "d2b7876e79ad2e02fd77221622774b511a8f0cb3"
SOURCES = {
    "ultralytics/data/utils.py": "12ce815a5e1055b337ee7ad1613e4ddcc4c909322b674e283837183f2ac61c18",
    "ultralytics/utils/ops.py": "50267c27f8364c3c895b78986c7006b72d9c26d273ccd422cba5776d26c29039",
    "ultralytics/nn/autobackend.py": "e4086e22403440f8293815ee585d41865a6d360deb82776ed01b3245a5c64c26",
    "ultralytics/utils/__init__.py": "33647de7592615e51485e9c577c92bdb31670ecf373b8e333aa3448df92dd0fc",
    "ultralytics/data/converter.py": "132871c377275302830ad976d9c11107f19e842e9ad1ff784e6362182fe38580",
}
LIFT = {
    "ultralytics/data/utils.py": ["IMG_FORMATS", "VID_FORMATS", "FORMATS_HELP_MSG", "DATASET_KEY_TYPES", "img2label_paths",
                                  "exif_size", "check_image", "_exif_jpeg", "_replace_image", "verify_image_label"],
    "ultralytics/utils/ops.py": ["xyxy2xywh", "segments2boxes", "empty_like"],
    "ultralytics/nn/autobackend.py": ["check_class_names"],
    "ultralytics/utils/__init__.py": ["YAML"],
    "ultralytics/data/converter.py": ["min_index", "merge_multi_segment"],
}


def fetch(path):
    url = f"https://raw.githubusercontent.com/ultralytics/ultralytics/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCES[path]:
        sys.exit(f"{path}: SHA-256 {digest}, expected {SOURCES[path]}")
    return data.decode("utf-8")


def lift(text, names):
    """The top-level functions, classes and assignments named, in the file's order"""
    out = []
    for node in ast.parse(text).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            out.append((name, ast.get_source_segment(text, node)))
    missing = set(names) - {n for n, _ in out}
    if missing:
        sys.exit(f"not found: {sorted(missing)}")
    return out


def no_imread(*_):
    raise AssertionError("imread is only for rotated WebP and TIFF images, which these datasets don't have")


ns = {
    "np": np, "os": os, "re": re, "json": json, "Path": Path, "Image": Image, "ImageOps": ImageOps, "uuid4": uuid4,
    "imread": no_imread, "torch": types.SimpleNamespace(Tensor=type("Tensor", (), {})),
}
for path, names in LIFT.items():
    for _, code in lift(fetch(path), names):
        exec("from __future__ import annotations\n" + code, ns)
YAML, check_class_names, verify_image_label, img2label_paths = ns["YAML"], ns["check_class_names"], ns["verify_image_label"], ns["img2label_paths"]
merge_multi_segment = ns["merge_multi_segment"]
IMG_FORMATS, DATASET_KEY_TYPES = ns["IMG_FORMATS"], ns["DATASET_KEY_TYPES"]


def fail(message):
    sys.exit(f"FAIL {message}")


def joined_row(cls, parts, width, height):
    """An object's parts, each kept within the image as Seamtag's COCO export keeps them (its points past an
    edge moved onto it), so as convert_coco would read them, joined by merge_multi_segment as convert_coco
    joins them, then written as Seamtag writes a row: each point a share of the image with '%.6f', a point
    that repeats the one before it or the first point at the end left out"""
    inside = [[v for x, y in part for v in (min(max(x, 0), width), min(max(y, 0), height))] for part in parts]
    points = np.concatenate(merge_multi_segment(inside), axis=0)
    shares = [("%.6f" % (x / width), "%.6f" % (y / height)) for x, y in points.tolist()]
    kept = [p for i, p in enumerate(shares) if i == 0 or p != shares[i - 1]]
    while len(kept) > 1 and kept[0] == kept[-1]:
        kept.pop()
    return " ".join([str(cls), *(v for p in kept for v in p)])


def dataset_yaml(folder):
    """check_det_dataset's checks on data.yaml (ultralytics/data/utils.py), as far as a local dataset goes"""
    file = folder / "data.yaml"
    data = YAML.load(file, append_filename=True)
    for key, kinds in DATASET_KEY_TYPES.items():
        if data.get(key) is not None and not isinstance(data[key], kinds):
            fail(f"{file}: {key} is {type(data[key]).__name__}")
    for key in "train", "val":
        if key not in data:
            fail(f"{file}: no {key}")
    if "nc" in data and (not isinstance(data["nc"], int) or len(data["names"]) != data["nc"]):
        fail(f"{file}: nc {data['nc']!r} and {len(data['names'])} names")
    data["names"] = check_class_names(data["names"])
    if data.get("path"):
        fail(f"{file}: has path {data['path']!r}, so its folders depend on where Ultralytics runs")
    root = Path(data["yaml_file"]).parent
    for key in "train", "val", "test":
        if isinstance(data.get(key), str):
            x = (root / data[key]).resolve()
            if not x.exists() and data[key].startswith("../"):
                x = (root / data[key][3:]).resolve()
            data[key] = str(x)
    return data


def images_of(entry):
    """get_img_files (ultralytics/data/base.py) for a folder: every file below it whose extension is an image's"""
    files = glob.glob(str(Path(glob.escape(entry)) / "**" / "*.*"), recursive=True) if os.path.isdir(entry) else [entry]
    return sorted(f for f in files if f.rpartition(".")[-1].lower() in IMG_FORMATS)


def check(folder):
    folder = Path(folder.rstrip("/\\")).resolve()
    with open(f"{folder}.expected.json", encoding="utf-8") as f:
        expected = json.load(f)
    task = expected["task"]
    data = dataset_yaml(folder)
    want_names = {i: n for i, n in enumerate(expected["names"])}
    if data["names"] != want_names:
        fail(f"{folder}: Ultralytics reads names {data['names']}, Seamtag wrote {want_names}")
    with open(folder / "data.yaml", encoding="utf-8") as f:
        pyyaml_names = check_class_names(yaml.safe_load(f)["names"])
    if pyyaml_names != want_names:
        fail(f"{folder}: PyYAML reads names {pyyaml_names}, Seamtag wrote {want_names}")
    nkpt, ndim = data.get("kpt_shape") or (0, 0)
    if task == "pose":
        # ultralytics/data/dataset.py checks kpt_shape; augment.py checks flip_idx against it when it mirrors
        if nkpt <= 0 or ndim not in {2, 3} or [nkpt, ndim] != expected["kpt_shape"]:
            fail(f"{folder}: kpt_shape {data.get('kpt_shape')}")
        if data.get("flip_idx", []) != (expected["flip_idx"] or []) or (data.get("flip_idx") and len(data["flip_idx"]) != nkpt):
            fail(f"{folder}: flip_idx {data.get('flip_idx')}")

    found = sorted({f for key in ("train", "val", "test") if data.get(key) for f in images_of(data[key])})
    listed = sorted(str(folder / rel) for rel in expected["images"])
    if found != listed:
        fail(f"{folder}: Ultralytics finds images {found}, Seamtag wrote {listed}")
    boxes = segments = rows = merged = 0
    for rel, want in expected["images"].items():
        image = str(folder / rel)
        label = img2label_paths([image])[0]
        im, lb, shape, segs, kpts, nm, nf, ne, nc, msg = verify_image_label((image, label, "", task == "pose", len(want_names), nkpt, ndim, False))
        if nc:
            fail(f"{rel}: {msg}")
        if msg:
            fail(f"{rel}: Ultralytics says {msg}")
        status = "missing" if nm else "empty" if ne else "ok"
        if status != want["status"]:
            fail(f"{rel}: {status}, expected {want['status']}")
        # The image upright, as its EXIF orientation shows it, which Seamtag's shares are of: orientations 5 to 8 swap its sides
        with Image.open(image) as pil:
            upright = (pil.width, pil.height) if pil.getexif().get(274, 1) in (5, 6, 7, 8) else (pil.height, pil.width)
        if shape != upright:
            fail(f"{rel}: check_image gives {shape}, not the upright image's {upright}")
        written = [np.asarray(r, dtype=np.float32) for r in want["rows"]]
        if len(lb) != len(written):
            fail(f"{rel}: {len(lb)} rows, {len(written)} written")
        for i, row in enumerate(written):
            if task == "detect":
                same = np.array_equal(lb[i], row)
            elif task == "pose":
                same = np.array_equal(lb[i], row[:5]) and np.array_equal(kpts[i], row[5:].reshape(nkpt, ndim))
            else:
                same = lb[i][0] == row[0] and np.array_equal(segs[i], row[1:].reshape(-1, 2))
            if not same:
                fail(f"{rel} row {i + 1}: Ultralytics reads {lb[i]}{'' if task in ('detect', 'pose') else f' {segs[i]}'}, Seamtag wrote {row}")
        if want.get("joined"):
            with open(label, encoding="utf-8") as f:
                text_rows = f.read().splitlines()
            for obj in want["joined"]:
                row = joined_row(obj["class"], obj["parts"], shape[1], shape[0])
                if row not in text_rows:
                    fail(f"{rel}: no row joins class {obj['class']}'s {len(obj['parts'])} parts as merge_multi_segment does: {row}")
                merged += 1
        boxes += len(lb)
        segments += len(segs)
        rows += len(written)
    # verify_labels (ultralytics/data/dataset.py): segment and OBB datasets stop when boxes and outlines differ in number
    if task in ("segment", "obb") and boxes != segments:
        fail(f"{folder}: {boxes} boxes and {segments} outlines")
    if task == "detect" and segments:
        fail(f"{folder}: {segments} outlines in a detect dataset")
    joins = f", {merged} joined as merge_multi_segment joins them" if merged else ""
    print(f"ok {folder.name} ({task}): {len(listed)} images, {rows} rows{joins}, names {list(want_names.values())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        check(arg)
