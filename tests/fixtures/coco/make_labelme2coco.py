"""Runs labelme 7.9.1's own labelme2coco.py on LabelMe files holding every shape type with an
inside, for tests/unit/coco.test.ts: Seamtag's COCO export must outline and group them as it
does (areas and boxes aside, which labelme2coco takes from Pillow's drawing).

    uv run --no-project --python 3.12 --with labelme==7.9.1 --with pycocotools==2.0.11 --with numpy==2.5.4 --with pillow==13.0.0 --with imgviz==2.1.0 python -I make_labelme2coco.py [EXAMPLES]

labelme2coco.py and the examples/utils.py it imports aren't in labelme's package, so they're
read from EXAMPLES, the examples folder of a labelme v7.9.1 checkout, or else fetched from that
tag; either way they must match the SHA-256 sums below. Writes labelme2coco.json beside this
file: the LabelMe files, the pixels of their masks, and labelme2coco's annotations.json.
"""
import hashlib
import importlib.metadata
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import PIL.Image

here = Path(__file__).parent
TAG = "v7.9.1"
SOURCES = {
    "instance_segmentation/labelme2coco.py": "457e789696223f81c2659b2a908b601863e1ebbdfacf67c4611fb485236414ff",
    "utils.py": "e9f97a2e3d1872c5c4dbcd09c9fa9a93650d8195034f7436aee5079acbc23c2d",
}
LABELS = ["cat", "dog", "ball", "plate", "blob"]


def fetch(examples: Path):
    for name, digest in SOURCES.items():
        if len(sys.argv) > 1:
            data = (Path(sys.argv[1]) / name).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/wkentaro/labelme/{TAG}/examples/{name}"
            data = urllib.request.urlopen(url, timeout=30).read()
        assert hashlib.sha256(data).hexdigest() == digest, f"{name} isn't labelme {TAG}'s"
        (examples / name).parent.mkdir(parents=True, exist_ok=True)
        (examples / name).write_bytes(data)


def png_b64(mask):
    import base64

    out = io.BytesIO()
    # A greyscale PNG of 0s and 1s, as Seamtag writes a mask shape
    PIL.Image.fromarray(mask.astype(np.uint8)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def shape(label, points, shape_type, group_id=None, mask=None):
    return {"label": label, "points": [[float(x), float(y)] for x, y in points], "group_id": group_id, "description": "",
            "shape_type": shape_type, "flags": {}, "mask": None if mask is None else png_b64(mask)}


def rotated(cx, cy, half_w, half_h, degrees):
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return [[cx + dx * c - dy * s, cy + dx * s + dy * c] for dx, dy in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]]


def pattern(h, w, seed):
    yy, xx = np.mgrid[:h, :w]
    return ((xx * 3 + yy * 5 + seed) % 7 < 4) | ((xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(h, w) / 3) ** 2)


masks = {"a": pattern(8, 10, 1), "b": pattern(8, 10, 4)}
files = {
    "a": (64, 48, [
        shape("cat", [[5.5, 4.25], [25, 6], [20.75, 24.5], [7, 20]], "polygon"),
        shape("dog", [[40, 30], [30.5, 10.25]], "rectangle"),
        shape("ball", [[50, 38], [54, 41]], "circle"),
        shape("plate", rotated(18, 33, 6.5, 4, 20), "oriented_rectangle"),
        shape("dog", [[2, 40], [12, 40], [8, 46]], "polygon", 1),
        shape("cat", [[30, 40], [36, 40], [33, 46]], "polygon", 1),
        shape("dog", [[14, 41], [20, 46]], "rectangle", 1),
        shape("blob", [[44, 2], [53, 9]], "mask", mask=masks["a"]),
    ]),
    "b": (160, 120, [
        shape("ball", [[80, 60], [80, 5]], "circle"),
        shape("blob", [[10, 10], [19, 17]], "mask", 2, mask=masks["b"]),
        shape("blob", [[15, 15], [30, 15], [30, 30], [15, 30]], "polygon", 2),
        shape("ball", [[30.3, 90.7], [38.1, 95.2]], "circle"),
        shape("plate", rotated(120, 80, 20, 9, -35), "oriented_rectangle", 5),
        shape("plate", [[100, 100], [130, 100], [115, 115]], "polygon", 5),
    ]),
}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fetch(tmp / "examples")
    given = tmp / "in"
    given.mkdir()
    docs = {}
    for stem, (width, height, shapes) in files.items():
        PIL.Image.new("RGB", (width, height), (128, 128, 128)).save(given / f"{stem}.png")
        doc = {"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": f"{stem}.png", "imageData": None,
               "imageHeight": height, "imageWidth": width}
        (given / f"{stem}.json").write_text(json.dumps(doc, indent=2))
        docs[f"{stem}.json"] = doc
    (tmp / "labels.txt").write_text("\n".join(["__ignore__", "_background_", *LABELS]) + "\n")
    script = tmp / "examples" / "instance_segmentation" / "labelme2coco.py"
    subprocess.run([sys.executable, "-I", str(script), str(given), str(tmp / "out"), "--labels", str(tmp / "labels.txt"), "--noviz"],
                   check=True, capture_output=True)
    coco = json.loads((tmp / "out" / "annotations.json").read_text())
    shutil.rmtree(tmp / "out")

# The moment it ran, so the file stays the same from run to run
coco["info"]["date_created"] = "2026-11-12 00:00:00.000000"
coco["info"]["year"] = 2026
fixture = {
    "labelme": TAG.lstrip("v"),
    "pycocotools": importlib.metadata.version("pycocotools"),
    "labels": LABELS,
    "files": docs,
    # Each mask shape's pixels, as rows of 0 and 1, so the test needn't decode PNGs
    "masks": {f"{stem}.json": {str(i): ["".join("1" if v else "0" for v in row) for row in masks[stem]]
                               for i, s in enumerate(files[stem][2]) if s["shape_type"] == "mask"} for stem in files},
    "coco": coco,
}
(here / "labelme2coco.json").write_text(json.dumps(fixture, indent=1) + "\n")
print("labelme2coco made", len(coco["annotations"]), "annotations")
