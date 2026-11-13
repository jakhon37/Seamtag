"""Runs labelme 7's AI text prompt on a photo with osam's YOLO-World, for e2e/labelme7.spec.ts:
Seamtag must find the same boxes. Needs yoloworld:latest in ~/.cache/osam (640 MB), so it's run
by hand:

    uv run --no-project --with labelme==7.9.1 python make_text_prompt.py [PHOTO]

The photo is labelme's examples/instance_segmentation/data_annotated/2011_000003.jpg,
fetched at v5.11.4's commit as the test fetches it, or read from PHOTO; either way it must
match the SHA-256 below. Writes text_prompt.json beside this file.
"""
import hashlib
import importlib.metadata
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import PIL.Image
from labelme._automation._osam_session import OsamSession
from labelme._automation._text_detection import propose_shapes_from_texts

here = Path(__file__).resolve().parent
COMMIT = "1a0daa1c90a082bb42b3b4df4f0e16e201838239"  # labelme v5.11.4
PHOTO = "examples/instance_segmentation/data_annotated/2011_000003.jpg"
PHOTO_SHA256 = "94bc550055447e1fab3982571d7114eb4a02a868054f34cb15f95cadf3e48c9a"


def photo():
    if len(sys.argv) > 1:
        data = Path(sys.argv[1]).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/wkentaro/labelme/{COMMIT}/{PHOTO}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PHOTO_SHA256:
        sys.exit(f"{PHOTO}: SHA-256 {digest}, expected {PHOTO_SHA256}")
    return PIL.Image.open(io.BytesIO(data))


def main():
    image = np.asarray(photo().convert("RGB"))
    out = {"labelme": importlib.metadata.version("labelme"), "osam": importlib.metadata.version("osam")}
    for texts in (["person"], ["person", "bottle"]):
        shapes = propose_shapes_from_texts(
            session=OsamSession(model_name="yoloworld:latest"),
            image=image,
            image_id="photo",
            texts=texts,
            shape_type="rectangle",
            existing_shapes=[],
            iou_threshold=0.5,
            score_threshold=0.1,
            image_size=(image.shape[1], image.shape[0]),
            polygon_detail=60,
        )
        out[",".join(texts)] = [
            {"label": s.label, "shape_type": s.shape_type, "points": np.asarray(s.points, dtype=np.float64).tolist(), "description": s.description}
            for s in shapes
        ]
    (here / "text_prompt.json").write_text(json.dumps(out) + "\n")


main()
