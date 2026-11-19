"""Runs Ultralytics' own label and data.yaml readers on test files, for tests/unit/ultralytics.test.ts:
Seamtag's reader must reach the same verdicts, and every file Seamtag writes must pass.

    uv run --no-project --python 3.12 --with numpy==2.3.4 --with pyyaml==6.0.3 python -I make_verify_cases.py

Ultralytics is AGPL-3.0, so none of its code is kept here. The script fetches data/utils.py,
utils/ops.py, nn/autobackend.py and utils/__init__.py of v8.4.211 (commit d2b7876, which the
annotated tag 483bc1b points to), checks their SHA-256, and lifts verify_image_label and the
helpers it calls, check_class_names and the YAML class out of them with ast. Only check_image
is replaced: it opens the image, and these cases have none. Pass a folder holding the four files
(named as on GitHub, with / as _) to read them from there instead.

Writes verify_cases.json beside this file:
- labels: a label file's text (or its bytes, when they aren't UTF-8; null for no file), the task,
  class count and kpt_shape, and what verify_image_label made of it: ok, missing, empty or
  corrupt, its message, the boxes (float32), segments and keypoints it kept, and how many
  duplicate rows it removed
- yaml_plain: what a name written without quotes in data.yaml reads as, in Ultralytics' own YAML
  reader (8.4.211 has its own, where yes, no, on and off stay text) and in PyYAML's safe_load
  (YAML 1.1, which X-AnyLabeling, CVAT and older Ultralytics read with), and how Ultralytics
  writes the name
- class_names: data.yaml names, read by Ultralytics' YAML reader and check_class_names
- data_yaml: whole data.yaml files as Ultralytics' YAML reader gives them back, or its error
"""
import ast
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np
import yaml

COMMIT = "d2b7876e79ad2e02fd77221622774b511a8f0cb3"
SOURCES = {
    "ultralytics/data/utils.py": "12ce815a5e1055b337ee7ad1613e4ddcc4c909322b674e283837183f2ac61c18",
    "ultralytics/utils/ops.py": "50267c27f8364c3c895b78986c7006b72d9c26d273ccd422cba5776d26c29039",
    "ultralytics/nn/autobackend.py": "e4086e22403440f8293815ee585d41865a6d360deb82776ed01b3245a5c64c26",
    "ultralytics/utils/__init__.py": "33647de7592615e51485e9c577c92bdb31670ecf373b8e333aa3448df92dd0fc",
}


def source(path, folder):
    if folder:
        data = (Path(folder) / path.replace("/", "_")).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/ultralytics/ultralytics/{COMMIT}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCES[path]:
        sys.exit(f"{path}: SHA-256 {digest}, expected {SOURCES[path]}")
    return data.decode("utf-8")


def lift(text, names):
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(text, node)
    missing = set(names) - set(found)
    if missing:
        sys.exit(f"not found: {sorted(missing)}")
    return found


folder = sys.argv[1] if len(sys.argv) > 1 else None
utils_py = source("ultralytics/data/utils.py", folder)
ops_py = source("ultralytics/utils/ops.py", folder)
autobackend_py = source("ultralytics/nn/autobackend.py", folder)
init_py = source("ultralytics/utils/__init__.py", folder)

ns = {
    "np": np,
    "os": os,
    # empty_like asks whether it has a torch tensor; these are numpy arrays
    "torch": types.SimpleNamespace(Tensor=type("Tensor", (), {})),
    "check_image": lambda f: ("", (480, 640)),
}
for code in lift(ops_py, ["empty_like", "xyxy2xywh", "segments2boxes"]).values():
    exec(code, ns)
exec(lift(utils_py, ["verify_image_label"])["verify_image_label"], ns)
exec(lift(autobackend_py, ["check_class_names"])["check_class_names"], ns)
verify = ns["verify_image_label"]
check_class_names = ns["check_class_names"]
yaml_ns = {"re": re, "json": json, "Path": Path}
exec("from __future__ import annotations\n" + lift(init_py, ["YAML"])["YAML"], yaml_ns)
UltralyticsYAML = yaml_ns["YAML"]

D, S, O, P = "detect", "segment", "obb", "pose"
# (name, task, class count, kpt_shape, text: str, bytes, or None for no file)
CASES = [
    ("a box", D, 3, None, "0 0.5 0.5 0.2 0.2\n"),
    ("two boxes, no line end", D, 3, None, "0 0.5 0.5 0.2 0.2\n2 0.1 0.2 0.1 0.1"),
    ("class written 1.0", D, 3, None, "1.0 0.5 0.5 0.2 0.2\n"),
    ("class 1.5", D, 3, None, "1.5 0.5 0.5 0.2 0.2\n"),
    ("class 3 of 3", D, 3, None, "3 0.5 0.5 0.2 0.2\n"),
    ("class 2 of 3", D, 3, None, "2 0.5 0.5 0.2 0.2\n"),
    ("class -0", D, 3, None, "-0 0.5 0.5 0.2 0.2\n"),
    ("class -1", D, 3, None, "-1 0.5 0.5 0.2 0.2\n"),
    ("class 1e10", D, 3, None, "1e10 0.5 0.5 0.2 0.2\n"),
    ("class 2.0000001, 2 in float32", D, 3, None, "2.0000001 0.5 0.5 0.2 0.2\n"),
    ("class 2.000001", D, 3, None, "2.000001 0.5 0.5 0.2 0.2\n"),
    ("class -0.005", D, 3, None, "-0.005 0.5 0.5 0.2 0.2\n"),
    ("exponents, tabs and spaces", D, 3, None, "0\t5e-1  0.5 \t2E-1 .2\n"),
    ("signs and bare points", D, 3, None, "+1 +.5 0.5e0 .2 0.\n"),
    ("5. is 5", D, 3, None, "0 0.5 0.5 5. 0.2\n"),
    ("spaces before and after", D, 3, None, "  0 0.5 0.5 0.2 0.2  \n"),
    ("a line of ideographic spaces", D, 3, None, "\u3000\n0 0.5 0.5 0.2 0.2\n\x1f\n"),
    ("commas", D, 3, None, "0,0.5,0.5,0.2,0.2\n"),
    ("a word", D, 3, None, "0 0.5 0.5 0.2 wide\n"),
    ("hex", D, 3, None, "0 0x1 0.5 0.2 0.2\n"),
    ("underscores", D, 3, None, "1_0 0.5 0.5 0.2 0.2\n"),
    ("underscores in a fraction", D, 11, None, "1_0 0.5_0 0.5 0.2 0.2\n"),
    ("nan", D, 3, None, "0 nan 0.5 0.2 0.2\n"),
    ("inf", D, 3, None, "0 0.5 0.5 inf 0.2\n"),
    ("-inf", D, 3, None, "0 0.5 0.5 0.2 -inf\n"),
    ("nan class", D, 3, None, "nan 0.5 0.5 0.2 0.2\n"),
    ("box edge past 1 (cx .995, w .02)", D, 3, None, "0 0.995 0.5 0.02 0.2\n"),
    ("w 1.2", D, 3, None, "0 0.5 0.5 1.2 0.2\n"),
    ("w 1.01", D, 3, None, "0 0.5 0.5 1.01 0.2\n"),
    ("w 1.0100001", D, 3, None, "0 0.5 0.5 1.0100001 0.2\n"),
    ("w 1.01000005", D, 3, None, "0 0.5 0.5 1.01000005 0.2\n"),
    ("w 1.0100004", D, 3, None, "0 0.5 0.5 1.0100004 0.2\n"),
    ("w 1.011", D, 3, None, "0 0.5 0.5 1.011 0.2\n"),
    ("cx -0.005", D, 3, None, "0 -0.005 0.5 0.2 0.2\n"),
    ("cx -0.01", D, 3, None, "0 -0.01 0.5 0.2 0.2\n"),
    ("cx -0.0100001", D, 3, None, "0 -0.0100001 0.5 0.2 0.2\n"),
    ("cx -0.0100005", D, 3, None, "0 -0.0100005 0.5 0.2 0.2\n"),
    ("cx -0.02", D, 3, None, "0 -0.02 0.5 0.2 0.2\n"),
    ("w -0.005", D, 3, None, "0 0.5 0.5 -0.005 0.2\n"),
    ("w 0", D, 3, None, "0 0.5 0.5 0 0\n"),
    ("6 values", D, 3, None, "0 0.1 0.2 0.3 0.4 0.5\n"),
    ("4 values", D, 3, None, "0 0.1 0.2 0.3\n"),
    ("1 value", D, 3, None, "0\n"),
    ("5 values and 6", D, 3, None, "0 0.5 0.5 0.2 0.2\n0 0.1 0.2 0.3 0.4 0.5\n"),
    ("duplicate rows", D, 3, None, "1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n"),
    ("duplicates once read: 0.5 and 0.50000001", D, 3, None, "0 0.5 0.5 0.2 0.2\n0 0.50000001 0.5 0.2 0.2\n"),
    ("duplicates once read: -0 and 0", D, 3, None, "0 0.5 0.5 0.2 0.2\n-0 0.5 0.5 0.2 0.2\n"),
    ("empty file", D, 3, None, ""),
    ("blank lines only", D, 3, None, "\n  \n\t\n"),
    ("no file", D, 3, None, None),
    ("blank lines around rows", D, 3, None, "\n\n0 0.5 0.5 0.2 0.2\n   \n1 0.1 0.1 0.1 0.1\n\n"),
    ("Windows line ends", D, 3, None, "0 0.5 0.5 0.2 0.2\r\n1 0.1 0.1 0.1 0.1\r\n"),
    ("old Mac line ends", D, 3, None, "0 0.5 0.5 0.2 0.2\r1 0.1 0.1 0.1 0.1\r"),
    ("byte order mark", D, 3, None, "\ufeff0 0.5 0.5 0.2 0.2\n"),
    ("form feed and vertical tab end lines", D, 3, None, "0 0.5 0.5 0.2 0.2\x0c1 0.1 0.1 0.1 0.1\x0b2 0.3 0.3 0.1 0.1\n"),
    ("file separators end lines", D, 3, None, "0 0.5 0.5 0.2 0.2\x1c1 0.1 0.1 0.1 0.1\x1d2 0.3 0.3 0.1 0.1\x1e"),
    ("unit separator splits values", D, 3, None, "0\x1f0.5 0.5 0.2 0.2\n"),
    ("next line ends a line", D, 3, None, "0 0.5 0.5 0.2 0.2\x851 0.1 0.1 0.1 0.1\n"),
    ("line separator ends a line", D, 3, None, "0 0.5 0.5 0.2 0.2\u20281 0.1 0.1 0.1 0.1\u2029"),
    ("no-break and ideographic spaces split values", D, 3, None, "0\xa00.5\u30000.5\u20030.2 0.2\n"),
    ("byte order mark between values", D, 3, None, "0\ufeff0.5 0.5 0.2 0.2\n"),
    ("zero-width space between values", D, 3, None, "0\u200b0.5 0.5 0.2 0.2\n"),
    ("Arabic-Indic digits", D, 3, None, "\u0661 0.5 0.5 0.2 0.2\n"),
    ("full-width digits", D, 3, None, "0 0.\uff15 0.5 0.2 0.2\n"),
    ("not UTF-8", D, 3, None, b"0 0.5 0.5 0.2 0.2\n\xff\xfe\n"),
    ("4-point outline in a detect run", D, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"),
    ("outlines and a 3-value row", D, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4\n1 0.5 0.5\n"),
    ("outline and a 1-value row", D, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4\n1\n"),
    ("outline and a 6-value row", D, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4\n1 0.1 0.2 0.3 0.4 0.5\n"),
    ("box rows and an outline row", S, 3, None, "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.3 0.4\n"),
    ("2-point outline (5 values)", S, 3, None, "0 0.1 0.1 0.2 0.2\n"),
    ("odd coordinate count", S, 3, None, "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4\n"),
    ("outline x 0.6 to 1.4", S, 3, None, "0 0.6 0.1 1.4 0.1 1.4 0.4\n"),
    ("outline x 0.95 to 1.2", S, 3, None, "0 0.95 0.1 1.2 0.1 1.2 0.4\n"),
    ("outline x -0.2 to 0.4", S, 3, None, "0 -0.2 0.1 0.4 0.1 0.4 0.4\n"),
    ("outline x -0.03 to 0.02", S, 3, None, "0 -0.03 0.1 0.02 0.1 0.02 0.4\n"),
    ("outline x 1 to 1.02: centre 1.01", S, 3, None, "0 1 0.1 1.02 0.1 1.02 0.4\n"),
    ("outline x 1 to 1.0200001", S, 3, None, "0 1 0.1 1.0200001 0.1 1.0200001 0.4\n"),
    ("outline x 1.0000001 to 1.02", S, 3, None, "0 1.0000001 0.1 1.02 0.1 1.02 0.4\n"),
    ("outline x 0.0100001 to 2.0099999", S, 3, None, "0 0.0100001 0.1 2.0099999 0.1 2.0099999 0.4\n"),
    ("same box, different outlines", S, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n0 0.1 0.1 0.3 0.3 0.1 0.3 0.3 0.1\n"),
    ("the same outline twice", S, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.3\n1 0.5 0.5 0.6 0.5 0.6 0.6\n0 0.1 0.1 0.3 0.1 0.3 0.3\n"),
    ("the same outline, once with -0", S, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.3\n-0 0.1 0.1 0.3 0.1 0.3 0.3\n"),
    ("outlines of different lengths", S, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.3\n2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6 0.55 0.55\n"),
    ("outline of class 1.5", S, 3, None, "1.5 0.1 0.1 0.3 0.1 0.3 0.3\n"),
    ("outline of class 3 of 3", S, 3, None, "3 0.1 0.1 0.3 0.1 0.3 0.3\n"),
    ("outline with nan", S, 3, None, "0 0.1 nan 0.3 0.1 0.3 0.3\n"),
    ("outline with inf", S, 3, None, "0 0.1 0.1 inf 0.1 0.3 0.3\n"),
    ("outline with a word", S, 3, None, "0 0.1 0.1 x 0.1 0.3 0.3\n"),
    ("outline whose class is a word", S, 3, None, "a 0.1 0.1 0.3 0.1 0.3 0.3\n"),
    ("outline of a 0-wide sliver", S, 3, None, "0 0.2 0.1 0.2 0.3 0.2 0.5\n"),
    ("outline of 3 equal points", S, 3, None, "0 0.2 0.2 0.2 0.2 0.2 0.2\n"),
    ("a triangle row in a segment run, kpt_shape [1, 2]: cut to its box", S, 3, [1, 2], "0 0.1 0.1 0.3 0.1 0.3 0.4\n"),
    ("a triangle and a square in a segment run, kpt_shape [1, 2]", S, 3, [1, 2], "0 0.1 0.1 0.3 0.1 0.3 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"),
    ("rotated box", O, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"),
    ("rotated box of 3 points", O, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4\n"),
    ("rotated box as an upright box row", O, 3, None, "0 0.5 0.5 0.2 0.2\n"),
    ("rotated box with corners past the edge", O, 3, None, "0 -0.05 0.5 0.5 -0.05 1.05 0.5 0.5 1.05\n"),
    ("the same rotated box twice", O, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"),
    ("rotated box and a box row", O, 3, None, "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n1 0.5 0.5 0.2 0.2\n"),
    ("keypoints [2, 3], one missing as 0 0 0", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 3] given x y only", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 0.5 0.5\n"),
    ("keypoints [2, 3], v 1.5 and x 1.3", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 1.3 0.45 1.5 0 0 0\n"),
    ("keypoints [2, 3], v 2.5", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2.5 0 0 0\n"),
    ("keypoints [2, 3], v -1", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 -1 0 0 0\n"),
    ("keypoints [2, 3], box w 1.5", P, 1, [2, 3], "0 0.5 0.5 1.5 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], box cx 1.2", P, 1, [2, 3], "0 1.2 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], box cx -0.005", P, 1, [2, 3], "0 -0.005 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], x -0.005", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 -0.005 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], x -0.02", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 -0.02 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], x 1.01", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 1.01 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], x nan", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 nan 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], v nan", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 nan 0 0 0\n"),
    ("keypoints [2, 3], class 1 of 1", P, 1, [2, 3], "1 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 3], rows of two lengths", P, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n0 0.5 0.5 0.2 0.2\n"),
    ("keypoints [2, 3], the same row twice", P, 2, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n1 0.3 0.3 0.2 0.2 0.3 0.3 1 0.35 0.35 2\n0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [2, 2], x -0.005", P, 1, [2, 2], "0 0.5 0.5 0.2 0.2 -0.005 0.45 0.5 0.5\n"),
    ("keypoints [2, 2], given with v", P, 1, [2, 2], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints [1, 3] and an empty line", P, 1, [1, 3], "\n0 0.5 0.5 0.2 0.2 0.45 0.45 2\n\n"),
    ("keypoints rows in a detect run, kpt_shape [2, 3]", D, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints rows in a segment run, kpt_shape [2, 3]", S, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints rows of another length in a detect run, kpt_shape [2, 3]", D, 1, [2, 3], "0 0.5 0.5 0.2 0.2 0.45 0.45 2\n"),
    ("keypoints rows in a detect run, no kpt_shape", D, 1, None, "0 0.5 0.5 0.2 0.2 0.45 0.45 2 0 0 0\n"),
    ("keypoints rows in a detect run, no kpt_shape, v 1", D, 1, None, "0 0.5 0.5 0.2 0.2 0.45 0.45 1 0.5 0.5 1\n"),
    ("a keypoints file, empty", P, 1, [17, 3], ""),
    ("a keypoints file, no file", P, 1, [17, 3], None),
]


def as_list(a):
    return [float(v) for v in np.asarray(a, dtype=np.float32).ravel()]


labels = []
with tempfile.TemporaryDirectory() as work:
    for i, (name, task, nc, kpt, content) in enumerate(CASES):
        lb = os.path.join(work, f"{i}.txt")
        if content is not None:
            with open(lb, "wb") as f:
                f.write(content if isinstance(content, bytes) else content.encode("utf-8"))
        nkpt, ndim = kpt or (0, 0)
        _, lb_out, _, segments, keypoints, nm, nf, ne, ncorrupt, msg = verify(("img.jpg", lb, "", task == P, nc, nkpt, ndim, False))
        status = "corrupt" if ncorrupt else "missing" if nm else "empty" if ne else "ok"
        case = {"name": name, "task": task, "nc": nc, "kpt_shape": kpt}
        if isinstance(content, bytes):
            case["bytes"] = base64.b64encode(content).decode("ascii")
        else:
            case["text"] = content
        case["status"] = status
        case["message"] = msg.split(": ", 1)[-1] if msg else ""
        duplicates = 0
        if msg and "duplicate labels removed" in msg:
            duplicates = int(msg.split(": ")[-1].split()[0])
        case["duplicates"] = duplicates
        if status == "ok":
            case["boxes"] = [as_list(row) for row in lb_out]
            case["segments"] = [[as_list(p) for p in s] for s in segments]
            if keypoints is not None:
                case["keypoints"] = [[as_list(p) for p in k] for k in keypoints]
        labels.append(case)

# Names as people write them in data.yaml without quotes: what each reader makes of them
PLAIN = ["person", "storage tank", "traffic-light", "a.b", "x/y", "_hidden", "Caf\u00e9", "\uace0\uc591\uc774", "e1",
         "nan", "inf", "Infinity", "y", "n", "Y", "N", "yes", "Yes", "YES", "no", "No", "on", "On", "ON", "off",
         "Off", "true", "True", "false", "False", "null", "Null", "NULL", "None", "none", "~", "1", "01",
         "007", "0x1f", "0o17", "0b11", "1_000", "1.5", "1e3", "1.0e+3", ".5", ".inf", "-.inf", ".nan",
         "190:20:30", "2001-12-14", "1:2", "a: b", "a:b", "a #b", "a#b", "#tag", "-x", "- x", "? x", "!x",
         "&x", "*x", "|", ">", "'q", '"q', "%x", "@x", "`x", "[x]", "{x}", "x, y", "x,y", "a  b", " lead",
         "trail ", "=", "<<", "-", "+1", "-1", "./x", "../x", "a\tb", "it's", 'say "hi"', "back\\slash",
         "line\nbreak", "bell\x07", "del\x7f", "nel\x85", "nbsp\xa0", "ls\u2028", "bom\ufeff", "e\u0301", "\U0001f600"]


def described(read):
    try:
        value = read()
        return {"type": type(value).__name__, "str": str(value)}
    except Exception as e:
        return {"type": "error", "str": str(e).splitlines()[0]}


def mapping_value(value):
    return value["k"] if isinstance(value, dict) and "k" in value else value


yaml_plain = []
for text in PLAIN:
    yaml_plain.append({
        "text": text,
        "ultralytics": described(lambda: mapping_value(UltralyticsYAML(f"k: {text}\n").parse())),
        "pyyaml": described(lambda: mapping_value(yaml.safe_load(f"k: {text}\n"))),
        "written": UltralyticsYAML._repr(text),
    })

# data.yaml names in the forms people write them, through check_class_names
NAMES = [
    "names: [a, b, c]",
    "names: {0: a, 1: b}",
    "names: {'0': a, '1': b}",
    "names: {1: b, 0: a}",
    "names: {0: a, 2: b}",
    "names: []",
    "names: {}",
    "names: [on, off, yes, no]",
    "names: [1, 2.5, null, None, true]",
    "names:\n  0: person\n  1: car\n",
    "names:\n  - person\n  - car\n",
    "names: ['head', 'helmet', 'person']",
    'names:\n  0: "yes"\n  1: "1"\n  2: "a: b"\n  3: "caf\\xe9"\n  4: "\\u2028"\n',
]
class_names = []
for text in NAMES:
    names = UltralyticsYAML(text).parse()["names"]
    try:
        out = check_class_names(names)
        class_names.append({"yaml": text, "names": {str(k): v for k, v in out.items()}})
    except Exception as e:
        class_names.append({"yaml": text, "error": f"{type(e).__name__}: {e}"})

# Whole data.yaml files as tools write them
DATA = {
    "roboflow": "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\nnc: 3\nnames: ['head', 'helmet', 'person']\n\n"
    "roboflow:\n  workspace: joseph-nelson\n  project: hard-hat-workers\n  version: 13\n  license: Public Domain\n"
    "  url: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/13\n",
    "cvat": "path: .\ntrain: train.txt\nval: val.txt\nnames:\n  0: car\n  1: person\n",
    "coco8-pose": "path: coco8-pose # dataset root dir\ntrain: images/train # train images (relative to 'path') 4 images\n"
    "val: images/val # val images (relative to 'path') 4 images\ntest: # test images (optional)\n\n# Keypoints\n"
    "kpt_shape: [17, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)\n"
    "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n\n# Classes\nnames:\n  0: person\n\n"
    "# Keypoint names per class\nkpt_names:\n  0:\n    - nose\n    - left_eye\n    - right_eye\n",
    "lists": "train: [images/a, images/b]\nval:\n  - images/c\n  - images/d\nnames: [x]\n",
    "nc only": "train: a\nval: b\nnc: 2\n",
    "nc as text": "train: a\nval: b\nnc: '2'\nnames: [x, y]\n",
    "validation key": "train: a\nvalidation: b\nnames: [x]\n",
    "byte order mark and CRLF": "\ufefftrain: a\r\nval: b\r\nnames: [x]\r\n",
    "anchor": "train: &t images/train\nval: *t\nnames: [x]\n",
    "tab indent": "names:\n\t- x\ntrain: a\nval: b\n",
    "not a mapping": "- a\n- b\n",
    "empty": "",
    "quoted names and escapes": 'train: a\nval: b\nnames:\n  0: "say \\"hi\\""\n  1: \'it\'\'s\'\n  2: "tab\\there"\n',
}
data_yaml = []
for name, text in DATA.items():
    try:
        data_yaml.append({"name": name, "yaml": text, "data": UltralyticsYAML(text).parse()})
    except Exception as e:
        data_yaml.append({"name": name, "yaml": text, "error": f"{type(e).__name__}: {e}"})

out = {
    "ultralytics": "8.4.211",
    "commit": COMMIT,
    "numpy": np.__version__,
    "pyyaml": yaml.__version__,
    "labels": labels,
    "yaml_plain": yaml_plain,
    "class_names": class_names,
    "data_yaml": data_yaml,
}
# One case a line, so a change shows as the cases it touches
lines = ["{"]
for key, value in out.items():
    if isinstance(value, list):
        items = ",\n".join(f"    {json.dumps(v)}" for v in value)
        lines.append(f'  "{key}": [\n{items}\n  ],')
    else:
        lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
lines[-1] = lines[-1].rstrip(",")
lines.append("}")
(Path(__file__).parent / "verify_cases.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(labels)} label files, {len(yaml_plain)} names, {len(class_names)} name lists, {len(data_yaml)} data.yaml files")
