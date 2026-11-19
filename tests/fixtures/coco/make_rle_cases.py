"""Runs pycocotools 2.0.11 on masks, polygons, boxes and runs, for tests/unit/rle.test.ts:
src/shared/rle.ts must give the same runs, compressed strings, areas and boxes, to the pixel.

    uv run --no-project --python 3.12 --with pycocotools==2.0.11 --with numpy==2.5.4 python -I make_rle_cases.py

Writes rle_cases.json beside this file. Masks are small enough to keep as rows of 0 and 1,
except the discs, which the test draws itself. Polygons are random and hand-picked: sub-pixel,
half-way coordinates that tie when scaled by 5, partly and wholly outside the image,
self-touching and self-crossing, repeated points, and long thin ones.
"""
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pycocotools._mask as _mask
import pycocotools.mask as M

here = Path(__file__).parent
rng = np.random.default_rng(2011)


def rows(mask):
    return ["".join("1" if v else "0" for v in row) for row in mask]


def described(rle):
    """What pycocotools says about one RLE"""
    return {
        "counts": rle["counts"].decode("ascii"),
        "area": int(M.area(rle)),
        "bbox": M.toBbox(rle).tolist(),
    }


def disc(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


def mask_cases():
    out = []

    def add(name, mask, keep_rows=True, **extra):
        mask = np.asarray(mask, dtype=np.uint8)
        rle = M.encode(np.asfortranarray(mask[:, :, None]))[0]
        assert (M.decode(rle) == mask).all()
        case = {"name": name, "h": mask.shape[0], "w": mask.shape[1], **extra, **described(rle)}
        if keep_rows:
            case["rows"] = rows(mask)
        out.append(case)

    add("one empty pixel", [[0]])
    add("one set pixel", [[1]])
    add("COCO's 3 x 4 example", [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    add("all set", np.ones((5, 7)))
    add("all clear", np.zeros((6, 3)))
    add("one column", (rng.random((17, 1)) < 0.5))
    add("one row", (rng.random((1, 19)) < 0.5))
    add("checkerboard", np.indices((8, 8)).sum(axis=0) % 2)
    add("last pixel only", np.pad([[1]], ((4, 0), (6, 0))))
    add("first pixel only", np.pad([[1]], ((0, 4), (0, 6))))
    for h, w, p in [(13, 17, 0.05), (13, 17, 0.5), (13, 17, 0.95), (31, 7, 0.3), (7, 31, 0.7), (40, 48, 0.5), (48, 40, 0.02)]:
        add(f"random {h} x {w} at {p}", rng.random((h, w)) < p)
    blobs = np.zeros((36, 44), np.uint8)
    blobs[3:9, 4:30] = 1
    blobs[20:33, 10:12] = 1
    blobs |= disc(36, 44, 24, 30, 9)
    add("blobs", blobs)
    # Big enough that runs and their differences take several characters, both signs
    for h, w, cy, cx, r in [(600, 800, 290, 410, 230), (1000, 37, 500, 18, 400), (3, 2000, 1, 1000, 900)]:
        add(f"disc {h} x {w}", disc(h, w, cy, cx, r), keep_rows=False, disc=[cy, cx, r])
    return out


def poly_case(name, h, w, poly):
    poly = [float(v) for v in poly]
    rle = _mask.frPoly([poly], h, w)[0]
    return {"name": name, "h": h, "w": w, "poly": poly, **described(rle)}


def polygon_cases():
    out = []
    add = lambda name, h, w, poly: out.append(poly_case(name, h, w, poly))
    add("square 10 to 20", 40, 40, [10, 10, 20, 10, 20, 20, 10, 20])
    add("square 10.5 to 20.5", 40, 40, [10.5, 10.5, 20.5, 10.5, 20.5, 20.5, 10.5, 20.5])
    add("square 10.2 to 19.7", 40, 40, [10.2, 10.2, 19.7, 10.2, 19.7, 19.7, 10.2, 19.7])
    add("triangle", 40, 40, [0, 0, 10, 0, 0, 10])
    add("triangle the other way round", 40, 40, [0, 10, 10, 0, 0, 0])
    add("whole image", 12, 9, [0, 0, 9, 0, 9, 12, 0, 12])
    add("past every edge", 12, 9, [-3, -4, 15, -2, 14, 20, -5, 18])
    add("wholly outside", 20, 20, [100, 100, 120, 100, 110, 120])
    add("wholly left of the image", 20, 20, [-30, 2, -10, 2, -20, 15])
    add("crossing the left edge", 20, 25, [-5.3, -2.1, 30.7, 4.2, 12.0, 40.5])
    add("bow tie", 20, 20, [2, 2, 15, 15, 15, 2, 2, 15])
    add("touching itself at a point", 30, 30, [5, 5, 15, 5, 15, 15, 10, 10, 20, 10, 20, 20, 5, 20])
    add("figure of eight", 30, 30, [5, 5, 25, 25, 25, 5, 5, 25, 5, 5])
    add("repeated points", 30, 30, [3, 3, 3, 3, 20, 4, 20, 4, 20, 4, 11, 22, 3, 3])
    add("collinear", 30, 30, [2, 2, 10, 10, 20, 20])
    add("one point", 10, 10, [4.2, 5.7])
    add("two points", 10, 10, [1, 1, 8, 8])
    add("odd number of numbers", 20, 20, [1, 1, 15, 2, 9, 14, 7])
    add("inside one pixel", 10, 10, [3.1, 3.1, 3.9, 3.2, 3.5, 3.8])
    add("round one pixel centre", 10, 10, [3.3, 3.3, 3.7, 3.3, 3.7, 3.7, 3.3, 3.7])
    add("thin and long", 50, 60, [1.2, 3.0, 58.7, 3.4, 58.9, 3.9, 1.0, 3.6])
    add("thin and tall", 60, 50, [3.0, 1.2, 3.4, 58.7, 3.9, 58.9, 3.6, 1.0])
    add("huge coordinates", 30, 40, [-1e4, -1e4, 1e4, -1e4, 1e4, 1e4, -1e4, 1e4])
    add("halves that tie at 5x", 30, 30, [2.5, 2.5, 17.5, 2.7, 22.1, 13.3, 8.9, 25.5, 1.3, 14.1])
    add("negative halves", 30, 30, [-0.1, -0.3, 12.3, -0.5, 13.7, 9.9, -0.7, 11.1])
    add("one pixel wide image", 25, 1, [-1, 2, 2, 3, 1, 22, 0.2, 20])
    add("one pixel high image", 1, 25, [2, -1, 3, 2, 22, 1, 20, 0.2])
    add("edge at the last column", 20, 20, [19, 2, 19.9, 2, 19.9, 18, 19, 18])
    add("edge at the bottom", 20, 20, [2, 19, 18, 19, 18, 20, 2, 20])
    # Random polygons over a margin around the image, at several kinds of coordinates
    for i in range(120):
        h, w = int(rng.integers(1, 60)), int(rng.integers(1, 60))
        k = int(rng.integers(3, 14))
        kind = i % 4
        if kind == 0:  # any float
            xs, ys = rng.uniform(-0.3 * w, 1.3 * w, k), rng.uniform(-0.3 * h, 1.3 * h, k)
        elif kind == 1:  # tenths, which tie at 5x
            xs, ys = rng.integers(-3 * w, 13 * w, k) / 10, rng.integers(-3 * h, 13 * h, k) / 10
        elif kind == 2:  # whole pixels
            xs, ys = rng.integers(-2, w + 3, k).astype(float), rng.integers(-2, h + 3, k).astype(float)
        else:  # a star around a point: what people draw
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            angles = np.sort(rng.uniform(0, 2 * math.pi, k))
            radii = rng.uniform(0.2, 0.7, k) * max(h, w)
            xs, ys = cx + radii * np.cos(angles), cy + radii * np.sin(angles)
        poly = np.column_stack([xs, ys]).ravel().tolist()
        add(f"random {i} ({['floats', 'tenths', 'whole', 'star'][kind]})", h, w, poly)
    # Long polygons in a large image: many runs, several characters each
    for i in range(3):
        h, w, k = 900, 700, 40
        angles = np.sort(rng.uniform(0, 2 * math.pi, k))
        radii = rng.uniform(150, 400, k)
        poly = np.column_stack([350 + radii * np.cos(angles), 450 + radii * np.sin(angles)]).ravel().tolist()
        add(f"large star {i}", h, w, poly)
    return out


def box_cases():
    out = []
    boxes = [[10, 10, 10, 10], [0, 0, 12, 9], [-3.5, 2.25, 8, 30], [5.2, 5.2, 0.4, 0.4], [3, 3, 0, 5], [7.7, 1.1, 12.6, 3.3]]
    for _ in range(30):
        boxes.append([float(v) for v in [*rng.uniform(-5, 30, 2), *rng.uniform(0, 25, 2)]])
    for i, box in enumerate(boxes):
        h, w = (20, 30) if i % 2 else (31, 17)
        rle = _mask.frBbox(np.array([box], dtype=np.double), h, w)[0]
        out.append({"name": f"box {i}", "h": h, "w": w, "box": [float(v) for v in box], **described(rle)})
    return out


def merge_cases():
    out = []
    for i in range(25):
        h, w = int(rng.integers(5, 50)), int(rng.integers(5, 50))
        polys = []
        for _ in range(int(rng.integers(2, 5))):
            k = int(rng.integers(3, 8))
            polys.append(np.column_stack([rng.uniform(-0.2 * w, 1.2 * w, k), rng.uniform(-0.2 * h, 1.2 * h, k)]).ravel().tolist())
        rles = _mask.frPoly(polys, h, w)
        out.append({
            "name": f"merge {i}", "h": h, "w": w, "polys": polys,
            "union": described(M.merge(rles, intersect=0)),
            "intersection": described(M.merge(rles, intersect=1)),
        })
    return out


def segmentation_cases():
    """annToRLE's way with a polygon list: frPyObjects, then merge"""
    out = []
    segs = [
        [[0, 0, 10, 0, 0, 10], [20, 20, 30, 20, 30, 30, 20, 30]],
        [[0, 0, 10, 0, 0, 10], [20, 20, 30, 30]],
        [[5.5, 5.5, 25.5, 5.5, 25.5, 25.5, 5.5, 25.5], [10, 10, 20, 10, 20, 20, 10, 20]],
    ]
    for seg in segs:
        rle = M.merge(M.frPyObjects(seg, 40, 40))
        out.append({"name": f"{len(seg)} polygons", "h": 40, "w": 40, "segmentation": seg, **described(rle)})
    return out


def run_cases():
    """Uncompressed runs, written as they are, including ones pycocotools never makes itself"""
    out = []
    lists = [
        (3, 4, [3, 2, 1, 2, 4]),
        (3, 4, [0, 12]),
        (3, 4, [12]),
        (3, 4, [0, 5, 7]),
        (4, 3, [2, 5, 5]),  # a run of 1s from one column into the next
        (4, 3, [2, 0, 3, 0, 7]),  # empty runs of 1s
        (4, 3, [0, 0, 2, 3, 7]),
        (5, 5, [1, 23, 1]),
        (5, 5, [4, 1, 3, 2, 15]),
        (5, 5, [3, 30]),  # more than the mask holds
        (2, 2, [1, 1]),  # less than the mask holds
        (100, 100, [5000, 1, 4998, 1]),
        (1000, 1000, [123456, 400000, 2, 476542]),
        (1000, 1000, [999999, 1]),
    ]
    for h, w, counts in lists:
        rle = M.frPyObjects({"size": [h, w], "counts": counts}, h, w)
        case = {"name": f"runs {counts}", "h": h, "w": w, "runs": counts, **described(rle)}
        if sum(counts) > h * w:
            try:
                M.decode(rle)
                raise AssertionError("pycocotools decoded runs past the mask")
            except ValueError:
                case["decodes"] = False
        elif sum(counts) == h * w and h * w <= 2500:
            # Runs that stop short leave the rest of pycocotools' mask as whatever was in memory
            case["rows"] = rows(M.decode(rle))
        out.append(case)
    return out


cases = {
    "pycocotools": importlib.metadata.version("pycocotools"),
    "numpy": importlib.metadata.version("numpy"),
    "masks": mask_cases(),
    "polygons": polygon_cases(),
    "boxes": box_cases(),
    "merges": merge_cases(),
    "segmentations": segmentation_cases(),
    "runs": run_cases(),
}
lines = ["{"]
for i, (key, value) in enumerate(cases.items()):
    end = "," if i < len(cases) - 1 else ""
    if isinstance(value, list):
        lines.append(f'  "{key}": [')
        lines.append(",\n".join("    " + json.dumps(v, separators=(",", ":")) for v in value))
        lines.append(f"  ]{end}")
    else:
        lines.append(f'  "{key}": {json.dumps(value)}{end}')
lines.append("}")
(here / "rle_cases.json").write_text("\n".join(lines) + "\n")
print("wrote", sum(len(v) for v in cases.values() if isinstance(v, list)), "cases")
