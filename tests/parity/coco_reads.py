"""Reads COCO files Seamtag wrote with pycocotools and checks them as COCO's own tools use them.

    uv run --no-project --python 3.12 --with pycocotools==2.0.11 --with numpy==2.5.4 python -I tests/parity/coco_reads.py FOLDER

Loads every annotations/*.json under FOLDER (tests/unit/coco.test.ts writes a sample there when
COCO_PARITY_DIR names it) and checks: image, annotation and category ids are whole numbers from
1, none repeated; every image and category an annotation names exists; the types COCO's page
gives; annToRLE works for every annotation, and pycocotools' mask.area of it is `area` exactly
(a posed object with only its box has the box's width x height); boxes are the vertices' for
polygons and toBbox's for masks; keypoints are 3 numbers each, 0, 0, 0 when not labelled, with
num_keypoints counting the rest and the skeleton within them; and COCOeval of the annotations
against themselves as detections scoring 1 gives AP 1.0 for bbox and segm, and for keypoints
in a file of posed objects. Fails at the first problem.
"""
import contextlib
import importlib.metadata
import io
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as M
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(path: Path) -> str:
    def expect(ok, message):
        if not ok:
            raise AssertionError(f"{path.name}: {message}")

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(path))
    d = coco.dataset
    for key in ("info", "licenses", "images", "annotations", "categories"):
        expect(key in d, f"there's no {key}")
    expect(isinstance(d["info"].get("year"), int), "info's year isn't a whole number")
    for key in ("images", "annotations", "categories"):
        ids = [x["id"] for x in d[key]]
        expect(all(type(i) is int and i >= 1 for i in ids), f"{key} ids must be whole numbers from 1")
        expect(len(set(ids)) == len(ids), f"{key} ids repeat")
    images = {im["id"]: im for im in d["images"]}
    for im in d["images"]:
        expect(type(im["width"]) is int and type(im["height"]) is int and im["width"] > 0 and im["height"] > 0, f"image {im['id']} has no size")
        expect(isinstance(im["file_name"], str) and im["file_name"], f"image {im['id']} has no file_name")
    cats = {c["id"]: c for c in d["categories"]}
    posed = any("keypoints" in c for c in d["categories"])
    for c in d["categories"]:
        expect(isinstance(c["name"], str) and isinstance(c["supercategory"], str), f"category {c['id']} needs a name and a supercategory")
        if "keypoints" in c:
            k = len(c["keypoints"])
            expect(all(isinstance(n, str) for n in c["keypoints"]), f"category {c['id']}'s keypoints must be names")
            expect(all(len(e) == 2 and all(type(i) is int and 1 <= i <= k for i in e) for e in c["skeleton"]), f"category {c['id']}'s skeleton must join keypoints 1 to {k}")
    for a in d["annotations"]:
        what = f"annotation {a['id']}"
        expect(a["image_id"] in images, f"{what} names image {a['image_id']}, which isn't listed")
        expect(a["category_id"] in cats, f"{what} names category {a['category_id']}, which isn't listed")
        expect(a["iscrowd"] in (0, 1) and type(a["iscrowd"]) is int, f"{what}: iscrowd must be 0 or 1")
        expect(is_number(a["area"]) and a["area"] > 0, f"{what}: area must be a number above 0")
        bbox = a["bbox"]
        expect(len(bbox) == 4 and all(is_number(v) for v in bbox) and bbox[2] > 0 and bbox[3] > 0, f"{what}: bbox must be x, y, width, height")
        im = images[a["image_id"]]
        seg = a["segmentation"]
        if isinstance(seg, list):
            expect(len(seg) > 0 and all(len(p) >= 6 and len(p) % 2 == 0 and all(is_number(v) for v in p) for p in seg), f"{what}: polygons need 3 points or more")
        else:
            expect(seg["size"] == [im["height"], im["width"]], f"{what}: the mask isn't the image's size")
            expect(isinstance(seg["counts"], str), f"{what}: the mask's counts aren't compressed")
        rle = coco.annToRLE(a)
        area = int(M.area(rle))
        x, y, w, h = bbox
        corners = [x, y, x + w, y, x + w, y + h, x, y + h]
        box_only = posed and len(seg) == 1 and len(seg[0]) == 8 and np.allclose(seg[0], corners, rtol=0, atol=1e-9) and a["area"] == w * h
        expect(area == a["area"] or box_only, f"{what}: area {a['area']} isn't pycocotools' {area}")
        expect(x >= 0 and y >= 0 and x + w <= im["width"] + 1e-9 and y + h <= im["height"] + 1e-9, f"{what}: bbox {bbox} reaches past the image")
        if not posed:
            if isinstance(seg, list):
                xs = [v for p in seg for v in p[0::2]]
                ys = [v for p in seg for v in p[1::2]]
                want = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                expect(np.allclose(bbox, want, rtol=0, atol=1e-9), f"{what}: bbox {bbox} isn't its polygons' {want}")
            else:
                expect(bbox == M.toBbox(rle).tolist(), f"{what}: bbox {bbox} isn't toBbox's {M.toBbox(rle).tolist()}")
        cat = cats[a["category_id"]]
        if "keypoints" in cat:
            expect("keypoints" in a and "num_keypoints" in a, f"{what}: keypoints and num_keypoints are missing")
            kp, k = a["keypoints"], len(cat["keypoints"])
            expect(len(kp) == 3 * k and all(is_number(v) for v in kp), f"{what}: keypoints must be {k} triplets")
            vis = kp[2::3]
            expect(all(v in (0, 1, 2) for v in vis), f"{what}: visibility must be 0, 1 or 2")
            expect(all(kp[3 * i] == 0 and kp[3 * i + 1] == 0 for i in range(k) if vis[i] == 0), f"{what}: a keypoint not labelled must be 0, 0, 0")
            expect(a["num_keypoints"] == sum(v > 0 for v in vis), f"{what}: num_keypoints isn't the keypoints labelled")
        elif posed:
            expect("num_keypoints" in a, f"{what}: COCOeval reads num_keypoints from every annotation")
    results = []
    kinds = ["bbox", "segm"] + (["keypoints"] if posed else [])
    for kind in kinds:
        groups = [(None, None)]
        if kind == "keypoints":
            groups = [(c["id"], len(c["keypoints"])) for c in d["categories"] if "keypoints" in c]
        for cat_id, k in groups:
            anns = [a for a in d["annotations"] if cat_id is None or a["category_id"] == cat_id]
            if kind == "keypoints":
                # Posed objects with no keypoints are ignored as ground truth, and have nothing to find
                anns = [a for a in anns if a["num_keypoints"] > 0]
            if not anns:
                continue
            dts = []
            for a in anns:
                dt = {"image_id": a["image_id"], "category_id": a["category_id"], "score": 1.0}
                if kind == "bbox":
                    dt["bbox"] = a["bbox"]
                elif kind == "segm":
                    dt["segmentation"] = coco.annToRLE(a)
                else:
                    dt["keypoints"] = a["keypoints"]
                dts.append(dt)
            with contextlib.redirect_stdout(io.StringIO()):
                found = coco.loadRes(dts)
                e = COCOeval(coco, found, kind)
                if kind == "keypoints":
                    e.params.catIds = [cat_id]
                    e.params.kpt_oks_sigmas = np.full(k, 0.05)
                e.evaluate()
                e.accumulate()
                e.summarize()
            expect(e.stats[0] == 1.0, f"COCOeval {kind}{'' if cat_id is None else f' of category {cat_id}'} gives AP {e.stats[0]}, not 1.0")
            results.append(f"{kind} AP {e.stats[0]:.1f}")
    return f"{len(d['images'])} images, {len(d['annotations'])} annotations, {len(d['categories'])} categories: {', '.join(results)}"


def main() -> None:
    print("pycocotools", importlib.metadata.version("pycocotools"))
    files = sorted(Path(sys.argv[1]).glob("annotations/*.json"))
    if not files:
        raise SystemExit(f"There are no annotations/*.json files in {sys.argv[1]}")
    for path in files:
        print(f"ok {path.name}: {check(path)}")


if __name__ == "__main__":
    main()
