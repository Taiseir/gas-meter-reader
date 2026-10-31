"""
detection_local.py - dial detection using our own YOLO model, on this machine.

Drop-in replacement for detection_main.py. No API key, no internet.
"""

import os
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_BUNDLED = _HERE / "models" / "best.pt"
DEFAULT_MODEL_PATH = _BUNDLED if _BUNDLED.exists() else (
    Path.home() / "runs" / "detect" / "gm" / "v1" / "weights" / "best.pt")
MODEL_PATH = Path(os.environ.get("GM_MODEL_PATH", DEFAULT_MODEL_PATH))

CONFIDENCE = 0.40
BOX_PAD = 0.03

DEBUG_DIR = Path(__file__).resolve().parent / "debug_detection"

_model = None


def _save_debug(name, img):
    DEBUG_DIR.mkdir(exist_ok=True)
    path = DEBUG_DIR / name
    cv2.imwrite(str(path), img)
    return f"{DEBUG_DIR.name}/{name}"


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {MODEL_PATH}. Set GM_MODEL_PATH to an existing best.pt."
            )
        from ultralytics import YOLO
        print(f"      Loading model: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def _build_boxes(result, img_w, img_h, pad=BOX_PAD):
    boxes = []
    for b in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * pad)
        half = side / 2.0
        nx1 = int(max(0, cx - half))
        ny1 = int(max(0, cy - half))
        nx2 = int(min(img_w, cx + half))
        ny2 = int(min(img_h, cy + half))
        boxes.append({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                      "cx": cx, "cy": cy, "conf": conf,
                      "w": nx2 - nx1, "h": ny2 - ny1})
    return boxes


def _cluster_rows(boxes, height_multiplier=0.70):
    if not boxes:
        return []
    by_y = sorted(boxes, key=lambda b: (b["y1"] + b["y2"]) / 2)
    rows, current = [], [by_y[0]]
    for box in by_y[1:]:
        prev = current[-1]
        prev_cy = (prev["y1"] + prev["y2"]) / 2
        curr_cy = (box["y1"] + box["y2"]) / 2
        avg_h = ((prev["y2"] - prev["y1"]) + (box["y2"] - box["y1"])) / 2
        if abs(curr_cy - prev_cy) > height_multiplier * avg_h:
            rows.append(current)
            current = [box]
        else:
            current.append(box)
    rows.append(current)
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda b: (b["x1"] + b["x2"]) / 2)
    rows.sort(key=lambda r: sum((b["y1"] + b["y2"]) / 2 for b in r) / len(r))
    return rows


def _select_meter_dials(rows, expected_gauges):
    # Reading dials are the top row. The bottom row holds test dials
    # ("half foot", "two feet") that are not part of the reading.
    # Returned right -> left (1,000 place first), the order reading_main expects.
    if not rows:
        return []
    return sorted(rows[0], key=lambda b: (b["x1"] + b["x2"]) / 2, reverse=True)


def _save_debug_image(image, boxes, selected):
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (160, 160, 160), 1)
    for i, b in enumerate(selected):
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 220, 0), 2)
        cv2.putText(vis, f"D{i+1}", (b["x1"] + 4, b["y1"] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
    out = _save_debug("debug_detections.png", vis)
    print(f"      Saved: {out}")


def d_main(image_path, expected_gauges=5, debug=False):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    print(f"      Image: {image_path}  ({w}x{h})")

    model = _load_model()
    result = model.predict(source=str(image_path), conf=CONFIDENCE, verbose=False)[0]

    boxes = _build_boxes(result, w, h)
    print(f"      Detected {len(boxes)} dial(s) at conf >= {CONFIDENCE}")
    if not boxes:
        print("      Nothing found. Try lowering CONFIDENCE.")
        return []

    rows = _cluster_rows(boxes)
    print(f"      Clustered into {len(rows)} row(s): {[len(r) for r in rows]}")

    selected = _select_meter_dials(rows, expected_gauges)
    print(f"\n      Selected {len(selected)} dial(s) (top row, right -> left):")
    for i, b in enumerate(selected):
        print(f"        Dial {i+1}: cx={b['cx']:.0f} cy={b['cy']:.0f} "
              f"box=[{b['x1']},{b['y1']} -> {b['x2']},{b['y2']}] conf={b['conf']:.2f}")

    if len(selected) != expected_gauges:
        print(f"      Warning: expected {expected_gauges}, selected {len(selected)}. "
              "Dial count differs between meters - check EXPECTED_GAUGES.")

    if debug:
        _save_debug_image(image, boxes, selected)

    cropped = []
    for i, b in enumerate(selected):
        crop = image[b["y1"]:b["y2"], b["x1"]:b["x2"]]
        if crop.size == 0:
            print(f"      Dial {i+1} crop is empty - skipping.")
            continue
        cropped.append(crop)
        if debug:
            out = _save_debug(f"debug_crop_{i+1}.png", crop)
            print(f"      Saved: {out}")

    print(f"\n      Returning {len(cropped)} cropped dial(s).")
    return cropped


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python detection_local.py <image> [expected_gauges]")
        raise SystemExit(1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    d_main(sys.argv[1], expected_gauges=n, debug=True)
