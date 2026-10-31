"""
reading_main.py - standalone OpenCV needle reader

Recovered from the Spring 2026 PDF export on 2026-10-27.
Indentation and line breaks were rebuilt from the printout; logic is unchanged.
"""

"""
reading_main.py - Gas Meter Dial Reader
=======================================
Reads cropped analog gas meter dials ordered RIGHT -> LEFT.


Key idea
--------
Use only the middle annulus of the dial for angle reading so printed digits
and outer rim text do not dominate the result.


Methods used per dial
---------------------
1. Ray-strip continuity scorer
2. Contour direction estimator
3. Constrained Hough-line estimator


Then combine the angle candidates with a confidence score.
"""

from pathlib import Path

import cv2
import numpy as np
import math

DEBUG_DIR = Path(__file__).resolve().parent / "debug_reading"


def _save_debug(name, img):
    DEBUG_DIR.mkdir(exist_ok=True)
    path = DEBUG_DIR / name
    cv2.imwrite(str(path), img)
    return f"{DEBUG_DIR.name}/{name}"

# ── Tuning ─────────────────────────────────────────────────────────────────────

ON_THRESHOLD_DEG = 4.0
ZERO_ANGLE = 0.0

# Circle detection
HOUGH_DP = 1.2
HOUGH_PARAM1 = 100
HOUGH_PARAM2_LIST = (40, 30, 20)

# Full dark mask
FULL_OUTER_MASK_FRAC = 0.86
FULL_INNER_MASK_FRAC = 0.28

# Reading mask (middle annulus only)
READ_INNER_FRAC = 0.24
READ_OUTER_FRAC = 0.72
# Ray scorer
RAY_STEP_DEG = 1.0
RAY_SAMPLES = 80
RAY_MIN_HITS = 6
RAY_MIN_RUN = 4
STRIP_HALF_WIDTH = 2
NEIGHBOR_OFFSET_DEG = 3.0

# Hough line
LINE_HOUGH_THRESHOLD = 14
LINE_MIN_LEN_FRAC = 0.22
LINE_MAX_GAP = 6

# ── Geometry helpers ───────────────────────────────────────────────────────────

def _avg_circles(circles, n):
    avg_x = sum(circles[0][i][0] for i in range(n)) / n
    avg_y = sum(circles[0][i][1] for i in range(n)) / n
    avg_r = sum(circles[0][i][2] for i in range(n)) / n
    return int(avg_x), int(avg_y), int(avg_r)

def _point_angle(cx, cy, px, py):

    """
   Absolute angle in clockwise frame:
     0° = up
     90° = right
     180° = down
     270° = left
   """
    dx = px - cx
    dy = py - cy
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0

def _relative_angle(abs_ang):
    return (abs_ang - ZERO_ANGLE) % 360.0

def _to_unified_cw(rel_ang, counter_clockwise):
    if counter_clockwise:
        return (360.0 - rel_ang) % 360.0
    return rel_ang

def _digit_from_angle(rel_ang):
    return int(rel_ang // 36.0) % 10

def _on_digit(rel_ang, digit):
    target = (digit * 36.0) % 360.0
    diff = abs((rel_ang - target + 180.0) % 360.0 - 180.0)
    return diff <= ON_THRESHOLD_DEG

def _passed_zero(right_rel_ang):
    return right_rel_ang < 180.0

def _draw_ticks(img, cx, cy, r):
    for i in range(10):
        a = math.radians((ZERO_ANGLE + i * 36.0) % 360.0)
        inner = (int(cx + 0.85 * r * math.sin(a)), int(cy - 0.85 * r * math.cos(a)))
        outer = (int(cx + 1.00 * r * math.sin(a)), int(cy - 1.00 * r * math.cos(a)))
        label = (int(cx + 1.15 * r * math.sin(a)), int(cy - 1.15 * r * math.cos(a)))

        cv2.line(img, inner, outer, (0, 200, 0), 2)
        cv2.putText(img, str(i), label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2, cv2.LINE_AA, )

    # ── Rule 4 ─────────────────────────────────────────────────────────────────────

def _apply_rule4(digits, rel_angles):
    corrected = digits[:]
    for i in range(len(corrected) - 1):
        current_digit = corrected[i + 1]
        current_rel = rel_angles[i + 1]
        right_rel = rel_angles[i]

        if not _on_digit(current_rel, current_digit):
            continue

        if _passed_zero(right_rel):
            print(f"    ✅ Rule 4 - dial {i+2}: ON {current_digit}, " f"right passed 0 (rel={right_rel:.1f}°) -> keep {current_digit}" )
        else:
            adjusted = (current_digit - 1) % 10
            corrected[i + 1] = adjusted
            print(f"    ⚠️ Rule 4 - dial {i+2}: ON {current_digit}, " f"right NOT past 0 (rel={right_rel:.1f}°) -> use {adjusted}" )

    return corrected

    # ── Circle detection ───────────────────────────────────────────────────────────

def _calibrate(img_arr):
    h, w = img_arr.shape[:2]
    if h < 60 or w < 60:
        print(f"        ⚠️ Crop too small ({w}×{h}) - skipping.")
        return None

    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = None
    for param2 in HOUGH_PARAM2_LIST:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=min(h, w) // 2, param1=HOUGH_PARAM1, param2=param2, minRadius=int(0.28 * min(h, w)), maxRadius=int(0.55 * min(h, w)), )
        if circles is not None:
            break

    if circles is None:
        print("        ❌ No circle detected.")
        return None

    cx, cy, r = _avg_circles(circles, circles.shape[1])
    return cx, cy, r

    # ── Masks ──────────────────────────────────────────────────────────────────────

def _make_full_dark_mask(gray, cx, cy, r):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq = clahe.apply(blur)

    binary = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8, )

    annulus = np.zeros_like(binary)
    cv2.circle(annulus, (cx, cy), int(FULL_OUTER_MASK_FRAC * r), 255, -1)
    cv2.circle(annulus, (cx, cy), int(FULL_INNER_MASK_FRAC * r), 0, -1)
    binary = cv2.bitwise_and(binary, annulus)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary

def _make_reading_mask(full_mask, cx, cy, r):

    """
   Mask used only for angle reading: middle annulus only.
   This suppresses outer printed digits and inner hub shadow.
   """
    ring = np.zeros_like(full_mask)
    cv2.circle(ring, (cx, cy), int(READ_OUTER_FRAC * r), 255, -1)
    cv2.circle(ring, (cx, cy), int(READ_INNER_FRAC * r), 0, -1)

    reading_mask = cv2.bitwise_and(full_mask, ring)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reading_mask = cv2.morphologyEx(reading_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    reading_mask = cv2.morphologyEx(reading_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return reading_mask

    # ── Method 1: Ray-strip scorer ────────────────────────────────────────────────

def _ray_strip_score(binary, cx, cy, ang_deg, r0, r1, strip_half_width=STRIP_HALF_WIDTH, n=RAY_SAMPLES):
    h, w = binary.shape[:2]
    a = math.radians(ang_deg)

    ux = math.sin(a)
    uy = -math.cos(a)
    px = math.cos(a)
    py = math.sin(a)

    radii = np.linspace(r0, r1, n)

    support = []
    weighted_sum = 0.0

    for rr in radii:
        vals = []
        for off in range(-strip_half_width, strip_half_width + 1):
            x = int(round(cx + rr * ux + off * px))
            y = int(round(cy + rr * uy + off * py))
            if 0 <= x < w and 0 <= y < h:
                vals.append(1 if binary[y, x] > 0 else 0)

        hit = 1 if sum(vals) >= max(1, len(vals) // 2) else 0
        support.append(hit)

        if hit:
            weight = (rr / max(r1, 1.0)) ** 1.5
            weighted_sum += weight

    best_run = 0
    cur_run = 0
    for s in support:
        if s:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 0

    total_hits = sum(support)

    score = 1.6 * weighted_sum + 0.8 * total_hits + 2.0 * best_run
    return float(score), int(total_hits), int(best_run)

def _ray_angle(reading_mask, cx, cy, r):
    best_ang = None
    best_score = -1e9

    r0 = READ_INNER_FRAC * r
    r1 = READ_OUTER_FRAC * r

    for ang in np.arange(0.0, 360.0, RAY_STEP_DEG):
        score, hits, run = _ray_strip_score(reading_mask, cx, cy, ang, r0, r1)

        if hits < RAY_MIN_HITS or run < RAY_MIN_RUN:
            continue
        s1, _, _ = _ray_strip_score(reading_mask, cx, cy, (ang - NEIGHBOR_OFFSET_DEG) % 360.0, r0, r1, strip_half_width=1, n=60)
        s2, _, _ = _ray_strip_score(reading_mask, cx, cy, (ang + NEIGHBOR_OFFSET_DEG) % 360.0, r0, r1, strip_half_width=1, n=60)

        final_score = score + 0.20 * (s1 + s2)

        if final_score > best_score:
            best_score = final_score
            best_ang = float(ang)

    return best_ang

    # ── Method 2: Contour estimator ───────────────────────────────────────────────

def _contour_angle(reading_mask, cx, cy, r):
    contours, _ = cv2.findContours(reading_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ang = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8:
            continue

        pts = cnt[:, 0, :].astype(np.float32)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dists = np.sqrt(dx**2 + dy**2)

        keep = (dists > 0.30 * r) & (dists < 0.78 * r)
        outer_pts = pts[keep]
        if len(outer_pts) < 5:
            continue

        vx = outer_pts[:, 0] - cx
        vy = outer_pts[:, 1] - cy
        norms = np.sqrt(vx**2 + vy**2) + 1e-6
        ux = vx / norms
        uy = vy / norms

        mean_x = float(np.mean(ux))
        mean_y = float(np.mean(uy))
        strength = float(np.hypot(mean_x, mean_y))

        if strength < 0.15:
            continue

        tip_x = cx + mean_x * 0.60 * r
        tip_y = cy + mean_y * 0.60 * r
        ang = _point_angle(cx, cy, tip_x, tip_y)

        score = 3.0 * strength + 0.005 * area
        if score > best_score:
            best_score = score
            best_ang = float(ang)

    return best_ang

    # ── Method 3: Constrained Hough line estimator ────────────────────────────────

def _line_centre_dist(cx, cy, x1, y1, x2, y2):
    num = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1) + 1e-6
    return num / den

def _line_angle(reading_mask, cx, cy, r):
    lines = cv2.HoughLinesP(reading_mask, rho=1, theta=np.pi / 180, threshold=LINE_HOUGH_THRESHOLD, minLineLength=int(LINE_MIN_LEN_FRAC * r), maxLineGap=LINE_MAX_GAP, )

    if lines is None:
        return None
    best_ang = None
    best_score = -1e9

    for line in lines:
        x1, y1, x2, y2 = line[0]

        d1 = float(np.hypot(x1 - cx, y1 - cy))
        d2 = float(np.hypot(x2 - cx, y2 - cy))

        # one endpoint should be closer to center, the other farther out
        near = min(d1, d2)
        far = max(d1, d2)

        if near > 0.40 * r:
            continue
        if far < 0.35 * r or far > 0.80 * r:
            continue

        lc = _line_centre_dist(cx, cy, x1, y1, x2, y2)
        if lc > 0.18 * r:
            continue

        if d1 > d2:
            tip = (x1, y1)
        else:
            tip = (x2, y2)

        ang = _point_angle(cx, cy, tip[0], tip[1])

        length = float(np.hypot(x2 - x1, y2 - y1))
        score = length - 3.0 * lc

        if score > best_score:
            best_score = score
            best_ang = float(ang)

    return best_ang

    # ── Angle combination ──────────────────────────────────────────────────────────

def _circular_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)

def _mean_angle_deg(angles):
    rs = np.radians(angles)
    x = np.mean(np.cos(rs))
    y = np.mean(np.sin(rs))
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)

def _combine_angles(candidates):

    """
   candidates: dict like {'ray': angle_or_None, 'contour': angle_or_None, 'line':
    angle_or_None}
       """
    valid = {k: v for k, v in candidates.items() if v is not None}

    if not valid:
        return None, 0.0, []

    # Build groups of mutually agreeing angles
    names = list(valid.keys())
    angles = [valid[n] for n in names]

    best_group = []
    best_spread = 999.0

    for i in range(len(angles)):
        group = [names[i]]
        base = angles[i]

        for j in range(len(angles)):
            if i == j:
                continue
            if _circular_diff_deg(base, angles[j]) <= 18.0:
                group.append(names[j])

        group_angles = [valid[n] for n in group]
        if len(group) > len(best_group):
            best_group = group
            diffs = []
            m = _mean_angle_deg(group_angles)
            for a in group_angles:
                diffs.append(_circular_diff_deg(a, m))
            best_spread = max(diffs) if diffs else 0.0
        elif len(group) == len(best_group) and len(group) > 0:
            group_angles = [valid[n] for n in group]
            m = _mean_angle_deg(group_angles)
            diffs = [_circular_diff_deg(a, m) for a in group_angles]
            spread = max(diffs) if diffs else 0.0
            if spread < best_spread:
                best_group = group
                best_spread = spread

    group_angles = [valid[n] for n in best_group]
    combined = _mean_angle_deg(group_angles)

    # confidence
    method_count = len(best_group)
    spread_penalty = min(best_spread / 18.0, 1.0)

    # ray+contour are correlated, so cap them lower if line is absent
    if set(best_group) == {"ray", "contour"}:
        base_conf = 0.65
    elif method_count == 3:
        base_conf = 0.95
    elif method_count == 2:
        base_conf = 0.78
    else:
        base_conf = 0.45

    conf = max(0.05, base_conf * (1.0 - 0.6 * spread_penalty))
    return combined, conf, best_group

    # ── Needle reading ─────────────────────────────────────────────────────────────

def _read_needle(img_arr, cx, cy, r, gauge_index, debug):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    full_mask = _make_full_dark_mask(gray, cx, cy, r)
    reading_mask = _make_reading_mask(full_mask, cx, cy, r)

    ray_ang = _ray_angle(reading_mask, cx, cy, r)
    contour_ang = _contour_angle(reading_mask, cx, cy, r)
    line_ang = _line_angle(reading_mask, cx, cy, r)

    print(f"      📎 Candidates: " f"ray={f'{ray_ang:.1f}°' if ray_ang is not None else 'None'}      " f"contour={f'{contour_ang:.1f}°' if contour_ang is not None else 'None'}   " f"line={f'{line_ang:.1f}°' if line_ang is not None else 'None'}" )

    combined_ang, conf, used = _combine_angles({"ray": ray_ang, "contour": contour_ang, "line": line_ang} )

    if combined_ang is None:
        print("     ❌ No usable angle candidates.")
        if debug:
            _save_debug(f"failed_binary_{gauge_index}.png", reading_mask)
        return None, None, None, 0.0

    raw_rel_ang = _relative_angle(combined_ang)

    tip_x = int(round(cx + 0.65 * r * math.sin(math.radians(combined_ang))))
    tip_y = int(round(cy - 0.65 * r * math.cos(math.radians(combined_ang))))
    tip = (tip_x, tip_y)

    print(f"      🎯 Combined: abs={combined_ang:.1f}° " f"raw_rel={raw_rel_ang:.1f}°     confidence={conf:.2f}   using={used}" )

    if debug:
        vis = img_arr.copy()
        cv2.circle(vis, (cx, cy), r, (0, 0, 220), 2)
        _draw_ticks(vis, cx, cy, r)
        cv2.arrowedLine(vis, (cx, cy), tip, (220, 0, 0), 2, tipLength=0.15)
        cv2.circle(vis, tip, 5, (0, 0, 220), -1)
        cv2.putText(vis, f"conf={conf:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 0), 2, )

        bin_bgr = cv2.cvtColor(reading_mask, cv2.COLOR_GRAY2BGR)
        board = np.hstack((bin_bgr, vis))
        path = _save_debug(f"storyboard_dial_{gauge_index}.png", board)
        print(f"     Saved: {path}")

    return tip, combined_ang, raw_rel_ang, conf

    # ── Single dial pipeline ───────────────────────────────────────────────────────

def _run_dial(img_arr, counter_clockwise, gauge_index, debug):
    calib = _calibrate(img_arr)
    if calib is None:
        if debug:
            _save_debug(f"failed_crop_{gauge_index}.png", img_arr)
        return 0, 0.0, 0.0

    cx, cy, r = calib
    print(f"        ⭕ Circle: centre=({cx},{cy}) r={r}px")
    tip, abs_ang, raw_rel_ang, conf = _read_needle(img_arr, cx, cy, r, gauge_index, debug)

    if tip is None:
        if debug:
            _save_debug(f"failed_crop_{gauge_index}.png", img_arr)
        return 0, 0.0, 0.0

    rel_ang_cw = _to_unified_cw(raw_rel_ang, counter_clockwise)
    digit = _digit_from_angle(rel_ang_cw)

    print(f"        🔄 Dial direction: {'CCW ↺' if counter_clockwise else 'CW ↻'}")
    print(f"        📐 Unified CW angle={rel_ang_cw:.1f}° -> lower digit={digit}")
    return int(digit), float(rel_ang_cw), float(conf)
    # ── Public API ─────────────────────────────────────────────────────────────────

def r_main(cropped_gauges, debug=False):
    n = len(cropped_gauges)
    if n == 0:
        print("   ⚠️ No gauge crops to read.")
        return 0, []

    print(f"\n{'═' * 60}")
    print(f"     Reading {n} dial(s)     [right -> left]")
    print(f"{'═' * 60}")

    digits = []
    rel_angles = []
    confidences = []

    for i, crop in enumerate(cropped_gauges):
        ccw = (i % 2 == 1)        # 1k=CW, 10k=CCW, 100k=CW, 1M=CCW
        place = 1000 * (10 ** i)

        print(f"\n    ── Dial {i+1}   [{place:>12,} place]   " f"({'CCW ↺' if ccw else 'CW ↻'})    " f"{crop.shape[1]}×{crop.shape[0]} px" )

        d, a, c = _run_dial(crop, counter_clockwise=ccw, gauge_index=i + 1, debug=debug, )

        digits.append(d)
        rel_angles.append(a)
        confidences.append(c)

    print(f"\n     Raw digits    (right->left) : {digits}")
    print(f"     Rel angles     (right->left) : {[f'{a:.1f}°' for a in rel_angles]}")
    print(f"     Confidence     (right->left) : {[f'{c:.2f}' for c in confidences]}")
    print("\n   Applying Rule 4 ...")
    corrected = _apply_rule4(digits, rel_angles)
    print(f"    Corrected   (right->left) : {corrected}")

    reading = sum(d * 1000 * (10 ** i) for i, d in enumerate(corrected))
    left_to_right = list(reversed(corrected))
    left_str = "".join(str(d) for d in left_to_right)

    print(f"\n{'═' * 60}")
    print(f"    Dials (left->right) : {left_to_right}")
    print(f"    Meter reading      : {left_str},000 cubic feet")
    print(f"    Numeric value      : {reading:,}")
    print(f"{'═' * 60}\n")

    return int(reading), left_to_right
