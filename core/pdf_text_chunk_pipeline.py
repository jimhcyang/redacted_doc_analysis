from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from core.pdf_docs_pipeline import PDFPair, collect_pdf_pairs, render_pdf_to_images

Rect = tuple[int, int, int, int]  # xyxy, inclusive


def _rect_w(rect: Rect) -> int:
    return max(0, rect[2] - rect[0] + 1)


def _rect_h(rect: Rect) -> int:
    return max(0, rect[3] - rect[1] + 1)


def _rect_area(rect: Rect) -> int:
    return _rect_w(rect) * _rect_h(rect)


def _intersect(a: Rect, b: Rect) -> Rect | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return None
    return (x1, y1, x2, y2)


def _clip_rect(rect: Rect, w: int, h: int) -> Rect:
    x1 = min(max(0, int(rect[0])), max(0, w - 1))
    y1 = min(max(0, int(rect[1])), max(0, h - 1))
    x2 = min(max(x1, int(rect[2])), max(0, w - 1))
    y2 = min(max(y1, int(rect[3])), max(0, h - 1))
    return (x1, y1, x2, y2)


def _dedupe_rects(rects: Iterable[Rect]) -> list[Rect]:
    seen = set()
    out: list[Rect] = []
    for r in rects:
        k = tuple(int(v) for v in r)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    # Otsu threshold + hard dark threshold union.
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total <= 0:
        return np.zeros_like(gray, dtype=bool)

    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    weight_b = 0.0
    sum_b = 0.0
    max_var = -1.0
    thresh = 127
    for t in range(256):
        weight_b += hist[t]
        if weight_b <= 0:
            continue
        weight_f = total - weight_b
        if weight_f <= 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t

    ink_otsu = gray <= thresh
    ink_hard = gray < 200
    return np.logical_or(ink_otsu, ink_hard)


def _integral_image(mask: np.ndarray) -> np.ndarray:
    # mask bool -> integral int32 with top/left zero pad
    arr = mask.astype(np.int32)
    ii = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    return np.pad(ii, ((1, 0), (1, 0)), mode="constant")


def _rect_sum(ii: np.ndarray, x1: int, y1: int, x2_exclusive: int, y2_exclusive: int) -> int:
    return int(
        ii[y2_exclusive, x2_exclusive]
        - ii[y1, x2_exclusive]
        - ii[y2_exclusive, x1]
        + ii[y1, x1]
    )


def _rect_white_ratio(ink_ii: np.ndarray, rect: Rect) -> float:
    area = max(1, _rect_area(rect))
    ink_count = _rect_sum(ink_ii, rect[0], rect[1], rect[2] + 1, rect[3] + 1)
    return 1.0 - (ink_count / float(area))


def _center_seed_rect_for_edge(
    *,
    w: int,
    h: int,
    edge: str,
    depth: int,
    seed_span: int,
) -> Rect:
    half = max(4, seed_span // 2)
    if edge == "top":
        return _clip_rect((w // 2 - half, 0, w // 2 + half, max(seed_span - 1, depth)), w, h)
    if edge == "bottom":
        return _clip_rect((w // 2 - half, max(0, h - 1 - max(seed_span - 1, depth)), w // 2 + half, h - 1), w, h)
    if edge == "left":
        return _clip_rect((0, h // 2 - half, max(seed_span - 1, depth), h // 2 + half), w, h)
    if edge == "right":
        return _clip_rect((max(0, w - 1 - max(seed_span - 1, depth)), h // 2 - half, w - 1, h // 2 + half), w, h)
    raise ValueError(f"Unknown edge: {edge}")


def _find_edge_anchor_depth(
    ink_ii: np.ndarray,
    *,
    w: int,
    h: int,
    edge: str,
    search_ratio: float = 0.05,
    window_size: int = 32,
    anchor_white_ratio: float = 0.90,
) -> int:
    ws = max(8, int(window_size))
    sr = float(max(0.005, min(0.25, search_ratio)))
    aw = float(max(0.5, min(1.0, anchor_white_ratio)))
    max_depth = max(1, int(round((h if edge in {"top", "bottom"} else w) * sr)))

    if edge in {"top", "bottom"}:
        cx1 = max(0, w // 2 - ws // 2)
        cx2 = min(w - 1, cx1 + ws - 1)
        for d in range(max_depth):
            if edge == "top":
                y = d
            else:
                y = h - 1 - d
            y1 = max(0, y - ws // 2)
            y2 = min(h - 1, y1 + ws - 1)
            rect = _clip_rect((cx1, y1, cx2, y2), w, h)
            if _rect_white_ratio(ink_ii, rect) >= aw:
                return d
    else:
        cy1 = max(0, h // 2 - ws // 2)
        cy2 = min(h - 1, cy1 + ws - 1)
        for d in range(max_depth):
            if edge == "left":
                x = d
            else:
                x = w - 1 - d
            x1 = max(0, x - ws // 2)
            x2 = min(w - 1, x1 + ws - 1)
            rect = _clip_rect((x1, cy1, x2, cy2), w, h)
            if _rect_white_ratio(ink_ii, rect) >= aw:
                return d

    return max_depth - 1


def _expand_edge_empty_rect(
    ink_ii: np.ndarray,
    *,
    w: int,
    h: int,
    edge: str,
    rect: Rect,
    white_ratio_target: float = 0.997,
    step: int = 16,
) -> Rect:
    wr = float(max(0.8, min(1.0, white_ratio_target)))
    st = max(2, int(step))
    cur = _clip_rect(rect, w, h)

    def candidates(r: Rect) -> list[Rect]:
        x1, y1, x2, y2 = r
        out: list[Rect] = []
        if edge == "top":
            out.extend([
                (x1 - st, y1, x2, y2),
                (x1, y1, x2 + st, y2),
                (x1, y1, x2, y2 + st),
            ])
        elif edge == "bottom":
            out.extend([
                (x1 - st, y1, x2, y2),
                (x1, y1, x2 + st, y2),
                (x1, y1 - st, x2, y2),
            ])
        elif edge == "left":
            out.extend([
                (x1, y1 - st, x2, y2),
                (x1, y1, x2, y2 + st),
                (x1, y1, x2 + st, y2),
            ])
        elif edge == "right":
            out.extend([
                (x1, y1 - st, x2, y2),
                (x1, y1, x2, y2 + st),
                (x1 - st, y1, x2, y2),
            ])
        return [_clip_rect(c, w, h) for c in out]

    while True:
        best = cur
        best_area = _rect_area(cur)
        best_ratio = _rect_white_ratio(ink_ii, cur)
        improved = False

        for cand in candidates(cur):
            if _rect_area(cand) <= best_area:
                continue
            white = _rect_white_ratio(ink_ii, cand)
            if white < wr:
                continue
            area = _rect_area(cand)
            if area > best_area or (area == best_area and white > best_ratio):
                best = cand
                best_area = area
                best_ratio = white
                improved = True

        if not improved:
            break
        cur = best

    return cur


def detect_initial_trim_from_edge_empty_rects(
    ink: np.ndarray,
    *,
    search_ratio: float = 0.05,
    anchor_window: int = 32,
    anchor_white_ratio: float = 0.90,
    expand_white_ratio: float = 0.997,
    expand_step: int = 16,
    min_empty_area_ratio: float = 0.05,
) -> tuple[Rect, dict[str, Rect | None]]:
    h, w = ink.shape
    ink_ii = _integral_image(ink)
    page_area = max(1, w * h)
    min_empty_area = int(round(float(max(0.005, min(0.6, min_empty_area_ratio))) * page_area))

    edge_rects: dict[str, Rect | None] = {"top": None, "bottom": None, "left": None, "right": None}
    for edge in ("top", "bottom", "left", "right"):
        depth = _find_edge_anchor_depth(
            ink_ii,
            w=w,
            h=h,
            edge=edge,
            search_ratio=search_ratio,
            window_size=anchor_window,
            anchor_white_ratio=anchor_white_ratio,
        )
        seed = _center_seed_rect_for_edge(w=w, h=h, edge=edge, depth=depth, seed_span=anchor_window)
        grown = _expand_edge_empty_rect(
            ink_ii,
            w=w,
            h=h,
            edge=edge,
            rect=seed,
            white_ratio_target=expand_white_ratio,
            step=expand_step,
        )
        if _rect_area(grown) < min_empty_area:
            edge_rects[edge] = seed
        else:
            edge_rects[edge] = grown

    top = edge_rects["top"]
    bottom = edge_rects["bottom"]
    left = edge_rects["left"]
    right = edge_rects["right"]
    if top is None or bottom is None or left is None or right is None:
        return (0, 0, w - 1, h - 1), edge_rects

    x1 = max(0, left[2] + 1)
    x2 = min(w - 1, right[0] - 1)
    y1 = max(0, top[3] + 1)
    y2 = min(h - 1, bottom[1] - 1)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, w - 1, h - 1), edge_rects

    return (x1, y1, x2, y2), edge_rects


def detect_typing_border(
    ink: np.ndarray,
    *,
    window_size: int = 32,
    step: int = 16,
    window_ink_ratio: float = 0.25,
    border_percentile_q: float = 1.0,
) -> tuple[Rect, list[Rect]]:
    h, w = ink.shape
    ws = max(8, int(window_size))
    st = max(4, int(step))
    ratio = float(max(0.0, min(1.0, window_ink_ratio)))
    q = float(max(0.0, min(20.0, border_percentile_q)))

    ii = _integral_image(ink)
    provisional: list[Rect] = []

    y_max = max(1, h - ws + 1)
    x_max = max(1, w - ws + 1)
    for y in range(0, y_max, st):
        y2 = min(h, y + ws)
        if y2 - y < ws:
            continue
        for x in range(0, x_max, st):
            x2 = min(w, x + ws)
            if x2 - x < ws:
                continue
            ink_count = _rect_sum(ii, x, y, x2, y2)
            if ink_count / float(ws * ws) >= ratio:
                provisional.append((x, y, x2 - 1, y2 - 1))

    if not provisional:
        ys, xs = np.where(ink)
        if xs.size == 0 or ys.size == 0:
            return (0, 0, max(0, w - 1), max(0, h - 1)), []
        provisional = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]

    ux1 = min(r[0] for r in provisional)
    uy1 = min(r[1] for r in provisional)
    ux2 = max(r[2] for r in provisional)
    uy2 = max(r[3] for r in provisional)
    union = _clip_rect((ux1, uy1, ux2, uy2), w, h)

    region = ink[union[1] : union[3] + 1, union[0] : union[2] + 1]
    ys, xs = np.where(region)
    if xs.size == 0 or ys.size == 0:
        return union, provisional

    left = union[0] + int(np.floor(np.percentile(xs, q)))
    right = union[0] + int(np.ceil(np.percentile(xs, 100.0 - q)))
    top = union[1] + int(np.floor(np.percentile(ys, q)))
    bottom = union[1] + int(np.ceil(np.percentile(ys, 100.0 - q)))

    border = _clip_rect((left, top, right, bottom), w, h)
    return border, provisional


def _connected_components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            comps.append(comp)
    return comps


def detect_no_text_zones(
    ink: np.ndarray,
    border: Rect,
    *,
    cell_size: int = 32,
    white_ratio: float = 0.997,
    min_zone_side: int = 32,
    min_zone_area_ratio: float = 0.01,
    min_zone_area_page_ratio: float = 0.05,
    max_zone_area_ratio: float = 0.35,
    max_zone_count: int = 64,
) -> list[Rect]:
    h, w = ink.shape
    bx1, by1, bx2, by2 = _clip_rect(border, w, h)
    region = ink[by1 : by2 + 1, bx1 : bx2 + 1]
    rh, rw = region.shape
    if rh <= 0 or rw <= 0:
        return []

    cs = max(8, int(cell_size))
    wr = float(max(0.5, min(1.0, white_ratio)))
    min_zone_area_ratio = float(max(0.0, min(0.8, min_zone_area_ratio)))
    min_zone_area_page_ratio = float(max(0.0, min(0.8, min_zone_area_page_ratio)))
    max_zone_area_ratio = float(max(0.01, min(0.95, max_zone_area_ratio)))
    max_zone_count = max(1, int(max_zone_count))

    gh = max(1, rh // cs)
    gw = max(1, rw // cs)
    grid = np.zeros((gh, gw), dtype=bool)
    ii = _integral_image(region)

    for gy in range(gh):
        for gx in range(gw):
            x1 = gx * cs
            y1 = gy * cs
            x2 = min(rw, x1 + cs)
            y2 = min(rh, y1 + cs)
            area = max(1, (x2 - x1) * (y2 - y1))
            ink_count = _rect_sum(ii, x1, y1, x2, y2)
            white_frac = 1.0 - (ink_count / float(area))
            if white_frac >= wr:
                grid[gy, gx] = True

    zones: list[Rect] = []
    border_area = max(1, _rect_area(border))
    page_area = max(1, w * h)
    min_zone_area_abs = max(
        int(round(min_zone_area_ratio * border_area)),
        int(round(min_zone_area_page_ratio * page_area)),
    )
    for comp in _connected_components(grid):
        gxs = [p[0] for p in comp]
        gys = [p[1] for p in comp]
        rx1 = bx1 + min(gxs) * cs
        ry1 = by1 + min(gys) * cs
        rx2 = bx1 + min(rw - 1, (max(gxs) + 1) * cs - 1)
        ry2 = by1 + min(rh - 1, (max(gys) + 1) * cs - 1)
        rect = _clip_rect((rx1, ry1, rx2, ry2), w, h)
        if _rect_w(rect) < int(min_zone_side) or _rect_h(rect) < int(min_zone_side):
            continue
        if _rect_area(rect) < min_zone_area_abs:
            continue
        if _rect_area(rect) > int(round(max_zone_area_ratio * border_area)):
            continue
        # Keep mostly-internal zones; discard near-full border fill.
        if _rect_w(rect) >= int(0.98 * _rect_w(border)) and _rect_h(rect) >= int(0.98 * _rect_h(border)):
            continue
        zones.append(rect)

    zones.sort(key=lambda r: _rect_area(r), reverse=True)
    return _dedupe_rects(zones)[:max_zone_count]


def _split_rect_by_zone(rect: Rect, zone: Rect, min_side: int) -> list[Rect]:
    inter = _intersect(rect, zone)
    if inter is None:
        return [rect]

    rx1, ry1, rx2, ry2 = rect
    zx1, zy1, zx2, zy2 = inter
    out: list[Rect] = []

    candidates = [
        (rx1, ry1, rx2, zy1 - 1),  # top
        (rx1, zy2 + 1, rx2, ry2),  # bottom
        (rx1, max(ry1, zy1), zx1 - 1, min(ry2, zy2)),  # left
        (zx2 + 1, max(ry1, zy1), rx2, min(ry2, zy2)),  # right
    ]
    for c in candidates:
        if c[2] < c[0] or c[3] < c[1]:
            continue
        if _rect_w(c) < min_side or _rect_h(c) < min_side:
            continue
        out.append(c)
    return out


def detect_text_chunks(
    ink: np.ndarray,
    border: Rect,
    no_text_zones: list[Rect],
    *,
    min_side: int = 32,
    min_text_ink_ratio: float = 0.01,
) -> list[Rect]:
    h, w = ink.shape
    min_side = max(8, int(min_side))
    min_ink = float(max(0.0, min(1.0, min_text_ink_ratio)))
    work: list[Rect] = [_clip_rect(border, w, h)]

    for zone in no_text_zones:
        next_work: list[Rect] = []
        for rect in work:
            next_work.extend(_split_rect_by_zone(rect, zone, min_side=min_side))
        if next_work:
            work = next_work

    ii = _integral_image(ink)
    filtered: list[Rect] = []
    for rect in _dedupe_rects(work):
        x1, y1, x2, y2 = rect
        area = max(1, _rect_area(rect))
        ink_count = _rect_sum(ii, x1, y1, x2 + 1, y2 + 1)
        if (ink_count / float(area)) < min_ink:
            continue
        filtered.append(rect)

    filtered.sort(key=lambda r: (r[1], r[0]))
    return filtered


def draw_debug_overlay(
    src_img: Image.Image,
    *,
    initial_trim: Rect,
    edge_empty_rects: dict[str, Rect | None],
    border: Rect,
    no_text_zones: list[Rect],
    text_chunks: list[Rect],
) -> Image.Image:
    vis = src_img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)

    # Edge-empty rectangles: orange
    for key, rect in edge_empty_rects.items():
        if rect is None:
            continue
        d.rectangle(rect, outline=(255, 140, 0), width=2)
        d.text((rect[0] + 2, rect[1] + 2), f"E-{key[0].upper()}", fill=(255, 140, 0))

    # Initial trim from edge-empty rectangles: magenta
    d.rectangle(initial_trim, outline=(255, 0, 255), width=3)
    d.text((initial_trim[0] + 4, max(0, initial_trim[1] - 14)), "TRIM", fill=(255, 0, 255))

    # Typing border: cyan
    d.rectangle(border, outline=(0, 255, 255), width=3)
    d.text((border[0] + 4, max(0, border[1] - 14)), "BORDER", fill=(0, 255, 255))

    # No-text zones: yellow
    for i, r in enumerate(no_text_zones, start=1):
        d.rectangle(r, outline=(255, 220, 0), width=2)
        d.text((r[0] + 2, r[1] + 2), f"N{i}", fill=(255, 220, 0))

    # Text chunks: green
    for i, r in enumerate(text_chunks, start=1):
        d.rectangle(r, outline=(0, 255, 120), width=3)
        d.text((r[0] + 3, r[1] + 3), f"T{i}", fill=(0, 255, 120))

    return vis


def _canonical_pair_key(pair: PDFPair) -> str:
    return f"{pair.unredacted_numeric_id}_{pair.redacted_doc_id}"


def run_pdf_text_chunk_pipeline(
    *,
    docs_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    out_dir: Path,
    dpi: int,
    max_pairs: int | None,
    max_pages: int | None,
    window_size: int,
    window_step: int,
    window_ink_ratio: float,
    border_percentile_q: float,
    initial_trim_search_ratio: float,
    initial_trim_anchor_white_ratio: float,
    initial_trim_expand_white_ratio: float,
    initial_trim_min_empty_area_ratio: float,
    white_cell_size: int,
    white_ratio: float,
    min_zone_side: int,
    min_zone_area_ratio: float,
    min_zone_area_page_ratio: float,
    max_zone_area_ratio: float,
    min_text_side: int,
    min_text_ink_ratio: float,
) -> None:
    docs_root = docs_root.resolve()
    csv_path = docs_root / csv_name
    red_dir = docs_root / redacted_dir_name
    unred_dir = docs_root / unredacted_dir_name

    pairs = collect_pdf_pairs(
        csv_path=csv_path,
        redacted_dir=red_dir,
        unredacted_dir=unred_dir,
        max_pairs=max_pairs,
    )
    if not pairs:
        raise SystemExit(f"No valid pairs found under {docs_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    render_root = out_dir / "_rendered_pages"
    docs_out_root = out_dir / "documents"
    render_root.mkdir(parents=True, exist_ok=True)
    docs_out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for pair_idx, pair in enumerate(pairs, start=1):
        pair_key = _canonical_pair_key(pair)
        render_dir = render_root / pair_key
        if render_dir.exists():
            shutil.rmtree(render_dir)
        (render_dir / "redacted").mkdir(parents=True, exist_ok=True)
        (render_dir / "unredacted").mkdir(parents=True, exist_ok=True)

        red_pages = render_pdf_to_images(
            pdf_path=pair.redacted_pdf,
            out_dir=render_dir / "redacted",
            prefix=f"{pair_key}_redacted",
            dpi=dpi,
            max_pages=max_pages,
        )
        unred_pages = render_pdf_to_images(
            pdf_path=pair.unredacted_pdf,
            out_dir=render_dir / "unredacted",
            prefix=f"{pair_key}_unredacted",
            dpi=dpi,
            max_pages=max_pages,
        )
        page_count = min(len(red_pages), len(unred_pages))
        if page_count <= 0:
            continue

        pair_out = docs_out_root / pair_key
        if pair_out.exists():
            shutil.rmtree(pair_out)
        pair_out.mkdir(parents=True, exist_ok=True)
        pages_dir = pair_out / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        pair_manifest = {
            "pair_index_1based": pair_idx,
            "pair_key": pair_key,
            "unredacted_id": pair.unredacted_numeric_id,
            "redacted_doc_id": pair.redacted_doc_id,
            "source_unredacted_pdf": str(pair.unredacted_pdf),
            "source_redacted_pdf": str(pair.redacted_pdf),
            "page_count": page_count,
            "pages": [],
        }

        for pidx in range(page_count):
            pno = pidx + 1
            page_key = f"page_{pno:04d}"
            page_dir = pages_dir / page_key
            chunks_dir = page_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            unred_img = Image.open(unred_pages[pidx]).convert("RGB")
            red_img = Image.open(red_pages[pidx]).convert("RGB")
            if red_img.size != unred_img.size:
                red_img = red_img.resize(unred_img.size, Image.Resampling.BILINEAR)

            w, h = unred_img.size
            gray = np.array(unred_img.convert("L"))
            ink = _binarize_ink(gray)

            initial_trim, edge_rects = detect_initial_trim_from_edge_empty_rects(
                ink,
                search_ratio=initial_trim_search_ratio,
                anchor_window=window_size,
                anchor_white_ratio=initial_trim_anchor_white_ratio,
                expand_white_ratio=initial_trim_expand_white_ratio,
                expand_step=window_step,
                min_empty_area_ratio=initial_trim_min_empty_area_ratio,
            )
            tx1, ty1, tx2, ty2 = _clip_rect(initial_trim, w, h)
            masked_ink = np.zeros_like(ink, dtype=bool)
            masked_ink[ty1 : ty2 + 1, tx1 : tx2 + 1] = ink[ty1 : ty2 + 1, tx1 : tx2 + 1]

            border, provisional = detect_typing_border(
                masked_ink,
                window_size=window_size,
                step=window_step,
                window_ink_ratio=window_ink_ratio,
                border_percentile_q=border_percentile_q,
            )
            border = _clip_rect(border, w, h)
            border_inter = _intersect(border, initial_trim)
            if border_inter is not None:
                border = border_inter
            else:
                border = initial_trim
            zones = detect_no_text_zones(
                ink,
                border,
                cell_size=white_cell_size,
                white_ratio=white_ratio,
                min_zone_side=min_zone_side,
                min_zone_area_ratio=min_zone_area_ratio,
                min_zone_area_page_ratio=min_zone_area_page_ratio,
                max_zone_area_ratio=max_zone_area_ratio,
            )
            chunks = detect_text_chunks(
                ink,
                border,
                zones,
                min_side=min_text_side,
                min_text_ink_ratio=min_text_ink_ratio,
            )

            unred_page_path = page_dir / f"{page_key}.unredacted.png"
            red_page_path = page_dir / f"{page_key}.redacted.png"
            unred_img.save(unred_page_path)
            red_img.save(red_page_path)

            # Save a quick binary debug image.
            bin_img = Image.fromarray((~ink).astype(np.uint8) * 255, mode="L")
            bin_img.save(page_dir / f"{page_key}.binary_bw.png")

            overlay = draw_debug_overlay(
                unred_img,
                initial_trim=initial_trim,
                edge_empty_rects=edge_rects,
                border=border,
                no_text_zones=zones,
                text_chunks=chunks,
            )
            overlay.save(page_dir / f"{page_key}.debug_overlay.png")

            page_entry = {
                "page_no_1based": pno,
                "page_key": page_key,
                "page_size_wh": [w, h],
                "initial_trim_xyxy": [int(v) for v in initial_trim],
                "edge_empty_rectangles_xyxy": {
                    k: ([int(v) for v in r] if r is not None else None) for k, r in edge_rects.items()
                },
                "typing_border_xyxy": [int(v) for v in border],
                "provisional_window_count": len(provisional),
                "no_text_zones_xyxy": [[int(v) for v in z] for z in zones],
                "text_chunks_xyxy": [[int(v) for v in c] for c in chunks],
                "chunk_count": len(chunks),
                "files": {
                    "unredacted_page_png": str(unred_page_path),
                    "redacted_page_png": str(red_page_path),
                    "binary_bw_png": str(page_dir / f"{page_key}.binary_bw.png"),
                    "debug_overlay_png": str(page_dir / f"{page_key}.debug_overlay.png"),
                },
            }

            for cidx, rect in enumerate(chunks, start=1):
                crop_box = (rect[0], rect[1], rect[2] + 1, rect[3] + 1)
                u_crop = unred_img.crop(crop_box)
                r_crop = red_img.crop(crop_box)
                stem = f"{page_key}_chunk_{cidx:03d}"
                up = chunks_dir / f"{stem}.unredacted.png"
                rp = chunks_dir / f"{stem}.redacted.png"
                u_crop.save(up)
                r_crop.save(rp)

            page_manifest_path = page_dir / f"{page_key}.layout.json"
            page_manifest_path.write_text(json.dumps(page_entry, ensure_ascii=False, indent=2), encoding="utf-8")
            pair_manifest["pages"].append(page_entry)

            manifest_rows.append(
                {
                    "pair_key": pair_key,
                    "page_no_1based": pno,
                    "chunk_count": len(chunks),
                    "initial_trim_xyxy": [int(v) for v in initial_trim],
                    "typing_border_xyxy": [int(v) for v in border],
                    "page_layout_json": str(page_manifest_path),
                    "page_folder": str(page_dir),
                }
            )
            print(f"[PAGE] {pair_key} {page_key} chunks={len(chunks)}")

        pair_manifest_path = pair_out / "pair_manifest.json"
        pair_manifest_path.write_text(json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_jsonl = out_dir / "chunk_manifest.jsonl"
    manifest_json = out_dir / "chunk_manifest.json"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] pairs_processed={len(pairs)} pages_processed={len(manifest_rows)}")
    print(f"[INFO] saved={manifest_jsonl}")
    print(f"[INFO] saved={manifest_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract page text chunks from unredacted PDFs and apply same crops to redacted PDFs.")
    parser.add_argument("--docs_root", default="docs", help="Root containing cibcia.csv, redacted_pdfs, unredacted_pdfs")
    parser.add_argument("--csv_name", default="cibcia.csv", help="CSV filename under docs_root")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs", help="Redacted pdf folder name")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs", help="Unredacted pdf folder name")
    parser.add_argument("--out", default="docs_text_chunks", help="Output root")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional pair cap")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional pages per pdf cap")

    parser.add_argument("--window_size", type=int, default=32, help="Sliding ink window size")
    parser.add_argument("--window_step", type=int, default=16, help="Sliding ink window stride")
    parser.add_argument("--window_ink_ratio", type=float, default=0.25, help="Ink ratio threshold for provisional text windows")
    parser.add_argument("--border_percentile_q", type=float, default=1.0, help="Percentile q for border from union ink pixels")
    parser.add_argument("--initial_trim_search_ratio", type=float, default=0.05, help="Max depth ratio for edge anchor search")
    parser.add_argument("--initial_trim_anchor_white_ratio", type=float, default=0.90, help="Whitespace threshold for edge-anchor detection")
    parser.add_argument("--initial_trim_expand_white_ratio", type=float, default=0.997, help="Whitespace threshold when expanding edge empty rectangles")
    parser.add_argument("--initial_trim_min_empty_area_ratio", type=float, default=0.05, help="Minimum page-area ratio for each edge empty rectangle")

    parser.add_argument("--white_cell_size", type=int, default=32, help="Grid cell size for no-text-zone detection")
    parser.add_argument("--white_ratio", type=float, default=0.997, help="White ratio threshold for no-text-zone cells")
    parser.add_argument("--min_zone_side", type=int, default=32, help="Minimum side length for no-text zones")
    parser.add_argument("--min_zone_area_ratio", type=float, default=0.01, help="Minimum no-text-zone area as ratio of typing border area")
    parser.add_argument("--min_zone_area_page_ratio", type=float, default=0.05, help="Minimum no-text-zone area as ratio of full page area")
    parser.add_argument("--max_zone_area_ratio", type=float, default=0.35, help="Maximum no-text-zone area as ratio of typing border area")
    parser.add_argument("--min_text_side", type=int, default=32, help="Minimum side length for final text chunks")
    parser.add_argument("--min_text_ink_ratio", type=float, default=0.01, help="Minimum ink ratio for final text chunks")
    args = parser.parse_args()

    run_pdf_text_chunk_pipeline(
        docs_root=Path(args.docs_root),
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        out_dir=Path(args.out),
        dpi=args.dpi,
        max_pairs=args.max_pairs,
        max_pages=args.max_pages,
        window_size=args.window_size,
        window_step=args.window_step,
        window_ink_ratio=args.window_ink_ratio,
        border_percentile_q=args.border_percentile_q,
        initial_trim_search_ratio=args.initial_trim_search_ratio,
        initial_trim_anchor_white_ratio=args.initial_trim_anchor_white_ratio,
        initial_trim_expand_white_ratio=args.initial_trim_expand_white_ratio,
        initial_trim_min_empty_area_ratio=args.initial_trim_min_empty_area_ratio,
        white_cell_size=args.white_cell_size,
        white_ratio=args.white_ratio,
        min_zone_side=args.min_zone_side,
        min_zone_area_ratio=args.min_zone_area_ratio,
        min_zone_area_page_ratio=args.min_zone_area_page_ratio,
        max_zone_area_ratio=args.max_zone_area_ratio,
        min_text_side=args.min_text_side,
        min_text_ink_ratio=args.min_text_ink_ratio,
    )


if __name__ == "__main__":
    main()
