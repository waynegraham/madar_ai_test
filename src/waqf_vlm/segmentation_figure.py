"""Deterministic publication figures from existing normalized Region coordinates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from src.images import load_image, normalized_bbox_to_pixels
from src.segmentation import Region, validate_bbox

# One accent; type codes remain identical across systems and review states.
REGION_STYLES = {
    "main_text": ("MT", "Main text"),
    "marginal_text": ("MG", "Marginal text"),
    "stamp_or_seal": ("SS", "Stamp or seal"),
    "handwritten_annotation": ("HA", "Handwritten annotation"),
    "archival_mark": ("AM", "Archival mark"),
    "illustration": ("IL", "Illustration"),
    "unknown": ("UN", "Unclassified region"),
}
PAPER = "#f8f5ee"
INK = "#292b28"
ACCENT = "#31594c"
MUTED = "#5d6058"


def _stroke(draw, points, *, reviewed: bool, width: int = 3):
    """Solid reference boundaries and dashed predictions, with a fine light halo."""
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if not length:
            continue
        pieces = [(0, length)] if reviewed else [(s, min(s + 13, length)) for s in range(0, math.ceil(length), 21)]
        for first, last in pieces:
            line = [(start[0] + dx * t / length, start[1] + dy * t / length) for t in (first, last)]
            draw.line(line, fill=PAPER, width=width + 2)
            draw.line(line, fill=ACCENT, width=width)


def render_segmentation(
    original_image: str | Path,
    regions: Sequence[Region],
    system_name: str,
    human_ground_truth: Sequence[Region] | None = None,
    *,
    output_dir: str | Path = "reports/generated/assets",
    show_confidence: bool = False,
    reviewed: bool = False,
    max_dimension: int = 1800,
) -> Path:
    """Return a PNG path; never mutate the image or Region objects.

    Coordinates must use the project's normalized 0–1000 bbox convention.
    Polygons are deliberately not substituted for boxes, keeping report inventories
    and illustrations aligned. `human_ground_truth` must be explicitly reviewed
    annotations, never machine ALTO simply originating in eScriptorium. `reviewed`
    marks the primary list as reviewed for a reference-only figure.

    Labels and the legend sit on paper outside the manuscript. Fine leader lines
    connect box edges to an external label rail; no text or opaque label panels
    cover the manuscript. Type codes and numbered references are redundant with
    the external type names. Confidence is neither rendered nor hashed by default.

    Filename and PNG bytes are deterministic for the same inputs and Pillow
    version. No timestamps, randomized IDs, model calls, or remote fonts are used.
    """
    if max_dimension < 300:
        raise ValueError("max_dimension must be at least 300")
    source = Path(original_image)
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    entries = [(r, reviewed, f"{'H' if reviewed else 'P'}{i:02d}") for i, r in enumerate(regions, 1)]
    entries += [(r, True, f"H{i:02d}") for i, r in enumerate(human_ground_truth or [], 1)]
    if reviewed and human_ground_truth is not None:
        raise ValueError("For a reference-only figure, pass reviewed=True without human_ground_truth")
    for region, _, _ in entries:
        validate_bbox(region.bbox)
    image = load_image(source).convert("RGB")
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=21)
    heading = ImageFont.load_default(size=32)
    # Extra canvas height handles dense pages without overlapping label text.
    margin, rail, header, row = 48, 510, 150, 56
    body_height = max(image.height, len(entries) * row + 20)
    types = [key for key in REGION_STYLES if any((r.region_type if r.region_type in REGION_STYLES else "unknown") == key for r, _, _ in entries)]
    footer_height = 135 + 34 * math.ceil(len(types) / 2)
    canvas = Image.new("RGB", (max(950, image.width + 2 * margin + rail), header + body_height + footer_height), PAPER)
    image_y = header + (body_height - image.height) // 2
    canvas.paste(image, (margin, image_y))
    draw = ImageDraw.Draw(canvas)
    # Wrap long system identifiers instead of clipping them.
    title_words, title_lines = system_name.split(), [""]
    for word in title_words:
        candidate = (title_lines[-1] + " " + word).strip()
        if draw.textlength(candidate, font=heading) > canvas.width - 2 * margin and title_lines[-1]:
            title_lines.append(word)
        else:
            title_lines[-1] = candidate
    # Keep the title restrained even for an unusually long identifier.
    title = " / ".join(title_lines)
    while draw.textlength(title, font=heading) > canvas.width - 2 * margin:
        heading = ImageFont.load_default(size=max(8, heading.size - 1))
        if heading.size == 8:
            break
    draw.text((margin, 30), title, fill=INK, font=heading)
    status = "Human-reviewed annotations" if reviewed else "Machine predictions · not reviewed"
    draw.text((margin, 83), status, fill=MUTED, font=small)
    if human_ground_truth is not None:
        draw.text((margin, 112), "With human-reviewed reference annotations", fill=MUTED, font=small)
    label_x = margin + image.width + 65
    projected = []
    for region, is_reviewed, reference in entries:
        x1, y1, x2, y2 = normalized_bbox_to_pixels(region.bbox, image.width, image.height)
        bounds = (margin + x1, image_y + y1, margin + x2, image_y + y2)
        projected.append((bounds, region, is_reviewed, reference))
    # Vertical sorting affects layout only; identifiers preserve input list order.
    projected.sort(key=lambda item: ((item[0][1] + item[0][3]) / 2, item[3]))
    positions = []
    for i, (bounds, *_rest) in enumerate(projected):
        desired = (bounds[1] + bounds[3]) / 2
        lower = header + 20 if not positions else positions[-1] + row
        upper = header + body_height - 25 - (len(projected) - 1 - i) * row
        positions.append(max(lower, min(desired, upper)))
    # Leaders first so all region boundaries stay visually above them.
    for (bounds, region, is_reviewed, reference), label_y in zip(projected, positions):
        x1, y1, x2, y2 = bounds
        cy = (y1 + y2) / 2
        draw.line([(x2, cy), (margin + image.width + 18, cy), (label_x - 18, label_y), (label_x - 6, label_y)], fill=MUTED, width=1)
        code, label = REGION_STYLES.get(region.region_type, REGION_STYLES["unknown"])
        draw.text((label_x, label_y - 24), f"{reference} · {code}  {label}", fill=INK, font=small)
        detail = "Human reviewed" if is_reviewed else "Machine suggested"
        if show_confidence and region.confidence is not None:
            try:
                confidence = float(region.confidence)
                detail += f" · confidence {confidence:.2f}" if math.isfinite(confidence) else " · confidence unavailable"
            except (ValueError, TypeError):
                detail += " · confidence unavailable"
        draw.text((label_x, label_y + 3), detail, fill=MUTED, font=ImageFont.load_default(size=17))
    for bounds, _, is_reviewed, _ in projected:
        x1, y1, x2, y2 = bounds
        _stroke(draw, [(x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)], reviewed=is_reviewed)
    legend_y = header + body_height + 38
    draw.line((margin, legend_y - 17, canvas.width - margin, legend_y - 17), fill="#ccc8bd", width=1)
    draw.text((margin, legend_y), "REGION KEY", fill=INK, font=small)
    for i, key in enumerate(types):
        code, label = REGION_STYLES[key]
        draw.text((margin + (i % 2) * ((canvas.width - 2 * margin) // 2), legend_y + 38 + (i // 2) * 34), f"{code}  {label}", fill=INK, font=small)
    style_y = legend_y + 45 + 34 * math.ceil(len(types) / 2)
    _stroke(draw, [(margin, style_y + 10), (margin + 55, style_y + 10)], reviewed=False)
    draw.text((margin + 68, style_y - 2), "Machine prediction", fill=MUTED, font=small)
    start = margin + 350
    _stroke(draw, [(start, style_y + 10), (start + 55, style_y + 10)], reviewed=True)
    draw.text((start + 68, style_y - 2), "Human reviewed", fill=MUTED, font=small)
    payload = {"renderer": 1, "system": system_name, "size": max_dimension, "reviewed": reviewed,
               "reference_supplied": human_ground_truth is not None, "confidence": show_confidence,
               "regions": [(r.id, r.region_type, r.bbox, h, ref, str(r.confidence) if show_confidence else None) for r,h,ref in entries]}
    digest = hashlib.sha256(source.read_bytes() + json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"segmentation-{digest}.png"
    if destination.is_symlink() or destination.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite an image source or symlink")
    canvas.save(destination, format="PNG", compress_level=9, dpi=(180, 180))
    return destination
