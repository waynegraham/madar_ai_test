"""Saved-coordinate illustrations and descriptive comparisons, never inference."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from itertools import combinations
from pathlib import Path

from src.evaluation import intersection_over_union, match_regions
from src.images import crop_normalized_bbox
from src.segmentation import Region
from .segmentation_figure import render_segmentation


@dataclass
class SegmentationSystem:
    name: str
    source: str
    regions: list[Region]
    reviewed: bool = False
    skipped_blocks: int = 0
    overlay: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(r.region_type for r in self.regions))


@dataclass
class RegionObservation:
    system: str
    label: str
    relationship: str
    region_id: str | None = None
    source: str | None = None
    overlap: float | None = None
    context: str | None = None


@dataclass
class Disagreement:
    heading: str
    explanation: str
    bbox: tuple[float, float, float, float]
    pair: tuple[str, str]
    kind: str
    observations: list[RegionObservation] = field(default_factory=list)
    human_review: list[RegionObservation] = field(default_factory=list)
    crop: str | None = None
    crop_preview: str | None = None
    crop_width: int | None = None
    crop_height: int | None = None

    @property
    def identifier(self) -> str:
        return hashlib.sha256(repr((self.pair, self.kind, self.bbox,
            [(o.source, o.region_id, o.relationship) for o in self.observations])).encode()).hexdigest()[:16]


def system_name(model: str) -> str:
    return {
        "qwen3-vl-8b-instruct-mlx": "Qwen3-VL 8B",
        "qwen3-vl-30b-a3b-instruct-mlx": "Qwen3-VL 30B",
    }.get(model, model)


def _boxes(system: SegmentationSystem) -> list[Region]:
    # Explicit bbox policy: do not compare one system's polygons with another's boxes.
    return [replace(r, polygon=[]) for r in system.regions]


def _observation(system: SegmentationSystem, anchor: Region, labels: dict[str, str],
                 exact: Region | None = None, *, unmatched: bool = False) -> RegionObservation:
    """Keep the original pairwise assignment separate from contextual overlap."""
    if exact is not None:
        return RegionObservation(system.name, labels.get(exact.region_type, exact.region_type),
                                 "Region in this comparison", exact.id, system.source,
                                 intersection_over_union(anchor, replace(exact, polygon=[])))
    candidates = [(intersection_over_union(anchor, r), r) for r in _boxes(system)]
    overlap, nearby = max(candidates, key=lambda value: value[0], default=(0, None))
    if not unmatched and nearby is not None and overlap >= .5:
        return RegionObservation(system.name, labels.get(nearby.region_type, nearby.region_type),
                                 "Overlapping region (compared independently with the crop area)",
                                 nearby.id, system.source, overlap)
    context = None
    if nearby is not None and overlap > 0:
        label = labels.get(nearby.region_type, nearby.region_type)
        context = f"A nearby or differently grouped box is labeled {label} (region {nearby.id}). This is not a matched counterpart."
    return RegionObservation(system.name, "No matched region", "No one-to-one counterpart in this pair" if unmatched
                             else "No box meets the overlap threshold for this crop area",
                             source=system.source, overlap=overlap, context=context)


def select_disagreements(disagreements: list[Disagreement], *, limit: int = 4) -> list[Disagreement]:
    """A bounded, spatially varied selection; the full pairwise list is untouched."""
    if limit < 0:
        raise ValueError("limit must not be negative")
    selected = []
    for candidate in disagreements:
        if len(selected) >= limit:
            break
        anchor = Region("crop", "unknown", candidate.bbox)
        if any(intersection_over_union(anchor, Region("other", "unknown", prior.bbox)) >= .4 for prior in selected):
            continue
        selected.append(candidate)
    return selected


def find_disagreements(systems: list[SegmentationSystem], labels: dict[str, str], *, limit: int | None = 4) -> list[Disagreement]:
    """Describe pairwise prediction differences using bbox IoU >= .5.

    Pass limit=None for every pairwise label conflict and unmatched region,
    including repeated features across different pairs. Default selection retains
    at most four spatially varied examples. References never create disagreements.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")
    predictions = [s for s in systems if not s.reviewed]
    references = [s for s in systems if s.reviewed]
    candidates = []
    for left, right in combinations(predictions, 2):
        matches, left_only, right_only = match_regions(_boxes(left), _boxes(right), iou_threshold=0.5, require_same_type=False)
        priority = 0 if all(s.name.startswith("Qwen3-VL") for s in (left, right)) else 1
        pair = (left.name, right.name)
        for match in matches:
            if not match.type_match:
                bounds = (min(match.truth.bbox[0], match.prediction.bbox[0]),
                          min(match.truth.bbox[1], match.prediction.bbox[1]),
                          max(match.truth.bbox[2], match.prediction.bbox[2]),
                          max(match.truth.bbox[3], match.prediction.bbox[3]))
                candidate = Disagreement("Different descriptions of an overlapping region",
                    f"{left.name} and {right.name} assigned different types to boxes that meet the overlap threshold. Neither label is endorsed by this comparison.",
                    bounds, pair, "different_types")
                candidates.append((priority, 0, -match.iou, candidate, left, right, match.truth, match.prediction))
        for owner, other, regions in ((left, right, left_only), (right, left, right_only)):
            for region in regions:
                x1, y1, x2, y2 = region.bbox
                candidate = Disagreement("A region without a matching counterpart",
                    f"{owner.name} recorded this region, but it has no one-to-one counterpart in {other.name} at the chosen overlap threshold. A difference in extent or grouping can produce this outcome; it does not establish that a feature was overlooked.",
                    region.bbox, pair, "unmatched")
                candidates.append((priority, 1, -(x2-x1)*(y2-y1), candidate, owner, other, region, None))
    all_disagreements = []
    for _, _, _, candidate, owner, other, owner_region, other_region in sorted(candidates, key=lambda item: item[:3]):
        anchor = Region("crop", "unknown", candidate.bbox)
        for system in predictions:
            if system is owner:
                observation = _observation(system, anchor, labels, owner_region)
            elif system is other:
                observation = _observation(system, anchor, labels, other_region, unmatched=other_region is None)
            else:
                observation = _observation(system, anchor, labels)
            candidate.observations.append(observation)
        for reference in references:
            observation = _observation(reference, anchor, labels)
            if observation.region_id is None:
                observation.label = "No matching reviewed region"
            else:
                observation.relationship = "Human-reviewed reference overlapping this crop area"
            candidate.human_review.append(observation)
        all_disagreements.append(candidate)
    return all_disagreements if limit is None else select_disagreements(all_disagreements, limit=limit)


def _destination(output: Path, identity: str) -> tuple[str, Path]:
    relative = "images/" + hashlib.sha256(identity.encode()).hexdigest()[:20] + ".png"
    path = output / relative
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output: {path}")
    return relative, path


def illustrate_experiment(systems: list[SegmentationSystem], disagreements: list[Disagreement], source: Path, output: Path) -> None:
    """Draw numbered boxes over a resized original; crop from the full-resolution source."""
    for system in systems:
        destination = render_segmentation(source, system.regions, system.name,
                                          output_dir=output / "assets", reviewed=system.reviewed)
        system.overlay = destination.relative_to(output).as_posix()
    crops = {}
    for disagreement in disagreements:
        if disagreement.bbox not in crops:
            relative, destination = _destination(output, f"disagreement-original:{source.name}:{disagreement.bbox}")
            preview_relative, preview_destination = _destination(output, f"disagreement-preview:{source.name}:{disagreement.bbox}")
            # Preserve native source pixels for inspection; only the display copy shrinks.
            crop = crop_normalized_bbox(source, disagreement.bbox, padding=30)
            crop.save(destination)
            width, height = crop.size
            preview = crop.copy()
            preview.thumbnail((900, 900))
            preview.save(preview_destination)
            crops[disagreement.bbox] = (relative, preview_relative, width, height)
        disagreement.crop, disagreement.crop_preview, disagreement.crop_width, disagreement.crop_height = crops[disagreement.bbox]
