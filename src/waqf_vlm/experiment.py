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
class Disagreement:
    heading: str
    explanation: str
    bbox: tuple[float, float, float, float]
    observations: list[tuple[str, str]] = field(default_factory=list)
    crop: str | None = None


def system_name(model: str) -> str:
    return {
        "qwen3-vl-8b-instruct-mlx": "Qwen3-VL 8B",
        "qwen3-vl-30b-a3b-instruct-mlx": "Qwen3-VL 30B",
    }.get(model, model)


def _boxes(system: SegmentationSystem) -> list[Region]:
    # Explicit bbox policy: do not compare one system's polygons with another's boxes.
    return [replace(r, polygon=[]) for r in system.regions]


def find_disagreements(systems: list[SegmentationSystem], labels: dict[str, str], *, limit: int = 4) -> list[Disagreement]:
    """Illustrative differences, not errors. No system is treated as ground truth.

    Compare predictions with greedy one-to-one bbox IoU >= 0.5 matching. Prioritize
    the two Qwen variants, then classification differences, then larger unmatched
    regions. Deduplicate overlapping examples. No review or correctness is inferred.
    """
    predictions = [s for s in systems if not s.reviewed]
    candidates = []
    for left, right in combinations(predictions, 2):
        matches, left_only, right_only = match_regions(_boxes(left), _boxes(right), iou_threshold=0.5, require_same_type=False)
        priority = 0 if all(s.name.startswith("Qwen3-VL") for s in (left, right)) else 1
        for match in matches:
            if not match.type_match:
                bounds = (min(match.truth.bbox[0], match.prediction.bbox[0]),
                          min(match.truth.bbox[1], match.prediction.bbox[1]),
                          max(match.truth.bbox[2], match.prediction.bbox[2]),
                          max(match.truth.bbox[3], match.prediction.bbox[3]))
                explanation = (f"{left.name} labels an overlapping region as {labels.get(match.truth.region_type, match.truth.region_type).lower()}, "
                               f"while {right.name} labels it as {labels.get(match.prediction.region_type, match.prediction.region_type).lower()}. "
                               "Their boxes meet the overlap threshold used for this comparison.")
                candidates.append((priority, 0, -match.iou, Disagreement("Different descriptions of an overlapping region", explanation, bounds)))
        for owner, other, regions in ((left, right, left_only), (right, left, right_only)):
            for region in regions:
                x1, y1, x2, y2 = region.bbox
                explanation = (f"{owner.name} identifies this region as {labels.get(region.region_type, region.region_type).lower()}. "
                               f"It has no one-to-one counterpart in {other.name} at the chosen overlap threshold. "
                               "This may reflect a difference in extent or grouping, rather than an absent detection.")
                candidates.append((priority, 1, -(x2-x1)*(y2-y1), Disagreement("A region without a matching counterpart", explanation, region.bbox)))
    selected = []
    for _, _, _, candidate in sorted(candidates, key=lambda item: item[:3]):
        anchor = Region("crop", "unknown", candidate.bbox)
        if any(intersection_over_union(anchor, Region("other", "unknown", prior.bbox)) >= .4 for prior in selected):
            continue
        for system in predictions:
            overlapping = [(intersection_over_union(anchor, region), region) for region in _boxes(system)]
            overlap, region = max(overlapping, key=lambda item: item[0], default=(0, None))
            if region is None or overlap == 0:
                observation = "No saved region box overlaps this crop area."
            else:
                observation = (f"The most-overlapping saved box is labeled {labels.get(region.region_type, region.region_type).lower()} "
                               f"(region {region.id}). Its boundaries may extend beyond this crop.")
            candidate.observations.append((system.name, observation))
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


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
    for disagreement in disagreements:
        relative, destination = _destination(output, f"disagreement:{source.name}:{disagreement.bbox}")
        crop = crop_normalized_bbox(source, disagreement.bbox, padding=30)
        crop.thumbnail((1200, 1200))
        crop.save(destination)
        disagreement.crop = relative
