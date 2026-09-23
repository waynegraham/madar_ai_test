from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .segmentation import Region, VALID_REGION_TYPES


SCHEMA_VERSION = "0.1"


@dataclass
class HumanAnnotation:
    id: str
    region_type: str
    bbox: tuple[float, float, float, float]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.region_type,
            "bbox": list(self.bbox),
        }

        if self.notes:
            result["notes"] = self.notes

        return result


@dataclass
class AnnotationDocument:
    image: str
    image_width: int
    image_height: int
    coordinate_system: str
    schema_version: str
    regions: list[HumanAnnotation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_system": self.coordinate_system,
            "schema_version": self.schema_version,
            "regions": [
                region.to_dict()
                for region in self.regions
            ],
        }


def make_annotation(
    region_type: str,
    bbox: tuple[float, float, float, float],
    *,
    notes: str | None = None,
) -> HumanAnnotation:
    """
    Create a human annotation with a stable unique ID.
    """

    if region_type not in VALID_REGION_TYPES:
        raise ValueError(
            f"Unknown region type: {region_type}"
        )

    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box: {bbox}"
        )

    return HumanAnnotation(
        id=f"human-{uuid4().hex[:8]}",
        region_type=region_type,
        bbox=bbox,
        notes=notes,
    )


def pixels_to_normalized_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> tuple[float, float, float, float]:
    """
    Convert image-pixel coordinates to normalized 0-1000
    coordinates.
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 / width) * coordinate_max,
        (y1 / height) * coordinate_max,
        (x2 / width) * coordinate_max,
        (y2 / height) * coordinate_max,
    )


def normalized_to_pixels_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> tuple[float, float, float, float]:

    x1, y1, x2, y2 = bbox

    return (
        (x1 / coordinate_max) * width,
        (y1 / coordinate_max) * height,
        (x2 / coordinate_max) * width,
        (y2 / coordinate_max) * height,
    )


def save_annotations(
    path: str | Path,
    *,
    image: str,
    image_width: int,
    image_height: int,
    regions: list[HumanAnnotation],
) -> Path:
    """
    Save human annotations as normalized JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = AnnotationDocument(
        image=image,
        image_width=image_width,
        image_height=image_height,
        coordinate_system="normalized-1000",
        schema_version=SCHEMA_VERSION,
        regions=regions,
    )

    path.write_text(
        json.dumps(
            document.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def load_annotations(
    path: str | Path,
) -> AnnotationDocument:
    """
    Load an existing human annotation file.
    """

    path = Path(path)

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    regions = []

    for item in data.get("regions", []):
        regions.append(
            HumanAnnotation(
                id=item["id"],
                region_type=item["type"],
                bbox=tuple(item["bbox"]),
                notes=item.get("notes"),
            )
        )

    return AnnotationDocument(
        image=data["image"],
        image_width=data["image_width"],
        image_height=data["image_height"],
        coordinate_system=data.get(
            "coordinate_system",
            "normalized-1000",
        ),
        schema_version=data.get(
            "schema_version",
            "0.1",
        ),
        regions=regions,
    )


def annotations_to_regions(
    document: AnnotationDocument,
) -> list[Region]:
    """
    Convert human annotations to the same Region schema used
    by eScriptorium and VLM predictions.

    This lets evaluation.py compare all three systems directly.
    """

    return [
        Region(
            id=annotation.id,
            region_type=annotation.region_type,
            bbox=annotation.bbox,
            source="human",
            metadata={
                "notes": annotation.notes,
            },
        )
        for annotation in document.regions
    ]