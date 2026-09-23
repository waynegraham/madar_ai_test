# src/segmentation.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

from src.alto import AltoPage


RegionType = Literal[
    "main_text",
    "marginal_text",
    "stamp_or_seal",
    "handwritten_annotation",
    "archival_mark",
    "illustration",
    "unknown",
]

VALID_REGION_TYPES = {
    "main_text",
    "marginal_text",
    "stamp_or_seal",
    "handwritten_annotation",
    "archival_mark",
    "illustration",
    "unknown",
}


@dataclass
class Region:
    """
    Model-independent representation of a document region.

    Coordinates are normalized to a 0-1000 coordinate system by
    default so results can be compared across derivative image sizes.
    """

    id: str
    region_type: str

    bbox: tuple[float, float, float, float]

    polygon: list[tuple[float, float]] = field(
        default_factory=list
    )

    confidence: float | None = None

    text: str | None = None
    language: str | None = None
    script: str | None = None

    source: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> tuple[float, float, float, float]:
    """
    Convert pixel coordinates to normalized coordinates.
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 / width) * coordinate_max,
        (y1 / height) * coordinate_max,
        (x2 / width) * coordinate_max,
        (y2 / height) * coordinate_max,
    )


def normalize_polygon(
    polygon: Iterable[tuple[float, float]],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> list[tuple[float, float]]:
    """
    Convert a pixel-coordinate polygon to normalized coordinates.
    """

    return [
        (
            (x / width) * coordinate_max,
            (y / height) * coordinate_max,
        )
        for x, y in polygon
    ]


def validate_bbox(
    bbox: tuple[float, float, float, float],
    *,
    coordinate_max: int = 1000,
) -> None:
    """
    Validate normalized bounding coordinates.
    """

    x1, y1, x2, y2 = bbox

    if not (
        0 <= x1 <= coordinate_max
        and 0 <= y1 <= coordinate_max
        and 0 <= x2 <= coordinate_max
        and 0 <= y2 <= coordinate_max
    ):
        raise ValueError(
            f"BBox outside coordinate range: {bbox}"
        )

    if x2 <= x1:
        raise ValueError(
            f"BBox has invalid horizontal extent: {bbox}"
        )

    if y2 <= y1:
        raise ValueError(
            f"BBox has invalid vertical extent: {bbox}"
        )


def normalize_region_type(
    value: str | None,
) -> str:
    """
    Normalize model/ALTO labels into our shared vocabulary.
    """

    if not value:
        return "unknown"

    normalized = (
        value
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "main": "main_text",
        "text": "main_text",
        "body": "main_text",
        "body_text": "main_text",

        "commentary": "marginal_text",
        "margin": "marginal_text",
        "marginalia": "marginal_text",

        "stamp": "stamp_or_seal",
        "seal": "stamp_or_seal",
        "stamp_seal": "stamp_or_seal",

        "note": "handwritten_annotation",
        "annotation": "handwritten_annotation",
        "handwriting": "handwritten_annotation",

        "archive_mark": "archival_mark",
        "catalog_mark": "archival_mark",
        "cataloguing_mark": "archival_mark",

        "image": "illustration",
        "figure": "illustration",
    }

    normalized = aliases.get(
        normalized,
        normalized,
    )

    if normalized not in VALID_REGION_TYPES:
        return "unknown"

    return normalized


def alto_to_regions(
    page: AltoPage,
    *,
    coordinate_max: int = 1000,
) -> list[Region]:
    """
    Convert an AltoPage into the normalized Region schema.

    Important:

    These are eScriptorium predictions, not ground truth.
    """

    regions: list[Region] = []

    for index, block in enumerate(page.blocks):

        if block.bbox is None:
            continue

        bbox = normalize_bbox(
            block.bbox,
            page.width,
            page.height,
            coordinate_max=coordinate_max,
        )

        polygon = normalize_polygon(
            block.polygon,
            page.width,
            page.height,
            coordinate_max=coordinate_max,
        )

        text_parts = [
            line.text
            for line in block.lines
            if line.text
        ]

        text = (
            "\n".join(text_parts)
            if text_parts
            else None
        )

        line_confidences = [
            line.confidence
            for line in block.lines
            if line.confidence is not None
        ]

        confidence = (
            sum(line_confidences)
            / len(line_confidences)
            if line_confidences
            else None
        )

        regions.append(
            Region(
                id=block.id or f"alto-{index}",
                region_type=normalize_region_type(
                    block.block_type
                ),
                bbox=bbox,
                polygon=polygon,
                confidence=confidence,
                text=text,
                source="escriptorium",
                metadata={
                    "alto_type": block.block_type,
                    "tagrefs": block.tagrefs,
                },
            )
        )

    return regions


def vlm_json_to_regions(
    data: dict[str, Any],
    *,
    source: str,
) -> list[Region]:
    """
    Convert structured VLM JSON into Region objects.

    Expected structure:

        {
            "regions": [
                {
                    "type": "stamp_or_seal",
                    "bbox": [100, 200, 400, 500],
                    "confidence": 0.91
                }
            ]
        }
    """

    regions: list[Region] = []

    raw_regions = data.get("regions", [])

    if not isinstance(raw_regions, list):
        raise ValueError(
            "'regions' must be a list"
        )

    for index, item in enumerate(raw_regions):

        if not isinstance(item, dict):
            continue

        raw_bbox = item.get("bbox")

        if (
            not isinstance(raw_bbox, list)
            or len(raw_bbox) != 4
        ):
            continue

        bbox = tuple(
            float(value)
            for value in raw_bbox
        )

        validate_bbox(bbox)

        region_type = normalize_region_type(
            item.get("type")
        )

        regions.append(
            Region(
                id=str(
                    item.get(
                        "id",
                        f"{source}-{index:04d}",
                    )
                ),
                region_type=region_type,
                bbox=bbox,
                confidence=item.get("confidence"),
                text=item.get("text"),
                language=item.get("language"),
                script=item.get("script"),
                source=source,
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {
                        "id",
                        "type",
                        "bbox",
                        "confidence",
                        "text",
                        "language",
                        "script",
                    }
                },
            )
        )

    return regions