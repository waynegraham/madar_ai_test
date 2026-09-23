# src/images.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    mode: str
    format: str | None


def image_info(path: str | Path) -> ImageInfo:
    """
    Return basic metadata without modifying the image.
    """
    path = Path(path)

    with Image.open(path) as image:
        return ImageInfo(
            path=path,
            width=image.width,
            height=image.height,
            mode=image.mode,
            format=image.format,
        )


def load_image(
    path: str | Path,
    *,
    apply_exif_orientation: bool = True,
) -> Image.Image:
    """
    Load an image into memory.

    The returned image is detached from the underlying file, so it
    remains usable after Image.open() closes the file.
    """
    with Image.open(path) as image:
        if apply_exif_orientation:
            image = ImageOps.exif_transpose(image)

        return image.copy()


def create_vlm_derivative(
    source: str | Path,
    destination: str | Path,
    *,
    max_dimension: int = 2048,
    format: str = "PNG",
) -> Path:
    """
    Create a VLM-friendly derivative while preserving aspect ratio.

    The original TIFF remains untouched.

    If the image is already smaller than max_dimension, it is not
    enlarged.
    """
    source = Path(source)
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = load_image(source)

    # Avoid problems with unusual TIFF modes.
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")

    scale = min(
        1.0,
        max_dimension / max(image.width, image.height),
    )

    if scale < 1.0:
        new_size = (
            round(image.width * scale),
            round(image.height * scale),
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    image.save(
        destination,
        format=format,
    )

    return destination


def normalized_bbox_to_pixels(
    bbox: BBox,
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> tuple[int, int, int, int]:
    """
    Convert normalized 0-coordinate_max coordinates to pixels.

    Example:

        bbox = (250, 100, 750, 900)

    on a 2000 x 3000 image becomes approximately:

        (500, 300, 1500, 2700)
    """
    x1, y1, x2, y2 = bbox

    return (
        round((x1 / coordinate_max) * width),
        round((y1 / coordinate_max) * height),
        round((x2 / coordinate_max) * width),
        round((y2 / coordinate_max) * height),
    )


def pixel_bbox_to_normalized(
    bbox: BBox,
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> BBox:
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


def normalized_polygon_to_pixels(
    polygon: Sequence[Point],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> list[tuple[int, int]]:
    """
    Convert normalized polygon coordinates to image pixels.
    """
    return [
        (
            round((x / coordinate_max) * width),
            round((y / coordinate_max) * height),
        )
        for x, y in polygon
    ]


def pixel_polygon_to_normalized(
    polygon: Sequence[Point],
    width: int,
    height: int,
    *,
    coordinate_max: int = 1000,
) -> list[Point]:
    """
    Convert pixel polygon coordinates to normalized coordinates.
    """
    return [
        (
            (x / width) * coordinate_max,
            (y / height) * coordinate_max,
        )
        for x, y in polygon
    ]


def clamp_bbox(
    bbox: BBox,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Ensure a bounding box remains inside the image.
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(round(x1), width))
    y1 = max(0, min(round(y1), height))
    x2 = max(0, min(round(x2), width))
    y2 = max(0, min(round(y2), height))

    if x2 <= x1:
        raise ValueError(
            f"Invalid bbox after clamping: {bbox}"
        )

    if y2 <= y1:
        raise ValueError(
            f"Invalid bbox after clamping: {bbox}"
        )

    return x1, y1, x2, y2


def expand_bbox(
    bbox: BBox,
    *,
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Add padding around a pixel-coordinate bounding box.

    Useful for HTR because some surrounding context often helps.
    """
    x1, y1, x2, y2 = bbox

    expanded = (
        x1 - padding,
        y1 - padding,
        x2 + padding,
        y2 + padding,
    )

    return clamp_bbox(
        expanded,
        width,
        height,
    )


def crop_bbox(
    source: str | Path,
    bbox: BBox,
    *,
    padding: int = 0,
) -> Image.Image:
    """
    Crop a pixel-coordinate region from the original image.

    The crop comes from the source image, not from the smaller
    VLM derivative.
    """
    image = load_image(source)

    bbox = clamp_bbox(
        bbox,
        image.width,
        image.height,
    )

    if padding:
        bbox = expand_bbox(
            bbox,
            padding=padding,
            width=image.width,
            height=image.height,
        )

    return image.crop(bbox)


def crop_normalized_bbox(
    source: str | Path,
    bbox: BBox,
    *,
    coordinate_max: int = 1000,
    padding: int = 0,
) -> Image.Image:
    """
    Crop a normalized VLM bounding box from the original image.

    This is the important bridge between:

        Qwen:
            [x1, y1, x2, y2] on a 0-1000 scale

    and:

        original archival TIFF:
            native pixel coordinates
    """
    info = image_info(source)

    pixel_bbox = normalized_bbox_to_pixels(
        bbox,
        info.width,
        info.height,
        coordinate_max=coordinate_max,
    )

    return crop_bbox(
        source,
        pixel_bbox,
        padding=padding,
    )


def save_crop(
    source: str | Path,
    bbox: BBox,
    destination: str | Path,
    *,
    normalized: bool = False,
    coordinate_max: int = 1000,
    padding: int = 0,
    format: str = "PNG",
) -> Path:
    """
    Crop a region and save it as a separate image.
    """
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if normalized:
        crop = crop_normalized_bbox(
            source,
            bbox,
            coordinate_max=coordinate_max,
            padding=padding,
        )
    else:
        crop = crop_bbox(
            source,
            bbox,
            padding=padding,
        )

    if crop.mode not in ("RGB", "RGBA", "L"):
        crop = crop.convert("RGB")

    crop.save(
        destination,
        format=format,
    )

    return destination