# src/visualization.py

from __future__ import annotations

from os import PathLike

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .segmentation import Region
from .images import normalized_bbox_to_pixels


def display_regions(
    image: str | PathLike[str] | Image.Image,
    regions: Sequence[Region],
    *,
    coordinate_max: int = 1000,
    show_confidence: bool = True,
    show_ids: bool = False,
    figsize: tuple[int, int] = (12, 16),
):
    """
    Display an image with region bounding boxes.

    Assumes Region.bbox uses normalized coordinates from
    0-coordinate_max.
    """

    if isinstance(image, (str, PathLike)):
        img = Image.open(image)
    else:
        img = image

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img)
    ax.axis("off")

    for region in regions:

        x1, y1, x2, y2 = normalized_bbox_to_pixels(
            region.bbox,
            img.width,
            img.height,
            coordinate_max=coordinate_max,
        )

        width = x2 - x1
        height = y2 - y1

        rect = Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        label_parts = [
            region.region_type
        ]

        if show_confidence and region.confidence is not None:
            label_parts.append(
                f"{region.confidence:.2f}"
            )

        if show_ids:
            label_parts.append(
                f"[{region.id}]"
            )

        label = " ".join(label_parts)

        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            fontsize=9,
            verticalalignment="bottom",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
                "pad": 2,
            },
        )

    plt.tight_layout()

    return fig, ax

def compare_regions(
    image: str | PathLike[str] | Image.Image,
    region_sets: Sequence[
        tuple[str, Sequence[Region]]
    ],
    *,
    coordinate_max: int = 1000,
    figsize: tuple[int, int] = (12, 16),
):
    """
    Overlay multiple segmentation systems.

    Example:

        compare_regions(
            image,
            [
                ("eScriptorium", escriptorium_regions),
                ("Qwen", qwen_regions),
            ],
        )
    """

    if isinstance(image, (str, PathLike)):
        img = Image.open(image)
    else:
        img = image

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img)
    ax.axis("off")

    for source_name, regions in region_sets:

        for region in regions:

            x1, y1, x2, y2 = normalized_bbox_to_pixels(
                region.bbox,
                img.width,
                img.height,
                coordinate_max=coordinate_max,
            )

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
            )

            ax.add_patch(rect)

            label = (
                f"{source_name}: "
                f"{region.region_type}"
            )

            ax.text(
                x1,
                max(0, y1 - 5),
                label,
                fontsize=8,
                verticalalignment="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )

    plt.tight_layout()

    return fig, ax
