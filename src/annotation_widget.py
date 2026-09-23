from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt

from IPython.display import display
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image

from .annotation import (
    HumanAnnotation,
    load_annotations,
    make_annotation,
    normalized_to_pixels_bbox,
    pixels_to_normalized_bbox,
    save_annotations,
)
from .segmentation import VALID_REGION_TYPES


class AnnotationWidget:
    def __init__(
        self,
        image_path: str | Path,
        annotation_path: str | Path,
    ):
        self.image_path = Path(image_path)
        self.annotation_path = Path(annotation_path)

        self.image = Image.open(self.image_path)

        self.annotations: list[HumanAnnotation] = []

        self.selected_index: int | None = None

        # Keep track of matplotlib objects separately.
        self.region_patches = []
        self.region_labels = []

        self._load_existing_annotations()
        self._build_controls()
        self._build_figure()
        self._refresh()

    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------

    def _load_existing_annotations(self):

        if not self.annotation_path.exists():
            return

        document = load_annotations(
            self.annotation_path
        )

        self.annotations = document.regions.copy()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_controls(self):

        self.region_type = widgets.Dropdown(
            options=sorted(VALID_REGION_TYPES),
            value="main_text",
            description="Region:",
            layout=widgets.Layout(
                width="350px"
            ),
        )

        self.undo_button = widgets.Button(
            description="Undo",
            icon="undo",
        )

        self.delete_button = widgets.Button(
            description="Delete selected",
            icon="trash",
            button_style="danger",
        )

        self.save_button = widgets.Button(
            description="Save",
            icon="save",
            button_style="success",
        )

        self.status = widgets.HTML()

        self.region_list = widgets.Select(
            options=[],
            description="Regions:",
            rows=8,
            layout=widgets.Layout(
                width="600px"
            ),
        )

        self.undo_button.on_click(
            self._undo
        )

        self.delete_button.on_click(
            self._delete_selected
        )

        self.save_button.on_click(
            self._save
        )

        self.region_list.observe(
            self._region_selected,
            names="value",
        )

        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.region_type,
                        self.undo_button,
                        self.delete_button,
                        self.save_button,
                    ]
                ),
                self.status,
            ]
        )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    def _build_figure(self):

        self.fig, self.ax = plt.subplots(
            figsize=(10, 15)
        )

        self.ax.imshow(self.image)

        self.ax.set_axis_off()

        self.selector = RectangleSelector(
            self.ax,
            self._rectangle_created,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )

    # ---------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------

    def _rectangle_created(
        self,
        eclick,
        erelease,
    ):

        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return

        x1 = min(
            eclick.xdata,
            erelease.xdata,
        )

        y1 = min(
            eclick.ydata,
            erelease.ydata,
        )

        x2 = max(
            eclick.xdata,
            erelease.xdata,
        )

        y2 = max(
            eclick.ydata,
            erelease.ydata,
        )

        bbox = pixels_to_normalized_bbox(
            (x1, y1, x2, y2),
            self.image.width,
            self.image.height,
        )

        annotation = make_annotation(
            self.region_type.value,
            bbox,
        )

        self.annotations.append(
            annotation
        )

        self.selected_index = (
            len(self.annotations) - 1
        )

        self._refresh()

    # ---------------------------------------------------------
    # Selection
    # ---------------------------------------------------------

    def _region_selected(
        self,
        change,
    ):

        value = change["new"]

        if value is None:
            return

        self.selected_index = value

        annotation = self.annotations[
            self.selected_index
        ]

        self.region_type.value = (
            annotation.region_type
        )

        self._draw_regions()

    # ---------------------------------------------------------
    # Undo / Delete
    # ---------------------------------------------------------

    def _undo(
        self,
        button,
    ):

        if not self.annotations:
            return

        self.annotations.pop()

        self.selected_index = None

        self._refresh()

    def _delete_selected(
        self,
        button,
    ):

        if self.selected_index is None:
            return

        if self.selected_index >= len(
            self.annotations
        ):
            return

        del self.annotations[
            self.selected_index
        ]

        self.selected_index = None

        self._refresh()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def _save(
        self,
        button,
    ):

        save_annotations(
            self.annotation_path,
            image=self.image_path.name,
            image_width=self.image.width,
            image_height=self.image.height,
            regions=self.annotations,
        )

        self.status.value = (
            f"<b>Saved:</b> "
            f"{len(self.annotations)} regions "
            f"→ {self.annotation_path}"
        )

    # ---------------------------------------------------------
    # Refresh
    # ---------------------------------------------------------

    def _refresh(self):

        self._refresh_list()
        self._draw_regions()

        self.status.value = (
            f"<b>{len(self.annotations)}</b> "
            f"regions annotated"
        )

    def _refresh_list(self):

        options = []

        for index, annotation in enumerate(
            self.annotations
        ):

            label = (
                f"{index + 1}. "
                f"{annotation.region_type}"
            )

            options.append(
                (
                    label,
                    index,
                )
            )

        self.region_list.options = options

    # ---------------------------------------------------------
    # Rendering annotations
    # ---------------------------------------------------------

    def _draw_regions(self):

        # Remove old patches.
        for patch in self.region_patches:
            patch.remove()

        for label in self.region_labels:
            label.remove()

        self.region_patches.clear()
        self.region_labels.clear()

        for index, annotation in enumerate(
            self.annotations
        ):

            x1, y1, x2, y2 = (
                normalized_to_pixels_bbox(
                    annotation.bbox,
                    self.image.width,
                    self.image.height,
                )
            )

            selected = (
                index == self.selected_index
            )

            linewidth = (
                4 if selected else 2
            )

            rectangle = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=linewidth,
            )

            self.ax.add_patch(
                rectangle
            )

            self.region_patches.append(
                rectangle
            )

            label = self.ax.text(
                x1,
                max(0, y1 - 5),
                (
                    f"{index + 1}. "
                    f"{annotation.region_type}"
                ),
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )

            self.region_labels.append(
                label
            )

        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    def show(self):

        display(
            self.controls,
            self.fig.canvas,
            self.region_list,
        )