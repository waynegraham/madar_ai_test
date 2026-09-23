"""Build an offline report from saved artifacts. No inference clients are imported."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.alto import load_alto
from src.annotation import load_annotations
from src.images import create_vlm_derivative, image_info
from src.segmentation import vlm_json_to_regions


LABELS = {
    "main_text": "Main text",
    "marginal_text": "Marginal text",
    "stamp_or_seal": "Stamp or seal",
    "handwritten_annotation": "Handwritten annotation",
    "archival_mark": "Archival mark",
    "illustration": "Illustration",
    "unknown": "Unclassified region",
}


@dataclass
class Artifact:
    path: str
    kind: str
    origin: str
    review: str
    role: str
    counts: dict[str, int] = field(default_factory=dict)
    transcriptions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ManuscriptImage:
    source: str
    preview: str
    thumbnail: str
    width: int
    height: int
    alt: str


@dataclass
class Manuscript:
    """One image/page, not necessarily an entire codex."""
    id: str
    images: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    figures: list[ManuscriptImage] = field(default_factory=list)

    @property
    def filename(self) -> str:
        # IDs remain display text; filenames never contain user-supplied paths.
        return hashlib.sha256(self.id.encode()).hexdigest()[:20] + ".html"

    @property
    def has_human_layout(self) -> bool:
        return any(a.role == "reference" for a in self.artifacts)

    @property
    def has_reviewed_text(self) -> bool:
        return any(a.role == "reference" and a.transcriptions for a in self.artifacts)


@dataclass
class ReportData:
    pages: list[Manuscript]
    issues: list[str]


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(LABELS.get(v, v or "Unclassified region") for v in values).items()))


def load_report_data(data_dir: Path) -> ReportData:
    """Read legacy results and explicitly separated prediction/reference ALTO.

    Human JSON records document manual layout work, but do not establish a
    subsequent independent review. Only ground-truth/alto is designated as
    human-corrected ALTO. The XML format itself never determines its role.
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    pages: dict[str, Manuscript] = {}
    issues: list[str] = []

    def page(identifier: str) -> Manuscript:
        return pages.setdefault(identifier, Manuscript(identifier))

    def relative(path: Path) -> str:
        return path.relative_to(data_dir).as_posix()

    for path in sorted((data_dir / "images").glob("*")):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            page(path.stem).images.append(relative(path))

    alto_locations = [
        (data_dir / "alto", False),
        (data_dir / "predictions" / "escriptorium", False),
        (data_dir / "ground-truth" / "alto", True),
    ]
    for directory, reviewed in alto_locations:
        for path in sorted(directory.glob("*.xml")):
            target = page(path.stem)
            try:
                alto = load_alto(path)
                artifact = Artifact(
                    relative(path), "eScriptorium ALTO", "eScriptorium",
                    "Human reviewed" if reviewed else "Not yet reviewed",
                    "reference" if reviewed else "prediction",
                    counts=_counts(b.block_type for b in alto.blocks),
                    transcriptions=[line.text for b in alto.blocks for line in b.lines if line.text],
                    details={"Image dimensions": f"{alto.width} × {alto.height} pixels",
                             "Source image": alto.image_filename or "Unavailable",
                             "Text blocks": len(alto.blocks),
                             "Text lines": sum(len(b.lines) for b in alto.blocks)},
                )
                if reviewed:
                    artifact.notes.append("Designated human-corrected by its location in ground-truth/alto. Reviewer and correction scope are not recorded.")
                else:
                    artifact.notes.append("Machine suggested. This ALTO export is a prediction, not ground truth.")
                without_box = sum(b.bbox is None for b in alto.blocks)
                if without_box:
                    artifact.notes.append(f"{without_box} block(s) lack a bounding box. Their text is retained here.")
                target.artifacts.append(artifact)
            except Exception as exc:
                issues.append(f"{relative(path)}: could not load ALTO ({exc}).")

    for path in sorted((data_dir / "ground-truth").glob("*.json")):
        target = page(path.stem)
        try:
            document = load_annotations(path)
            if document.coordinate_system != "normalized-1000":
                raise ValueError(f"Unsupported coordinates: {document.coordinate_system}")
            target.artifacts.append(Artifact(
                relative(path), "Human layout annotations", "Human annotated",
                "Not yet reviewed", "reference",
                counts=_counts(r.region_type for r in document.regions),
                details={"Image dimensions": f"{document.image_width} × {document.image_height} pixels",
                         "Schema version": document.schema_version,
                         "Regions": len(document.regions)},
                notes=["Manually drawn regions, stored as layout references. Independent review is not recorded; page completeness is not recorded.",
                       "No reference transcription, script, or language is stored in this annotation format."],
            ))
        except Exception as exc:
            issues.append(f"{relative(path)}: could not load human annotations ({exc}).")

    results = data_dir / "results"
    for path in sorted(results.rglob("*.json")):
        parts = path.relative_to(results).parts
        if len(parts) < 3:
            issues.append(f"{relative(path)}: expected results/<page>/<model>/…/*.json.")
            continue
        target = page(parts[0])
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            response = saved.get("response")
            if not isinstance(response, dict):
                raise ValueError("Saved response is unavailable or is not a JSON object")
            task = saved.get("task") or "Unspecified task"
            artifact = Artifact(
                relative(path), {"htr": "Transcription, script and language", "segmentation": "Page layout"}.get(task, task),
                str(saved.get("model") or parts[1]), "Not yet reviewed", "prediction",
                details={"Task": task, "Prompt version": saved.get("prompt_version") or "Unavailable",
                         "Input image": saved.get("image") or "Unavailable",
                         "Recorded duration (seconds)": saved.get("elapsed_seconds", "Unavailable")},
                notes=["Machine suggested. These suggestions have not been verified against a human-reviewed reference."],
            )
            if task == "segmentation":
                raw = response.get("regions")
                if not isinstance(raw, list):
                    raise ValueError("Segmentation response has no region list")
                regions = vlm_json_to_regions(response, source=artifact.origin)
                artifact.counts = _counts(r.region_type for r in regions)
                artifact.details["Regions"] = len(regions)
                if len(regions) != len(raw):
                    artifact.notes.append(f"{len(raw) - len(regions)} malformed region(s) were excluded by the region parser.")
            elif task == "htr":
                transcription = response.get("transcription")
                if isinstance(transcription, str) and transcription:
                    artifact.transcriptions = [transcription]
                artifact.details.update({"Suggested script": response.get("script") or "Unavailable",
                                         "Suggested language": response.get("language") or "Unavailable"})
                artifact.notes.append("Script and language are shown exactly as supplied by the model; a script name and a calligraphic style may not be equivalent.")
            else:
                artifact.notes.append("This task has no report adapter yet; no results have been inferred.")
            target.artifacts.append(artifact)
        except Exception as exc:
            issues.append(f"{relative(path)}: could not load saved experiment ({exc}).")
    return ReportData([pages[key] for key in sorted(pages)], issues)


def _asset_root() -> Path:
    repository_assets = Path(__file__).resolve().parents[2] / "reports"
    if (repository_assets / "templates" / "base.html").is_file():
        return repository_assets
    packaged = Path(__file__).parent / "report_assets"
    if packaged.is_dir():
        return packaged
    return repository_assets


def prepare_figures(report: ReportData, data_dir: Path, output_dir: Path) -> None:
    """Make whole-page display copies; never crop, annotate, or change originals."""
    destination_dir = output_dir / "images"
    if destination_dir.is_symlink():
        raise ValueError(f"Refusing symlink output: {destination_dir}")
    destination_dir.mkdir(exist_ok=True)
    for manuscript in report.pages:
        for source in manuscript.images:
            image_id = hashlib.sha256(source.encode()).hexdigest()[:20]
            preview = f"images/{image_id}.png"
            thumbnail = f"images/{image_id}-small.png"
            for relative in (preview, thumbnail):
                if (output_dir / relative).is_symlink():
                    raise ValueError(f"Refusing symlink output: {relative}")
            try:
                create_vlm_derivative(data_dir / source, output_dir / preview, max_dimension=1800)
                create_vlm_derivative(data_dir / source, output_dir / thumbnail, max_dimension=640)
                info = image_info(output_dir / preview)
                manuscript.figures.append(ManuscriptImage(
                    source, preview, thumbnail, info.width, info.height,
                    f"Full manuscript page, image identifier {manuscript.id}. An unannotated reproduction of the source image.",
                ))
            except (OSError, ValueError) as exc:
                report.issues.append(f"{source}: display image unavailable ({exc}).")


def build_report(data_dir: Path, output_dir: Path, *, assets: Path | None = None) -> ReportData:
    """Render local HTML/CSS. Output must be separate from all source data."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    assets = Path(assets).resolve() if assets else _asset_root().resolve()
    source_root = Path(__file__).resolve().parents[1]
    for protected in (data_dir, assets / "templates", assets / "static", source_root):
        if output_dir == protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError(f"Output overlaps a protected input directory: {protected}")
    marker = output_dir / ".waqf-report.json"
    if output_dir.exists():
        existing = {p.name for p in output_dir.iterdir()} - {".gitkeep", ".gitignore"}
        if existing and not marker.is_file():
            raise ValueError("Output directory is not empty and is not a previously generated report.")
    report = load_report_data(data_dir)
    environment = Environment(loader=FileSystemLoader(assets / "templates"),
                              autoescape=select_autoescape(["html", "xml"]))
    index_template = environment.get_template("index.html")
    page_template = environment.get_template("manuscript.html")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manuscripts").mkdir(exist_ok=True)
    # Refuse symlink destinations rather than following them outside output.
    for destination in [output_dir / "static", output_dir / "manuscripts", marker,
                        output_dir / "index.html"]:
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink output: {destination}")
    static_destination = output_dir / "static"
    for source in sorted((assets / "static").rglob("*")):
        if source.is_file() and source.name != ".gitkeep":
            destination = static_destination / source.relative_to(assets / "static")
            if any(parent.is_symlink() for parent in [destination, *destination.parents] if parent != output_dir):
                raise ValueError(f"Refusing symlink output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    prepare_figures(report, data_dir, output_dir)
    filenames = []
    for manuscript in report.pages:
        destination = output_dir / "manuscripts" / manuscript.filename
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink output: {destination}")
        destination.write_text(page_template.render(page=manuscript, asset_prefix="../", title=manuscript.id), encoding="utf-8")
        filenames.append(manuscript.filename)
    (output_dir / "index.html").write_text(index_template.render(report=report, asset_prefix="", title="Manuscript research report"), encoding="utf-8")
    # Only remove stale detail pages explicitly owned by a previous build.
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        for filename in previous.get("pages", []):
            if isinstance(filename, str) and len(filename) == 25 and filename.endswith(".html") and all(c in "0123456789abcdef" for c in filename[:-5]):
                stale = output_dir / "manuscripts" / filename
                if filename not in filenames and stale.is_file() and not stale.is_symlink():
                    stale.unlink()
    marker.write_text(json.dumps({"pages": filenames}, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a static manuscript research report from saved files; never runs models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Render a static HTML site")
    build.add_argument("--data", type=Path, default=Path("data"))
    build.add_argument("--output", type=Path, default=Path("reports/generated"))
    args = parser.parse_args(argv)
    try:
        report = build_report(args.data, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot build report: {exc}\n")
    print(f"Built {len(report.pages)} manuscript page(s): {args.output / 'index.html'}")
    for issue in report.issues:
        print(f"Data issue: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
