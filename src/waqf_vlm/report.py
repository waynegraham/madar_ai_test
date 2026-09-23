"""Build an offline report from saved artifacts. No inference clients are imported."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import re
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateError, select_autoescape

from src.alto import load_alto
from src.annotation import load_annotations
from src.images import create_vlm_derivative, image_info
from src.segmentation import alto_to_regions, vlm_json_to_regions
from .experiment import SegmentationSystem, Disagreement, find_disagreements, select_disagreements, illustrate_experiment, system_name
from .htr import HTRComparison, load_htr_comparisons, render_htr_crops
from .overview import corpus_overview


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
    segmentations: list[SegmentationSystem] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    all_disagreements: list[Disagreement] = field(default_factory=list)
    htr_comparisons: list[HTRComparison] = field(default_factory=list)

    @property
    def predictions(self) -> list[SegmentationSystem]:
        return [system for system in self.segmentations if not system.reviewed]

    @property
    def corrected_alto(self) -> list[SegmentationSystem]:
        return [system for system in self.segmentations if system.reviewed]

    @property
    def filename(self) -> str:
        # IDs remain display text; filenames never contain user-supplied paths.
        return hashlib.sha256(self.id.encode()).hexdigest()[:20] + ".html"

    @property
    def has_human_layout(self) -> bool:
        return any(a.role == "reference" for a in self.artifacts)

    @property
    def has_reviewed_text(self) -> bool:
        return any(a.role == "reference" and a.transcriptions for a in self.artifacts) or any(g.human is not None for g in self.htr_comparisons)


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
                target.segmentations.append(SegmentationSystem(
                    "Human-corrected eScriptorium" if reviewed else "eScriptorium",
                    relative(path), alto_to_regions(alto), reviewed, without_box,
                ))
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
            if task == "segmentation":
                target.segmentations.append(SegmentationSystem(system_name(artifact.origin), relative(path), regions))
            artifact.details.update({
                "Model name": saved.get("model") or "Unavailable",
                "Quantization": saved.get("quantization") or "Unavailable — not recorded",
                "Temperature": saved.get("temperature", "Unavailable — not recorded"),
                "Maximum tokens": saved.get("max_tokens", "Unavailable — not recorded"),
                "Input dimensions": "Unavailable",
            })
            # Legacy paths are notebook-relative; never resolve arbitrary paths outside data.
            recorded_image = saved.get("image")
            if isinstance(recorded_image, str):
                parts_image = Path(recorded_image).parts
                if len(parts_image) >= 3 and parts_image[:2] == ("..", "data"):
                    image_path = data_dir.joinpath(*parts_image[2:]).resolve()
                else:
                    image_path = (data_dir / recorded_image).resolve()
                if image_path.is_relative_to(data_dir) and image_path.is_file():
                    try:
                        info = image_info(image_path)
                        artifact.details["Input dimensions"] = f"{info.width} × {info.height} pixels (from the current saved input file)"
                    except (OSError, ValueError):
                        pass
        except Exception as exc:
            issues.append(f"{relative(path)}: could not load saved experiment ({exc}).")
    for manuscript in pages.values():
        manuscript.images.sort(key=lambda name: (Path(name).suffix.lower() not in {".tif", ".tiff"}, name))
        manuscript.segmentations.sort(key=lambda s: (s.reviewed, {"eScriptorium": 0, "Qwen3-VL 8B": 1, "Qwen3-VL 30B": 2}.get(s.name, 3), s.source))
        manuscript.all_disagreements = find_disagreements(manuscript.segmentations, LABELS, limit=None)
        manuscript.disagreements = select_disagreements(manuscript.all_disagreements)
        manuscript.htr_comparisons = load_htr_comparisons(data_dir, manuscript.id, issues)
        for artifact in manuscript.artifacts:
            if artifact.kind == "Human layout annotations" and any(g.human is not None and g.human.source == artifact.path for g in manuscript.htr_comparisons):
                artifact.notes = [note for note in artifact.notes if not note.startswith("No reference transcription")]
                artifact.notes.append("Explicitly human-corrected text is available for some regions in the HTR comparison; this does not establish whole-page review.")
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


def _render_report(report: ReportData, selected: list[Manuscript], data_dir: Path,
                   destination: Path, assets: Path, min_reviewed_pages: int) -> dict[str, list[str]]:
    environment = Environment(loader=FileSystemLoader(assets / "templates"),
                              autoescape=select_autoescape(["html", "xml"]))
    templates = {name: environment.get_template(name + ".html")
                 for name in ("index", "manuscript", "method", "overview")}
    shutil.copytree(assets / "static", destination / "static", ignore=shutil.ignore_patterns('.gitkeep'))
    (destination / "manuscripts").mkdir()
    page_files = {}
    for page in selected:
        prepare_figures(ReportData([page], report.issues), data_dir, destination)
        if not page.images:
            report.issues.append(f"{page.id}: original image unavailable; image figures and crops are omitted.")
        if page.figures:
            original = data_dir / page.figures[0].source
            for action in (
                lambda: render_htr_crops(page.htr_comparisons, original, destination),
                lambda: illustrate_experiment(page.segmentations, page.all_disagreements, original, destination),
            ):
                try:
                    action()
                except (OSError, ValueError) as exc:
                    report.issues.append(f"{page.id}: some illustrations are unavailable ({exc}).")
        (destination / "manuscripts" / page.filename).write_text(templates['manuscript'].render(
            page=page, labels=LABELS, asset_prefix="../", title=f"Manuscript {page.id}"), encoding="utf-8")
        owned = {f'manuscripts/{page.filename}'}
        for figure in page.figures:
            owned.update((figure.preview, figure.thumbnail))
        owned.update(s.overlay for s in page.segmentations if s.overlay)
        for item in page.all_disagreements:
            owned.update(value for value in (item.crop, item.crop_preview) if value)
        for item in page.htr_comparisons:
            owned.update(value for value in (item.crop, item.preview) if value)
        page_files[page.id] = sorted(owned)
    overview = corpus_overview(report, data_dir, min_reviewed_pages=min_reviewed_pages)
    for name, context in (
        ('index', {'report': report, 'title': 'Manuscript research report'}),
        ('method', {'title': 'How this experiment works'}),
        ('overview', {'overview': overview, 'report': report, 'labels': LABELS, 'title': 'Corpus overview'}),
    ):
        (destination / f'{name}.html').write_text(templates[name].render(
            asset_prefix='', current_page=name, **context), encoding='utf-8')
    return page_files


def _safe_relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and '..' not in path.parts and path.as_posix() == value


def build_report(data_dir: Path, output_dir: Path, *, assets: Path | None = None,
                 min_reviewed_pages: int = 5, manuscript_id: str | None = None) -> ReportData:
    """Stage an offline build, then publish only generated files. Never write inputs."""
    if min_reviewed_pages < 1:
        raise ValueError('min_reviewed_pages must be positive')
    raw_output = Path(output_dir).absolute()
    if raw_output.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    data_dir, output_dir = Path(data_dir).resolve(), raw_output.resolve()
    assets = Path(assets).resolve() if assets else _asset_root().resolve()
    for protected in (data_dir, assets / 'templates', assets / 'static', Path(__file__).resolve().parents[1]):
        if output_dir == protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError(f'Output overlaps a protected input directory: {protected}')
    marker = output_dir / '.waqf-report.json'
    previous = {}
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError('Output must be a directory')
        if any(p.is_symlink() for p in output_dir.rglob('*')):
            raise ValueError('Output contains symlinks; refusing to overwrite or follow them')
        existing = {p.name for p in output_dir.iterdir()} - {'.gitignore', '.gitkeep'}
        if existing and not marker.is_file():
            raise ValueError('Output directory is not empty and is not a previously generated report.')
        if marker.is_file():
            try:
                previous = json.loads(marker.read_text(encoding='utf-8'))
                if not isinstance(previous, dict):
                    raise ValueError('expected an object')
                if previous.get('version') not in (None, 2):
                    raise ValueError('unsupported manifest version')
                if not isinstance(previous.get('files', []), list):
                    raise ValueError('files must be a list')
                if any(not isinstance(v, str) or not _safe_relative(v) for v in previous.get('files', [])):
                    raise ValueError('unsafe output file path')
                if previous.get('version') == 2:
                    if not isinstance(previous.get('page_files'), dict) or not isinstance(previous.get('figures'), dict):
                        raise ValueError('page ownership and figures must be mappings')
                    for values in previous['page_files'].values():
                        if not isinstance(values, list) or any(v not in previous['files'] for v in values):
                            raise ValueError('page ownership must reference generated files')
                    for figures in previous['figures'].values():
                        if not isinstance(figures, list):
                            raise ValueError('page figures must be lists')
                        for figure in figures:
                            if not isinstance(figure, dict) or any(figure.get(key) not in previous['files'] for key in ('preview','thumbnail')):
                                raise ValueError('figure paths must reference generated files')
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f'Invalid report ownership manifest {marker}: {exc}') from exc
    report = load_report_data(data_dir)
    all_pages = report.pages
    chosen = [p for p in all_pages if p.id == manuscript_id] if manuscript_id else all_pages
    if manuscript_id and not chosen:
        raise ValueError(f'No manuscript page found for {manuscript_id!r}; check --data and the page identifier')
    if manuscript_id and previous and previous.get('version') != 2:
        raise ValueError('Run a full build once to upgrade this report before using --manuscript')
    retained_ids = set(previous.get('page_files', {})) if manuscript_id else set()
    report.pages = [p for p in all_pages if not manuscript_id or p.id == manuscript_id or p.id in retained_ids]
    # Restore only visual metadata for retained pages; their HTML/assets are not regenerated.
    if manuscript_id:
        for page in report.pages:
            if page.id != manuscript_id:
                for figure in previous.get('figures', {}).get(page.id, []):
                    page.figures.append(ManuscriptImage(**figure))
    with tempfile.TemporaryDirectory(prefix='waqf-report-') as temporary:
        stage = Path(temporary)
        per_page = _render_report(report, chosen, data_dir, stage, assets, min_reviewed_pages)
        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file())
        previous_files = set(previous.get('files', []))
        if previous and previous.get('version') is None:
            # Migrate only known generated paths from the original report builder.
            previous_files.update(f'manuscripts/{name}' for name in previous.get('pages', [])
                                  if isinstance(name, str) and re.fullmatch(r'[0-9a-f]{20}\.html', name))
            for directory in ('images', 'assets'):
                previous_files.update(p.relative_to(output_dir).as_posix() for p in (output_dir / directory).glob('*.png')
                    if re.fullmatch(r'(?:segmentation-|htr-)?[0-9a-f]{20,24}(?:-small|-preview)?\.png', p.name))
            previous_files.update(name for name in ('index.html','method.html','overview.html','static/css/site.css') if (output_dir/name).is_file())
        retained_files = set()
        retained_pages = {}
        if manuscript_id:
            for key, values in previous.get('page_files', {}).items():
                if key != manuscript_id and key in {p.id for p in report.pages}:
                    if any(not isinstance(v, str) or not _safe_relative(v) for v in values):
                        raise ValueError('Invalid retained page ownership paths')
                    if any(not (output_dir / value).is_file() for value in values):
                        raise ValueError(f'Retained output for {key} is missing; run a full build to restore it')
                    retained_files.update(values)
                    retained_pages[key] = values
        generated = set(files) | retained_files
        for relative in generated:
            destination = output_dir / relative
            if destination.exists() and relative not in previous_files:
                raise ValueError(f'Refusing to overwrite an unowned file: {destination}')
        output_dir.mkdir(parents=True, exist_ok=True)
        # Byte-identical files are not touched; atomic replacement protects hard-linked inputs.
        for relative in files:
            destination = output_dir / relative
            content = (stage / relative).read_bytes()
            if destination.is_file() and destination.read_bytes() == content:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
                staged.write(content)
                temp_path = Path(staged.name)
            temp_path.replace(destination)
        for relative in sorted(previous_files - generated):
            stale = output_dir / relative
            if stale.is_file():
                stale.unlink()
        manifest = {'version': 2, 'files': sorted(generated), 'page_files': {**retained_pages, **per_page},
                    'figures': {p.id: [asdict(f) for f in p.figures] for p in report.pages}}
        content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        if not marker.is_file() or marker.read_text(encoding='utf-8') != content:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_dir, delete=False) as staged:
                staged.write(content)
                temp_path = Path(staged.name)
            temp_path.replace(marker)
    return report


class ReportHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(403, 'Directory listing is disabled')
        return None

    def send_head(self):
        target = Path(self.translate_path(self.path))
        root = Path(self.directory).resolve()
        if not target.resolve().is_relative_to(root) or any(p.is_symlink() for p in (target, *target.parents) if p != root):
            self.send_error(403, 'Path is outside the generated report')
            return None
        return super().send_head()


def serve_report(output_dir: Path, *, port: int = 8000) -> None:
    root = Path(output_dir).resolve()
    if not (root / 'index.html').is_file() or not (root / '.waqf-report.json').is_file():
        raise ValueError('Generated report not found; run waqf-report build first')
    if not 0 <= port <= 65535:
        raise ValueError('Port must be between 0 and 65535')
    with ThreadingHTTPServer(('127.0.0.1', port), partial(ReportHandler, directory=str(root))) as server:
        print(f'Serving {root} at http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a static manuscript research report from saved files; never runs models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Render a static HTML site")
    build.add_argument("--data", type=Path, default=Path("data"))
    build.add_argument("--output", type=Path, default=Path("reports/generated"))
    build.add_argument("--min-reviewed-pages", type=int, default=5, help="Minimum paired, completely reviewed layout pages per run for aggregate metrics (default: 5; not a statistical guarantee)")
    build.add_argument("--manuscript", help="Rebuild one exact page identifier; refresh shared indexes without rerendering other pages")
    serve = subparsers.add_parser("serve", help="Serve the generated report on localhost; never builds or runs models")
    serve.add_argument("--output", type=Path, default=Path("reports/generated"))
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == 'serve':
            serve_report(args.output, port=args.port)
            return 0
        report = build_report(args.data, args.output, min_reviewed_pages=args.min_reviewed_pages, manuscript_id=args.manuscript)
    except (OSError, ValueError, TypeError, KeyError, TemplateError) as exc:
        parser.exit(1, f"Cannot {args.command} report: {exc}\n")
    print(f"Built {1 if args.manuscript else len(report.pages)} manuscript page(s): {args.output / 'index.html'}")
    for issue in report.issues:
        print(f"Data issue: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
