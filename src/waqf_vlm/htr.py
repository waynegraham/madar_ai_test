"""Region-linked saved HTR comparisons; reviewed text alone enables error metrics."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import unicodedata

from src.alto import load_alto
from src.annotation import annotations_to_regions, load_annotations
from src.evaluation import HTRMetrics, htr_metrics, intersection_over_union
from src.images import crop_normalized_bbox
from src.segmentation import Region, alto_to_regions
from .experiment import system_name


@dataclass
class Reading:
    system: str
    text: str
    source: str
    script: str | None = None
    language: str | None = None
    metrics: HTRMetrics | None = None
    linkage: str = "Saved region identifier"

    @property
    def direction(self) -> str:
        # Inspect Unicode directionality without modifying the text itself.
        for character in self.text:
            direction = unicodedata.bidirectional(character)
            if direction in {"R", "AL"}:
                return "rtl"
            if direction == "L":
                return "ltr"
        return "auto"


@dataclass
class HTRComparison:
    region: Region
    region_source: str
    human: Reading | None = None
    readings: list[Reading] = field(default_factory=list)
    crop: str | None = None
    preview: str | None = None
    crop_width: int | None = None
    crop_height: int | None = None

    @property
    def identifier(self) -> str:
        return hashlib.sha256((self.region_source + self.region.id).encode()).hexdigest()[:16]


def _equivalent(region: Region, candidates: list[Region]) -> Region | None:
    by_id = [r for r in candidates if r.id == region.id]
    if len(by_id) == 1:
        return by_id[0]
    close = [r for r in candidates if intersection_over_union(replace(region, polygon=[]), replace(r, polygon=[])) >= .95]
    return close[0] if len(close) == 1 else None


def _alto_regions(path: Path) -> list[Region]:
    page = load_alto(path)
    regions = alto_to_regions(page)
    blocks = {b.id: b for b in page.blocks}
    for region in regions:
        lines = blocks[region.id].lines
        # Keep XML line order and blank lines; the existing parser handles String content.
        region.text = "\n".join(line.text for line in lines) if any(line.text for line in lines) else None
    return regions


def load_htr_comparisons(data: Path, page_id: str, issues: list[str]) -> list[HTRComparison]:
    groups: list[HTRComparison] = []
    annotation_path = data / 'ground-truth' / f'{page_id}.json'
    if annotation_path.is_file():
        try:
            document = load_annotations(annotation_path)
            if document.coordinate_system != 'normalized-1000':
                raise ValueError('Unsupported annotation coordinate system')
            raw = json.loads(annotation_path.read_text(encoding='utf-8'))
            items = {r['id']: r for r in raw.get('regions', [])}
            for region in annotations_to_regions(document):
                group = HTRComparison(region, annotation_path.relative_to(data).as_posix())
                item = items[region.id]
                # Optional extension: text is not ground truth merely because someone drew a box.
                if item.get('review_status') == 'human_corrected' and isinstance(item.get('transcription'), str):
                    group.human = Reading('Human transcription', item['transcription'], group.region_source,
                                          item.get('script'), item.get('language'), linkage='Explicit human_corrected transcription')
                groups.append(group)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            issues.append(f'{annotation_path.relative_to(data)}: HTR region references unavailable ({exc}).')

    corrected_path = data / 'ground-truth' / 'alto' / f'{page_id}.xml'
    if corrected_path.is_file():
        try:
            corrected = _alto_regions(corrected_path)
            for region in corrected:
                match = _equivalent(region, [g.region for g in groups])
                # Require a unique match in both directions to avoid reusing reviewed text.
                reciprocal = _equivalent(match, corrected) if match else None
                if match and reciprocal is region:
                    group = next(g for g in groups if g.region is match)
                else:
                    group = HTRComparison(region, corrected_path.relative_to(data).as_posix())
                    groups.append(group)
                if region.text is not None and group.human is None:
                    group.human = Reading('Human transcription', region.text, corrected_path.relative_to(data).as_posix(),
                                          linkage='Corrected ALTO: stable region ID or unique reciprocal bbox IoU ≥ 0.95')
        except Exception as exc:
            issues.append(f'{corrected_path.relative_to(data)}: corrected HTR reference unavailable ({exc}).')

    # Resolve only an explicit region ID or the current legacy per-region crop path.
    for path in sorted((data / 'results' / page_id).rglob('*.json')):
        is_htr = path.parent.name == 'htr'
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved.get('task') != 'htr':
                continue
            is_htr = True
            region_id = saved.get('region_id')
            if not region_id:
                parts = Path(saved.get('image') or '').parts
                if len(parts) >= 3 and parts[-3:-1] == ('crops', page_id):
                    region_id = Path(parts[-1]).stem
            matches = [g for g in groups if g.region.id == region_id]
            if len(matches) != 1:
                issues.append(f'{path.relative_to(data)}: HTR result has no unique human-defined region link; excluded from comparison.')
                continue
            response = saved.get('response')
            if not isinstance(response, dict) or not isinstance(response.get('transcription'), str):
                issues.append(f'{path.relative_to(data)}: saved HTR transcription unavailable.')
                continue
            matches[0].readings.append(Reading(system_name(str(saved.get('model') or path.parent.parent.name)),
                response['transcription'], path.relative_to(data).as_posix(), response.get('script'), response.get('language')))
        except (ValueError, TypeError, OSError) as exc:
            if is_htr:
                issues.append(f'{path.relative_to(data)}: HTR comparison could not load this result ({exc}).')

    # Machine ALTO is a prediction. Never substitute a broad page block for a human crop.
    for path in (data / 'alto' / f'{page_id}.xml', data / 'predictions' / 'escriptorium' / f'{page_id}.xml'):
        if not path.is_file():
            continue
        try:
            predictions = _alto_regions(path)
            for group in groups:
                match = _equivalent(group.region, predictions)
                reciprocal = _equivalent(match, [g.region for g in groups]) if match else None
                if match and reciprocal is group.region and match.text is not None:
                    group.readings.append(Reading('eScriptorium', match.text, path.relative_to(data).as_posix(),
                        linkage='Machine ALTO: stable region ID or unique reciprocal bbox IoU ≥ 0.95'))
        except Exception as exc:
            issues.append(f'{path.relative_to(data)}: eScriptorium region transcription unavailable ({exc}).')
    for group in groups:
        group.readings.sort(key=lambda r: ({'eScriptorium': 0, 'Qwen3-VL 8B': 1, 'Qwen3-VL 30B': 2}.get(r.system, 3), r.source))
        if group.human is not None:
            for reading in group.readings:
                reading.metrics = htr_metrics(group.human.text, reading.text)
    return [g for g in groups if g.readings]


def render_htr_crops(groups: list[HTRComparison], original: Path, output: Path) -> None:
    directory = output / 'assets'
    if directory.is_symlink():
        raise ValueError('Refusing symlink assets directory')
    directory.mkdir(exist_ok=True)
    for group in groups:
        stem = 'htr-' + hashlib.sha256(f'{original.name}:{group.region.id}:{group.region.bbox}'.encode()).hexdigest()[:24]
        crop_path, preview_path = directory / f'{stem}.png', directory / f'{stem}-preview.png'
        if crop_path.is_symlink() or preview_path.is_symlink():
            raise ValueError('Refusing symlink crop destination')
        crop = crop_normalized_bbox(original, group.region.bbox, padding=30)
        if crop.mode not in ('RGB', 'RGBA', 'L'):
            crop = crop.convert('RGB')
        crop.save(crop_path)
        group.crop_width, group.crop_height = crop.size
        crop.thumbnail((1200, 1200))
        crop.save(preview_path)
        group.crop = crop_path.relative_to(output).as_posix()
        group.preview = preview_path.relative_to(output).as_posix()
