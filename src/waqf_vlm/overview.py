"""Corpus coverage and explicitly scoped evaluation from available records only."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

from src.evaluation import metrics_by_type


def corpus_overview(report, data_dir: Path, *, min_reviewed_pages: int = 5) -> dict:
    if min_reviewed_pages < 1:
        raise ValueError('min_reviewed_pages must be positive')
    page_ids = {p.id for p in report.pages}
    manifest = {}
    manifest_path = data_dir / 'corpus.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict) or not isinstance(manifest.get('pages', {}), dict):
                raise ValueError('Expected an object with a pages mapping')
        except (OSError, ValueError) as exc:
            report.issues.append(f'corpus.json: overview metadata unavailable ({exc}).')
            manifest = {}
    mappings = manifest.get('pages', {})
    mapped = {key: value['manuscript_id'] for key, value in mappings.items()
              if key in page_ids and isinstance(value, dict) and isinstance(value.get('manuscript_id'), str) and value['manuscript_id'].strip()}
    complete_layout = {key for key, value in mappings.items() if key in page_ids and isinstance(value, dict)
                       and value.get('layout_review_complete') is True}
    coverage = {'eScriptorium': set(), 'Qwen3-VL 8B': set(), 'Qwen3-VL 30B': set()}
    reviewed = set()
    all_counts = Counter()
    per_system = defaultdict(Counter)
    run_pages = defaultdict(set)
    run_counts = Counter()
    prediction_records = 0
    htr_available = set()
    htr_evaluated = set()
    htr_pairs = 0
    htr_pages = set()
    htr_runs = defaultdict(Counter)
    paired = defaultdict(list)
    excluded_ambiguous = 0
    for page in report.pages:
        if page.corrected_alto or page.has_reviewed_text:
            reviewed.add(page.id)
        annotation_path = data_dir / 'ground-truth' / f'{page.id}.json'
        if annotation_path.is_file():
            try:
                annotations = json.loads(annotation_path.read_text(encoding='utf-8'))
                if any(isinstance(item, dict) and item.get('review_status') == 'human_corrected'
                       and isinstance(item.get('transcription'), str) for item in annotations.get('regions', [])):
                    reviewed.add(page.id)
            except (OSError, ValueError, AttributeError, TypeError):
                pass  # The report loader records unreadable annotation files.
        for artifact in page.artifacts:
            if artifact.role != 'prediction':
                continue
            if artifact.origin == 'eScriptorium': coverage['eScriptorium'].add(page.id)
            elif artifact.origin == 'qwen3-vl-8b-instruct-mlx': coverage['Qwen3-VL 8B'].add(page.id)
            elif artifact.origin == 'qwen3-vl-30b-a3b-instruct-mlx': coverage['Qwen3-VL 30B'].add(page.id)
        local_runs = defaultdict(list)
        for system in page.predictions:
            artifact = next((a for a in page.artifacts if a.path == system.source), None)
            prompt = artifact.details.get('Prompt version', 'unrecorded') if artifact else 'unrecorded'
            run = f'{system.name} · {prompt}'
            counts = Counter(r.region_type for r in system.regions)
            all_counts.update(counts)
            per_system[run].update(counts)
            run_pages[run].add(page.id)
            run_counts[run] += 1
            prediction_records += 1
            local_runs[run].append(system)
        for group in page.htr_comparisons:
            key = (page.id, group.region_source, group.region.id)
            if group.readings:
                htr_available.add(key)
            for reading in group.readings:
                htr_runs[reading.system]['readings'] += 1
            for failure in group.failures:
                htr_runs[failure['system']]['failures'] += 1
            evaluated = [r for r in group.readings if r.metrics is not None]
            if evaluated:
                htr_evaluated.add(key)
                htr_pages.add(page.id)
                htr_pairs += len(evaluated)
        # Corrected ALTO alone does not document that all layout regions were reviewed.
        if page.id in complete_layout and len(page.corrected_alto) == 1:
            reference = page.corrected_alto[0]
            if reference.skipped_blocks:
                continue  # Do not call an incomplete visual conversion complete reference coverage.
            for run, systems in local_runs.items():
                if len(systems) == 1:
                    paired[run].append((page.id, reference, systems[0]))
                else:
                    excluded_ambiguous += 1
    evaluations = []
    eligible = []
    for run, samples in sorted(paired.items()):
        eligible.append({'run': run, 'pages': len(samples)})
        if len(samples) < min_reviewed_pages:
            continue
        totals = defaultdict(lambda: Counter(tp=0, fp=0, fn=0, truth_pages=0, predicted_pages=0))
        for page_id, reference, prediction in samples:
            truth = [replace(r, polygon=[]) for r in reference.regions]
            hypothesis = [replace(r, polygon=[]) for r in prediction.regions]
            for kind, metric in metrics_by_type(truth, hypothesis, iou_threshold=.5).items():
                row = totals[kind]
                row.update(tp=metric.true_positives, fp=metric.false_positives, fn=metric.false_negatives,
                           truth_pages=int(any(r.region_type == kind for r in truth)),
                           predicted_pages=int(any(r.region_type == kind for r in hypothesis)))
        for kind, values in sorted(totals.items()):
            tp, fp, fn = values['tp'], values['fp'], values['fn']
            evaluations.append({'run': run, 'kind': kind, 'pages': len(samples),
                'reference_regions': tp+fn, 'predicted_regions': tp+fp, **values,
                'precision': tp/(tp+fp) if tp+fp else None,
                'recall': tp/(tp+fn) if tp+fn else None,
                'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    return {
        'page_count': len(page_ids), 'manuscript_count': len(set(mapped.values())) if mapped else None,
        'mapped_pages': len(mapped), 'mapping_complete': len(mapped) == len(page_ids) and bool(page_ids),
        'coverage': {name: len(ids) for name, ids in coverage.items()}, 'reviewed_pages': len(reviewed),
        'layout_declared_pages': len(complete_layout),
        'total_regions': sum(all_counts.values()), 'region_counts': dict(sorted(all_counts.items())),
        'runs': [{'name': name, 'pages': len(run_pages[name]), 'records': run_counts[name],
                  'counts': dict(per_system[name]), 'regions': sum(per_system[name].values())} for name in sorted(per_system)],
        'prediction_records': prediction_records, 'segmented_pages': len(set().union(*run_pages.values())) if run_pages else 0,
        'htr_available': len(htr_available), 'htr_evaluated': len(htr_evaluated),
        'htr_evaluated_pages': len(htr_pages), 'htr_pairs': htr_pairs,
        'htr_runs': dict(sorted(htr_runs.items())),
        'minimum_pages': min_reviewed_pages, 'evaluations': evaluations, 'eligible': eligible,
        'excluded_ambiguous': excluded_ambiguous,
    }
