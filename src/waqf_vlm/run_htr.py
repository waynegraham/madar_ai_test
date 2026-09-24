"""Resumable, identical-region Qwen HTR runs, independent of reviewed text."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.annotation import annotations_to_regions, load_annotations
from src.images import save_crop
from src.lmstudio import analyze_image, extract_json, get_client, load_prompt
from src.segmentation import Region
from .htr import _alto_regions

MODELS = {'qwen3-vl-8b': 'qwen3-vl-8b-instruct-mlx',
          'qwen3-vl-30b': 'qwen3-vl-30b-a3b-instruct-mlx'}


def prepare(data: Path) -> list[dict]:
    jobs = []
    images = sorted(p for p in (data / 'images').iterdir()
                    if p.suffix.lower() in {'.tif', '.tiff', '.jpg', '.jpeg', '.png'})
    for image in images:
        manifest = data / 'htr-inputs' / f'{image.stem}.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
        else:
            annotation = data / 'ground-truth' / f'{image.stem}.json'
            alto = data / 'alto' / f'{image.stem}.xml'
            if annotation.exists():
                regions = annotations_to_regions(load_annotations(annotation))
                source = annotation.relative_to(data).as_posix()
            elif alto.exists():
                regions = _alto_regions(alto)
                source = alto.relative_to(data).as_posix() + ' (machine layout; unreviewed)'
            else:
                regions = [Region('whole-page', 'main_text', (0, 0, 1000, 1000))]
                source = 'Whole image (no region layout available)'
            # Include seals and unknown regions, which can also contain writing.
            regions = [r for r in regions if r.region_type != 'illustration']
            saved = {'image': image.name, 'coordinate_system': 'normalized-1000',
                     'region_source': source, 'padding': 30,
                     'regions': [{'id': r.id, 'type': r.region_type, 'bbox': list(r.bbox)} for r in regions]}
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2))
        for region in saved['regions']:
            crop = data / 'crops' / image.stem / f"{region['id']}.png"
            save_crop(image, tuple(region['bbox']), crop, normalized=True, padding=saved['padding'])
            jobs.append({'page_id': image.stem, 'region_id': region['id'],
                         'region_bbox': region['bbox'], 'crop': crop,
                         'region_source': manifest.relative_to(data).as_posix(),
                         'image_sha256': hashlib.sha256(crop.read_bytes()).hexdigest()})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--prompt', type=Path, default=Path('prompts/htr-v1.txt'))
    parser.add_argument('--base-url', default='http://localhost:1234/v1')
    parser.add_argument('--model', choices=list(MODELS), action='append')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    jobs = prepare(args.data)
    print(f'Prepared {len(jobs)} regions across {len(set(j["page_id"] for j in jobs))} images.', flush=True)
    if args.prepare_only:
        return
    prompt = load_prompt(args.prompt)
    client = get_client(args.base_url).with_options(timeout=600, max_retries=0)
    failures = []
    for label in args.model or MODELS:
        for index, job in enumerate(jobs, 1):
            destination = args.data / 'results' / job['page_id'] / label / 'htr' / f"{job['region_id']}-{args.prompt.stem}.json"
            if destination.exists():
                old = json.loads(destination.read_text())
                if (old.get('image_sha256') == job['image_sha256'] and old.get('prompt') == prompt
                        and old.get('model') == MODELS[label] and isinstance(old.get('response', {}).get('transcription'), str)):
                    print(f'Skip {label} {index}/{len(jobs)} {job["page_id"]} {job["region_id"]}', flush=True)
                    continue
                raise ValueError(f'Existing result differs from this input; preserve or move it before rerunning: {destination}')
            print(f'Run {label} {index}/{len(jobs)} {job["page_id"]} {job["region_id"]}', flush=True)
            result = None
            try:
                result = analyze_image(job['crop'], prompt, model=MODELS[label], max_tokens=4096, client=client, parse_json=False)
                result.parsed = extract_json(result.content)
                if not isinstance(result.parsed.get('transcription'), str):
                    raise ValueError('Missing transcription string')
                if result.raw_response.choices[0].finish_reason != 'stop':
                    raise ValueError('Incomplete generation: ' + str(result.raw_response.choices[0].finish_reason))
                payload = {k: v for k, v in job.items() if k != 'crop'}
                payload.update(model=result.model, task='htr', prompt_version=args.prompt.stem,
                               image=str(job['crop']), prompt=prompt, response=result.parsed,
                               raw_content=result.content, elapsed_seconds=result.elapsed_seconds,
                               temperature=0.0, max_tokens=4096)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(destination)
                print(f'Saved ({result.elapsed_seconds:.1f}s)', flush=True)
            except Exception as exc:
                error_path = args.data / 'htr-errors' / job['page_id'] / label / destination.name
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(json.dumps({
                    'model': MODELS[label], 'region_id': job['region_id'],
                    'image_sha256': job['image_sha256'], 'prompt': prompt,
                    'error': str(exc), 'raw_content': result.content if result else None,
                    'finish_reason': result.raw_response.choices[0].finish_reason if result else None,
                }, ensure_ascii=False, indent=2), encoding='utf-8')
                failures.append(f'{label}/{job["page_id"]}/{job["region_id"]}: {type(exc).__name__} (see {error_path})')
                print(f'FAILED: {failures[-1]}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} requests failed; rerun to retry missing results.')


if __name__ == '__main__':
    main()
