# Waqf VLM

Experimental tooling for evaluating vision-language models (VLMs) on historical Arabic-script archival documents, with particular attention to **document segmentation, handwritten text recognition (HTR), language identification, stamps and seals, annotations, and waqf-related content**.

The project is intended to support the development of a human-verified dataset that can eventually be used for benchmarking, model evaluation, and an AI challenge focused on historical document understanding.

## Research Goals

Historical archival documents often contain multiple overlapping layers of information:

* Primary document text
* Marginal text
* Handwritten annotations
* Stamps and seals
* Later archival or cataloguing marks
* Multiple scripts or languages
* Waqf/endowment statements and other semantic information

Traditional HTR pipelines generally treat layout analysis and text recognition as separate tasks and may have difficulty distinguishing these layers or interpreting their function.

This project explores whether modern VLMs can complement specialist tools such as eScriptorium by performing several distinct tasks:

1. **Document segmentation**
2. **Region classification**
3. **Script and language identification**
4. **Handwritten text recognition**
5. **Semantic classification**
6. **Identification of waqf-related evidence**

These tasks are intentionally evaluated separately. For example, recognizing that an object is a seal is a visual segmentation/classification problem, while determining that a text region provides evidence of waqf status is a semantic interpretation problem.

## Initial Experimental Pipeline

```text
Original TIFF
      │
      ├───────────────┐
      │               │
      ▼               ▼
eScriptorium      VLM derivative
ALTO XML              PNG
      │               │
      │               ▼
      │          Vision-language model
      │               │
      │               ▼
      │          detected regions
      │               │
      │               ▼
      │       crop original TIFF
      │               │
      │               ▼
      │          HTR / language ID
      │               │
      │               ▼
      │      semantic classification
      │               │
      └───────┬───────┘
              ▼
         evaluation
              │
              ▼
       human verification
```

The original TIFF is always treated as the authoritative image. Smaller PNG derivatives may be generated for whole-page VLM analysis, but region crops for HTR are taken from the original TIFF whenever possible.

## Experimental Tasks

### 1. Segmentation

Given a complete page image, identify visually distinct document regions.

Initial region vocabulary:

* `main_text`
* `marginal_text`
* `stamp_or_seal`
* `handwritten_annotation`
* `archival_mark`
* `illustration`
* `unknown`

The initial segmentation task should be based only on visually observable evidence. It should not attempt to identify waqf content.

### 2. Script and Language Identification

For each text-bearing region, identify where possible:

* Script
* Probable language
* Confidence or uncertainty

For the initial corpus this may include Arabic-script material in languages such as Arabic and Ottoman Turkish.

Script and language should be recorded separately.

### 3. HTR

Text-bearing regions are cropped from the original high-resolution TIFF and submitted independently for transcription.

The goal is a **diplomatic transcription** rather than modernization or semantic correction.

Models should be instructed to preserve uncertainty rather than invent plausible text.

### 4. Semantic Classification

Semantic analysis occurs after segmentation and HTR.

Possible semantic classes may eventually include:

* Waqf/endowment
* Ownership/provenance
* Personal name
* Place
* Institution
* Date
* Administrative annotation
* Cataloguing or archival mark
* Other
* Uncertain

This vocabulary is experimental and should evolve through examination of the corpus and expert annotation.

### 5. Waqf Identification

`waqf` is deliberately **not treated as a visual segmentation class**.

Evidence of waqf status might occur in:

* Main document text
* Marginal annotations
* Seals
* Stamps
* Formulaic language
* Multiple regions considered together

Waqf detection is therefore treated as a semantic document-understanding task.

## Models

The initial experiments use Qwen3-VL running locally on Apple Silicon.

Models currently under consideration include:

* Qwen3-VL-8B-Instruct
* Qwen3-VL-30B-A3B-Instruct

The initial local environment uses MLX models through LM Studio.

LM Studio acts as an inference server:

```text
Python / Jupyter
       │
       │ HTTP
       ▼
localhost:1234
       │
       ▼
LM Studio
       │
       ▼
Qwen3-VL
       │
       ▼
Apple Silicon GPU
```

This separates the experimental code from the inference backend. The same evaluation pipeline can later be used with direct MLX inference, CUDA-hosted models, or external APIs.

## Project Structure

```text
waqf-vlm/
├── notebooks/
│   ├── 01_vlm_sandbox.ipynb
│   ├── 02_segmentation.ipynb
│   ├── 03_htr.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── lmstudio.py
│   ├── images.py
│   ├── alto.py
│   ├── segmentation.py
│   └── evaluation.py
│
├── data/
│   ├── images/
│   ├── alto/
│   ├── derivatives/
│   ├── crops/
│   └── results/
│
├── prompts/
│   ├── segmentation.txt
│   ├── htr.txt
│   └── classification.txt
│
├── pyproject.toml
└── README.md
```

## Installation

The project uses Python and [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

Install `uv` on macOS:

```bash
brew install uv
```

Clone or create the project and install the initial dependencies:

```bash
uv add jupyterlab \
       pillow \
       matplotlib \
       pandas \
       requests \
       openai \
       pydantic \
       shapely \
       lxml
```

Start JupyterLab:

```bash
uv run jupyter lab
```

## LM Studio

Start LM Studio and load a vision-capable model such as:

```text
Qwen3-VL-8B-Instruct-MLX
```

The model should report:

```text
format: mlx
vision: true
```

The default LM Studio API endpoint is:

```text
http://localhost:1234/v1
```

A Python client can be created with:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
```

Model identifiers can then be treated as experimental variables:

```python
MODEL_8B = "qwen3-vl-8b-instruct-mlx"
MODEL_30B = "qwen3-vl-30b-a3b-instruct-mlx"
```

## Image Handling

Original archival TIFFs should be retained without modification.

For whole-page VLM processing, create a PNG derivative:

```python
from src.images import create_vlm_derivative

create_vlm_derivative(
    "data/images/example.tif",
    "data/derivatives/example.png",
    max_dimension=2048,
)
```

VLM segmentation coordinates use a normalized `0–1000` coordinate system.

For example:

```json
{
  "type": "stamp_or_seal",
  "bbox": [390, 80, 735, 290],
  "confidence": 0.91
}
```

These coordinates can then be mapped back onto the original TIFF for high-resolution cropping.

```python
from src.images import crop_normalized_bbox

crop = crop_normalized_bbox(
    "data/images/example.tif",
    (390, 80, 735, 290),
    padding=50,
)
```

This avoids performing HTR on a downscaled whole-page image.

## ALTO

Existing eScriptorium output is imported from ALTO XML.

The ALTO parser preserves:

* Page dimensions
* Text blocks
* Region types
* Polygons
* Text lines
* Baselines
* Existing HTR output
* Confidence values

Example:

```python
from src.alto import load_alto

page = load_alto(
    "data/alto/example.xml"
)

print(page.image_filename)

for block in page.blocks:
    print(block.block_type)

    for line in block.lines:
        print(line.text)
```

eScriptorium output is treated as a **model prediction**, not as ground truth.

## Ground Truth

The eventual evaluation architecture is:

```text
                     eScriptorium
                          │
                          ▼
                     predictions
                          │
                          │
Human annotation ─────────┼──── evaluation
                          │
                          │
                     predictions
                          ▲
                          │
                       VLMs
```

Human-reviewed annotations should ultimately serve as the gold standard against which all systems are evaluated.

Existing eScriptorium output can nevertheless provide useful weak labels and can help prioritize material for human review.

## Segmentation Evaluation

Region detection can initially be evaluated using Intersection over Union (IoU).

```python
from src.evaluation import match_regions

matches, missed, extra = match_regions(
    human_regions,
    qwen_regions,
    iou_threshold=0.5,
)
```

Detection and classification should be evaluated independently.

This distinguishes:

> Did the model find the object?

from:

> Did the model correctly identify the object?

Metrics can be calculated separately for classes such as:

```text
main_text
marginal_text
stamp_or_seal
handwritten_annotation
archival_mark
```

Potential metrics include:

* Precision
* Recall
* F1
* IoU
* Mean IoU
* AP/mAP

## HTR Evaluation

Initial HTR evaluation uses:

* Character Error Rate (CER)
* Word Error Rate (WER)

Example:

```python
from src.evaluation import htr_metrics

metrics = htr_metrics(
    reference=human_transcription,
    hypothesis=model_transcription,
)

print(metrics.cer)
print(metrics.wer)
```

### Unicode Normalization

Arabic-script historical material requires an explicit normalization policy.

The project should eventually distinguish between:

**Diplomatic CER**

Measures the model against the transcription exactly as encoded by the annotator.

**Normalized CER**

Applies a documented normalization policy before comparison.

Possible normalization decisions include:

* Unicode NFC/NFKC
* Combining marks
* Diacritics
* Arabic/Persian character variants
* Alef forms
* Whitespace
* Punctuation

Normalization should never occur silently.

## Prompt Versioning

Prompts are part of the experimental methodology and should be versioned.

For example:

```text
prompts/
├── segmentation-v1.txt
├── segmentation-v2.txt
├── htr-v1.txt
└── classification-v1.txt
```

An initial segmentation prompt might be:

```text
You are analyzing a historical archival document.

Identify visually distinct regions on the page.

Use only these classes:

main_text
marginal_text
stamp_or_seal
handwritten_annotation
archival_mark
illustration
unknown

For every region return:

{
  "type": "...",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}

Coordinates must be normalized from 0–1000.

Do not transcribe text.
Do not infer semantic meaning.
Identify only visually observable regions.
```

## Experiment Provenance

Every model run should preserve enough information to reproduce the experiment.

Example:

```json
{
  "image": "example.tif",
  "model": "qwen3-vl-8b-instruct-mlx",
  "quantization": "8bit",
  "task": "segmentation",
  "prompt_version": "segmentation-v1",
  "coordinate_system": "normalized-1000",
  "regions": []
}
```

Additional metadata should eventually include:

* Model revision
* Inference backend
* Runtime version
* Image derivative dimensions
* Temperature
* Maximum tokens
* Prompt hash/version
* Date/time
* Processing duration

## Experimental Comparisons

The initial benchmark should compare identical pages and prompts across models.

For example:

| Model            | Segmentation | Seal Detection | Annotation Detection | Language ID | HTR |
| ---------------- | ------------ | -------------- | -------------------- | ----------- | --- |
| eScriptorium     | ✓            | ✓              | ✓                    | —           | ✓   |
| Qwen3-VL-8B      | ✓            | ✓              | ✓                    | ✓           | ✓   |
| Qwen3-VL-30B-A3B | ✓            | ✓              | ✓                    | ✓           | ✓   |

Model comparisons should use the same human-reviewed test set.

The project should also investigate whether existing HTR can improve VLM performance:

### Image only

```text
image → VLM → transcription
```

### Image + eScriptorium hypothesis

```text
image
   +
eScriptorium HTR
   ↓
VLM correction
   ↓
transcription
```

### eScriptorium baseline

```text
image → eScriptorium → transcription
```

This allows us to test whether specialist HTR and general-purpose multimodal models are complementary.

## Toward an AI Challenge

A mature dataset could support independent challenge tracks:

### Track 1 — Layout Analysis

Detect and segment document regions.

### Track 2 — Stamp and Seal Detection

Locate and classify stamps and seals.

### Track 3 — Script and Language Identification

Identify scripts and languages at page or region level.

### Track 4 — Historical HTR

Produce diplomatic transcriptions of text-bearing regions.

### Track 5 — Waqf Detection

Identify regions or documents containing evidence of waqf/endowment status.

### Track 6 — Semantic Document Understanding

Extract entities and document functions such as people, institutions, dates, ownership, provenance, and administrative annotations.

This modular structure allows systems to participate in individual tasks without requiring a single model to solve the entire document-understanding problem.

## Development Principles

1. **Preserve original images.** Never modify source TIFFs.
2. **Separate observation from interpretation.** Visual segmentation and semantic classification are distinct tasks.
3. **Preserve uncertainty.** Models and annotators should be able to mark uncertain or illegible material.
4. **Do not treat model output as ground truth.**
5. **Version prompts and model configurations.**
6. **Retain raw model responses.**
7. **Evaluate against human-reviewed annotations.**
8. **Keep the pipeline model-independent.**
9. **Document normalization decisions.**
10. **Design the dataset for scholarly reuse, not only model performance.**

## Status

This project is currently exploratory.

Initial work focuses on a small number of representative historical documents processed previously with eScriptorium. The immediate goal is to determine where modern VLMs improve on or complement existing segmentation and HTR pipelines before defining a larger annotation campaign or challenge dataset.

## Static research report

Build an offline HTML site from the saved experiments, without running any models:

```bash
uv run waqf-report build
```

Open `reports/generated/index.html` in a browser, or publish the **contents** of
`reports/generated/` to GitHub Pages or any static host. Links and styles are
relative, so the report also works beneath a project URL. No JavaScript, remote
fonts, inference server, or network access is required by the report generator.
`uv` may need to install Python dependencies on first use.

Alternative locations:

```bash
uv run waqf-report build --data data --output /tmp/waqf-report
```

Run commands from the repository root, or supply explicit paths. Output must be
separate from source data, templates, and Python code. A nonempty output folder
must belong to a previous report build. Builds update generated pages and remove
stale detail pages listed in the previous build's ownership file.

Report sources:

- `src/waqf_vlm/report.py`: saved-data loading and CLI rendering.
- `reports/templates/`: Jinja2 base, index, and manuscript-image templates.
- `reports/static/css/site.css`: responsive site styles.
- `reports/static/js/`: reserved for future optional enhancements.
- `reports/generated/`: ignored build output; not research source data.

The existing `src` package remains available to notebooks. The wheel also
includes the `waqf_vlm` command package and report templates/static assets.

### Evidence and review status

`data/alto/*.xml` and `data/predictions/escriptorium/*.xml` are treated as
**machine predictions**, never as ground truth. Only explicitly human-corrected
exports placed in `data/ground-truth/alto/*.xml` are designated **human reviewed**.
Do not put uncorrected machine exports in that reference directory. The report
cannot infer review history from the ALTO format and notes that correction scope
and reviewer identity are not recorded.

`data/ground-truth/*.json` contains human-drawn layout references. These are
identified as human annotations, with independent review and annotation
completeness unrecorded; they are not labeled as reviewed transcriptions.
Saved Qwen experiments under `data/results/<page>/<model>/` are **machine
suggested** and **not yet reviewed**. These labels describe separate facts:
origin and review status.

The foundation lists saved region counts, ALTO text, and saved HTR/script/language
suggestions. It does not compute accuracy, infer semantic findings, copy original
TIFFs, or invent missing metadata. Proportional PNG display copies (at most 1800
pixels on the longer edge, with 640-pixel thumbnails) are generated from source
images for publication. Invalid records appear in a data-quality
notice; unavailable references are explicitly marked. A manuscript detail page
represents one source image, not necessarily a complete manuscript.

### Publication design

Manuscript images lead the index and detail pages. Warm paper tones, dark text,
serif reading typography, and one muted green accent keep the presentation
focused on the source material. No remote fonts, animation, or JavaScript are
required. Desktop and tablet layouts adapt to a single column on narrow screens.

`reports/templates/components.html` provides reusable manuscript figures and
review labels. Figure descriptions come from `ManuscriptImage.alt`; the default
identifies the image without inventing a description of its contents. Captions
name the source file and describe the display transformation. Catalogue metadata
and image credits remain explicitly unavailable until supplied by the research
data; a local source filename is not presented as collection provenance.

Reusable CSS components in `reports/static/css/site.css` include:

- `.manuscript-figure`, `.figure-caption`, and `.provenance` for visual evidence.
- `.scholarly-note` and `.experimental-warning` for editorial context and limits.
- `.model-label`, `.human-reviewed-label`, and `.review-pending-label` for status.
- `.comparison-table` within a keyboard-focusable `.table-wrap` for inventories.
- `.technical-details` on native `details`/`summary` elements for disclosures.
- `.callout-question` within `.interpretation-section` for open research questions.

Machine suggestions use dashed rules, human reference material uses solid rules,
and interpretation has a separate labeled section. These distinctions do not
depend on color alone. Images preserve their full extent, text readings retain
line breaks and automatic text direction, and skip links and visible focus
indicators support keyboard navigation.

Run focused checks with:

```bash
uv run python -m unittest discover -s tests -v
```

### Manuscript experiment narrative

Each detail page now follows the original manuscript image with available system
segmentations, a shared region inventory, automatically selected disagreement
crops, human-review status, and research context. The saved experiment for
`1280_AB010309_0005` includes eScriptorium, Qwen3-VL 8B, and Qwen3-VL 30B.
Only systems with readable saved segmentation data appear.

`src/waqf_vlm/experiment.py` reuses the existing region conversion, matching, and
image helpers. Numbered overlays are generated from saved bounding boxes;
original-resolution crops have 30 pixels of context before display resizing.
The inventory excludes ALTO blocks without bounding boxes and discloses that
exclusion. Text from those blocks remains in the saved-record disclosure.

Disagreement examples compare predictions using greedy one-to-one bounding-box
IoU matching at 0.5, without requiring equal labels. The Qwen pair is prioritized,
then label differences and larger unmatched boxes; up to four examples are
shown, omitting examples overlapping an earlier selection at IoU 0.4 or above.
The method is disclosed on the page. These are descriptive differences, never
error judgments; a differently sized or grouped region may have no counterpart
under this matching rule. Human references are excluded from this selection.

Only corrected ALTO in `data/ground-truth/alto/` activates the human-reviewed
section. Otherwise the page states “Human review not yet available.” Existing
human-drawn JSON boxes are acknowledged separately. Technical disclosures retain
saved model names, prompt versions, inference durations, and available metadata.
Input dimensions are read from the current saved derivative/crop file; missing
quantization and other run settings remain unavailable rather than inferred.

### Reusable publication segmentation figures

```python
from waqf_vlm.segmentation_figure import render_segmentation

figure = render_segmentation(
    "data/images/1280_AB010309_0005.tif",
    regions,                         # existing normalized Region objects
    "Qwen3-VL 30B",
    human_ground_truth=None,         # supply only explicitly reviewed Regions
    output_dir="reports/generated/assets",
    show_confidence=False,
)
```

This function writes a publication PNG and returns its path. It preserves the
whole manuscript and its aspect ratio, with external numbered labels and a
compact legend. Stable type codes use one restrained accent across systems.
Dashed boundaries identify machine predictions; solid boundaries identify
human-reviewed annotations. Fine leader lines connect boundaries to the external
labels; label text never covers the manuscript. `reviewed=True` renders a
reference-only figure. Merely originating in eScriptorium does not imply review.

Figures use the existing normalized-bbox conversion utilities and keep the
report's bounding-box policy (polygons are not substituted). TIFF input is
supported. Confidence appears only with `show_confidence=True`, including legacy
numeric strings. Identical inputs and rendering-library versions produce the
same content-derived filename and PNG bytes; no timestamps or inference are
involved. Default figures are unaffected by changes to undisplayed confidence.

The report builder automatically uses this renderer for segmentation figures in
`reports/generated/assets/`. Source manuscript images are never overwritten.

### Inspecting every disagreement

The reusable component in `reports/templates/disagreements.html` presents one
crop with a row per prediction system and a separate **Human review** row.
Without a human-reviewed reference it explicitly says **Not yet available**.
Reference labels, when available, are shown as evidence without automatically
judging a prediction right or wrong.

`find_disagreements(systems, labels, limit=None)` returns the complete pairwise
list: every geometrically matched region with differing types and every
unmatched region, using the existing IoU matcher. `select_disagreements(...)`
selects at most four spatially varied examples for the main narrative. The
**Inspect all … pairwise disagreements** disclosure includes the complete list,
including repeated features compared across different pairs of systems.

Rows preserve the actual pairwise assignments. An unmatched region remains
**No matched region** even when a differently grouped box overlaps it; contextual
overlap is explained separately. Other systems and reviewed references are
compared independently with the crop area at IoU ≥ 0.5. Comparison source paths,
region identifiers, crop bounds, and overlaps are available in a disclosure.

Disagreement crops retain native pixels from the original TIFF (preferred when
multiple source formats are present), with up to 30 pixels of edge-clamped
padding. Each crop links to that full-resolution PNG. A separate preview is
limited to 900 pixels for inline display; no model inference is involved.

### HTR comparisons

Manuscript pages now show a crop followed by the human transcription (when
available) and each saved system reading. `src/waqf_vlm/htr.py` groups results by
an explicit `region_id` or the legacy `crops/<page>/<region-id>.png` input path.
Only human-defined JSON regions or human-corrected ALTO blocks qualify; ambiguous
or missing links are reported instead of guessed. Larger machine ALTO blocks are
not substituted for a passage. Matching ALTO blocks require the same stable ID
or unique reciprocal bounding-box IoU of at least 0.95.

Human-corrected ALTO from `ground-truth/alto/` supplies reference text, never
machine ALTO from `alto/` or `predictions/escriptorium/`. The HTR reader also
supports an optional explicit extension to a human JSON region: `transcription`
(string), `review_status: "human_corrected"`, and optional `script`/`language`.
A transcription without that review status is not used as ground truth. The
existing rectangle annotation widget does not author or preserve these extension
fields; use corrected ALTO for its normal workflow.

CER and WER are calculated with the existing evaluation helpers only when a
linked human-corrected transcription exists, and are placed in a collapsed
technical disclosure. No Unicode or editorial normalization is applied. CER uses
code points and WER uses the evaluator's whitespace tokenization. Empty saved
hypotheses are retained; unavailable text is not replaced by an empty string.
ALTO lines are assembled in XML order with newline separators. UTF-8 HTML,
Unicode-aware direction selection, and preserved whitespace keep Arabic-script
readings and line breaks intact. Script and language suggestions are displayed
as supplied, without treating a model's script/style vocabulary as authoritative.

The existing example has two Qwen readings for `human-382299a4`, but no linked
human transcription or unambiguous eScriptorium region reading. No error rates
are reported for it. Crops are regenerated at native resolution from the original
image using the human region and 30 pixels of context, with a separate display
preview; they do not reconstruct unrecorded historical crop settings.

### Guide for nontechnical readers

`reports/templates/method.html` generates **How this experiment works** at
`reports/generated/method.html`. The publication navigation links to it from
every page. It explains the intended workflow, the current experiments' scope,
and seven key terms in plain language, keeping machine suggestions, reviewed
references, and historical interpretation distinct.
