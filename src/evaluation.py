from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from shapely.geometry import Polygon, box


@dataclass
class Region:
    """
    Normalized representation of a document region.

    bbox uses absolute image coordinates:

        (x1, y1, x2, y2)

    polygon is optional. If present, polygon IoU is preferred.
    """
    id: str
    region_type: str
    bbox: tuple[float, float, float, float]
    polygon: list[tuple[float, float]] | None = None
    text: str | None = None
    language: str | None = None
    script: str | None = None
    confidence: float | None = None


@dataclass
class RegionMatch:
    """
    Match between a predicted region and a ground-truth region.
    """
    truth: Region
    prediction: Region
    iou: float
    type_match: bool


@dataclass
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass
class HTRMetrics:
    character_errors: int
    reference_characters: int
    cer: float
    word_errors: int
    reference_words: int
    wer: float


def _geometry(region: Region):
    """
    Return a Shapely geometry for a region.

    Prefer the polygon if one exists; otherwise use bbox.
    """
    if region.polygon and len(region.polygon) >= 3:
        polygon = Polygon(region.polygon)

        # Historical-document polygons can occasionally be invalid.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if not polygon.is_empty:
            return polygon

    x1, y1, x2, y2 = region.bbox
    return box(x1, y1, x2, y2)


def intersection_over_union(
    a: Region,
    b: Region,
) -> float:
    """
    Calculate IoU between two document regions.

        intersection
    -------------------
          union
    """
    geom_a = _geometry(a)
    geom_b = _geometry(b)

    intersection = geom_a.intersection(geom_b).area
    union = geom_a.union(geom_b).area

    if union == 0:
        return 0.0

    return intersection / union


def match_regions(
    truth: Sequence[Region],
    predictions: Sequence[Region],
    iou_threshold: float = 0.5,
    require_same_type: bool = False,
) -> tuple[list[RegionMatch], list[Region], list[Region]]:
    """
    Greedily match predicted regions against ground truth.

    Returns:

        matches
        unmatched_truth
        unmatched_predictions

    Each region may be matched only once.
    """

    candidates = []

    for truth_index, truth_region in enumerate(truth):
        for prediction_index, prediction_region in enumerate(predictions):

            if (
                require_same_type
                and truth_region.region_type
                != prediction_region.region_type
            ):
                continue

            iou = intersection_over_union(
                truth_region,
                prediction_region,
            )

            if iou >= iou_threshold:
                candidates.append(
                    (
                        iou,
                        truth_index,
                        prediction_index,
                    )
                )

    # Highest IoU wins.
    candidates.sort(reverse=True)

    used_truth = set()
    used_predictions = set()

    matches = []

    for iou, truth_index, prediction_index in candidates:

        if truth_index in used_truth:
            continue

        if prediction_index in used_predictions:
            continue

        truth_region = truth[truth_index]
        prediction_region = predictions[prediction_index]

        matches.append(
            RegionMatch(
                truth=truth_region,
                prediction=prediction_region,
                iou=iou,
                type_match=(
                    truth_region.region_type
                    == prediction_region.region_type
                ),
            )
        )

        used_truth.add(truth_index)
        used_predictions.add(prediction_index)

    unmatched_truth = [
        region
        for i, region in enumerate(truth)
        if i not in used_truth
    ]

    unmatched_predictions = [
        region
        for i, region in enumerate(predictions)
        if i not in used_predictions
    ]

    return (
        matches,
        unmatched_truth,
        unmatched_predictions,
    )


def detection_metrics(
    truth: Sequence[Region],
    predictions: Sequence[Region],
    iou_threshold: float = 0.5,
    require_same_type: bool = True,
) -> DetectionMetrics:
    """
    Calculate region-detection precision, recall and F1.
    """

    matches, missed, extra = match_regions(
        truth,
        predictions,
        iou_threshold=iou_threshold,
        require_same_type=require_same_type,
    )

    tp = len(matches)
    fp = len(extra)
    fn = len(missed)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def metrics_by_type(
    truth: Sequence[Region],
    predictions: Sequence[Region],
    iou_threshold: float = 0.5,
) -> dict[str, DetectionMetrics]:
    """
    Calculate detection metrics separately for every region type.

    Useful for results such as:

        main_text       F1 .91
        stamp_or_seal   F1 .84
        annotation      F1 .62
    """

    region_types = sorted(
        {
            region.region_type
            for region in [*truth, *predictions]
        }
    )

    results = {}

    for region_type in region_types:

        truth_subset = [
            r for r in truth
            if r.region_type == region_type
        ]

        prediction_subset = [
            r for r in predictions
            if r.region_type == region_type
        ]

        results[region_type] = detection_metrics(
            truth_subset,
            prediction_subset,
            iou_threshold=iou_threshold,
            require_same_type=True,
        )

    return results


def classification_accuracy(
    matches: Iterable[RegionMatch],
) -> float:
    """
    Among geometrically matched regions, calculate how often
    the predicted region class matches the reference class.

    This intentionally separates:

        "Did you find the region?"

    from:

        "Did you correctly identify what it is?"
    """

    matches = list(matches)

    if not matches:
        return 0.0

    correct = sum(
        match.type_match
        for match in matches
    )

    return correct / len(matches)


# ---------------------------------------------------------------------
# HTR evaluation
# ---------------------------------------------------------------------

def levenshtein_distance(
    reference: Sequence,
    hypothesis: Sequence,
) -> int:
    """
    Standard Levenshtein edit distance.

    Works with strings for CER and lists of words for WER.
    """

    previous = list(range(len(hypothesis) + 1))

    for i, reference_item in enumerate(reference, start=1):

        current = [i]

        for j, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            substitution_cost = (
                0
                if reference_item == hypothesis_item
                else 1
            )

            current.append(
                min(
                    previous[j] + 1,             # deletion
                    current[j - 1] + 1,          # insertion
                    previous[j - 1]
                    + substitution_cost,         # substitution
                )
            )

        previous = current

    return previous[-1]


def character_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    """
    CER = character edit distance / reference characters.

    No Unicode normalization is performed here intentionally.
    For diplomatic transcription evaluation we want normalization
    policy to be explicit rather than silently modifying text.
    """

    if not reference:
        return 0.0 if not hypothesis else 1.0

    errors = levenshtein_distance(
        reference,
        hypothesis,
    )

    return errors / len(reference)


def word_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    """
    WER = word edit distance / reference words.
    """

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    errors = levenshtein_distance(
        reference_words,
        hypothesis_words,
    )

    return errors / len(reference_words)


def htr_metrics(
    reference: str,
    hypothesis: str,
) -> HTRMetrics:

    character_errors = levenshtein_distance(
        reference,
        hypothesis,
    )

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    word_errors = levenshtein_distance(
        reference_words,
        hypothesis_words,
    )

    return HTRMetrics(
        character_errors=character_errors,
        reference_characters=len(reference),
        cer=(
            character_errors / len(reference)
            if reference
            else 0.0 if not hypothesis else 1.0
        ),
        word_errors=word_errors,
        reference_words=len(reference_words),
        wer=(
            word_errors / len(reference_words)
            if reference_words
            else 0.0 if not hypothesis_words else 1.0
        ),
    )