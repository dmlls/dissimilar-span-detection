"""Calculate inter-annotator agreements."""

import itertools
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import List

from models import InterAnnotatorAgreementResults
from utils import AnnotationEvaluator, calculate_ssd_cohen_kappa

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.data_loading import SSDTextPairDataLoader
from dataset.models import RawTextPair

ANNOTATIONS_PATH: str = "../../span-similarity-dataset/annotations"
OUTPUT_DIR: str = "./results"


def calculate_agreement(
    file_a: Path, file_b: Path, log_dir: Path
) -> InterAnnotatorAgreementResults:
    annotations_a: List[RawTextPair] = list(
        SSDTextPairDataLoader.load_raw(file_a, skip_header=False)
    )
    annotations_b: List[RawTextPair] = list(
        SSDTextPairDataLoader.load_raw(file_b, skip_header=False)
    )
    agreement = calculate_ssd_cohen_kappa(annotations_a, annotations_b)
    suffix_a = file_a.stem.split("_")[-1] if "_" in file_a.stem else file_a.stem
    suffix_b = file_b.stem.split("_")[-1] if "_" in file_b.stem else file_b.stem
    with open(
        log_dir / f"{suffix_a}-{suffix_b}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "kappa_span_boundaries": agreement.span_boundaries,
                "kappa_span_labels": agreement.span_labels,
                "kappa_n_spans": agreement.n_spans,
            },
            f,
            indent=4,
        )
    return agreement


def main():
    annotations_path: Path = Path(ANNOTATIONS_PATH)
    output_dir: Path = Path(OUTPUT_DIR)
    agreement_dir: Path = output_dir / "agreement"
    evaluation_dir: Path = output_dir / "evaluation"
    detailed_dir: Path = agreement_dir / "detailed"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)

    annotation_files: List[Path] = [
        f for f in annotations_path.iterdir() if f.is_file() and "gold" not in f.name
    ]
    gold_file: Path = [
        f for f in annotations_path.iterdir() if f.is_file() and "gold" in f.name
    ]
    if gold_file:
        gold_file = gold_file[0]
    else:
        raise RuntimeError(
            "file named 'gold.tsv' containing the gold annotations is missing"
        )

    # Inter-annotator agreements.
    span_boundaries_inter_agreements: List[float] = []
    span_labels_inter_agreements: List[float] = []
    n_spans_inter_agreements: List[float] = []
    for file_b, file_a in itertools.combinations(annotation_files, r=2):
        agreement = calculate_agreement(file_a, file_b, log_dir=detailed_dir)
        span_boundaries_inter_agreements.append(agreement.span_boundaries)
        span_labels_inter_agreements.append(agreement.span_labels)
        n_spans_inter_agreements.append(agreement.n_spans)

    # Annotator-gold results.
    span_boundaries_gold_agreements: List[float] = []
    span_labels_gold_agreements: List[float] = []
    n_spans_gold_agreements: List[float] = []
    for annotation_file in annotation_files:
        agreement = calculate_agreement(
            annotation_file, gold_file, log_dir=detailed_dir
        )
        span_boundaries_gold_agreements.append(agreement.span_boundaries)
        span_labels_gold_agreements.append(agreement.span_labels)
        n_spans_gold_agreements.append(agreement.n_spans)

    with open(
        agreement_dir / "aggregated_annotator_agreements.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "inter_annotator_kappa_scores": {
                    "span_boundaries_mean": round(
                        mean(span_boundaries_inter_agreements), 3
                    ),
                    "span_boundaries_stdev": round(
                        stdev(span_boundaries_inter_agreements), 3
                    ),
                    "span_labels_mean": round(mean(span_labels_inter_agreements), 3),
                    "span_labels_stdev": round(stdev(span_labels_inter_agreements), 3),
                    "n_spans_mean": round(mean(n_spans_inter_agreements), 3),
                    "n_spans_stdev": round(stdev(n_spans_inter_agreements), 3),
                },
                "annotator_gold_kappa_scores": {
                    "span_boundaries_mean": round(
                        mean(span_boundaries_gold_agreements), 3
                    ),
                    "span_boundaries_stdev": round(
                        stdev(span_boundaries_gold_agreements), 3
                    ),
                    "span_labels_mean": round(mean(span_labels_gold_agreements), 3),
                    "span_labels_stdev": round(stdev(span_labels_gold_agreements), 3),
                    "n_spans_mean": round(mean(n_spans_gold_agreements), 3),
                    "n_spans_stdev": round(stdev(n_spans_gold_agreements), 3),
                },
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
