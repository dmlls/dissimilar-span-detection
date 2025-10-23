"""Annotation evaluation utilities."""

import itertools
import json
import string
import sys
import time
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, List, Optional, Set, Tuple

import nltk
from models import InterAnnotatorAgreementResults
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.data_loading import SSDTextPairDataLoader
from dataset.models import RawTextPair, Span
from evaluation.models import EvaluationResults


def get_span_vector(sentence: str) -> List[int]:
    """Compute word-in-span vector for a sentence.

    This vector contains, for each word (whitespace tokenized), whether the
    word is outside a span (0), or inside (1). Punctuation is ignored.

    E.g., for the sentence ``"It's {{not cold}}."``, the vector ``[0, 1, 1]``
    is returned.
    """
    # Add whitespace around span markers, and word-tokenize.
    sentence_words = sentence.replace("{{", " {{ ").replace("}}", " }} ").split()
    in_span: bool = False
    word_in_span_vector: List[int] = []
    for word in sentence_words:
        word = word.strip()
        if word == "{{":
            in_span = True
        elif word == "}}":
            in_span = False
        elif any(ch not in string.punctuation for ch in word):
            word_in_span_vector.append(int(in_span))
    return word_in_span_vector


def calculate_ssd_cohen_kappa(
    annotations_a: List[RawTextPair], annotations_b: List[RawTextPair]
) -> InterAnnotatorAgreementResults:
    """Calculate the Cohen's kappa score between two SSD annotations."""

    span_data_a: List[int] = []
    span_data_b: List[int] = []

    label_data_a: List[int] = []
    label_data_b: List[int] = []

    n_annotated_spans_data_a: List[int] = []
    n_annotated_spans_data_b: List[int] = []

    for pair_a, pair_b in zip(annotations_a, annotations_b):
        span_data_a += get_span_vector(pair_a.premise) + get_span_vector(
            pair_a.hypothesis
        )
        span_data_b += get_span_vector(pair_b.premise) + get_span_vector(
            pair_b.hypothesis
        )

        shorter_list = min(len(pair_a.span_labels), len(pair_b.span_labels))
        label_data_a += pair_a.span_labels[:shorter_list]
        label_data_b += pair_b.span_labels[:shorter_list]

        n_annotated_spans_data_a.append(len(pair_a.span_labels))
        n_annotated_spans_data_b.append(len(pair_b.span_labels))

    return InterAnnotatorAgreementResults(
        span_boundaries=cohen_kappa_score(span_data_a, span_data_b),
        span_labels=cohen_kappa_score(label_data_a, label_data_b),
        n_spans=cohen_kappa_score(n_annotated_spans_data_a, n_annotated_spans_data_b),
    )


class AnnotationEvaluator:
    """Annotation evaluator.

    This class is meant to evaluate the SSD against an annotation. Therefore,
    the annotation is taken as the reference.
    """

    @classmethod
    def evaluate(
        cls,
        dataset_path: str,
        annotation_path: str,
        **kwargs,
    ) -> EvaluationResults:
        """See base class."""
        # Scores for sentence pairs with no dissimilar spans.
        precisions_no_diff: List[float] = []
        recalls_no_diff: List[float] = []
        f1s_no_diff: List[float] = []
        # Scores for sentence pairs with dissimilar spans.
        precisions_diff: List[float] = []
        recalls_diff: List[float] = []
        f1s_diff: List[float] = []

        logging_path = cls.set_up_logging_path(annotation_path)
        step_logs = open(
            logging_path / "step_logs.tsv",
            "w",
            encoding="utf-8",
        )
        step_logs.write(cls.format_log_step_header("precision", "recall", "f1"))

        start_time = time.perf_counter()
        for sdd_data_pair, annotation_data_pair in tqdm(
            zip(
                SSDTextPairDataLoader.load_raw(dataset_path, skip_header=False),
                SSDTextPairDataLoader.load(annotation_path, skip_header=False),
            ),
            ncols=60,
        ):
            premise = annotation_data_pair.premise
            hypothesis = annotation_data_pair.hypothesis
            spans_premise: List[Span] = [
                span.premise_span
                for span in annotation_data_pair.span_pairs
                if span.label == "0"
            ]
            spans_hypothesis: List[Span] = [
                span.hypothesis_span
                for span in annotation_data_pair.span_pairs
                if span.label == "0"
            ]
            # We get predictions for both premise and hypothesis.
            for prediction in [sdd_data_pair.hypothesis, sdd_data_pair.premise]:
                if prediction:
                    predicted_spans: List[Span] = [
                        span
                        for span, label in zip(
                            SSDTextPairDataLoader.get_spans(prediction),
                            sdd_data_pair.span_labels,
                        )
                        if label == 0
                    ]
                    aligned_spans = cls.align_spans(spans_hypothesis, predicted_spans)
                    aligned_spans_unigrams: List[
                        Tuple[Optional[List[str]], Optional[List[str]]]
                    ] = []
                    for span_premise, span_hypothesis in aligned_spans:
                        span_premise_unigrams = (
                            cls.get_unigrams(span_premise.text)
                            if span_premise
                            else None
                        )
                        span_hypothesis_unigrams = (
                            cls.get_unigrams(span_hypothesis.text)
                            if span_hypothesis
                            else None
                        )
                        aligned_spans_unigrams.append(
                            (span_premise_unigrams, span_hypothesis_unigrams)
                        )
                    precision, recall, f1 = cls.calculate_precision_recall_f1(
                        aligned_spans_unigrams
                    )
                else:  # The model able to provide a valid annotation.
                    predicted_spans = None
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                # Sentence pair with no differing spans?
                if not [span for span in spans_hypothesis if span]:
                    precisions_no_diff.append(precision)
                    recalls_no_diff.append(recall)
                    f1s_no_diff.append(f1)
                else:
                    precisions_diff.append(precision)
                    recalls_diff.append(recall)
                    f1s_diff.append(f1)
                step_logs.write(
                    cls.format_log_step(
                        premise,
                        hypothesis,
                        ["{{" + span.text + "}}" for span in spans_hypothesis if span],
                        (
                            [
                                "{{" + span.text + "}}"
                                for span in predicted_spans
                                if span
                            ]
                            if predicted_spans is not None
                            else None
                        ),
                        precision,
                        recall,
                        f1,
                    )
                )
                # Swap things around.
                spans_premise, spans_hypothesis = spans_hypothesis, spans_premise
                premise, hypothesis = hypothesis, premise

        elapsed_seconds = time.perf_counter() - start_time

        step_logs.close()

        mean_no_diff_precision = mean(precisions_no_diff) if precisions_no_diff else 0.0
        se_no_diff_precision = (
            pstdev(precisions_no_diff) / sqrt(len(precisions_no_diff))
            if precisions_no_diff
            else 0.0
        )
        mean_no_diff_recall = mean(recalls_no_diff) if recalls_no_diff else 0.0
        se_no_diff_recall = (
            pstdev(recalls_no_diff) / sqrt(len(recalls_no_diff))
            if recalls_no_diff
            else 0.0
        )
        mean_no_diff_f1 = mean(f1s_no_diff) if f1s_no_diff else 0.0
        se_no_diff_f1 = (
            pstdev(f1s_no_diff) / sqrt(len(f1s_no_diff)) if f1s_no_diff else 0.0
        )

        mean_diff_precision = mean(precisions_diff) if precisions_diff else 0.0
        se_diff_precision = (
            pstdev(precisions_diff) / sqrt(len(precisions_diff))
            if precisions_diff
            else 0.0
        )
        mean_diff_recall = mean(recalls_diff) if recalls_diff else 0.0
        se_diff_recall = (
            pstdev(recalls_diff) / sqrt(len(recalls_diff)) if recalls_diff else 0.0
        )
        mean_diff_f1 = mean(f1s_diff) if f1s_diff else 0.0
        se_diff_f1 = pstdev(f1s_diff) / sqrt(len(f1s_diff)) if f1s_diff else 0.0

        all_precisions: List[float] = precisions_no_diff + precisions_diff
        mean_precision = mean(all_precisions) if all_precisions else 0.0
        se_precision = (
            pstdev(all_precisions) / sqrt(len(all_precisions))
            if all_precisions
            else 0.0
        )
        all_recalls: List[float] = recalls_no_diff + recalls_diff
        mean_recall = mean(all_recalls) if all_recalls else 0.0
        se_recall = pstdev(all_recalls) / sqrt(len(all_recalls)) if all_recalls else 0.0
        all_f1s: List[float] = f1s_no_diff + f1s_diff
        mean_f1 = mean(all_f1s) if all_f1s else 0.0
        se_f1 = pstdev(all_f1s) / sqrt(len(all_f1s)) if all_f1s else 0.0

        results: EvaluationResults = EvaluationResults(
            precision_mean=mean_precision,
            recall_mean=mean_recall,
            f1_mean=mean_f1,
            precision_se=se_precision,
            recall_se=se_recall,
            f1_se=se_f1,
            no_diff_precision_mean=mean_no_diff_precision,
            no_diff_recall_mean=mean_no_diff_recall,
            no_diff_f1_mean=mean_no_diff_f1,
            no_diff_precision_se=se_no_diff_precision,
            no_diff_recall_se=se_no_diff_recall,
            no_diff_f1_se=se_no_diff_f1,
            diff_precision_mean=mean_diff_precision,
            diff_recall_mean=mean_diff_recall,
            diff_f1_mean=mean_diff_f1,
            diff_precision_se=se_diff_precision,
            diff_recall_se=se_diff_recall,
            diff_f1_se=se_diff_f1,
            config=kwargs,
        )
        cls.log_results(
            path=logging_path / "results.json",
            elapsed_seconds=elapsed_seconds,
            results=results,
            **kwargs,
        )
        cls.print_results(results)

    @classmethod
    def align_spans(
        cls, spans_premise: List[Span], spans_hypothesis: List[Span]
    ) -> List[Tuple[Optional[Span], Optional[Span]]]:
        spans_premise, spans_hypothesis = cls._fill_span_lists(
            spans_premise, spans_hypothesis
        )
        all_hypothesis: Set[List[Span]] = set(itertools.permutations(spans_hypothesis))
        best_offset_score: int = 999999  # minimize
        best_alignment: List[Optional[Span]] = [None] * len(spans_premise)
        for candidate_hypothesis in all_hypothesis:
            offset_score: int = sum(
                cls._calculate_span_offset(span_premise, span_hypothesis)
                for span_premise, span_hypothesis in zip(
                    spans_premise, candidate_hypothesis
                )
            )
            if offset_score < best_offset_score:
                best_offset_score = offset_score
                best_alignment = candidate_hypothesis
        return list(zip(spans_premise, best_alignment))

    @classmethod
    def _fill_span_lists(
        cls,
        spans_premise: List[Span],
        spans_hypothesis: List[str],
        fillvalue: Optional[Any] = None,
    ):
        len_premise = len(spans_premise)
        len_hypothesis = len(spans_hypothesis)
        if len_premise < len_hypothesis:
            spans_premise += [fillvalue] * (len_hypothesis - len_premise)
        elif len_premise > len_hypothesis:
            spans_hypothesis += [fillvalue] * (len_premise - len_hypothesis)
        return spans_premise, spans_hypothesis

    @classmethod
    def _calculate_span_offset(cls, span_premise: Span, span_hypothesis: Span) -> int:
        if span_premise is None and span_hypothesis is None:
            return 0
        if span_premise is None or span_hypothesis is None:
            return 999
        return abs(span_premise.start_index - span_hypothesis.start_index) + abs(
            span_premise.end_index - span_hypothesis.end_index
        )

    @classmethod
    def calculate_recall(
        cls, aligned_spans: List[Tuple[Optional[List[str]], Optional[List[str]]]]
    ) -> float:
        recall_scores: List[float] = []
        for reference_unigrams, predicted_unigrams in aligned_spans:
            if None in (reference_unigrams, predicted_unigrams):
                continue
            recall: float = len(
                [w for w in predicted_unigrams if w in reference_unigrams]
            ) / len(reference_unigrams)
            recall_scores.append(recall)
        return mean(recall_scores) if recall_scores else 0.0

    @classmethod
    def calculate_precision_recall_f1(
        cls,
        aligned_spans_unigrams: List[Tuple[Optional[List[str]], Optional[List[str]]]],
    ) -> float:
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        f1_scores: List[float] = []
        for reference_unigrams, candidate_unigrams in aligned_spans_unigrams:
            if reference_unigrams and candidate_unigrams:
                # Calculated following METEOR
                # (https://en.wikipedia.org/wiki/METEOR#Algorithm).
                precision: float = len(
                    [w for w in candidate_unigrams if w in reference_unigrams]
                ) / len(candidate_unigrams)
                recall: float = len(
                    [w for w in candidate_unigrams if w in reference_unigrams]
                ) / len(reference_unigrams)
            elif not reference_unigrams and not candidate_unigrams:
                precision: float = 1.0
                recall: float = 1.0
            else:
                precision: float = 0.0
                recall: float = 0.0
            if precision + recall != 0:
                f1: float = 2 * (precision * recall) / (precision + recall)
            else:
                f1: float = 0.0
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        mean_precision = mean(precision_scores) if precision_scores else 1.0
        mean_recall = mean(recall_scores) if recall_scores else 1.0
        mean_f1 = mean(f1_scores) if f1_scores else 1.0
        return mean_precision, mean_recall, mean_f1

    @classmethod
    def log_results(
        cls,
        path: Path,
        elapsed_seconds: float,
        results: EvaluationResults,
        **kwargs,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "n/a",
                    "model_name": "n/a",
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "precision_mean": round(results.precision_mean, 3),
                    "recall_mean": round(results.recall_mean, 3),
                    "f1_mean": round(results.f1_mean, 3),
                    "precision_se": round(results.precision_se, 3),
                    "recall_se": round(results.recall_se, 3),
                    "f1_se": round(results.f1_se, 3),
                    "no_diff_precision_mean": round(results.no_diff_precision_mean, 3),
                    "no_diff_recall_mean": round(results.no_diff_recall_mean, 3),
                    "no_diff_f1_mean": round(results.no_diff_f1_mean, 3),
                    "no_diff_precision_se": round(results.no_diff_precision_se, 3),
                    "no_diff_recall_se": round(results.no_diff_recall_se, 3),
                    "no_diff_f1_se": round(results.no_diff_f1_se, 3),
                    "diff_precision_mean": round(results.diff_precision_mean, 3),
                    "diff_recall_mean": round(results.diff_recall_mean, 3),
                    "diff_f1_mean": round(results.diff_f1_mean, 3),
                    "diff_precision_se": round(results.diff_precision_se, 3),
                    "diff_recall_se": round(results.diff_recall_se, 3),
                    "diff_f1_se": round(results.diff_f1_se, 3),
                    "config": {},
                },
                f,
                indent=4,
            )

    @classmethod
    def set_up_logging_path(cls, annotation_path: str) -> Path:
        """Create necessary directories for evaluation logging."""
        # Create a dir named after the annotation and the timestamp.
        logging_path = Path(f'{annotation_path}_{time.strftime("%Y%m%d-%H%M%S")}')
        logging_path.mkdir(parents=True, exist_ok=True)
        return logging_path

    @classmethod
    def format_log_step_header(cls, *score_names: Tuple[str, ...]) -> str:
        """Format the header of the evaluation step log file.

        Args:
            score_names (:obj:`Tuple[str, ...]`):
                The names of the scores.

        Returns:
            :obj:`str`: A ``.tsv`` suitable header.
        """
        formatted_score_names = "\t".join(score_names)
        return f"premise\thypothesis\treference_spans\tpredicted_spans\t{formatted_score_names}\n"

    @classmethod
    def format_log_step(
        cls,
        premise: str,
        hypothesis: str,
        reference_spans: str,
        predicted_spans: str,
        *scores: Tuple[str, ...],
        **kwargs,
    ) -> str:
        """Format an evaluation step in order to be logged.

        Args:
            premise (:obj:`str`):
                The premise text.
            hypothesis (:obj:`str`):
                The hypothesis text.
            reference_spans (:obj:`str`):
                The reference spans.
            predicted_spans (:obj:`str`):
                The predicted spans.
            scores (:obj:`Tuple[str, ...]`):
                The scores to log. Each score in the list will be a column in
                the ``.tsv`` file.
            **kwargs (:obj:`Dict`):
                The evaluation configuration.

        Returns:
            :obj:`str`: The ``.tsv`` formatted evaluation step.
        """
        formatted_scores = "\t".join(f"{s:.3f}" for s in scores)
        formatted_reference_spans = json.dumps(reference_spans)
        formatted_predicted_spans = json.dumps(predicted_spans)
        return (
            f"{premise}\t{hypothesis}\t{formatted_reference_spans}"
            f"\t{formatted_predicted_spans}\t{formatted_scores}\n"
        )

    @classmethod
    def get_unigrams(
        cls,
        sentence: str,
        tokenization_method: Optional[Callable[[str], List[str]]] = None,
        keep_punctuation: Optional[bool] = None,
    ) -> List[str]:
        """Split a sentence into unigrams.

        Args:
            sentence (:obj:`str`):
                The sentence to get the unigrams from.
            tokenization_method (:obj:`Optional[Callable[[str], List[str]]]`, `optional`):
                The tokenization method to use in order to get the unigrams.
            keep_punctuation (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to keep tokens that are punctuation or not.

        Returns:
            :obj:`List[str]`: The sentence unigrams.
        """
        if tokenization_method is None:
            tokenization_method = nltk.tokenize.word_tokenize
        if keep_punctuation is None:
            keep_punctuation = False

        if keep_punctuation:
            return tokenization_method(sentence)
        return [
            tk for tk in tokenization_method(sentence) if tk not in string.punctuation
        ]

    @classmethod
    def print_results(cls, results: EvaluationResults) -> None:
        print()
        print(f"    Precision Mean:               {results.precision_mean:.3f}")
        print(f"    Recall Mean:                  {results.recall_mean:.3f}")
        print(f"    F1 Mean:                      {results.f1_mean:.3f}")
        print()
        print(f"    Precision Std. Error:         {results.precision_se:.3f}")
        print(f"    Recall Std. Error:            {results.recall_se:.3f}")
        print(f"    F1 Std. Error:                {results.f1_se:.3f}")
        print()
        print(f"    NoDiff Precision Mean:        {results.no_diff_precision_mean:.3f}")
        print(f"    NoDiff Recall Mean:           {results.no_diff_recall_mean:.3f}")
        print(f"    NoDiff F1 Mean:               {results.no_diff_f1_mean:.3f}")
        print()
        print(f"    NoDiff Precision Std. Error:  {results.no_diff_precision_se:.3f}")
        print(f"    NoDiff Recall Std. Error:     {results.no_diff_recall_se:.3f}")
        print(f"    NoDiff F1 Std. Error:         {results.no_diff_f1_se:.3f}")
        print()
        print(f"    Diff Precision Mean:          {results.diff_precision_mean:.3f}")
        print(f"    Diff Recall Mean:             {results.diff_recall_mean:.3f}")
        print(f"    Diff F1 Mean:                 {results.diff_f1_mean:.3f}")
        print()
        print(f"    Diff Precision Std. Error:    {results.diff_precision_se:.3f}")
        print(f"    Diff Recall Std. Error:       {results.diff_recall_se:.3f}")
        print(f"    Diff F1 Std. Error:           {results.diff_f1_se:.3f}")
        print("\n")
