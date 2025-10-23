"""Span Similarity Evaluation."""

import itertools
import json
import string
import sys
import time
from abc import ABC, abstractmethod
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import nltk
from baselines_sentence_diff import NaiveSentenceDiff, NoSentenceDiff
from config import EVAL_CONFIG, ModelType
from embedding_sentence_diff import EmbeddingSentenceDiff
from lime_sentence_diff import LimeSentenceDiff
from llm_sentence_diff import LLMProvider, LLMSentenceDiff
from models import EvaluationResults
from shap_sentence_diff import ShapSentenceDiff
from token_classification_sentence_diff import TokenClassificationSentenceDiff
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.data_loading import KFoldDataLoader, TextPairDataLoader
from dataset.models import Span
from training.token_classification.train_cross_eval import Trainer


class SpanSimilarityEvaluator(ABC):
    """High-level sentence similarity evaluator.

    Concrete data loaders should inherit from this class.

    Attributes:
        model_type(:obj:`ModelType`):
            The type of the model
        model_name (:obj:`str`):
            The method or model to initialize SentenceDiff with.
        data_loader (:obj:`TextPairDataLoader`):
            The data loader used to load the dataset.
        logging_path (:obj:`Union[pathlib.Path, None]`):
            The path where the evaluation logs will be saved. Set to :obj:`None`
            to disable logging.
    """

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        data_loader: TextPairDataLoader,
        logging_path: Union[str, None] = None,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.data_loader = data_loader
        self.logging_path = Path(logging_path) if logging_path else None
        self.set_up_logging_path()
        if self.model_type is ModelType.BASELINE:
            if self.model_name == "no-sentence-diff":
                self.model = NoSentenceDiff()
            elif self.model_name == "naive-sentence-diff":
                self.model = NaiveSentenceDiff()
            else:
                raise NotImplementedError(
                    f"baseline method '{self.model_name}' not supported."
                    " Supported methods are 'no-sentence-diff' and"
                    " 'naive-sentence-diff'"
                )
        elif self.model_type is ModelType.SENTENCE_TRANSFORMER:
            self.model = EmbeddingSentenceDiff(self.model_name)
        elif self.model_type is ModelType.SHAP:
            self.model = ShapSentenceDiff(self.model_name)
        elif self.model_type is ModelType.LIME:
            self.model = LimeSentenceDiff(self.model_name)
        elif self.model_type is ModelType.TRANSFORMER:
            self.model = None  # updated later
        elif self.model_type.value in [provider.value for provider in LLMProvider]:
            self.model = LLMSentenceDiff(
                LLMProvider(self.model_type.value), self.model_name
            )
        else:
            raise NotImplementedError(
                "looks like you forgot to add the model initialization code"
            )

    @abstractmethod
    def evaluate(
        self,
        dataset_path: str,
        **kwargs,
    ) -> EvaluationResults:
        """Evaluate initialized model on the Span Similarity Dataset.

        Args:
            dataset_path (:obj:`str`):
                The path to the dataset.
            **kwargs (:obj:`Dict`):
                The model-specific arguments to pass to the model.
        """

    def set_up_logging_path(self) -> None:
        """Create necessary directories for evaluation logging."""
        if self.logging_path:
            model_name = self.model_name.replace("/", "_")
            # Create a dir named after the model and the timestamp.
            self.logging_path /= (
                f'{self.model_type.value}_{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
            )
            self.logging_path.mkdir(parents=True, exist_ok=True)

    def format_log_step_header(self, *score_names: Tuple[str, ...]) -> str:
        """Format the header of the evaluation step log file.

        Args:
            score_names (:obj:`Tuple[str, ...]`):
                The names of the scores.

        Returns:
            :obj:`str`: A ``.tsv`` suitable header.
        """
        formatted_score_names = "\t".join(score_names)
        return f"premise\thypothesis\treference_spans\tpredicted_spans\t{formatted_score_names}\n"

    def format_log_step(
        self,
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

    def get_unigrams(
        self,
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


class SSDSpanSimilarityEvaluator(SpanSimilarityEvaluator):
    """Span similarity evaluator for the Span Similarity Dataset."""

    def evaluate(
        self,
        dataset_path: str,
        k_folds: Optional[int] = None,
        **kwargs,
    ) -> EvaluationResults:
        """See base class."""
        if not k_folds:
            k_folds = 1  # evaluate on the entire dataset

        total_elapsed_time: float = 0.0
        k_results: List[EvaluationResults] = []
        for k, (_, test_data) in enumerate(
            KFoldDataLoader.load(
                dataset_path=dataset_path,
                k_folds=k_folds,
            ),
            1,
        ):
            print(f"\n\n{'='*12}  FOLD {k}  {'='*12}\n")
            logging_path = self.logging_path / f"{k:02}"
            logging_path.mkdir(parents=True, exist_ok=True)
            # Supervised models: Train first on the training folds.
            if self.model_type is ModelType.TRANSFORMER:
                training_arguments = EVAL_CONFIG[self.model_type]["training_arguments"]
                training_arguments.output_dir = str(logging_path / "trained_model")
                trainer = Trainer(
                    base_model=self.model_name,
                    dataset_path=Path(dataset_path),
                    training_args=training_arguments,
                )
                print("\n  >>> Running training...")
                trained_model_path = trainer.train(k, k_folds)
                self.model = TokenClassificationSentenceDiff(trained_model_path)

            # Scores for sentence pairs with no dissimilar spans.
            precisions_no_diff: List[float] = []
            recalls_no_diff: List[float] = []
            f1s_no_diff: List[float] = []
            # Scores for sentence pairs with dissimilar spans.
            precisions_diff: List[float] = []
            recalls_diff: List[float] = []
            f1s_diff: List[float] = []

            step_logs = None
            if self.logging_path:
                step_logs = open(
                    logging_path / "step_logs.tsv",
                    "w",
                    encoding="utf-8",
                )
                step_logs.write(
                    self.format_log_step_header("precision", "recall", "f1")
                )

            start_time = time.perf_counter()

            for pair in tqdm(test_data, ncols=60):
                premise = pair.premise
                hypothesis = pair.hypothesis
                spans_premise: List[Span] = [
                    span.premise_span for span in pair.span_pairs if span.label == "0"
                ]
                spans_hypothesis: List[Span] = [
                    span.hypothesis_span
                    for span in pair.span_pairs
                    if span.label == "0"
                ]
                # We get predictions for both premise and hypothesis.
                for _ in range(2):
                    prediction = self.model.annotate_diff(
                        premise=premise,
                        hypothesis=hypothesis,
                        start_marker="{{",
                        end_marker="}}",
                        **kwargs,
                    )
                    if prediction:
                        predicted_spans: List[Span] = self.data_loader.get_spans(
                            prediction
                        )
                        aligned_spans = self.align_spans(
                            spans_hypothesis, predicted_spans
                        )
                        aligned_spans_unigrams: List[
                            Tuple[Optional[List[str]], Optional[List[str]]]
                        ] = []
                        for span_premise, span_hypothesis in aligned_spans:
                            span_premise_unigrams = (
                                self.get_unigrams(span_premise.text)
                                if span_premise
                                else None
                            )
                            span_hypothesis_unigrams = (
                                self.get_unigrams(span_hypothesis.text)
                                if span_hypothesis
                                else None
                            )
                            aligned_spans_unigrams.append(
                                (span_premise_unigrams, span_hypothesis_unigrams)
                            )
                        precision, recall, f1 = self.calculate_precision_recall_f1(
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
                        self.format_log_step(
                            premise,
                            hypothesis,
                            [
                                "{{" + span.text + "}}"
                                for span in spans_hypothesis
                                if span
                            ],
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
            total_elapsed_time += elapsed_seconds

            if step_logs:
                step_logs.close()

            mean_no_diff_precision = (
                mean(precisions_no_diff) if precisions_no_diff else 0.0
            )
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
            se_recall = (
                pstdev(all_recalls) / sqrt(len(all_recalls)) if all_recalls else 0.0
            )
            all_f1s: List[float] = f1s_no_diff + f1s_diff
            mean_f1 = mean(all_f1s) if all_f1s else 0.0
            se_f1 = pstdev(all_f1s) / sqrt(len(all_f1s)) if all_f1s else 0.0

            results: EvaluationResults = EvaluationResults(
                k_fold=k,
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
            if self.logging_path:
                self.log_results(
                    path=logging_path / "results.json",
                    elapsed_seconds=elapsed_seconds,
                    results=results,
                    **kwargs,
                )
            k_results.append(results)
            self.print_results(results)

        k_precisions = [r.precision_mean for r in k_results]
        k_recalls = [r.recall_mean for r in k_results]
        k_f1s = [r.f1_mean for r in k_results]

        k_no_diff_precisions = [r.no_diff_precision_mean for r in k_results]
        k_no_diff_recalls = [r.no_diff_recall_mean for r in k_results]
        k_no_diff_f1s = [r.no_diff_f1_mean for r in k_results]

        k_diff_precisions = [r.diff_precision_mean for r in k_results]
        k_diff_recalls = [r.diff_recall_mean for r in k_results]
        k_diff_f1s = [r.diff_f1_mean for r in k_results]

        mean_results: EvaluationResults = EvaluationResults(
            precision_mean=mean(k_precisions),
            recall_mean=mean(k_recalls),
            f1_mean=mean(k_f1s),
            precision_se=pstdev(k_precisions) / sqrt(k_folds),
            recall_se=pstdev(k_recalls) / sqrt(k_folds),
            f1_se=pstdev(k_f1s) / sqrt(k_folds),
            no_diff_precision_mean=mean(k_no_diff_precisions),
            no_diff_recall_mean=mean(k_no_diff_recalls),
            no_diff_f1_mean=mean(k_no_diff_f1s),
            no_diff_precision_se=pstdev(k_no_diff_precisions) / sqrt(k_folds),
            no_diff_recall_se=pstdev(k_no_diff_recalls) / sqrt(k_folds),
            no_diff_f1_se=pstdev(k_no_diff_f1s) / sqrt(k_folds),
            diff_precision_mean=mean(k_diff_precisions),
            diff_recall_mean=mean(k_diff_recalls),
            diff_f1_mean=mean(k_diff_f1s),
            diff_precision_se=pstdev(k_diff_precisions) / sqrt(k_folds),
            diff_recall_se=pstdev(k_diff_recalls) / sqrt(k_folds),
            diff_f1_se=pstdev(k_diff_f1s) / sqrt(k_folds),
            config=kwargs,
        )
        if self.logging_path:
            logging_path: Path = self.logging_path / "mean"
            logging_path.mkdir(parents=True, exist_ok=True)
            self.log_results(
                path=logging_path / "results.json",
                elapsed_seconds=total_elapsed_time,
                results=mean_results,
                **kwargs,
            )
        print(f"\n\n{'='*9}  MEAN RESULTS  {'='*9}\n")
        self.print_results(mean_results)

    def align_spans(
        self, spans_premise: List[Span], spans_hypothesis: List[Span]
    ) -> List[Tuple[Optional[Span], Optional[Span]]]:
        spans_premise, spans_hypothesis = self._fill_span_lists(
            spans_premise, spans_hypothesis
        )
        all_hypothesis: Set[List[Span]] = set(itertools.permutations(spans_hypothesis))
        best_offset_score: int = 999999  # minimize
        best_alignment: List[Optional[Span]] = [None] * len(spans_premise)
        for candidate_hypothesis in all_hypothesis:
            offset_score: int = sum(
                self._calculate_span_offset(span_premise, span_hypothesis)
                for span_premise, span_hypothesis in zip(
                    spans_premise, candidate_hypothesis
                )
            )
            if offset_score < best_offset_score:
                best_offset_score = offset_score
                best_alignment = candidate_hypothesis
        return list(zip(spans_premise, best_alignment))

    def _fill_span_lists(
        self,
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

    def _calculate_span_offset(self, span_premise: Span, span_hypothesis: Span) -> int:
        if span_premise is None and span_hypothesis is None:
            return 0
        if span_premise is None or span_hypothesis is None:
            return 999
        return abs(span_premise.start_index - span_hypothesis.start_index) + abs(
            span_premise.end_index - span_hypothesis.end_index
        )

    def calculate_recall(
        self, aligned_spans: List[Tuple[Optional[List[str]], Optional[List[str]]]]
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

    def calculate_precision_recall_f1(
        self,
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

    def log_results(
        self,
        path: Path,
        elapsed_seconds: float,
        results: EvaluationResults,
        **kwargs,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": self.model_type,
                    "model_name": self.model_name,
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
                    "config": {
                        k: v
                        for k, v in kwargs.items()
                        if self.model_type in EVAL_CONFIG
                        and k in EVAL_CONFIG[self.model_type]
                    },
                },
                f,
                indent=4,
            )

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
