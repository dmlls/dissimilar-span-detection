"""Data loading."""

import csv
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .models import RawTextPair, Span, SpanPair, TextPair
except ImportError:
    from models import RawTextPair, Span, SpanPair, TextPair


class TextPairDataLoader(ABC):
    """High-level sentence pair data loader.

    Concrete data loaders should inherit from this class.
    """

    @classmethod
    @abstractmethod
    def load(
        cls,
        dataset_path: str,
        **kwargs,
    ) -> Iterator[TextPair]:
        """Load the sentence pairs from a dataset.

        Args:
            dataset_path (:obj:`dataset`):
                The path to the dataset to load the pairs from.

        Returns:
            :obj:`Iterator[TextPair]`: An iterator to the text pairs.
        """

    @classmethod
    @abstractmethod
    def load_raw(
        cls,
        dataset_path: str,
        **kwargs,
    ) -> Iterator[RawTextPair]:
        """Load the raw sentence pairs from a dataset.

        Args:
            dataset_path (:obj:`dataset`):
                The path to the dataset to load the pairs from.

        Returns:
            :obj:`Iterator[RawTextPair]`: An iterator to the text pairs.
        """

    @classmethod
    def _remove_span_markers(
        cls, sentence: str, markers: Optional[List[str]] = None
    ) -> str:
        """Remove the characters that mark the start and end of a span in a sentence.

        Args:
            sentence (:obj:`str`):
                The sentence to remove the span markers from.
            markers (:obj:`Optional[List[str]]`, defaults to ``["{", "}"]``):
                A list containing two elements:

                   * The marker signaling the beginning of the span.
                   * The marker signaling the end of the span.

        Returns:
            :obj:`str`: The input sentence without the span markers.
        """
        if markers is None:
            markers = ["{", "}"]
        for marker in markers:
            sentence = sentence.replace(marker, "")
        return sentence


class SSDTextPairDataLoader(TextPairDataLoader):
    """Text pair data loader for the Span Similarity Dataset."""

    @classmethod
    def load(
        cls, dataset_path: str, skip_header: bool = True, **kwargs
    ) -> Iterator[TextPair]:
        """Load the text pairs from the dataset.

        See base class.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            if skip_header:
                try:
                    next(reader)  # skip header
                except StopIteration:
                    return  # empty file
            for line in reader:
                try:
                    premise, hypothesis, span_labels, text_label = line
                except ValueError:
                    premise, hypothesis, span_labels = line
                    text_label = None
                spans_premise = cls.get_spans(premise)
                spans_hypothesis = cls.get_spans(hypothesis)
                span_labels = cls._get_span_labels(span_labels)
                span_pairs = [
                    SpanPair(
                        premise_span=premise_span,
                        hypothesis_span=hypothesis_span,
                        label=span_label,
                    )
                    for premise_span, hypothesis_span, span_label in zip(
                        spans_premise, spans_hypothesis, span_labels
                    )
                ]
                yield TextPair(
                    premise=cls._remove_span_markers(premise),
                    hypothesis=cls._remove_span_markers(hypothesis),
                    span_pairs=span_pairs,
                    label=text_label,
                )

    @classmethod
    def load_raw(
        cls, dataset_path: str, skip_header: bool = True, **kwargs
    ) -> Iterator[RawTextPair]:
        """Load the raw text pairs from the dataset.

        See base class.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            if skip_header:
                try:
                    next(reader)  # skip header
                except StopIteration:
                    return  # empty file
            for line in reader:
                try:
                    premise, hypothesis, raw_span_labels, _ = line
                except ValueError:  # sentence label not included
                    premise, hypothesis, raw_span_labels = line
                span_labels = [int(label) for label in raw_span_labels.split(",")]
                yield RawTextPair(
                    premise=premise,
                    hypothesis=hypothesis,
                    span_labels=span_labels,
                )

    @classmethod
    def get_spans(cls, string_: str) -> List[Span]:
        """Get annotated spans in a string."""
        span_pattern = re.compile(r"{{.+?}}")
        start_indexes = [index for index, char in enumerate(string_) if char == "{"]
        # Don't consider span markers in indexes.
        start_indexes = [
            index - i * 2 for i, index in enumerate(start_indexes) if i % 2 == 0
        ]
        end_indexes = [index for index, char in enumerate(string_) if char == "}"]
        end_indexes = [
            index - i * 2 for i, index in enumerate(end_indexes, 1) if i % 2 != 0
        ]
        return [
            Span(
                start_index=start_index,
                end_index=end_index,
                text=cls._remove_span_markers(span_text),
            )
            for start_index, end_index, span_text in zip(
                start_indexes, end_indexes, span_pattern.findall(string_)
            )
        ]

    @classmethod
    def _get_span_labels(cls, string_: str) -> List[int]:
        """Get the span labels from a string."""
        return list(string_.split(","))


class SSDSentenceDataLoader(TextPairDataLoader):
    """Sentence data loader for the Span Similarity Dataset."""

    @classmethod
    def load(cls, dataset_path: str, **kwargs) -> List[TextPair]:
        """Load the sentence pairs from the dataset.

        See base class.
        """
        with open(Path(dataset_path), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            return [
                TextPair(
                    premise=cls._remove_span_markers(premise),
                    hypothesis=cls._remove_span_markers(hypothesis),
                    span_pairs=[],
                    label=sentence_label,
                )
                for premise, hypothesis, _, sentence_label in reader
            ]


class SSDSpanDataLoader(TextPairDataLoader):
    """Span data loader for the Span Similarity Dataset."""

    @classmethod
    def load(cls, dataset_path: str, **kwargs) -> List[TextPair]:
        """Load the span pairs from the dataset.

        See base class.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            input_examples = []
            span_pattern = re.compile(r"{{.+?}}")
            for premise, hypothesis, span_labels, _ in reader:
                spans_premise = cls._get_spans(premise, span_pattern)
                spans_hypothesis = cls._get_spans(hypothesis, span_pattern)
                span_labels = cls._get_span_labels(span_labels)
                for span_premise, span_hypothesis, span_label in zip(
                    spans_premise, spans_hypothesis, span_labels
                ):
                    input_examples.append(
                        TextPair(
                            premise=span_premise,
                            hypothesis=span_hypothesis,
                            label=span_label,
                        )
                    )
        return input_examples

    @classmethod
    def _get_spans(
        cls, string_: str, span_pattern: Optional[re.Pattern] = None
    ) -> List[str]:
        """Get annotated spans in a string."""
        if span_pattern is None:
            span_pattern = re.compile(r"{{.+?}}")
        return [
            cls._remove_span_markers(span) for span in span_pattern.findall(string_)
        ]

    @classmethod
    def _get_span_labels(cls, string_: str) -> List[int]:
        """Get span labels from a string."""
        return list(string_.split(","))


class KFoldDataLoader:
    """Data loader suitable for k-fold cross-validation."""

    @classmethod
    @abstractmethod
    def load(
        cls,
        dataset_path: str,
        k_folds: int,
    ) -> Iterator[Tuple[np.ndarray[TextPair], np.ndarray[TextPair]]]:
        """Load the sentence pairs from a dataset.

        Args:
            dataset_path (:obj:`dataset`):
                The path to the dataset to load the pairs from.

        Returns:
            :obj:`Iterator[Tuple[np.ndarray[TextPair], np.ndarray[TextPair]]`]:
            An iterator yielding a tuple containing the split for the training
            data (X) and the test data (y), for each of the k-folds.
        """
        assert k_folds > 0, "`k_folds` must be a positive number"
        loaded_samples: np.ndarray[TextPair] = np.array(
            list(SSDTextPairDataLoader.load(dataset_path))
        )
        split_samples = np.array_split(loaded_samples, k_folds)
        for k in range(k_folds):
            test_data: np.ndarray[TextPair] = split_samples[k]
            train_data = split_samples.copy()
            del train_data[k]  # delete test split
            train_data: np.ndarray[TextPair] = np.concatenate(train_data, axis=0)
            yield (train_data, test_data)

    @classmethod
    @abstractmethod
    def load_raw(
        cls,
        dataset_path: str,
        k_folds: int,
    ) -> Iterator[Tuple[np.ndarray[RawTextPair], np.ndarray[RawTextPair]]]:
        """Load the raw sentence pairs from a dataset.

        Args:
            dataset_path (:obj:`dataset`):
                The path to the dataset to load the pairs from.

        Returns:
            :obj:`Iterator[Tuple[np.ndarray[RawTextPair], np.ndarray[RawTextPair]]`]:
            An iterator yielding a tuple containing the split for the training
            data (X) and the test data (y), for each of the k-folds.
        """
        assert k_folds > 0, "`k_folds` must be a positive number"
        loaded_samples: np.ndarray[RawTextPair] = np.array(
            list(SSDTextPairDataLoader.load_raw(dataset_path))
        )
        split_samples = np.array_split(loaded_samples, k_folds)
        for k in range(k_folds):
            test_data: np.ndarray[RawTextPair] = split_samples[k]
            train_data = split_samples.copy()
            del train_data[k]  # delete test split
            train_data: np.ndarray[RawTextPair] = np.concatenate(train_data, axis=0)
            yield (train_data, test_data)
