"""Evaluation module."""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation.embedding_sentence_diff import EmbeddingSentenceDiff


class Evaluator:
    """Evaluator."""

    def __init__(self, model: str):
        self.sts_model = SentenceTransformer(model)
        self.dsd_model = EmbeddingSentenceDiff(model)

    def evaluate(
        self,
        dataset: str,
        dsd: bool,
        sts_threshold: float = None,
        dsd_threshold: float = None,
        labels: List[int] = None,
        **kwargs
    ) -> float:
        """Evaluate a model on a dataset.

        Args:
            dataset (:obj:`str`):
                The dataset to evaluate the model on.
            dsd (:obj:`bool`):
                Whether to run the evaluation applying Dissimilar Span Detection
                (DSD) or not. If set to ``False`` only Semantic Textual
                Similarity (STS) is used for the predictions.

        Returns:
            :obj:`float`: The accuracy of the model on the dataset.
        """
        if sts_threshold is None:
            sts_threshold = 0.5
        sentences_a, sentences_b, loaded_labels = self._load_samples(dataset, labels)
        sts_scores = self._calculate_sts(sentences_a, sentences_b)
        if not dsd:
            predictions: List[int] = [
                int(score >= sts_threshold) for score in sts_scores
            ]
        else:
            predictions = []
            for sent_a, sent_b, sts in zip(sentences_a, sentences_b, sts_scores):
                annotation: str = self.dsd_model.annotate_diff(
                    sent_a, sent_b, token_diff_threshold=dsd_threshold
                )
                # print(annotation)
                predictions.append(int(sts >= sts_threshold and "{{" not in annotation))
        accuracy: float = sum(
            int(prediction == reference)
            for prediction, reference in zip(predictions, loaded_labels)
        ) / len(loaded_labels)
        return accuracy

    def _load_samples(
        self, dataset: str, labels: List[int] = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """Load the the dataset."""
        if labels is None:
            labels = ["0", "1"]
        sentences_a: List[str] = []
        sentences_b: List[str] = []
        loaded_labels: List[str] = []
        with open(Path(dataset), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for line in reader:
                _, sentence_a, sentence_b, label = line
                if int(label) in labels:
                    sentences_a.append(sentence_a)
                    sentences_b.append(sentence_b)
                    loaded_labels.append(int(label))

        return sentences_a, sentences_b, loaded_labels

    def _calculate_sts(self, sentences_a: List[str], sentences_b: List[str]):
        if len(sentences_a) != len(sentences_b):
            raise ValueError("Both sentence lists must have the same length.")

        embeddings_a = self.sts_model.encode(sentences_a, convert_to_tensor=True)
        embeddings_b = self.sts_model.encode(sentences_b, convert_to_tensor=True)

        cosine_similarities = self.sts_model.similarity(embeddings_a, embeddings_b)

        return cosine_similarities.diagonal()
