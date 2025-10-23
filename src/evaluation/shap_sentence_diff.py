"""Sentence Diff with SHAP."""

from io import StringIO
from typing import List, Optional

import shap
import torch
from sentence_transformers import SentenceTransformer, util

try:
    from base_sentence_diff import BaseSentenceDiff
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff
from config import EVAL_CONFIG, ModelType


class ShapSentenceDiff(BaseSentenceDiff):
    """ShapSentenceDiff."""

    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.model_type = ModelType.SHAP
        self.model = SentenceTransformer(model, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.shap_explainer = shap.Explainer(self._f, self.tokenizer)

    def compute_similarities(
        self, premises: List[str], hypothesis: List[str]
    ) -> torch.Tensor:
        embeddings = self.model.encode(premises + hypothesis)
        return util.cos_sim(
            embeddings[: len(premises)], embeddings[len(hypothesis) :]
        ).diagonal()

    def annotate_diff(
        self,
        premise: str,
        hypothesis: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Annotate the differing spans in the hypothesis, compared to a premise.

        Args:
            premise (:obj:`str`):
                The first sentence.
            hypothesis (:obj:`str`):
                The second sentence.
            start_marker (:obj:`Optional[str]`, defaults to ``{{``):
                The marker to signal the beginning of a differing span.
            end_marker (:obj:`Optional[str]`, defaults to ``}}``):
                The marker to signal the end of a differing span.
            kwargs (:obj:`Dict`):
                Additional arguments to the annotation. Possible values are:

                   * dissimilarity_threshold (:obj:`Optional[float]`, defaults to ``0.4``):
                         The minimum dissimilarity (inclusive) a unigram must
                         have in order to be annotated as dissimilar.

        Returns:
            :obj:`str`: The hypothesis with the differing spans annotated.
        """
        if start_marker is None:
            start_marker = "{{"
        if end_marker is None:
            end_marker = "}}"
        dissimilarity_threshold = kwargs.get("shap_value_threshold")
        if dissimilarity_threshold is None:
            dissimilarity_threshold = EVAL_CONFIG[self.model_type][
                "shap_value_threshold"
            ]
        input_pair = f"{premise}{self.tokenizer.sep_token}{hypothesis}"
        shap_values = self.shap_explainer([input_pair])

        # Get only the hypothesis SHAP values and tokens.
        values = shap_values.values[0]
        tokens = shap_values.data[0].tolist()
        # Index where the hypothesis begins.
        start_index = tokens.index(self.tokenizer.sep_token) + 1
        diff = {
            tk.strip(): value
            for tk, value in zip(tokens[start_index:], values[start_index:])
            if tk.strip()
        }
        in_span: bool = False  # are we inside a span?
        annotated_hypothesis = StringIO()
        for token, dissimilarity in diff.items():
            if not in_span and dissimilarity >= dissimilarity_threshold:
                annotated_hypothesis.write(f" {start_marker}")
                in_span = True
            elif in_span and dissimilarity < dissimilarity_threshold:
                annotated_hypothesis.write(f"{end_marker} ")
                in_span = False
            else:
                annotated_hypothesis.write(" ")
            annotated_hypothesis.write(token)
        if in_span:
            annotated_hypothesis.write(end_marker)
            in_span = False
        return annotated_hypothesis.getvalue().strip()

    def _f(self, x) -> torch.Tensor:
        premises: List[str] = []
        hypotheses: List[str] = []
        # Collect premises and hypotheses.
        for x_ in x:
            p, h = x_.split(self.tokenizer.sep_token)
            premises.append(p)
            hypotheses.append(h)
        dissimilarities: torch.Tensor = 1 - self.compute_similarities(
            premises, hypotheses
        )
        return dissimilarities
