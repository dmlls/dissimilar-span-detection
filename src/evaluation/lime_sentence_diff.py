"""Sentence Diff with LIME."""

from io import StringIO
from typing import Dict, List, Optional

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer, util

try:
    from base_sentence_diff import BaseSentenceDiff
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff
from config import EVAL_CONFIG, ModelType


class LimeSentenceDiff(BaseSentenceDiff):
    """LimeSentenceDiff."""

    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.model_type = ModelType.LIME
        self.model = SentenceTransformer(model, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.sep_token: str = "<&>"
        self.mask_token: str = self.model.tokenizer.mask_token
        self.class_names: List[str] = ["similar", "dissimilar"]
        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names, mask_string=self.mask_token, bow=False
        )

    def compute_similarities(
        self, premises: List[str], hypothesis: List[str]
    ) -> torch.Tensor:
        embeddings = self.model.encode(premises + hypothesis)
        return util.cos_sim(
            embeddings[: len(premises)], embeddings[len(hypothesis) :]
        ).diagonal()

    def compute_similarity(self, a: str, b: str) -> float:
        # Compute sentence embeddings.
        sim_embeddings_cands = self.model.encode([a], convert_to_tensor=True)
        sim_embeddings_refs = self.model.encode([b], convert_to_tensor=True)

        # Compute cosine similarities.
        return util.cos_sim(sim_embeddings_cands, sim_embeddings_refs).diagonal().item()

    def predict_proba(self, texts: List[str]):
        probas = []
        for text in texts:
            premise, hypothesis = text.split(self.sep_token)
            similarity = self.compute_similarity(premise, hypothesis)
            probas.append([similarity, 1.0 - similarity])
        return np.asarray(probas)

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
        dissimilarity_threshold = kwargs.get("lime_weight_threshold")
        if dissimilarity_threshold is None:
            dissimilarity_threshold = EVAL_CONFIG[self.model_type][
                "lime_weight_threshold"
            ]
        num_samples = kwargs.get("lime_num_samples")
        if num_samples is None:
            num_samples = EVAL_CONFIG[self.model_type]["lime_num_samples"]
        input_pair = f"{premise} {self.sep_token} {hypothesis}"
        exp = self.lime_explainer.explain_instance(
            input_pair,
            self.predict_proba,
            num_features=len(hypothesis.split()),
            num_samples=num_samples,
        )
        indexed_string = exp.domain_mapper.indexed_string
        sep_token_raw_idx = indexed_string.raw_string().index(self.sep_token) + len(
            self.sep_token
        )
        # Sort explanations by the order tney appear in the input pair, and
        # discard explanations that do not belong to the hypothesis.
        explanations = [
            (
                indexed_string.word(x[0]),
                indexed_string.string_position(x[0])[0] - sep_token_raw_idx - 1,
                x[1],
            )
            for x in exp.as_map()[1]
        ]
        explanations = [
            explanation
            for explanation in sorted(explanations, key=lambda x: x[1])
            if explanation[1] >= 0
        ]
        # We now get the tokenized hypothesis. Unfortunately, I didn't find a
        # neater way to do this.
        tokenized_input: List[str] = indexed_string.as_list
        tokenized_hypothesis: List[str] = []
        sep_token_found: bool = False
        for tk in tokenized_input:
            if self.sep_token in tk:
                sep_token_found = True
                continue
            if not sep_token_found:
                continue
            tokenized_hypothesis.append(tk)

        # Get diff of all hypothesis tokens.
        if not explanations:
            diff: Dict[str, float] = {tk: 0.0 for tk in tokenized_hypothesis}
        else:
            diff: Dict[str, float] = {}
            str_i = 0
            exp_i = 0
            tk_i = 0
            while tk_i < len(tokenized_hypothesis):
                token, index, weight = explanations[exp_i]
                if index == str_i:
                    diff[token] = weight
                    if exp_i + 1 < len(explanations):
                        exp_i += 1
                else:  # no weights available
                    token = tokenized_hypothesis[tk_i]
                    if token.strip():  # discard empty tokens
                        diff[token] = 0.0
                str_i += len(token)
                tk_i += 1

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
