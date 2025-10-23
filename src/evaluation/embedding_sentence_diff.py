"""EmbeddingSentenceDiff."""

from io import StringIO
from typing import Callable, Dict, FrozenSet, List, Optional

import nltk
import torch
from coloraide import Color
from IPython.display import HTML, display
from sentence_transformers import SentenceTransformer, util

try:
    from base_sentence_diff import BaseSentenceDiff
    from config import EVAL_CONFIG, ModelType
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff
    from .config import EVAL_CONFIG, ModelType


class EmbeddingSentenceDiff(BaseSentenceDiff, SentenceTransformer):
    """EmbeddingSentenceDiff

    Computes local meaning differences between sentences."""

    def __init__(self, model: str, **kwargs):
        # nltk.download("punkt")
        self.model_name = model
        self.model_type = ModelType.SENTENCE_TRANSFORMER
        i = Color.interpolate(["#34ff00", "#ff2b00"], space="hsl", hue="shorter")
        self.steps = 100
        self.colors = [
            i(x / self.steps).convert("srgb").to_string(hex=True)
            for x in range(self.steps + 1)
        ]
        kwargs["trust_remote_code"] = True
        super().__init__(model, **kwargs)

    def diff(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculate meaning differences between sentences.

        Args:
            premise (:obj:`str`):
                The first sentence.
            hypothesis (:obj:`str`):
                The second sentence.

        Returns:
            :obj:`Dict[str, float]`: The per-token differences in meaning. The
            keys of the dictionary are the unigrams (as returned by
            :meth:`get_unigrams`), and the values, which range between
            ``[0, 1]``, the difference. The higher the number, the more
            different a token in the hypothesis is with respect to the premise.
        """
        premise_unigrams = self.get_unigrams(premise)
        hypothesis_unigrams = self.get_unigrams(hypothesis)
        base_similarity = util.cos_sim(
            *self._encode(
                [" ".join(premise_unigrams), " ".join(hypothesis_unigrams)],
                convert_to_tensor=True,
            )
        )
        replacements = self.get_replacements(premise, hypothesis)
        hypothesis_strings_only = [
            repl for repls in replacements.values() for repl in repls
        ]
        hypothesis_embeddings = self._encode(
            hypothesis_strings_only, convert_to_tensor=True
        )
        premise_embeddings = self._encode(
            [" ".join(premise_unigrams)], convert_to_tensor=True
        )
        # Repeat as many times as strings in hypothesis_strings_only.
        premise_embeddings = premise_embeddings.expand(
            len(hypothesis_strings_only), premise_embeddings.shape[1]
        )
        assert premise_embeddings.shape == hypothesis_embeddings.shape
        # Compute cosine similarities.
        new_similarities = util.cos_sim(
            premise_embeddings, hypothesis_embeddings
        ).diagonal()

        gains = {}
        start = 0
        end = 0
        for word_idxs, repl in replacements.items():
            end += len(repl)
            max_sim_gain = self.calculate_max_sim_gain(
                base_similarity, new_similarities[start:end]
            ).item()
            for word_idx in word_idxs:
                gains.setdefault(word_idx, []).append(max_sim_gain)
            start = end
        aggregated_gains = {
            unigram: self.exp_aggregation_function(values)
            for unigram, values in zip(hypothesis_unigrams, gains.values())
        }
        return aggregated_gains

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
        dissimilarity_threshold = kwargs.get("token_diff_threshold")
        if dissimilarity_threshold is None:
            dissimilarity_threshold = EVAL_CONFIG[self.model_type][
                "token_diff_threshold"
            ]
        diff = self.diff(premise, hypothesis)
        in_span: bool = False  # are we inside a span?
        annotated_hypothesis = StringIO()
        for unigram, dissimilarity in diff.items():
            if not in_span and dissimilarity >= dissimilarity_threshold:
                annotated_hypothesis.write(f" {start_marker}")
                in_span = True
            elif in_span and dissimilarity < dissimilarity_threshold:
                annotated_hypothesis.write(f"{end_marker} ")
                in_span = False
            else:
                annotated_hypothesis.write(" ")
            annotated_hypothesis.write(unigram)
        if in_span:
            annotated_hypothesis.write(end_marker)
            in_span = False
        return annotated_hypothesis.getvalue().strip()

    def _encode(self, strings_: List[str], **kwargs) -> torch.Tensor:
        """Encode strings.

        Args:
            strings_ (:obj:`List[str]`):
                The strings to encode.

        Returns:
            :obj:`torch.Tensor`: The encoded strings.
        """
        # Sentence Transformers
        return super().encode(strings_, convert_to_tensor=True)

        # Open AI
        # from embeddings import OpenAIEmbeddingGenerator
        # return OpenAIEmbeddingGenerator.generate(strings_)

        # Cohere
        # from embeddings import CohereEmbeddingGenerator
        # return CohereEmbeddingGenerator.generate(strings_)

        # Google
        # from embeddings import GoogleEmbeddingGenerator
        # return GoogleEmbeddingGenerator.generate(strings_)

    def exp_aggregation_function(self, scores: List[torch.Tensor]) -> float:
        """Aggregate scores across n-grams.

        Args:
            values (:obj:`List[torch.Tensor]):
                The difference scores of the different n-grams.
        """
        return 1 / len(scores) * sum(x / i for i, x in enumerate(scores, 1))

    def calculate_max_sim_gain(
        self, base_similarity: torch.Tensor, new_similarities: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the maximum similarity gain between similarities.

        Args:
            base_similarity (:obj:`torch.Tensor`):
                The base similarity score.
            new_similarities (:obj:`torch.Tensor`):
                The new similarity scores that come from altering the original
                sentence.
        """
        return torch.max(new_similarities - base_similarity)

    def get_unigrams(
        self,
        sentence: str,
        tokenization_method: Optional[Callable[[str], List[str]]] = None,
    ) -> List[str]:
        """Split a sentence into unigrams.

        Args:
            sentence (:obj:`str`):
                The sentence to get the unigrams from.
            tokenization_method (:obj:`Optional[Callable[[str], List[str]]]`, `optional`):
                The tokenization method to use in order to get the unigrams.

        Returns:
            :obj:`List[str]`: The sentence unigrams.
        """
        if tokenization_method is None:
            tokenization_method = nltk.tokenize.word_tokenize
        return tokenization_method(sentence)

    def get_n_grams(self, sentence: str, n_gram_size: int) -> List[List[str]]:
        """Get the n-grams of a sentence.

        :param:`n_gram_size` must be at least ``1``. If :param:`n_gram_size` is
        set to a value larger than the number of unigrams present in the
        sentence, an n_gram containing as many unigrams as the sentence has is
        returned.

        Args:
            sentence (:obj:`str`):
                The sentence to get the n-grams from.
            n_gram_size (:obj:`int`):
                The size of the n-grams to split the sentence into.

        Returns:
            :obj:`List[List[str]]`: A list containing the n-grams of the
            specified size, returned as list of strings.
        """
        if n_gram_size < 1:
            raise ValueError("n_gram_size cannot be smaller than 1")
        unigrams = self.get_unigrams(sentence)
        if n_gram_size > len(unigrams):
            n_gram_size = len(unigrams)
        n_grams = [unigrams[i : i + n_gram_size] for i, _ in enumerate(unigrams)]
        # We remove the list of unigrams that have less unigrams than required
        # by n_gram_size.
        return [n_gram for n_gram in n_grams if len(n_gram) == n_gram_size]

    def get_replacements(
        self, premise: str, hypothesis: str
    ) -> Dict[FrozenSet[int], List[str]]:
        """Replace each n-grams in the hypothesis by an n-grams from the premise.

        All possible (contiguous) n-grams are considered.

        Args:
            premise (:obj:`str`):
                The premise.
            hypothesis (:obj:`str`):
                The hypothesis in which unigrams will be replaced by unigrams
                coming from the premise.

        Returns:
            :obj:`Dict[FrozenSet[int], List[str]]`: A dictionary whose keys are
            the indexes of the replaced n-grams, and the values are the
            resultant sentences when replacing those n-grams.
        """
        b_unigrams = self.get_unigrams(hypothesis)
        replacements = {}
        for n_gram_size, _ in enumerate(b_unigrams, 1):
            a_n_grams = self.get_n_grams(premise, n_gram_size)
            for n_gram in a_n_grams:
                for i in range(0, len(b_unigrams) - len(n_gram) + 1):
                    replaced_idx: List[int] = []  # indexes of the replaced unigrams
                    replaced = b_unigrams[:]
                    for j, gram in enumerate(n_gram):
                        replaced_idx.append(i + j)
                        replaced[i + j] = gram
                    replacements.setdefault(frozenset(replaced_idx), []).append(
                        " ".join(str(repl) for repl in replaced)
                    )
        return replacements

    def _normalize_values(self, values, upper_bound=255):
        max_value = max(values)
        min_value = min(values)
        if min_value == max_value:  # avoid division by zero
            if max_value <= upper_bound:
                return [int(value) for value in values]
            return [upper_bound for _ in values]
        return [
            int(((value - min_value) / (max_value - min_value)) * upper_bound)
            for value in values
        ]

    def render_diff(self, a: str, b: str, html_tag: str = "span"):
        diff = self.diff(a, b)
        b_unigrams = self.get_unigrams(b)
        string = StringIO()
        normalized_values = self._normalize_values(list(diff.values()), self.steps)
        string.write(f"<{html_tag}>{a}</{html_tag}><br>")
        for unigram, value in zip(b_unigrams, normalized_values):
            string.write(
                f'<{html_tag} style="color: {self.colors[value]};">'
                f"{unigram} "
                f"</{html_tag}>"
            )
        display(HTML(string.getvalue().strip()))
