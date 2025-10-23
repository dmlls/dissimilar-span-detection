"""Baseline Sentence Diff methods."""

from io import StringIO
from typing import Optional

try:
    from base_sentence_diff import BaseSentenceDiff
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff


class NoSentenceDiff(BaseSentenceDiff):
    """NoSentenceDiff.

    This class returns all hypothesis unchanged, i.e., with no annotated spans.
    """

    def annotate_diff(
        self,
        premise: str,
        hypothesis: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        **kwargs,
    ) -> str:
        return hypothesis


class NaiveSentenceDiff(BaseSentenceDiff):
    """NaiveSentenceDiff.

    This class includes within dissimilar spans any words in the hypothesis that
    are not in the premise, no matter whether these differing words are actually
    semantically dissimilar or not.
    """

    def annotate_diff(
        self,
        premise: str,
        hypothesis: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        **kwargs,
    ) -> str:
        words_premise = premise.split()
        words_hypothesis = hypothesis.split()
        annotated_hypothesis = StringIO()
        in_span: bool = False
        for word in words_hypothesis:
            if word not in words_premise:
                if not in_span:
                    in_span = True
                    annotated_hypothesis.write(f"{{{{ {word} ")
                else:
                    annotated_hypothesis.write(f" {word} ")
            elif in_span:
                in_span = False
                annotated_hypothesis.write(f"}}}} {word}")
            else:
                annotated_hypothesis.write(f" {word} ")
        if in_span:
            in_span = False
            annotated_hypothesis.write(" }}")
        return " ".join(annotated_hypothesis.getvalue().replace("}}{{", "").split())
