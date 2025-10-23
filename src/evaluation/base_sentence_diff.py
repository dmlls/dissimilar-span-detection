"""BaseSentenceDiff."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseSentenceDiff(ABC):
    """BaseSentenceDiff.

    Methods implementing SentenceDiff should inherit from this base class.
    """

    @abstractmethod
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
            start_marker (:obj:`Optional[str]`):
                The marker to signal the beginning of a differing span.
            end_marker (:obj:`Optional[str]`):
                The marker to signal the end of a differing span.
            kwargs (:obj:`Dict`):
                Additional arguments to the annotation.

        Returns:
            :obj:`str`: The hypothesis with the differing spans annotated.
        """
        pass
