"""Data models for data loading."""

from typing import List, Optional

from pydantic import BaseModel


class HashableBaseModel(BaseModel):
    """A hashable Pydantic model.

    Meant to be inherited from other models that need to be hashable.
    """

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Span(HashableBaseModel):
    """Span.

    Attributes:
        start_index (:obj:`int`):
            The index where the span begins within the sentence (without
            considering span markers).
        end_index (:obj:`int`):
            The index where the span finishes within the sentence (without
            considering span markers).
        text (:obj:`str`):
            The text within the span.
    """

    start_index: int
    end_index: int
    text: str


class SpanPair(HashableBaseModel):
    """Span pair.

    Attributes:
        premise_span (:obj:`Span`):
            The premise span.
        premise_hypothesis (:obj:`Span`):
            The hypothesis span.
        label (:obj:`str`):
            The label of the span pair.
    """

    premise_span: Span
    hypothesis_span: Span
    label: str


class TextPair(BaseModel):
    """Text pair.

    Attributes:
        premise (:obj:`str`):
            The premise text without span markers.
        hypothesis (:obj:`str`):
            The hypothesis text without span markers.
        span_pairs (:obj:`List[SpanPair]`):
            The span pairs in the text pair.
        label (:obj:`Optional[str]`):
            The label of the text pair.
    """

    premise: str
    hypothesis: str
    span_pairs: List[SpanPair]
    label: Optional[str]


class RawTextPair(BaseModel):
    """Raw text pair.

    Attributes:
        premise (:obj:`str`):
            The premise text, containing the span markers.
        hypothesis (:obj:`str`):
            The hypothesis text, containing the span markers.
        span_labels (:obj:`List[int]`):
            The labels of the span pairs.
    """

    premise: str
    hypothesis: str
    span_labels: List[int]
