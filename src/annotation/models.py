"""Annotation models."""

from pydantic import BaseModel


class InterAnnotatorAgreementResults(BaseModel):
    """Inter annotator agreement results.

    Attributes:
        span_boundaries (:obj:`float`):
            The agreement in span boundaries, considered at the word level.
        span_labels (:obj:`float`):
            The agreement in span labels.
        n_spans (:obj:`float`):
            The agreement in the number of identified spans.
    """

    span_boundaries: float
    span_labels: float
    n_spans: float
