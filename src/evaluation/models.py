"""Data models for span detection evaluation."""

from typing import Dict, Optional

from pydantic import BaseModel


class EvaluationResults(BaseModel):
    """Results of an evaluation.

    Attributes:
        k_fold (:obj:`Optional[int]`, defaults to :obj:`None`):
            The k-fold the results refer to. Set to :obj:`None` if no k-fold
            cross-validation is being done.
        precision_mean (:obj:`float`):
            The mean precision.
        recall_mean (:obj:`float`):
            The mean recall.
        f1_mean (:obj:`float`):
            The mean F1 score.
        precision_se (:obj:`float`):
            The standard error of the precision.
        recall_se (:obj:`float`):
            The standard error of the recall.
        f1_se (:obj:`float`):
            The standard error of the F1 score.
        no_diff_precision_mean (:obj:`float`):
            The mean precision of sentence-pairs with _no_ differing spans.
        no_diff_recall_mean (:obj:`float`):
            The mean recall of sentence-pairs with _no_ differing spans.
        no_diff_f1_mean (:obj:`float`):
            The mean F1 score of sentence-pairs with _no_ differing spans.
        no_diff_precision_se (:obj:`float`):
            The standard error of the no-differing-span precision.
        no_diff_recall_se (:obj:`float`):
            The standard error of the no-differing-span recall.
        no_diff_f1_se (:obj:`float`):
            The standard error of the no-differing-span F1.
        diff_precision_mean (:obj:`float`):
            The mean precision of sentence-pairs with differing spans.
        diff_recall_mean (:obj:`float`):
            The mean recall of sentence-pairs with differing spans.
        diff_f1_mean (:obj:`float`):
            The mean F1 score of sentence-pairs with differing spans.
        diff_precision_se (:obj:`float`):
            The standard error of the differing-span precision.
        diff_recall_se (:obj:`float`):
            The standard error of the differing-span recall.
        diff_f1_se (:obj:`float`):
            The standard error of the differing-span F1.
        config (:obj:`Dict`):
            The configuration used for the evaluation.
    """

    k_fold: Optional[int] = None
    precision_mean: float
    recall_mean: float
    f1_mean: float
    precision_se: float
    recall_se: float
    f1_se: float
    no_diff_precision_mean: float
    no_diff_recall_mean: float
    no_diff_f1_mean: float
    no_diff_precision_se: float
    no_diff_recall_se: float
    no_diff_f1_se: float
    diff_precision_mean: float
    diff_recall_mean: float
    diff_f1_mean: float
    diff_precision_se: float
    diff_recall_se: float
    diff_f1_se: float
    config: Dict
