"""Sentence Diff with Token Classification."""

from io import StringIO
from typing import Callable, Dict, List, Optional

import nltk
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BatchEncoding

try:
    from base_sentence_diff import BaseSentenceDiff
    from config import ModelType
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff
    from .config import ModelType


class TokenClassificationSentenceDiff(BaseSentenceDiff):
    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.model_type = ModelType.TRANSFORMER
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.id2label: Dict[int, str] = {
            0: "O",
            1: "B",
            2: "I",
        }
        self.label2id: Dict[str, int] = {
            "O": 0,
            "B": 1,
            "I": 2,
        }

    def annotate_diff(
        self,
        premise: str,
        hypothesis: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        **kwargs,
    ) -> list[str, str]:
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
                Additional arguments to the annotation.

        Returns:
            :obj:`list[str, str]`: The premise and hypothesis with the differing spans
            annotated.
        """
        if start_marker is None:
            start_marker = "{{"
        if end_marker is None:
            end_marker = "}}"

        inputs = self._tokenize_sentence_pair(premise, hypothesis, self.tokenizer)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [
            self.model.config.id2label[t.item()] for t in predictions[0]
        ]

        input_ids = inputs["input_ids"].squeeze().tolist()
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        annotated_output = StringIO()

        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        previous_prediction: str = "1"

        for prediction, id_, tk in zip(
            predicted_token_class, input_ids, decoded_tokens
        ):
            if prediction in ["B", "I"] and previous_prediction == "O":
                annotated_output.write(f" {start_marker} ")
            # From the special tokens, we write only the separator token in
            # order to be able to split the pair afterwards.
            if (
                id_ == self.tokenizer.sep_token_id
                or id_ not in self.tokenizer.all_special_ids
            ):
                annotated_output.write(f" {tk.lstrip('##').lstrip('Ġ')} ")
            if prediction == "O" and previous_prediction in ["B", "I"]:
                annotated_output.write(f" {end_marker} ")
            previous_prediction = prediction

        annotated_output = annotated_output.getvalue()
        _, annotated_hypothesis, _ = annotated_output.split(self.tokenizer.sep_token)
        return self._fix_output(
            hypothesis, annotated_hypothesis, start_marker, end_marker
        )

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

    def _tokenize_sentence_pair(
        self, sentence_a: str, sentence_b: str, tokenizer: AutoTokenizer
    ) -> BatchEncoding:
        """Tokenize a sentence pair for the SentenceDiff model."""
        tokenized_sent_a = tokenizer(sentence_a, return_tensors="pt")
        tokenized_sent_b = tokenizer(sentence_b, return_tensors="pt")
        processed_sentence_pair = {}
        for (key, values_a), values_b in zip(
            tokenized_sent_a.items(), tokenized_sent_b.values()
        ):
            processed_sentence_pair[key] = torch.cat(
                (values_a.squeeze(), values_b.squeeze()[1:])
            ).unsqueeze(dim=0)
        return BatchEncoding(data=processed_sentence_pair)

    def _fix_output(
        self,
        input_: str,
        output: str,
        start_marker: str,
        end_marker: str,
    ) -> str:
        """Fix casing and whitespaces in the model decoded output."""
        fixed_output = StringIO()

        i = 0  # index for the input
        j = 0  # index for the output

        while i < len(input_):
            if output[j] == " ":
                j += 1
                continue
            # Start marker
            if (
                j + len(start_marker) - 1 < len(output)
                and "".join(output[j : j + len(start_marker)]) == start_marker
            ):
                fixed_output.write(f" {start_marker}")
                j += 2
                if input_[i] == " ":
                    i += 1
                continue
            # End marker
            if (
                j + len(end_marker) - 1 < len(output)
                and "".join(output[j : j + len(end_marker)]) == end_marker
            ):
                fixed_output.write(f"{end_marker}")
                j += 2
                continue
            fixed_output.write(input_[i])
            if input_[i].lower() == output[j].lower():
                j += 1
            i += 1
        fixed_output.write(output[j:].strip())
        return fixed_output.getvalue()
