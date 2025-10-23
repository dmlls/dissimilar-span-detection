"""Train a Token Classifier on the Sentence Similarity Dataset (SSD).

Code inpired by
`https://huggingface.co/docs/transformers/tasks/token_classification`__.
"""

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import ClassLabel, Dataset, Features, NamedSplit, Sequence, Value
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset.data_loading import KFoldDataLoader
from dataset.models import RawTextPair


class ModelFamily(str, Enum):
    """Supported model families."""

    BERT = "BERT"
    ROBERTA = "RoBERTa"


class Trainer:
    """Token Classification Trainer."""

    def __init__(
        self,
        base_model: str,
        dataset_path: str,
        training_args: TrainingArguments,
    ):
        self.base_model = base_model
        self.training_args = training_args
        prefix = f"{base_model.split('/')[1]}" if "/" in self.base_model else base_model
        if "roberta" in prefix:
            self.model_family: ModelFamily = ModelFamily.ROBERTA
        elif "bert" in prefix:
            self.model_family: ModelFamily = ModelFamily.BERT
        else:
            raise NotImplementedError(
                "only the following base models are supported: "
                f"{[model.value for model in ModelFamily]}"
            )
        self.output_model_name = f"{prefix}-sentence-diff"
        timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_save_path = Path(
            f"{training_args.output_dir}/{timestamp}/{self.output_model_name}"
        )
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, model_max_length=512, add_prefix_space=True
        )
        self.start_span: str = "{{"
        self.end_span: str = "}}"
        self.start_span_id: int = self._get_span_marker_id(
            self.tokenizer, self.start_span
        )
        self.end_span_id: int = self._get_span_marker_id(self.tokenizer, self.end_span)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
        self.dataset_path = Path(dataset_path)

    def train(self, k: int, k_folds: int) -> Path:
        """Run training."""
        dataset_train = self._process_training_data(
            self.dataset_path, k, k_folds, "train"
        )
        trainer = HFTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset_train,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
        trained_model_dir = [
            directory
            for directory in Path(self.training_args.output_dir).iterdir()
            if directory.name.startswith("checkpoint")
        ][-1]
        return Path(trained_model_dir)

    def _process_training_data(
        self,
        dataset_path: str,
        k: int,
        k_folds: int,
        split: Optional[str] = None,
    ) -> Dataset:
        sentences: List[str] = []
        span_labels: List[List[int]] = []

        training_data: np.ndarray[RawTextPair] = None
        for i, (train_data, _) in enumerate(
            KFoldDataLoader.load_raw(dataset_path, k_folds), 1
        ):
            if i == k:
                training_data = train_data
                break

        for text_pair in training_data:
            # if all(label == 1 for label in labels):  # discard sentences with no differing spans
            #    continue
            span_labels += [
                text_pair.span_labels,
                text_pair.span_labels,
            ]  # premise and hypothesis share the same labels
            sentences += [
                self._add_whitespace_around_span_markers(text_pair.premise),
                self._add_whitespace_around_span_markers(text_pair.hypothesis),
            ]

        # Tokenize sentences.
        tokenized_sentences = self.tokenizer(sentences, truncation=True)
        processed_sentences = self._process_sentences(tokenized_sentences, span_labels)

        return Dataset.from_dict(
            processed_sentences,
            features=Features(
                {
                    "input_ids": Sequence(Value("int32")),
                    "attention_mask": Sequence(Value("int8")),
                    "labels": Sequence(Value("int8")),
                    "similarity_labels": Sequence(ClassLabel(names=["O", "B", "I"])),
                }
            ),
            split=NamedSplit(split),
        )

    def _get_span_marker_id(
        self, tokenizer: PreTrainedTokenizerBase, marker: str
    ) -> int:
        """Get the token id corresponding to a span marker."""
        return tokenizer(marker)["input_ids"][1]

    def _process_tokenized_sentence(self, input_ids, attention_mask, span_labels):
        processed_sentence = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        current_label_idx = -1
        in_span: bool = False
        first_token_span: bool = True
        first_start_marker: bool = False
        for sentence_input_id, mask in zip(input_ids, attention_mask):
            if sentence_input_id == self.start_span_id:
                in_span = True
                # BERT will split the span marker "{{" into 2 tokens.
                if self.model_family is ModelFamily.BERT:
                    if not first_start_marker:
                        first_start_marker = True
                        current_label_idx += 1
                    else:
                        first_start_marker = False
                else:
                    current_label_idx += 1
            elif sentence_input_id == self.end_span_id:
                in_span = False
                first_token_span = True
            else:
                processed_sentence["input_ids"].append(sentence_input_id)
                processed_sentence["attention_mask"].append(mask)
                label = self.label2id["O"]
                # Ignore special tokens or sub-word tokens.
                if self._is_special_or_non_first_subword_tk(sentence_input_id):
                    # Label -100 ignored by PyTorch loss function.
                    label = -100
                elif in_span and span_labels[current_label_idx] == 0:  # dissimilar span
                    if first_token_span:
                        label = self.label2id["B"]
                        first_token_span = False
                    else:
                        label = self.label2id["I"]
                processed_sentence["labels"].append(label)
        processed_sentence["similarity_labels"] = [
            label for label in processed_sentence["labels"] if label != -100
        ]
        return processed_sentence

    def _process_sentences(self, tokenized_sentences, span_labels):
        processed_sentence_pairs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "similarity_labels": [],
        }
        input_ids = tokenized_sentences["input_ids"]
        attention_mask = tokenized_sentences["attention_mask"]
        for i, (sentence_input_ids, mask, labels) in enumerate(
            zip(input_ids, attention_mask, span_labels)
        ):
            processed_sentence = self._process_tokenized_sentence(
                sentence_input_ids, mask, labels
            )
            for key, sentence in processed_sentence_pairs.items():
                if i % 2 == 0:  # premise
                    sentence.append(processed_sentence[key])
                else:  # Hypothesis: remove BOS token and append to
                    # previous sentence (premise).
                    sentence[-1] += processed_sentence[key][1:]
        return processed_sentence_pairs

    def _is_special_or_non_first_subword_tk(self, token_id: int) -> bool:
        """Determine whether a token id represents a special or a non-first, sub-word token."""
        if token_id in self.tokenizer.all_special_ids:  # special token
            return True
        if self.model_family is ModelFamily.BERT:
            return self.tokenizer.convert_ids_to_tokens(token_id)[:2] == "##"
        if self.model_family is ModelFamily.ROBERTA:
            return self.tokenizer.convert_ids_to_tokens(token_id)[0] != "Ġ"
        return False

    def _add_whitespace_around_span_markers(self, sentence: str) -> str:
        """Wrap span markers within whitespaces.

        This makes it easier to identify which tokens are sub-word tokens.
        """
        return " ".join(
            sentence.replace(self.start_span, f" {self.start_span} ")
            .replace(self.end_span, f" {self.end_span} ")
            .split()
        )
