"""Span Similarity annotation validator.

Example usage::

   python validate_annotation.py annotation.tsv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List

arg_parser = argparse.ArgumentParser(
    description=("Validate a Span Similarity annotation."),
    formatter_class=argparse.RawTextHelpFormatter,
)
arg_parser.add_argument("annotation", help="the annotation to validate")


class AnnotationValidator:
    def __init__(self):
        self.error_count: int = 0
        self.span_pattern: str = re.compile(r"{{.+?}}")

    def validate(self, annotation_path: str, print_to_console: bool = True) -> int:
        annotation_path = Path(annotation_path)
        with open(annotation_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for i, line in enumerate(reader, 2):
                try:
                    premise, hypothesis, span_labels, sentence_label = line
                except ValueError:
                    try:  # allow sentence label to be missing
                        premise, hypothesis, span_labels = line
                        sentence_label = None
                    except ValueError:
                        self._log_error(f"[Line {i}] Malformed line", print_to_console)
                        continue
                (
                    opening_brackets_premise,
                    closing_brackets_premise,
                ) = self._validate_brackets(
                    premise,
                    line_number=i,
                    role="premise",
                    print_to_console=print_to_console,
                )
                (
                    opening_brackets_hypothesis,
                    closing_brackets_hypothesis,
                ) = self._validate_brackets(
                    hypothesis,
                    line_number=i,
                    role="hypothesis",
                    print_to_console=print_to_console,
                )
                spans_premise = None
                if opening_brackets_premise == closing_brackets_premise:
                    spans_premise = opening_brackets_premise // 2
                spans_hypothesis = None
                if opening_brackets_hypothesis == closing_brackets_hypothesis:
                    spans_hypothesis = opening_brackets_hypothesis // 2

                if (
                    spans_premise is not None
                    and spans_hypothesis is not None
                    and spans_premise != spans_hypothesis
                ):
                    self._log_error(
                        f"[Line {i}] Mismatching number of spans between premise "
                        "and hypothesis.",
                        print_to_console,
                    )

                span_label_count = None
                try:
                    span_labels = [int(l.strip()) for l in span_labels.split(",")]
                    if any(label not in (0, 1) for label in span_labels):
                        self._log_error(
                            f"[Line {i}] Span label different from '0' or '1'.",
                            print_to_console,
                        )
                    span_label_count = len(span_labels)
                except ValueError:
                    self._log_error(
                        f"[Line {i}] Malformed span labels.", print_to_console
                    )

                if (
                    span_label_count
                    and spans_premise is not None
                    and spans_hypothesis is not None
                    and spans_premise == spans_hypothesis
                ):
                    if spans_premise < span_label_count:
                        self._log_error(
                            f"[Line {i}] Less spans than span labels.", print_to_console
                        )
                    if spans_premise > span_label_count:
                        self._log_error(
                            f"[Line {i}] More spans than span labels.", print_to_console
                        )

                try:
                    if sentence_label and int(sentence_label) not in (0, 1):
                        self._log_error(
                            f"[Line {i}] Sentence label different from '0' or '1'.",
                            print_to_console,
                        )
                except ValueError:
                    self._log_error(
                        f"[Line {i}] Malformed sentence label.", print_to_console
                    )

        if print_to_console:
            if self.error_count == 0:
                print("✅ No errors found!")
            else:
                print(f"❌ {self.error_count} issue(s) found.")
        return self.error_count

    def _log_error(self, string_: str, print_to_console: bool):
        self.error_count += 1
        if print_to_console:
            print(string_)

    def _validate_brackets(
        self, string_: str, line_number: int, role: str, print_to_console: bool
    ):
        opening_brackets = string_.count("{")
        closing_brackets = string_.count("}")
        if opening_brackets % 2 != 0:
            self._log_error(
                f"[Line {line_number}] Odd number of opening brackets in {role}.",
                print_to_console,
            )
        if closing_brackets % 2 != 0:
            self._log_error(
                f"[Line {line_number}] Odd number of closing brackets in {role}.",
                print_to_console,
            )
        if opening_brackets != closing_brackets:
            self._log_error(
                f"[Line {line_number}] Mismatching number of brackets in {role}.",
                print_to_console,
            )
        return opening_brackets, closing_brackets

    def _get_non_spans(self, string_: str) -> List[str]:
        return re.sub(self.span_pattern, "", string_)


def main():
    args = arg_parser.parse_args()
    validator = AnnotationValidator()
    validator.validate(args.annotation)


if __name__ == "__main__":
    main()
