"""Span Similarity Dataset Statistics.

Example usage::

   python get_dataset_statistics.py ../../span-similarity-dataset/span_similarity_dataset_v1.0.0-train.tsv
"""

import argparse
import importlib
import re
import statistics
import string
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy
from data_loading import SSDTextPairDataLoader
from models import Span, SpanPair
from spacy.lang.en import stop_words
from validate_dataset import DatasetValidator


def validate_n_most_frequent(arg):
    """Validate the ``--n-most-frequent`` argument.

    This argument must be equal or greater than 1.
    """
    try:
        most_frequent = int(arg)
    except ValueError as ex:
        raise argparse.ArgumentTypeError("must be an integer.") from ex
    if most_frequent < 1:
        raise argparse.ArgumentTypeError("must be equal or greater than 1.")
    return arg


arg_parser = argparse.ArgumentParser(
    description=("Get statistics from the Span Similarity Dataset."),
    formatter_class=argparse.RawTextHelpFormatter,
)
arg_parser.add_argument("dataset", help="the dataset to get statistics from")
arg_parser.add_argument(
    "--keep-stop",
    action="store_true",
    help="consider stop words in the datasets statistics",
)
arg_parser.add_argument(
    "--keep-punct",
    action="store_true",
    help="consider punctuation in the datasets statistics",
)
arg_parser.add_argument(
    "--n-most-frequent",
    type=validate_n_most_frequent,
    default=15,
    help="how many of the most frequent words, POS tags, and\ndependencies tags to show. Defaults to 15.",
)


@dataclass
class Stats:
    dataset_name: str
    sentence_count: int = 0
    span_count: int = 0
    span_label_zero_count: int = 0
    span_label_one_count: int = 0
    sent_label_zero_count: int = 0
    sent_label_one_count: int = 0
    sentence_word_lengths: Dict[int, int] = field(default_factory=dict)
    span_word_lengths: Dict[int, int] = field(default_factory=dict)
    sentence_word_length_mean: int = 0
    sentence_word_length_std: int = 0
    span_word_length_mean: int = 0
    span_word_length_std: int = 0
    spans_per_sentence_counts: Dict[int, int] = field(default_factory=dict)
    spans_per_sentence_mean: int = 0
    spans_per_sentence_std: int = 0
    most_frequent_words_in_span: Dict[str, int] = field(default_factory=dict)
    most_frequent_pos_in_span: Dict[str, int] = field(default_factory=dict)
    most_frequent_dep_in_span: Dict[str, int] = field(default_factory=dict)


class DatasetStatistics:
    """Dataset Statistics."""

    def __init__(self):
        self.span_pattern: str = re.compile(r"{{.+?}}")
        self.spacy_model = spacy.load("en_core_web_md")
        self.stop_words: List[str] = stop_words.STOP_WORDS

    def get_statistics(
        self,
        dataset_path: str,
        keep_stop: Optional[bool] = None,
        keep_punct: Optional[bool] = None,
        n_most_frequent: Optional[int] = None,
    ) -> None:
        """Get the statistics of a dataset.

        Args:
            dataset_path (:obj:`str`):
                The path to the dataset to get the statistics of.
            keep_stop (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to consider stop words or not when generating the
                dataset statistics.
            keep_punct (:obj:`Optional[bool]`, defaults to :obj:`False`):
                Whether to consider punctuation or not when generating the
                dataset statistics.
            n_most_frequent (:obj:`Optional[int]`, defaults to ``15``):
                How many of the most frequent words, POS tags, and dependencies
                tags to show.
        """
        if keep_stop is None:
            keep_stop = False
        if keep_punct is None:
            keep_punct = False
        if n_most_frequent is None:
            n_most_frequent = 15

        self._validate_dataset(str(dataset_path))
        dataset = SSDTextPairDataLoader.load(dataset_path)
        stats = Stats(dataset_path)
        for text_pair in dataset:
            premise: str = text_pair.premise
            hypothesis: str = text_pair.hypothesis
            span_pairs: List[SpanPair] = text_pair.span_pairs
            sentence_label: str = text_pair.label
            stats.sentence_count += 2
            word_count_premise: int = self._get_word_count(premise)
            word_count_hypothesis: int = self._get_word_count(hypothesis)
            self._safe_plus_one(stats.sentence_word_lengths, word_count_premise)
            self._safe_plus_one(stats.sentence_word_lengths, word_count_hypothesis)
            spans_premise: List[Span] = [sp.premise_span for sp in span_pairs]
            spans_hypothesis: List[Span] = [sp.hypothesis_span for sp in span_pairs]
            stats.span_count += len(spans_premise) + len(spans_hypothesis)
            span_labels: List[str] = [sp.label for sp in span_pairs]
            span_zeroes_count = sum(l == "0" for l in span_labels)
            span_ones_count = sum(l == "1" for l in span_labels)
            stats.span_label_zero_count += span_zeroes_count
            stats.span_label_one_count += span_ones_count
            if sentence_label == "0":
                stats.sent_label_zero_count += 1
            else:
                stats.sent_label_one_count += 1
            for span_pair in span_pairs:
                premise_span: str = span_pair.premise_span.text
                hypothesis_span: str = span_pair.hypothesis_span.text
                for i in range(2):
                    span: str = hypothesis_span if i else premise_span
                    self._safe_plus_one(
                        stats.span_word_lengths, self._get_word_count(span)
                    )
                    for word in self._get_words(span):
                        if not word:
                            continue  # skip empty strings
                        if keep_stop or (not keep_stop and word not in self.stop_words):
                            self._safe_plus_one(stats.most_frequent_words_in_span, word)
            self._safe_plus_one(stats.spans_per_sentence_counts, len(span_pairs))
            self._calculate_pos_dep_statistics(
                premise, spans_premise, stats, keep_stop, keep_punct
            )
            self._calculate_pos_dep_statistics(
                hypothesis, spans_hypothesis, stats, keep_stop, keep_punct
            )
        stats.sentence_word_length_mean, stats.sentence_word_length_std = (
            self._calculate_mean_std(stats.sentence_word_lengths)
        )
        stats.span_word_length_mean, stats.span_word_length_std = (
            self._calculate_mean_std(stats.span_word_lengths)
        )
        stats.spans_per_sentence_mean, stats.spans_per_sentence_std = (
            self._calculate_mean_std(stats.spans_per_sentence_counts)
        )
        self._display_stats(stats, n_most_frequent=n_most_frequent)

    def _get_spans(self, string_: str) -> List[str]:
        return self.span_pattern.findall(string_)

    def _get_words(self, string_: str) -> List[str]:
        return [word.strip(string.punctuation).lower() for word in string_.split()]

    def _get_word_count(self, string_: str) -> int:
        return len(self._get_words(string_))

    def _safe_plus_one(self, dictionary: Dict[Any, Union[int, float]], key: Any):
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    def _get_span_label_count(self, label: str):
        zeroes = 0
        ones = 0
        for v in label.split(","):
            zeroes += int(v == "0")
            ones += int(v == "1")
        return zeroes, ones

    def _calculate_pos_dep_statistics(
        self,
        sentence: str,
        spans: List[Span],
        stats: Stats,
        keep_stop: bool,
        keep_punct: bool,
    ) -> None:
        """Calculate and update the statistics related to POS and dependency parsing.

        Args:
            sentence (:obj:`str`):
                The sentence to calculate and update the statistics with.
            spans (:obj:`List[str]`):
                The spans in the sentence.
            stats (:obj:`Stats`):
                The current dataset statistics. This object is modified in
                place.
            keep_stop (:obj:`bool`):
                Whether to consider stop words in statistics or not.
            keep_punct (:obj:`bool`):
                Whether to consider punctuation in statistics or not.
        """
        doc = self.spacy_model(sentence)
        span_indexes: List[Tuple[int, int]] = [
            (span.start_index, span.end_index) for span in spans
        ]
        for tk in doc:
            if not keep_punct and tk.is_punct:
                continue
            if not keep_stop and tk.text in self.stop_words:
                continue
            start = tk.i
            end = tk.i + len(tk)
            for span_start, span_end in span_indexes:
                if span_start <= start and span_end >= end:
                    self._safe_plus_one(stats.most_frequent_pos_in_span, tk.pos_)
                    self._safe_plus_one(stats.most_frequent_dep_in_span, tk.dep_)

    def _calculate_mean_std(
        self, counts: Dict[Union[int, float], Union[int, float]]
    ) -> Tuple[float, float]:
        all_values: List[float] = []
        for key, value in counts.items():
            all_values += [key] * value
        if not all_values:
            return 0.0
        return statistics.mean(all_values), statistics.pstdev(all_values)

    def _validate_dataset(self, dataset_path) -> None:
        """Validate the dataset.

        If the dataset has an invalid format, a message is printed and the
        script is exited.

        Args:
            dataset_path (:obj:`str`):
                The path to the dataset to validate.
        """
        validator = DatasetValidator()
        error_count = validator.validate(dataset_path, print_to_console=False)
        if error_count > 0:
            print(
                "❌ The dataset contains errors.\n\n"
                "Please, run 'validate_dataset.py' first",
                file=sys.stderr,
            )
            sys.exit(1)

    def _sort_most_frequent(
        self, most_frequent: Dict[Any, int], descending: bool = True
    ) -> Dict[Any, int]:
        """Sort the most frequent elements based on their counts.

        Args:
            most_frequent (:obj:`Dict[Any, int]`):
                The dictionary mapping element -> count.
            descending (:obj:`bool`):
                Whether to sort the elements by desceding counts (i.e., higher
                counts first), or not.

        Returns:
            :obj:`Dict[Any, int]`: The sorted elements.
        """
        return dict(
            sorted(
                list(most_frequent.items()),
                key=lambda item: item[1],
                reverse=descending,
            )
        )

    def _display_stats(
        self,
        stats: Stats,
        print_stats: Optional[bool] = None,
        plot_graphs: Optional[bool] = None,
        n_most_frequent: Optional[int] = None,
    ) -> None:
        """Display the dataset statistics.

        Args:
            stats (:obj:`Stats`):
                The dictionary mapping element -> count.
            print_stats (:obj:`Optional[bool]`, defaults to :obj:`True`):
                Wether to print the dataset statistics to the stdout or not.
            plot_graphs (:obj:`Optional[bool]`, defaults to :obj:`True`):
                Wether to plot the statistics or not.

                .. note::

                   The modules `matplotlib
                   <https://pypi.org/project/matplotlib/>`__, and `wordcloud
                   <https://pypi.org/project/wordcloud/>`__ are required if this
                   parameter is set to :obj:`True`.
            n_most_frequent (:obj:`Optional[int]`, defaults to ``15``):
                How many of the most frequent words, POS tags, and dependencies
                tags to show.
        """
        if print_stats is None:
            print_stats = True
        if plot_graphs is None:
            plot_graphs = True
        if n_most_frequent is None:
            n_most_frequent = 15

        frequent_words = self._sort_most_frequent(stats.most_frequent_words_in_span)
        total_words = sum(stats.most_frequent_words_in_span.values())

        frequent_pos = self._sort_most_frequent(stats.most_frequent_pos_in_span)
        total_pos = sum(stats.most_frequent_pos_in_span.values())

        frequent_dep = self._sort_most_frequent(stats.most_frequent_dep_in_span)
        total_dep = sum(stats.most_frequent_dep_in_span.values())
        title = f" | STATISTICS FOR {stats.dataset_name} |"
        title_frame = f" +{'='*(len(title)-3)}+"
        if print_stats:
            print()
            print(title_frame)
            print(title)
            print(title_frame)
            print()
            print(
                f"   -> Number of sentence pairs:             {stats.sentence_count // 2:>5}"
            )
            print(
                f"   -> Number of span pairs:                 {stats.span_count // 2:>5}"
            )
            print(
                f"   -> Number of span pairs labeled '0':     {stats.span_label_zero_count:>5}"
            )
            print(
                f"   -> Number of span pairs labeled '1':     {stats.span_label_one_count:>5}"
            )
            print(
                f"   -> Number of sentence pairs labeled '0': {stats.sent_label_zero_count:>5}"
            )
            print(
                f"   -> Number of sentence pairs labeled '1': {stats.sent_label_one_count:>5}"
            )
            print()
            print(
                f"   -> Avg. sentence length (words):  {stats.sentence_word_length_mean:>5.2f} ({stats.sentence_word_length_std:.2f})"
            )
            print(
                f"   -> Avg. span length (words):      {stats.span_word_length_mean:>5.2f} ({stats.span_word_length_std:.2f})"
            )
            print(
                f"   -> Avg. spans per sentence:       {stats.spans_per_sentence_mean:>5.2f} ({stats.spans_per_sentence_std:.2f})"
            )
            print()
            print("   -> Most frequent words in spans:")
            print()
            print("         +=============================+====================+")
            print("         |            WORD             |  COUNT  |   PERC.  |")
            print("         +=============================+====================+")
            for i, items in enumerate(frequent_words.items()):
                word, count = items
                word_percentage = (
                    round((count / total_words) * 100, 2) if total_words else 0
                )
                print(
                    f"         |{word:^29}| {count:>6}  |  {word_percentage:>5.2f} % |"
                )
                if i == n_most_frequent - 1:
                    break
            print("         +--------------------------------------------------+")
            print()
            print("   -> Most frequent POS in spans:")
            print()
            print("         +=============================+====================+")
            print("         |             POS             |  COUNT  |   PERC.  |")
            print("         +=============================+====================+")
            for i, items in enumerate(frequent_pos.items()):
                pos, count = items
                pos_percentage = round((count / total_pos) * 100, 2) if total_pos else 0
                print(f"         |{pos:^29}| {count:>6}  |  {pos_percentage:>5.2f} % |")
                if i == n_most_frequent - 1:
                    break
            print("         +--------------------------------------------------+")
            print()
            print("   -> Most frequent dependencies in spans:")
            print()
            print("         +=============================+====================+")
            print("         |             DEP             |  COUNT  |   PERC.  |")
            print("         +=============================+====================+")
            for i, items in enumerate(frequent_dep.items()):
                dep, count = items
                dep_percentage = round((count / total_dep) * 100, 2) if total_dep else 0
                print(f"         |{dep:^29}| {count:>6}  |  {dep_percentage:>5.2f} % |")
                if i == n_most_frequent - 1:
                    break
            print("         +--------------------------------------------------+")
            print()
        if plot_graphs:
            try:
                plt = importlib.import_module("matplotlib.pyplot")
                wordcloud = importlib.import_module("wordcloud")
            except ModuleNotFoundError:
                print(
                    'Some of the required modules ["matplotlib", "wordcloud"] are not installed.'
                )
                print("Please, install them and run again.")
                sys.exit(1)

            plt.style.use("ggplot")
            _, axes = plt.subplot_mosaic(
                "AB;CD;EF;GG",
                figsize=(8, 9),
                constrained_layout=True,
                gridspec_kw={"hspace": 0.15},
            )

            ax = axes["A"]
            span_label_counts = [
                stats.span_label_zero_count,
                stats.span_label_one_count,
            ]
            ax.bar(
                ["Label 0", "Label 1"], span_label_counts, color=["#6200ee", "#00fee6"]
            )
            for i, count in enumerate(span_label_counts):
                ax.text(
                    i,
                    count,
                    count,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    size=9,
                )
            ax.set_title("Span Label Distribution")

            ax = axes["B"]
            sent_label_counts = [
                stats.sent_label_zero_count,
                stats.sent_label_one_count,
            ]
            ax.bar(
                ["Label 0", "Label 1"], sent_label_counts, color=["#6200ee", "#00fee6"]
            )
            for i, count in enumerate(sent_label_counts):
                ax.text(
                    i,
                    count,
                    count,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    size=9,
                )
            ax.set_title("Sentence Label Distribution")

            ax = axes["C"]
            ax.bar(
                stats.sentence_word_lengths.keys(),
                stats.sentence_word_lengths.values(),
                color="#6200ee",
            )
            ax.set_title("Sentence Word Lengths")
            ax.set_xlabel("Word Length")
            ax.set_ylabel("Count")

            ax = axes["D"]
            ax.bar(
                stats.span_word_lengths.keys(),
                stats.span_word_lengths.values(),
                color="#6200ee",
            )
            ax.set_title("Span Word Lengths")
            ax.set_xlabel("Word Length")
            ax.set_ylabel("Count")

            if frequent_pos:
                wc = wordcloud.WordCloud(
                    background_color="white"
                ).generate_from_frequencies(frequent_pos)
                ax = axes["E"]
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title("Most Frequent POS in Spans")

            if frequent_dep:
                wc = wordcloud.WordCloud(
                    background_color="white"
                ).generate_from_frequencies(frequent_dep)
                ax = axes["F"]
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title("Most Frequent Dependencies in Spans")

            if frequent_words:
                wc = wordcloud.WordCloud(
                    background_color="white"
                ).generate_from_frequencies(frequent_words)
                ax = axes["G"]
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title("Most Frequent Words in Spans")

            plt.show()


def main():
    args = arg_parser.parse_args()
    stats = DatasetStatistics()
    stats.get_statistics(
        args.dataset, args.keep_stop, args.keep_punct, args.n_most_frequent
    )


if __name__ == "__main__":
    main()
