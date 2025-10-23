"""Transform data from iSTS format to SSD format.

For more info ont the iSTS dataset, see
`https://alt.qcri.org/semeval2016/task2/`__.
"""

import re
import sys
from io import StringIO
from pathlib import Path

# Directory with the iSTS dataset files.
ists_directory: Path = Path("ists_datasets")
output_file: Path = "ists_ssd.tsv"


def init_sentence_pair() -> dict:
    """Initialize a dictionary to hold a sentence pair."""
    return {"premise": {}, "hypothesis": {}, "relations": {"OPPO": []}}


def parse_data() -> list[dict]:
    """Parse the iSTS data into a list of dictionaries.

    The structure of these dictionaries is as returned by
    :func:`init_sentence_pair`.

    This function reads all files under :obj:`ists_directory`, ignoring any
    files that do not have the `".wa"` extension.
    """
    # Check if the directory exists
    if not ists_directory.exists():
        print(f"The directory '{ists_directory}' does not exist.")
        sys.exit(1)

    parsed_data: list[dict] = []
    # Iterate over all files in the directory.
    for file_path in ists_directory.iterdir():
        # Check if it's a ".wa" file.
        if file_path.is_file() and file_path.suffix == ".wa":
            print(f"Reading file: {file_path.name}")
            sentence_pair: dict = init_sentence_pair()
            in_sentence: bool = False
            in_source: bool = False
            in_translation: bool = False
            in_alignment: bool = False
            with file_path.open("r") as input_f:
                for line in input_f:
                    stripped = line.strip()
                    if stripped.startswith("<sentence id="):
                        in_sentence = True
                        continue
                    if stripped == "</sentence>":
                        in_sentence = False
                        continue
                    if stripped == "<source>":
                        in_source = True
                        continue
                    if stripped == "</source>":
                        in_source = False
                        continue
                    if stripped == "<translation>":
                        in_translation = True
                        continue
                    if stripped == "</translation>":
                        in_translation = False
                        continue
                    if stripped == "<alignment>":
                        in_alignment = True
                        continue
                    if stripped == "</alignment>":
                        in_alignment = False
                        continue
                    if not in_sentence:
                        if sentence_pair["premise"]:
                            parsed_data.append(sentence_pair)
                        sentence_pair = init_sentence_pair()
                        continue
                    if in_source:
                        token_number, token, _ = stripped.split()
                        sentence_pair["premise"][int(token_number)] = token
                    if in_translation:
                        token_number, token, _ = stripped.split()
                        sentence_pair["hypothesis"][int(token_number)] = token
                    if in_alignment:
                        premise_relations, rest, _ = stripped.split("<==>")
                        premise_relations = premise_relations.strip()
                        hypothesis_relations, label, _, _ = rest.split("//")
                        hypothesis_relations = hypothesis_relations.strip()
                        label = label.strip()
                        if label in sentence_pair["relations"]:
                            sentence_pair["relations"][label].append(
                                (
                                    tuple(
                                        int(rel) for rel in premise_relations.split()
                                    ),
                                    tuple(
                                        int(rel) for rel in hypothesis_relations.split()
                                    ),
                                )
                            )
        else:
            print(f'{file_path.name} is not a ".wa" file, skipping.')
    return parsed_data


def write_span_markers(
    parsed_sentence: dict, start_span_idx: list[int], end_span_idx: list[int]
) -> str:
    """Transform a parsed sentence into a formatted STS sentence.

    If no spans are passed, the whole sentence is taken as a span.
    """
    sentence = StringIO()
    for sentence_token_id, sentence_token in parsed_sentence.items():
        if sentence_token_id in start_span_idx:
            sentence.write(" {{")
        # Determine whether to add whitespace or not.
        if sentence_token[0] in ".,;:!?'":
            sentence.write(sentence_token)
        else:
            sentence.write(f" {sentence_token}")
        if sentence_token_id in end_span_idx:
            sentence.write(" }} ")
    return re.sub(
        r"\s*{{\s+", " {{", re.sub(r"\s+}}\s*", "}} ", sentence.getvalue().strip())
    ).strip() if start_span_idx else "{{" + sentence.getvalue().strip() + "}}"


def write_sts_dataset(parsed_data: dict) -> None:
    """Write the parsed iSTS data into an STS-formatted file."""
    if parsed_data:
        with open(output_file, "w", encoding="utf-8") as output_f:
            output_f.write("premise	hypothesis	span_similarity	sentence_similarity\n")
            for pair in parsed_data:
                premise_start_span_idx = []
                premise_end_span_idx = []
                hypothesis_start_span_idx = []
                hypothesis_end_span_idx = []
                for relation in pair["relations"]["OPPO"]:
                    premise_start_span_idx.append(relation[0][0])
                    premise_end_span_idx.append(relation[0][-1])
                    hypothesis_start_span_idx.append(relation[1][0])
                    hypothesis_end_span_idx.append(relation[1][-1])
                if not premise_end_span_idx:
                    continue  # we skip pairs without OPPO spans
                output_f.write(
                    write_span_markers(
                        pair["premise"], premise_start_span_idx, premise_end_span_idx
                    )
                )
                output_f.write("\t")
                output_f.write(
                    write_span_markers(
                        pair["hypothesis"],
                        hypothesis_start_span_idx,
                        hypothesis_end_span_idx,
                    )
                )
                if premise_start_span_idx:
                    # Write as many comma-separated zeros as spans.
                    output_f.write(f"\t{",".join(["0"]*len(premise_end_span_idx))}\t0\n")
                else:
                    output_f.write("\t1\t1\n")


if __name__ == "__main__":
    write_sts_dataset(parse_data())
    print(f"Done! Converted dataset in {output_file}.")
