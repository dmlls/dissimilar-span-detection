"""Parse annotated output from the LLM and convert it into the SSD format."""

import re
from typing import List, Tuple


def main():
    output: List[Tuple[str]] = []
    with open("chat-gpt-output.txt", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            split = re.split(r"=>", line)
            output.append(split)

    with open("annotated.tsv", "w", encoding="utf-8") as f:
        for premise, hypothesis in output:
            f.write(f"{premise.strip()}\t{hypothesis.strip()}\t0\n")
    print("Done! 🍻")


if __name__ == "__main__":
    main()
