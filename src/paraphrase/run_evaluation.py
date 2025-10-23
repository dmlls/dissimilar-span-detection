"""Run evaluation on the PAWS dataset."""

from timeit import default_timer as timer

from evaluationn import Evaluator

# DATASET_PATH: str = "./PAWS-Wiki-labeled-final/dev.tsv"
DATASET_PATH: str = "./PAWS-Wiki-labeled-final/test.tsv"
# MODEL: str = "all-MiniLM-L6-v2"
# MODEL: str = "all-mpnet-base-v2"
# MODEL: str = "./all-MiniLM-L6-v2_paws"
MODEL: str = "./all-mpnet-base-v2_paws"

STS_THRESHOLD: float = 0.63
DSD_THRESHOLD: float = 0.007


def main():
    print("Starting evaluation...")
    start = timer()
    evaluator = Evaluator(MODEL)
    accuracy = evaluator.evaluate(
        DATASET_PATH, dsd=False, sts_threshold=STS_THRESHOLD, labels=[0, 1]
    )
    print(f"Accuracy without DSD: {accuracy:.3f}")
    accuracy = evaluator.evaluate(
        DATASET_PATH,
        dsd=True,
        sts_threshold=STS_THRESHOLD,
        dsd_threshold=DSD_THRESHOLD,
        labels=[0, 1],
    )
    end = timer()
    print(f"Accuracy with DSD:    {accuracy:.3f}")
    print(f"Elapsed seconds: {end - start:.2f}")


if __name__ == "__main__":
    main()
