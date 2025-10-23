"""Evaluation of spans on the Sentence Similarity Dataset (SSD).

Example usage::

   python run_evaluation.py baseline no-sentence-diff ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
   python run_evaluation.py baseline naive-sentence-diff ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
   python run_evaluation.py sentence-transformer all-MiniLM-L6-v2 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
   python run_evaluation.py shap all-MiniLM-L6-v2 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
   python run_evaluation.py openai gpt-4-turbo-2024-04-09 ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
   python run_evaluation.py mistral mistral-medium ../../span-similarity-dataset/span_similarity_dataset_v1.0.0.tsv
"""

import argparse
import sys
from pathlib import Path

from config import EVAL_CONFIG, LLMProvider, ModelType, settings
from models import EvaluationResults

from evaluation import SSDSpanSimilarityEvaluator

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.data_loading import SSDTextPairDataLoader

arg_parser = argparse.ArgumentParser(
    description="Evaluate a model on the Span Similarity Dataset.",
    formatter_class=argparse.RawTextHelpFormatter,
)
arg_parser.add_argument(
    "model_type",
    metavar="model-type",
    help=(
        "the model type to evaluate. Possible values are:"
        f"\n{[model_type.value for model_type in ModelType]}."
    ),
)
arg_parser.add_argument(
    "model_name",
    metavar="model-name",
    help=(
        "the model or method to evaluate, e.g., 'all-MiniLM-L6-v2' if"
        " using the model\ntype 'sentence-transformers', 'gpt-4-turbo-preview'"
        " if using 'openai', or\n'no-sentence-diff' / 'naive-sentence-diff' if"
        " using 'baseline'"
    ),
)
arg_parser.add_argument("dataset", help="the path to the dataset to evaluate on")
arg_parser.add_argument(
    "--k-folds",
    type=str,
    default=settings.K_FOLDS,
    help=(
        "the number of folds for k-fold cross-validation. Defaults to "
        f"{settings.K_FOLDS}."
    ),
)
arg_parser.add_argument(
    "--logging-path",
    type=str,
    default=settings.LOGGING_PATH,
    help=(
        "path where to store the evaluation results. If no path is provided, logs"
        f"\nare stored in '{settings.LOGGING_PATH}/'"
    ),
)
arg_parser.add_argument(
    "--token-diff-threshold",
    type=float,
    default=EVAL_CONFIG[ModelType.SENTENCE_TRANSFORMER]["token_diff_threshold"],
    help=(
        "(only with model type 'sentence-transformer') the minimum value"
        " (inclusive)\nthe diff score of a unigram must have in order to be considered"
        " dissimilar.\nDefaults to"
        f" {EVAL_CONFIG[ModelType.SENTENCE_TRANSFORMER]['token_diff_threshold']}."
    ),
)

arg_parser.add_argument(
    "--shap_value_threshold",
    type=float,
    default=EVAL_CONFIG[ModelType.SHAP]["shap_value_threshold"],
    help=(
        "(only with model type 'shap') the minimum value (inclusive)\nthe SHAP"
        " value of a token must have in order to be considered dissimilar."
        " \nDefaults to"
        f" {EVAL_CONFIG[ModelType.SHAP]['shap_value_threshold']}."
    ),
)

arg_parser.add_argument(
    "--lime_weight_threshold",
    type=float,
    default=EVAL_CONFIG[ModelType.LIME]["lime_weight_threshold"],
    help=(
        "(only with model type 'lime') the minimum value (inclusive)\nthe LIME"
        " weight of a token must have in order to be considered dissimilar."
        "\nDefaults to"
        f" {EVAL_CONFIG[ModelType.LIME]['lime_weight_threshold']}."
    ),
)

arg_parser.add_argument(
    "--lime_num_samples",
    type=float,
    default=EVAL_CONFIG[ModelType.LIME]["lime_num_samples"],
    help=(
        "(only with model type 'lime') the size of the neighborhood to learn"
        " the\nlinear model. For more info, visit the LIME docs at"
        "\nhttps://lime-ml.readthedocs.io. Defaults to"
        f" {EVAL_CONFIG[ModelType.LIME]['lime_num_samples']}."
    ),
)

arg_parser.add_argument(
    "--max-retries",
    type=int,
    default=EVAL_CONFIG[ModelType.OPENAI]["max_retries"],
    help=(
        f"(only with model types {[provider.value for provider in LLMProvider]}) the"
        " maximum number\nof retries if the annotation fails due to malformed outputs"
        f" or API errors.\nDefaults to {EVAL_CONFIG[ModelType.OPENAI]['max_retries']}."
    ),
)


def main():
    """Run evaluation on the Sentence Similarity Dataset."""
    args = arg_parser.parse_args()
    try:
        model_type = ModelType(args.model_type)
    except ValueError:
        print(f"\n  ERROR: Model type '{args.model_type}' not supported.")
        print(
            "  Currently supported values are:"
            f" {[model_type.value for model_type in ModelType]}.\n"
        )
        sys.exit()
    evaluator = SSDSpanSimilarityEvaluator(
        model_type=model_type,
        model_name=args.model_name,
        data_loader=SSDTextPairDataLoader,
        logging_path=args.logging_path,
    )
    evaluator.evaluate(
        dataset_path=args.dataset,
        k_folds=args.k_folds,
        token_diff_threshold=args.token_diff_threshold,
        shap_value_threshold=args.shap_value_threshold,
        lime_weight_threshold=args.lime_weight_threshold,
        lime_num_samples=args.lime_num_samples,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
