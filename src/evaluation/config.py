"""Evaluation settings."""

from enum import Enum
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import TrainingArguments


class ModelType(str, Enum):
    """Model type to use for evaluation."""

    BASELINE = "baseline"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    SHAP = "shap"
    LIME = "lime"
    TRANSFORMER = "transformer"
    OPENAI = "openai"
    MISTRAL = "mistral"
    LLAMA = "llama"


class LLMProvider(str, Enum):
    """LLM Provider to use when evaluating with LLMs."""

    OPENAI: str = ModelType.OPENAI.value
    MISTRAL: str = ModelType.MISTRAL.value
    LLAMA: str = ModelType.LLAMA.value


# Model type specific configuration. Mapped as dictionaries with the keys being
# the setting names, and the values the default values.
EVAL_CONFIG: Dict[ModelType, Dict[str, Any]] = {
    ModelType.SENTENCE_TRANSFORMER: {
        "token_diff_threshold": 0.005,
    },
    ModelType.SHAP: {
        "shap_value_threshold": 0.03,
    },
    ModelType.LIME: {
        "lime_weight_threshold": 0.040,
        "lime_num_samples": 5000,
    },
    ModelType.TRANSFORMER: {
        "training_arguments": TrainingArguments(
            output_dir="./trained_models",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.005,
            eval_strategy="no",
            do_eval=False,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=False,
            report_to="tensorboard",
            use_cpu=False,
        )
    },
    # Common config for the following models added below.
    ModelType.OPENAI: {},
    ModelType.MISTRAL: {},
    ModelType.LLAMA: {},
}
# Set here configuration common to all LLM providers.
common_llm_config = {
    "max_retries": 10,
}
for llm_provider in LLMProvider:
    EVAL_CONFIG[llm_provider].update(common_llm_config)


class Settings(BaseSettings):
    """Evaluation settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    #######################
    # Evaluation Settings #
    #######################
    K_FOLDS: int = 5
    LOGGING_PATH: str = "eval_results"

    ###########
    # Open AI #
    ###########
    OPENAI_CHAT_API: str = "https://api.openai.com/v1/chat/completions"
    OPENAI_EMBED_API: str = "https://api.openai.com/v1/embeddings"
    OPENAI_API_KEY: str  # specified in .env

    ###########
    # Mistral #
    ###########
    MISTRAL_CHAT_API: str = "https://api.mistral.ai/v1/chat/completions"
    MISTRAL_API_KEY: str  # specified in .env

    ##########
    # Cohere #
    ##########
    COHERE_EMBED_API: str = "https://api.cohere.com/v1/embed"
    COHERE_API_KEY: str  # specified in .env

    ##########
    # Google #
    ##########
    GOOGLE_EMBED_API: str = "https://generativelanguage.googleapis.com/v1beta/models/"
    GOOGLE_API_KEY: str  # specified in .env

    #########
    # Llama #
    #########
    HUGGING_FACE_LLAMA_TOKEN: str  # specified in .env


load_dotenv()
settings = Settings()
