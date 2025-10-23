"""Embedding generation."""

import time
from typing import List

import requests
import torch
from config import settings


class OpenAIEmbeddingGenerator:
    """OpenAI embedding generator."""

    @classmethod
    def generate(
        cls,
        strings_: List[str],
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        }
        input_batches = [
            strings_[i : i + batch_size] for i in range(0, len(strings_), batch_size)
        ]
        embeddings = []
        for batch in input_batches:
            body = {"model": model, "input": batch}
            body.update(kwargs)
            tries = 0
            while True:
                try:
                    response = requests.post(
                        settings.OPENAI_EMBED_API,
                        headers=headers,
                        json=body,
                        timeout=240,
                    ).json()
                except Exception as ex:
                    response = str(ex)
                if isinstance(response, dict) and "data" in response:
                    embeddings += [data["embedding"] for data in response["data"]]
                    break  # success
                if tries == 10:
                    raise RuntimeError(
                        f"embedding generation request failed: {response}"
                    )
                tries += 1
                time.sleep(5)  # wait before retrying
        return torch.tensor(embeddings)


class CohereEmbeddingGenerator:
    """Cohere embedding generator."""

    @classmethod
    def generate(
        cls,
        strings_: List[str],
        model: str = "embed-english-v3.0",
        batch_size: int = 96,
        input_type: str = "classification",
        **kwargs,
    ) -> torch.Tensor:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.COHERE_API_KEY}",
        }
        input_batches = [
            strings_[i : i + batch_size] for i in range(0, len(strings_), batch_size)
        ]
        embeddings = []
        for batch in input_batches:
            body = {"model": model, "texts": batch, "input_type": input_type}
            body.update(kwargs)
            tries = 0
            while True:
                try:
                    response = requests.post(
                        settings.COHERE_EMBED_API,
                        headers=headers,
                        json=body,
                        timeout=240,
                    ).json()
                except Exception as ex:
                    response = str(ex)
                if isinstance(response, dict) and "embeddings" in response:
                    embeddings += response["embeddings"]
                    break  # success
                if tries == 10:
                    raise RuntimeError(
                        f"embedding generation request failed: {response}"
                    )
                tries += 1
                time.sleep(5)  # wait before retrying
        return torch.tensor(embeddings)


class GoogleEmbeddingGenerator:
    """Google embedding generator."""

    @classmethod
    def generate(
        cls,
        strings_: List[str],
        model: str = "text-embedding-004",
        batch_size: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        headers = {
            "Content-Type": "application/json",
        }
        input_batches = [
            strings_[i : i + batch_size] for i in range(0, len(strings_), batch_size)
        ]
        embeddings = []
        for batch in input_batches:
            body = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]},
                        "task_type": "SEMANTIC_SIMILARITY",
                    }
                    for text in batch
                ]
            }
            body.update(kwargs)
            tries = 0
            while True:
                try:
                    response = requests.post(
                        f"{settings.GOOGLE_EMBED_API.rstrip('/')}/{model}:batchEmbedContents",
                        headers=headers,
                        params={"key": settings.GOOGLE_API_KEY},
                        json=body,
                        timeout=240,
                    ).json()
                except Exception as ex:
                    response = str(ex)
                if isinstance(response, dict) and "embeddings" in response:
                    embeddings += [data["values"] for data in response["embeddings"]]
                    break  # success
                if tries == 30:
                    raise RuntimeError(
                        f"embedding generation request failed: {response}"
                    )
                tries += 1
                time.sleep(5)  # wait before retrying
        return torch.tensor(embeddings)
