"""Sentence Diff with LLMs."""

import json
import logging
import math
import re
from typing import Callable, Dict, List, Optional, Union

import nltk
import requests
import torch
import transformers
from transformers import AutoTokenizer

try:
    from base_sentence_diff import BaseSentenceDiff
except ImportError:
    from .base_sentence_diff import BaseSentenceDiff
from config import EVAL_CONFIG, LLMProvider, ModelType, settings

logger = logging.getLogger(__name__)


class LLMSentenceDiff(BaseSentenceDiff):
    """SentenceDiff powered by an LLM."""

    def __init__(self, provider: LLMProvider, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.model_type = ModelType(provider.value)
        self.api_url = {
            LLMProvider.OPENAI: settings.OPENAI_CHAT_API,
            LLMProvider.MISTRAL: settings.MISTRAL_CHAT_API,
        }
        self.api_key = {
            LLMProvider.OPENAI: settings.OPENAI_API_KEY,
            LLMProvider.MISTRAL: settings.MISTRAL_API_KEY,
        }
        self.request = {
            LLMProvider.OPENAI: self._request_openai,
            LLMProvider.MISTRAL: self._request_mistral,
            LLMProvider.LLAMA: self._request_llama,
        }
        if provider is LLMProvider.LLAMA:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, token=settings.HUGGING_FACE_LLAMA_TOKEN
            )
            self.llama = transformers.pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=torch.float16,
                device_map="auto",
                token=settings.HUGGING_FACE_LLAMA_TOKEN,
            )
        self.max_retries = kwargs.get("max_retries")
        if self.max_retries is None:
            self.max_retries = EVAL_CONFIG[self.model_type]["max_retries"]

    def annotate_diff(
        self,
        premise: str,
        hypothesis: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        **kwargs,
    ) -> Union[str, None]:
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
                Additional arguments to the annotation. Possible values are:

                   * max_retries (:obj:`Optional[float]`, defaults to ``0.4``):
                        The maximum number of retries if the annotation fails
                        due to malformed outputs or API errors.

        Returns:
            :obj:`Union[str, None]`: The hypothesis with the differing spans
            annotated, or :obj:`None` if the model failed to provide a valid
            annotation.
        """
        if start_marker is None:
            start_marker = "{{"
        if end_marker is None:
            end_marker = "}}"
        for _ in range(math.ceil(9 / 10 * self.max_retries)):
            annotated_hypothesis = self.annotate_hypothesis(
                premise, hypothesis, start_marker, end_marker
            )
            if annotated_hypothesis:
                return annotated_hypothesis
        logger.warning(
            "maximum retries exceeded with model type '%s' and model '%s'",
            self.model_type,
            self.model,
        )
        return None

    def annotate_hypothesis(
        self,
        premise: str,
        hypothesis: str,
        start_marker: str,
        end_marker: str,
        **kwargs,
    ) -> Optional[str]:
        system_prompt, user_message = self._get_prompt(
            premise, hypothesis, start_marker, end_marker
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        for _ in range(2):
            llm_outputs = self.request[self.provider](messages)
            llm_outputs = (
                list(llm_outputs)
                if self.provider is LLMProvider.LLAMA
                else [llm_outputs]
            )
            for llm_output in llm_outputs:
                print(llm_output)
                annotated_hypothesis = self._extract_code_block(llm_output)
                if self._validate_annotated_hypothesis(
                    hypothesis, annotated_hypothesis, start_marker, end_marker
                ):
                    return annotated_hypothesis
            messages += [
                {"role": "assistant", "content": llm_outputs[0]},
                {
                    "role": "user",
                    "content": (
                        "Your response is incorrect. Please, read my instructions"
                        " above again and provide ONLY the annotated hypothesis"
                        " within a code block."
                    ),
                },
            ]
        print(
            f'\n  Failed to extract annotation from model\'s response: "{llm_outputs}"'
        )
        return None

    def _extract_code_block(self, response: str) -> Optional[str]:
        if not response:
            return None
        pattern = r"`{3}(?:[\w]*)\n([\S\s]+?)\n`{3}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        return None

    def _request_openai(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key[LLMProvider.OPENAI]}",
        }
        body = {"model": self.model, "messages": messages}
        body.update(kwargs)
        try:
            response = requests.post(
                self.api_url[LLMProvider.OPENAI],
                headers=headers,
                json=body,
                timeout=240,
            )
            if response.ok:
                response_body = response.json()
                if "error" in response_body:
                    print(f"\n  API Request failed {response_body['error']}")
                else:
                    return response_body["choices"][0]["message"]["content"].strip()
            print(f"\n  API Request failed with status code {response.status_code}")
        except Exception as ex:  # pylint: disable=broad-except
            print(f"\n  Annotation failed: '{ex}'")
        return None

    def _request_mistral(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key[LLMProvider.MISTRAL]}",
        }
        body = {"model": self.model, "messages": messages}
        body.update(kwargs)
        try:
            response = requests.post(
                self.api_url[LLMProvider.MISTRAL],
                headers=headers,
                json=body,
                timeout=240,
            )
            if response.ok:
                return response.json()["choices"][0]["message"]["content"].strip()
            print(f"\n  API Request failed with status code {response.status_code}")
        except Exception as ex:  # pylint: disable=broad-except
            print(f"\n  Annotation failed: '{ex}'")
        return None

    def _request_llama(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Optional[str]:
        prompt = (
            "<s>[INST]"
            f" <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
            f"{messages[1]['content']} [/INST]"
        )
        if len(messages) == 4:
            prompt += (
                f" {messages[2]['content']} </s><s>[INST]"
                f" {messages[3]['content']} [/INST]"
            )
        try:
            sequences = self.llama(
                prompt,
                do_sample=False,
                top_k=None,
                top_p=None,
                temperature=None,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=2.0,
                num_return_sequences=5,
                eos_token_id=self.tokenizer.eos_token_id,
                # max_length=200,
                # max_new_tokens=200,
            )
            for seq in sequences:
                yield seq["generated_text"].split("[/INST]")[-1].strip()
        except Exception as ex:  # pylint: disable=broad-except
            print(f"\n  Annotation failed: '{ex}'")
        yield None

    def _get_prompt(
        self,
        premise: str,
        hypothesis: str,
        start_marker: str,
        end_marker: str,
    ) -> str:
        input_ = {"premise": premise, "hypothesis": hypothesis}
        system_prompt: str = (
            "You are an NLP model able to detect differences in meaning in textual"
            " pairs. More concretely, given a premise and a hypothesis, you are able"
            " to compare them and annotate in the hypothesis the spans that are"
            " differing in meaning to the information included in the premise.\n\nHere"
            " is an example of an input and the expected output:\n\n# INPUT"
            ' 1\n```json\n{\n  "premise": "There was international outrage for the'
            ' decision.",\n  "hypothesis": "There was no reaction to the'
            ' decision."\n}\n```\n\n#OUTPUT 1\n```\nThere was {{no reaction}} to the'
            " decision.\n```\n\nAs you can see, the inputs are formatted as a JSON"
            " blob containing the premise and hypothesis. The response is a code block"
            " containing the  hypothesis with the differing spans enclosed"
            ' within the markers "{{" and "}}" Note that without these markers, both the'
            " input hypothesis and the annotated hypothesis are identical.\n\nHere is"
            " another example:\n\n#"
            ' INPUT 2\n```json\n{\n  "premise": "Microorganisms are too small to be'
            ' seen by the naked eye.",\n  "hypothesis": "Microorganisms have'
            ' considerable size and can be seen with your eyes."\n}\n```\n\n# OUTPUT'
            " 2\n```\nMicroorganisms {{have considerable size}} and {{can be seen with"
            " your eyes}}.\n```\n\nOn the other hand, here goes an example with an"
            " incorrect output, since the annotated hypothesis includes words not"
            ' present in the input hypothesis:\n\n# INPUT 3\n```json\n{\n  "premise":'
            ' "It is much warmer here than it used to be.", \n  "hypothesis": "It is'
            ' way colder here than it used to be."\n}\n```\n\n# (INCORRECT) OUTPUT'
            " 3\n```\nI believe it is {{way colder}} here than it used to"
            " be.\n```\n\nLet me show you one last example of an erroneous output."
            " In this case, a span is annotated that does not differ in meaning"
            " with respect to the information included in the premise:"
            '\n\n# INPUT 4\n```json\n{\n  "premise": "The deputy was'
            ' urged to provide an immediate apology for his controversial comments.",'
            ' \n  "hypothesis": "The deputy was pressed to issue a prompt apology for'
            ' his controversial comments."\n}\n```\n\n# (INCORRECT) OUTPUT 4\n```\nThe '
            " deputy was {{pressed to issue a prompt apology}} for his controversial"
            " comments.\n```\n\nAs shown in the last example, bear in mind that there"
            " might be sentence pairs with spans containing different wording but that"
            " are still equivalent in meaning. In these cases, the hypothesis must"
            " be returned as is, with no annotated spans."
        )
        user_message: str = (
            "You are now given the following JSON"
            f" input:\n\n```json\n{json.dumps(input_, indent=2)}\n```\n\nPlease,"
            f' provide the annotated hypothesis using the start marker "{start_marker}"'
            f' and the end marker "{end_marker}". Enclose the annotated hypothesis within'
            ' a code block using "```" so I can easily identify it. Please, reason your answer.'
        )
        return system_prompt, user_message

    def _validate_annotated_hypothesis(
        self,
        hypothesis: str,
        annotated_hypothesis: Optional[str],
        start_marker: str,
        end_marker: str,
    ) -> bool:
        if annotated_hypothesis is None:
            return False
        annotated_hypothesis_no_markers = annotated_hypothesis.replace(
            start_marker, ""
        ).replace(end_marker, "")
        if self.get_unigrams(hypothesis) != self.get_unigrams(
            annotated_hypothesis_no_markers
        ):
            print("\n  Annotated hypothesis does not correspond to input hypothesis:")
            print(f'    - Input hypothesis:     "{hypothesis}"')
            print(
                "    - Annotated hypothesis:"
                f" \"{' '.join(annotated_hypothesis.split())}\""
            )
            return False
        if annotated_hypothesis.count(start_marker) == annotated_hypothesis.count(
            end_marker
        ):
            return True
        print(f'\n  Invalid annotation schema: "{annotated_hypothesis}"')
        return False

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
