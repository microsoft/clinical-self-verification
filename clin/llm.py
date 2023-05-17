import json
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain.cache import InMemoryCache
import re
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join, dirname
import os
import pickle as pkl
import langchain
from scipy.special import softmax
import openai
from langchain.llms.base import LLM
import hashlib
import torch
import time

# from mprompt.config import CACHE_DIR
CACHE_DIR = "/home/chansingh/clin/CACHE_OPENAI"
LLM_REPEAT_DELAY = 5  # how long to wait before recalling a failed llm call

# repo_dir = join(dirname(dirname(__file__)))


def get_llm(checkpoint, seed=1):
    if checkpoint.startswith("text-da"):
        return llm_openai(checkpoint, seed=seed)
    elif checkpoint.startswith("gpt-3") or checkpoint.startswith("gpt-4"):
        return llm_openai_chat(checkpoint, seed=seed)
    else:
        return llm_hf(checkpoint)


def repeatedly_call_with_delay(llm_call, delay=LLM_REPEAT_DELAY):
    """
    delay: float
        Number of seconds to wait between calls (None will not repeat)
    """
    if delay is None:
        return llm_call

    def wrapper(*args, **kwargs):
        response = None
        while response is None:
            try:
                response = llm_call(*args,  **kwargs)

                # fix for when this function was returning response rather than string
                if response is not None and not isinstance(response, str):
                    response = response["choices"][0]["message"]["content"]
            except:
                time.sleep(delay)
        return response

    return wrapper


def llm_openai(checkpoint="text-davinci-003", seed=1) -> LLM:
    class LLM_OpenAI:
        def __init__(self, checkpoint, seed):
            self.cache_dir = join(
                CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
            )
            self.checkpoint = checkpoint

        @repeatedly_call_with_delay
        def __call__(self, prompt: str, max_new_tokens=250, do_sample=True, stop=None):
            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            hash_str = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
            )
            if os.path.exists(cache_file):
                return pkl.load(open(cache_file, "rb"))

            response = openai.Completion.create(
                engine=self.checkpoint,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                stop=stop,
                # stop=["101"]
            )
            response_text = response["choices"][0]["text"]

            pkl.dump(response_text, open(cache_file, "wb"))
            return response_text

    return LLM_OpenAI(checkpoint, seed)


def llm_openai_chat(checkpoint="gpt-3.5-turbo", seed=1) -> LLM:
    class LLM_Chat:
        """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

        def __init__(self, checkpoint, seed):
            self.cache_dir = join(
                CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
            )
            self.checkpoint = checkpoint

        @repeatedly_call_with_delay
        def __call__(
            self,
            prompts_list: List[Dict[str, str]],
            max_new_tokens=250,
            do_sample=True,
            stop=None,
        ):
            """
            prompts_list: list of dicts, each dict has keys 'role' and 'content'
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
            prompts_list: str
                Alternatively, string which gets formatted into basic prompts_list:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": <<<<<prompts_list>>>>},
                ]
            """
            if isinstance(prompts_list, str):
                prompts_list = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompts_list},
                ]

            assert isinstance(prompts_list, list), prompts_list

            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            prompts_list_dict = {
                str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
            }
            if not self.checkpoint == "gpt-3.5-turbo":
                prompts_list_dict["checkpoint"] = self.checkpoint
            dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
            hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
            cache_file_raw = join(
                self.cache_dir,
                f"chat__raw_{hash_str}__num_tok={max_new_tokens}.pkl",
            )
            if os.path.exists(cache_file_raw):
                print("cached!")
                return pkl.load(open(cache_file_raw, "rb"))
            print("not cached")

            response = openai.ChatCompletion.create(
                model=self.checkpoint,
                messages=prompts_list,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                stop=stop,
                # stop=["101"]
            )

            pkl.dump(response, open(cache_file_raw, "wb"))
            return response["choices"][0]["message"]["content"]

    return LLM_Chat(checkpoint, seed)


def llm_hf(checkpoint="google/flan-t5-xl") -> LLM:
    def _get_tokenizer(checkpoint):
        if "facebook/opt" in checkpoint:
            # opt can't use fast tokenizer
            # https://huggingface.co/docs/transformers/model_doc/opt
            return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        else:
            return AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    class LLM_HF:
        def __init__(self, checkpoint):
            _checkpoint: str = checkpoint
            self._tokenizer = _get_tokenizer(_checkpoint)
            if "google/flan" in checkpoint:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )
            elif checkpoint == "gpt-xl":
                self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )

        def __call__(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            max_new_tokens=20,
            do_sample=False,
        ) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            inputs = self._tokenizer(
                prompt, return_tensors="pt", return_attention_mask=True
            ).to(
                self._model.device
            )  # .input_ids.to("cuda")
            # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
            # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
            # print('pad_token', self._tokenizer.pad_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                # pad_token=self._tokenizer.pad_token,
                pad_token_id=self._tokenizer.pad_token_id,
                # top_p=0.92,
                # top_k=0
            )
            out_str = self._tokenizer.decode(outputs[0])
            if "facebook/opt" in checkpoint:
                return out_str[len("</s>") + len(prompt) :]
            elif "google/flan" in checkpoint:
                print("full", out_str)
                return out_str[len("<pad>") : out_str.index("</s>")]
            else:
                return out_str[len(prompt) :]

        def _get_logit_for_target_token(
            self, prompt: str, target_token_str: str
        ) -> float:
            """Get logits target_token_str
            This is weird when token_output_ids represents multiple tokens
            It currently will only take the first token
            """
            # Get first token id in target_token_str
            target_token_id = self._tokenizer(target_token_str)["input_ids"][0]

            # get prob of target token
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                padding=False,
                truncation=False,
            ).to(self._model.device)
            # shape is (batch_size, seq_len, vocab_size)
            logits = self._model(**inputs)["logits"].detach().cpu()
            # shape is (vocab_size,)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            return probs_next_token[target_token_id]

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

        @property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

    return LLM_HF(checkpoint)


if __name__ == "__main__":
    llm = get_llm("text-davinci-003")
    text = llm("What do these have in common? Horse, ")
    print("text", text)
