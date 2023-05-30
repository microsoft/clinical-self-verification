import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import imodelsx.llm
import time


def get_multishot_prompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = ""
    for ex in examples_nums_shot:
        prompt += f"""Patient note: {df.iloc[ex]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

{clin.parse.list_medications(df.iloc[ex])}

"""
    prompt += f"""Patient note: {df.iloc[ex_num]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

-"""
    return prompt


def get_megaprompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = """Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

"""
    for ex in examples_nums_shot:
        prompt += f"""### Patient note: {df.iloc[ex]['snippet']}

{clin.parse.list_medications(df.iloc[ex])}

"""

    prompt += f"""

Before you provide your final response:
(1) Find any medications in the patient note that were missed.
(2) Find evidence for each medication as a text span in the input.
(3) Verify whether each extracted medication is actually a medication and that its status is correct.         

### Patient note: {df.iloc[ex_num]['snippet']}

-"""
    return prompt


class Extractor:
    def __call__(self, i, df, nums, n_shots, llm, use_megaprompt: bool) -> str:
        examples_nums_shot = clin.parse.sample_shots_excluding_i(i, nums, n_shots)
        if not use_megaprompt:
            prompt = get_multishot_prompt(df, examples_nums_shot, i)
        else:
            prompt = get_megaprompt(df, examples_nums_shot, i)

        response = llm(prompt)
        return response


if __name__ == "__main__":
    nums = np.arange(4).tolist()
    np.random.default_rng(seed=1).shuffle(nums)
