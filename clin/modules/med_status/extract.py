import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
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


class Extractor:
    def __call__(self, i, df, nums, n_shots, llm) -> str:
        examples_nums_shot = clin.parse.sample_shots_excluding_i(i, nums, n_shots)
        prompt = get_multishot_prompt(df, examples_nums_shot, i)

        response = llm(prompt)
        return response


if __name__ == "__main__":
    nums = np.arange(4).tolist()
    np.random.default_rng(seed=1).shuffle(nums)
