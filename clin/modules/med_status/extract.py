import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
import time

def get_multishot_prompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = ''
    for ex in examples_nums_shot:
        prompt += f'''Patient note: {df.iloc[ex]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

{clin.parse.list_medications(df.iloc[ex])}

'''
    prompt += f'''Patient note: {df.iloc[ex_num]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

-'''
    return prompt

class Extractor:
    def __call__(self, i, df, nums, n_shots, llm) -> str:
        if i - n_shots < 0:
            examples_nums_shot = nums[i - n_shots:] + nums[:i]
        else:
            examples_nums_shot = nums[i - n_shots: i]
        ex_num = nums[i]
        prompt = get_multishot_prompt(df, examples_nums_shot, ex_num)

        response = llm(prompt)
        return response
    
