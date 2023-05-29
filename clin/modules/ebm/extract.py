import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
import time
import joblib
from os.path import join
from clin.config import PATH_REPO


def get_multishot_prompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = "Read each patient note and create a bulleted list of the interventions in the clinical trial. Intervention names should be simple, avoid acronyms, and not contain any information in parentheses.\n"
    for ex in examples_nums_shot:
        prompt += f"""### Patient note
{df.iloc[ex]['doc']}

### Create a bulleted list of the arms in this trial.
{clin.parse.list_to_bullet_str(df.iloc[ex]['interventions'])}

"""
    prompt += f"""### Patient note
{df.iloc[ex_num]['doc']}

### Create a bulleted list of the arms in this trial.
-"""
    return prompt


def get_megaprompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = """Read each patient note and create a bulleted list of the interventions in the clinical trial. Intervention names should be simple, avoid acronyms, and not contain any information in parentheses.
"""
    for ex in examples_nums_shot:
        prompt += f"""### Patient note
{df.iloc[ex]['doc']}

### Create a bulleted list of the arms in this trial.
{clin.parse.list_to_bullet_str(df.iloc[ex]['interventions'])}

"""
    prompt += f"""### Patient note
{df.iloc[ex_num]['doc']}

Before you provide your final response:
(1) Find any interventions that were missed.
(2) Find evidence for each intervention as a text span in the input.
(3) Verify whether each extracted intervention is actually an intervention.

### Create a bulleted list of the arms in this trial.
-"""
    return prompt


class Extractor:
    def __call__(self, i, df, nums, n_shots, llm, use_megaprompt: bool) -> List[str]:
        examples_nums_shot = clin.parse.sample_shots_excluding_i(i, nums, n_shots)
        if not use_megaprompt:
            prompt = get_multishot_prompt(df, examples_nums_shot, i)
        else:
            prompt = get_megaprompt(df, examples_nums_shot, i)

        # print(prompt)
        bullet_str = llm(prompt)
        interventions = clin.parse.bullet_str_to_list(bullet_str)
        return interventions


if __name__ == "__main__":
    df = joblib.load(join(PATH_REPO, "data", "ebm", "ebm_interventions_cleaned.pkl"))
    nums = np.arange(len(df)).tolist()
    np.random.default_rng(seed=13).shuffle(nums)
    dfe = df.iloc[nums]
    n = len(dfe)
    llm = clin.llm.get_llm("text-davinci-003")

    i = 0
    extractor = Extractor()
    n_shots = 1
    extractor(i, df, nums, n_shots, llm)
