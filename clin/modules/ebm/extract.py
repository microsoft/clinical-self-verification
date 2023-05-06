import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
import time
import joblib
from os.path import join
from clin.config import PATH_REPO


def get_multishot_prompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = 'Read each patient note and create a bulleted list of the arms in the trial. Arm names should be simple and not contain any information in parentheses.\n'
    for ex in examples_nums_shot:
        prompt += f'''### Patient note ###
{df.iloc[ex]['doc']}

### Create a bulleted list of the arms in this trial.###
{clin.parse.list_to_bullet_str(df.iloc[ex]['interventions'])}

'''


    prompt += f'''### Patient note ###
{df.iloc[ex_num]['doc']}

### Create a bulleted list of the arms in this trial. ###
-'''
    return prompt

class Extractor:
    def __call__(self, i, df, nums, n_shots, llm) -> List[str]:
        if i - n_shots < 0:
            examples_nums_shot = nums[i - n_shots:] + nums[:i]
        else:
            examples_nums_shot = nums[i - n_shots: i]
        ex_num = nums[i]
        prompt = get_multishot_prompt(df, examples_nums_shot, ex_num)
        # print(prompt)
        bullet_str = llm(prompt)
        interventions = clin.parse.bullet_str_to_list(bullet_str)
        return interventions
    
if __name__ == '__main__':
    df = joblib.load(join(PATH_REPO, 'data', 'ebm', 'ebm_interventions_cleaned.pkl'))
    nums = np.arange(len(df)).tolist()
    np.random.default_rng(seed=13).shuffle(nums)
    dfe = df.iloc[nums]
    n = len(dfe)
    llm = clin.llm.get_llm('text-davinci-003')

    i = 0
    extractor = Extractor()
    n_shots = 1
    extractor(i, df, nums, n_shots, llm)