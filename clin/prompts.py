import numpy as np
from typing import List

def list_medications(row) -> str:
        def str_to_list(s):
            l = s.replace('[', '').replace(']', '').split(',')
            l = [val.strip() for val in l]
            if l == ['']:
                return []
            else:
                return l
            
        d = [('active', val) for val in str_to_list(row['active_medications'])] + \
            [('discontinued', val) for val in str_to_list(row['discontinued_medications'])] + \
            [('neither', val) for val in str_to_list(row['neither_medications'])]
        np.random.default_rng(seed=13).shuffle(d)
        # print(d)
        s = '- ' + '\n- '.join([f'{med} ({status})' for status, med in d])
        return s

def get_multishot_prompt(df, examples_nums_shot: List[int], ex_num: int):
    prompt = ''
    for ex in examples_nums_shot:
        prompt += f'''Patient note: {df.iloc[ex]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

{list_medications(df.iloc[ex])}

'''
    prompt += f'''Patient note: {df.iloc[ex_num]['snippet']}

Create a bulleted list of which medications are mentioned and whether they are active, discontinued, or neither.

-'''
    return prompt

# """
# - "Kadian" (active)
# -"Dilaudid" (discontinued)
# -"Levaquin" (active)
# """