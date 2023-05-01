import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import joblib
import sys
import datasets
import time
import openai
import numpy as np
from typing import List, Dict
from collections import defaultdict
import clin.prompts

def parse_response_medication_list(s: str) -> Dict[str, str]:
    """
    "Gatifloxacin" (initiated)
    - "Acyclovir" (prophylactic therapy through day 100)
    - "Bactrim" (active for PCP prophylaxis)
    - "Systemic steroids" (weaned)
    - "Cyclosporin" (discontinued)

    -> 

    {
        'Gatifloxacin': 'initiated',
        'Acyclovir': 'prophylactic therapy through day 100',
        'Bactrim': 'active for PCP prophylaxis',
        'Systemic steroids': 'weaned',
        'Cyclosporin': 'discontinued',
    }
    """
    s = s.replace('- ', '')
    s_list = s.split('\n')
    med_status_dict = {}
    for i, s in enumerate(s_list):
        # find second occurence of "
        idx = s.find('"', s.find('"') + 1)
        medication = s[:idx].strip('"')
        status = s[idx + 1:].strip().strip('()')
        med_status_dict[medication] = status
    return med_status_dict

def eval_med_extraction(med_status_dict: Dict[str, str], df_row: pd.Series) -> List[bool]:
    """
    Given a dictionary of medication status, and a row of the dataframe,
    return precision and recall
    """
    meds_retrieved = list(med_status_dict.keys())
    meds_true = clin.prompts.str_to_list(df_row['active_medications']) + \
        clin.prompts.str_to_list(df_row['discontinued_medications']) + \
        clin.prompts.str_to_list(df_row['neither_medications'])

    # clean up
    def drop_dosage(med: str) -> str:
        med_words = med.split()
        # if first word is numeric, can be okay (this case is for "6 mp")
        for j in range(1, len(med_words)):
            if med_words[j].isnumeric():
                return ' '.join(med_words[:j])
        return med
    meds_retrieved = [med.strip(' "').lower() for med in meds_retrieved]
    meds_retrieved = [drop_dosage(med) for med in meds_retrieved]
    meds_true = [med.strip(' "').lower() for med in meds_true]

    print(sorted(meds_retrieved))
    print(sorted(meds_true))
    print()    

    # compute precision and recall
    precision = len(set(meds_retrieved).intersection(set(meds_true))) / len(meds_retrieved)
    recall = len(set(meds_retrieved).intersection(set(meds_true))) / len(meds_true)
    return {
        'precision': precision,
        'recall': recall,
    }