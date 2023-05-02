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
import sklearn.metrics


def eval_med_extraction(med_status_dict: Dict[str, str], df_row: pd.Series) -> List[bool]:
    """
    Given a dictionary of medication status, and a row of the dataframe,
    return precision and recall
    """
    meds_retrieved = list(med_status_dict.keys())
    meds_true = clin.prompts.str_to_list(df_row['active_medications']) + \
        clin.prompts.str_to_list(df_row['discontinued_medications']) + \
        clin.prompts.str_to_list(df_row['neither_medications'])
    meds_true = [med.strip(' "').lower() for med in meds_true]

    # print(sorted(meds_retrieved))
    # print(sorted(meds_true))
    # print()    

    # compute precision and recall
    precision = len(set(meds_retrieved).intersection(set(meds_true))) / len(meds_retrieved)
    recall = len(set(meds_retrieved).intersection(set(meds_true))) / len(meds_true)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # tp = len(set(meds_retrieved).intersection(set(meds_true)))
    # fp = len(set(meds_retrieved).difference(set(meds_true)))
    # fn = len(set(meds_true).difference(set(meds_retrieved)))
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def calculate_metrics(med_status_dicts: List, dfe: pd.DataFrame):
    mets_dict = defaultdict(list)
    for i in range(len(dfe)):
        # print(i)
        # medications_list_resp = clin.parse.parse_response_medication_list(r['resps'][i])
        mets = eval_med_extraction(med_status_dicts[i], dfe.iloc[i])
        for k in mets.keys():
            mets_dict[k].append(mets[k])
    return mets_dict

def eval_medication_status(dfe: pd.DataFrame, r: pd.DataFrame):
    """Compute the metrics for medication status,
    conditioned on the medications that are retrieved by all models and are valid medications in the groundtruth
    """
    # get the common retrieved medications for each row by all models
    common_meds = []
    for i in range(len(r)):
        resps = r.iloc[i]['resps']
        n = len(resps)
        med_status_dicts = [clin.eval.parse_response_medication_list(resps[j]) for j in range(n)]

        if i == 0:
            common_meds = [set(med_status_dicts[j].keys()) for j in range(n)]
        else:
            common_meds = [common_meds[j].intersection(set(med_status_dicts[j].keys())) for j in range(n)]
    common_meds

    # add status and only keep medications that are present in the groundtruth
    def _get_status_of_med(row, med):
        if med in row['active_medications'].lower():
            return 'active'
        elif med in row['discontinued_medications'].lower():
            return 'discontinued'
        elif med in row['neither_medications'].lower():
            return 'neither'
        else:
            return None
    common_meds_status_dict = [
        {med: _get_status_of_med(dfe.iloc[i], med) for med in common_meds[i] if _get_status_of_med(dfe.iloc[i], med) is not None}
        for i in range(len(dfe))
    ]

    # compute conditional accuracy for all rows
    accs_cond = []
    f1s_macro_cond = []
    for i in tqdm(range(len(r))):
        resps = r.iloc[i]['resps']
        status_resp_list = []
        status_list = []
        for j in range(len(resps)):
            med_status_dict = clin.eval.parse_response_medication_list(resps[j])
            for med in common_meds_status_dict[j]:
                status_resp_list.append(med_status_dict[med])
                status_list.append(common_meds_status_dict[j][med])
        accs_cond.append(np.mean(np.array(status_resp_list) == np.array(status_list)))
        f1s_macro_cond.append(sklearn.metrics.f1_score(status_list, status_resp_list, average='macro'))

    return accs_cond, f1s_macro_cond