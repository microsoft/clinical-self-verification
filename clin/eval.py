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
import clin.parse
import sklearn.metrics


def eval_med_extraction(
    med_status_dict: Dict[str, str], df_row: pd.Series, verbose=False
) -> List[bool]:
    """
    Given a dictionary of medication status, and a row of the dataframe,
    return precision and recall
    """
    meds_retrieved = list(med_status_dict.keys())
    meds_retrieved = [med.strip(' "').lower() for med in meds_retrieved]
    meds_true = (
        clin.prompts.str_to_list(df_row["active_medications"])
        + clin.prompts.str_to_list(df_row["discontinued_medications"])
        + clin.prompts.str_to_list(df_row["neither_medications"])
    )
    meds_true = [med.strip(' "').lower() for med in meds_true]

    if verbose:
        if "".join(sorted(meds_retrieved)) == "".join(sorted(meds_true)):
            print("correct")
            # print("grt", sorted(meds_true))
        else:
            print("ret", sorted(meds_retrieved))
            print("grt", sorted(meds_true))
            print()

    true_pos = len(set(meds_retrieved).intersection(set(meds_true)))
    pred_pos = len(meds_retrieved)
    gt_pos = len(meds_true)
    return {
        "true_pos": true_pos,
        "pred_pos": pred_pos,
        "gt_pos": gt_pos,

        'fp_list': list(set(meds_retrieved) - set(meds_true)),
        'fn_list': list(set(meds_true) - set(meds_retrieved)),
    }


def calculate_metrics(med_status_dicts: List[Dict], dfe: pd.DataFrame, verbose=False):
    mets_dict_per_example = defaultdict(list)
    for i in range(len(dfe)):
        mets = eval_med_extraction(med_status_dicts[i], dfe.iloc[i], verbose=verbose)
        for k in mets.keys():
            mets_dict_per_example[k].append(mets[k])
    rec = np.sum(mets_dict_per_example["true_pos"]) / np.sum(mets_dict_per_example["gt_pos"])
    prec = np.sum(mets_dict_per_example["true_pos"]) / np.sum(mets_dict_per_example["pred_pos"])
    
    if verbose:
        print('fp', sum(mets_dict_per_example['fp_list'], []))
        print('fn', sum(mets_dict_per_example['fn_list'], []))

    mets_dict = {
        "recall": rec,
        "precision": prec,
        "f1": 2 * rec * prec / (rec + prec),
    }
    return mets_dict


def eval_medication_status(dfe: pd.DataFrame, r: pd.DataFrame):
    """Compute the metrics for medication status,
    conditioned on the medications that are retrieved by all models and are valid medications in the groundtruth
    """
    # get the common retrieved medications for each row by all models
    common_meds = []
    for i in range(len(r)):
        resps = r.iloc[i]["resps"]
        n = len(resps)
        med_status_dicts = [
            clin.parse.parse_response_medication_list(resps[j]) for j in range(n)
        ]

        if i == 0:
            common_meds = [set(med_status_dicts[j].keys()) for j in range(n)]
        else:
            common_meds = [
                common_meds[j].intersection(set(med_status_dicts[j].keys()))
                for j in range(n)
            ]

    # add status and only keep medications that are present in the groundtruth
    def _get_status_of_med(row, med):
        if med in row["active_medications"].lower():
            return "active"
        elif med in row["discontinued_medications"].lower():
            return "discontinued"
        elif med in row["neither_medications"].lower():
            return "neither"
        else:
            return None

    common_meds_status_dict = [
        {
            med: _get_status_of_med(dfe.iloc[i], med)
            for med in common_meds[i]
            if _get_status_of_med(dfe.iloc[i], med) is not None
        }
        for i in range(len(dfe))
    ]

    # compute conditional accuracy for all rows
    accs_cond = []
    f1s_macro_cond = []
    for i in tqdm(range(len(r))):
        resps = r.iloc[i]["resps"]
        status_resp_list = []
        status_list = []
        for j in range(len(resps)):
            med_status_dict = clin.parse.parse_response_medication_list(resps[j])
            for med in common_meds_status_dict[j]:
                status_resp_list.append(med_status_dict[med])
                status_list.append(common_meds_status_dict[j][med])
        accs_cond.append(np.mean(np.array(status_resp_list) == np.array(status_list)))
        f1s_macro_cond.append(
            sklearn.metrics.f1_score(status_list, status_resp_list, average="macro")
        )

    return accs_cond, f1s_macro_cond
