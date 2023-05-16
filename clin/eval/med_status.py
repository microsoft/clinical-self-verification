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
import clin.parse
import sklearn.metrics


def process_med_lists(
    med_status_dict: Dict[str, str], df_row: pd.Series, verbose=False
) -> List[bool]:
    """
    Given a dictionary of medication status, and a row of the dataframe,
    return precision and recall
    """

    # process meds_retrieved
    meds_retrieved = list(med_status_dict.keys())
    meds_retrieved = [med.strip(' "').lower() for med in meds_retrieved]

    # get meds_true
    meds_true = (
        clin.parse.str_to_list(df_row["active_medications"])
        + clin.parse.str_to_list(df_row["discontinued_medications"])
        + clin.parse.str_to_list(df_row["neither_medications"])
    )
    meds_true = [med.strip(' "').lower() for med in meds_true]
    return meds_retrieved, meds_true


def get_common_medications(
    med_status_dicts_list: List[List[Dict[str, str]]], dfe: pd.DataFrame
):
    # get the common retrieved medications for each row by all models
    n_runs_to_compare = len(med_status_dicts_list)
    n = len(dfe)
    common_meds = []
    for i in range(n_runs_to_compare):
        med_status_dicts = clin.parse.convert_keys_to_lowercase(
            med_status_dicts_list[i]
        )

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

    common_meds_status_gt_dict = [
        {
            med: _get_status_of_med(dfe.iloc[i], med)
            for med in common_meds[i]
            if _get_status_of_med(dfe.iloc[i], med) is not None
        }
        for i in range(len(dfe))
    ]
    return common_meds_status_gt_dict


def eval_medication_status(
    med_status_dicts_list: List[List[Dict[str, str]]],
    common_meds_status_gt_dict: Dict[str, str],
    verbose=False,
):
    """Compute the metrics for medication status,
    conditioned on the medications that are retrieved by all models and are valid medications in the groundtruth
    """
    n_runs_to_compare = len(med_status_dicts_list)
    n = len(common_meds_status_gt_dict)
    # compute conditional accuracy for all rows
    accs_cond = []
    f1s_macro_cond = []
    for i in tqdm(range(n_runs_to_compare)):
        status_extracted_list = []
        status_gt_list = []
        for j in range(n):
            med_status_dict = clin.parse.convert_keys_to_lowercase(
                [med_status_dicts_list[i][j]]
            )[0]
            for med in common_meds_status_gt_dict[j]:
                status_extracted_list.append(med_status_dict[med])
                status_gt_list.append(common_meds_status_gt_dict[j][med])
                if verbose:
                    if not med_status_dict[med] == common_meds_status_gt_dict[j][med]:
                        print(
                            "med",
                            med,
                            "\n\t",
                            "pred\t",
                            med_status_dict[med],
                            "\n\t",
                            "gt\t",
                            common_meds_status_gt_dict[j][med],
                        )
        accs_cond.append(
            np.mean(np.array(status_extracted_list) == np.array(status_gt_list))
        )
        f1s_macro_cond.append(
            sklearn.metrics.f1_score(
                status_gt_list, status_extracted_list, average="macro"
            )
        )

    return accs_cond, f1s_macro_cond