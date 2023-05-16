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

# note, eval should be case-insensitive, preprocess before passing to these funcs
def calculate_precision_recall_from_lists(
    pred_list: List[str], gt_list: List[str], verbose=False
):
    true_pos = len(set(pred_list).intersection(set(gt_list)))
    pred_pos = len(pred_list)
    gt_pos = len(gt_list)

    if verbose:
        if "".join(sorted(pred_list)) == "".join(sorted(gt_list)):
            print("correct")
            # print("grt", sorted(meds_true))
        else:
            print("pred", sorted(pred_list))
            print("true", sorted(gt_list))
            print()

    return {
        "true_pos": true_pos,
        "pred_pos": pred_pos,
        "gt_pos": gt_pos,
        "fp_list": list(set(pred_list) - set(gt_list)),
        "fn_list": list(set(gt_list) - set(pred_list)),
    }


def aggregate_precision_recall(mets_df: pd.DataFrame, verbose=False):
    rec = np.sum(mets_df["true_pos"]) / np.sum(mets_df["gt_pos"])
    prec = np.sum(mets_df["true_pos"]) / np.sum(mets_df["pred_pos"])
    mets_dict = {
        "recall": rec,
        "precision": prec,
        "f1": 2 * rec * prec / (rec + prec),
    }
    if verbose:
        print("fp", sum(mets_df["fp_list"], []))
        print("fn", sum(mets_df["fn_list"], []))
    return mets_dict
