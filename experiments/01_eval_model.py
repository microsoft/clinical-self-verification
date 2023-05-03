import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import inspect
import datasets
import pandas as pd
from tqdm import tqdm
import time
from typing import List
import torch

import clin.llm
import clin.parse
import clin.eval
import clin.modules.extract
import clin.modules.evidence
import clin.modules.omission
import clin.modules.prune
import clin.modules.status
from imodelsx import cache_save_utils


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name", type=str, default="medication_status", help="name of dataset"
    )

    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir", type=str, default="results", help="directory for saving"
    )

    # model args
    parser.add_argument(
        "--checkpoint",
        type=str,  # choices=['gpt-4-0314', 'gpt-3.5-turbo', 'text-davinci-003',],
        default="text-davinci-003",
        help="name of llm checkpoint",
    )

    # prompt args
    parser.add_argument("--n_shots", type=int, default=5, help="number of shots")
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )
    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load text data
    if args.dataset_name in ["medication_status", "medication_attr", "coreference"]:
        dset = datasets.load_dataset("mitclinicalml/clinical-ie", args.dataset_name)
    else:
        dset = datasets.load_dataset(args.dataset_name)
    # val = pd.DataFrame.from_dict(dset['validation'])
    df = pd.DataFrame.from_dict(dset["test"])
    nums = np.arange(len(df)).tolist()
    np.random.default_rng(seed=13).shuffle(nums)
    dfe = df.iloc[nums]
    n = len(dfe)

    # load model
    llm = clin.llm.get_llm(args.checkpoint)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # perform basic extraction
    extractor = clin.modules.extract.Extractor()
    r["extracted_strs"] = [
        extractor(i, df, nums, args.n_shots, llm) for i in range(len(df))
    ]
    medications_dict_list = [
        clin.parse.parse_response_medication_list(r["extracted_strs"][i])
        for i in range(len(df))
    ]

    extracted_strs_orig = r["extracted_strs"]
    med_status_dict_list_orig = [
        clin.parse.parse_response_medication_list(extracted_strs_orig[i])
        for i in range(n)
    ]
    llm_verify = clin.llm.get_llm("text-davinci-003")

    ov = clin.modules.omission.OmissionVerifier()
    pv = clin.modules.prune.PruneVerifier()
    ev = clin.modules.evidence.EvidenceVerifier()
    sv = clin.modules.status.StatusVerifier()

    # apply individual verifiers
    # apply omission verifier
    med_status_dict_list_ov = [
        ov(dfe.iloc[i]["snippet"], bulleted_str=extracted_strs_orig[i], llm=llm_verify)
        for i in tqdm(range(n))
    ]

    # apply prune verifier
    med_status_dict_list_pv = [
        pv(dfe.iloc[i]["snippet"], bulleted_str=extracted_strs_orig[i], llm=llm_verify)
        for i in tqdm(range(n))
    ]

    # apply evidence verifier
    med_status_and_evidence = [
        ev(
            snippet=dfe.iloc[i]["snippet"],
            bulleted_str=extracted_strs_orig[i],
            llm=llm_verify,
        )
        for i in tqdm(range(n))
    ]
    med_status_dict_list_ev = [med_status_and_evidence[i][0] for i in range(n)]
    med_evidence_dict_list_ev = [med_status_and_evidence[i][1] for i in range(n)]

    # apply sequential verifiers
    logging.info("sequential verifiers...")
    med_status_dict_list_ov_ = [
        ov(
            dfe.iloc[i]["snippet"],
            bulleted_str=extracted_strs_orig[i],
            llm=llm_verify,
            lower=False,
        )
        for i in tqdm(range(n))
    ]
    bulleted_str_list_ov_ = [
        clin.parse.medication_dict_to_bullet_str(med_status_dict_list_ov_[i])
        for i in range(n)
    ]

    med_status_dict_list_pv_ = [
        pv(
            dfe.iloc[i]["snippet"],
            bulleted_str=bulleted_str_list_ov_[i],
            llm=llm_verify,
            lower=False,
        )
        for i in tqdm(range(n))
    ]
    bulleted_str_list_pv_ = [
        clin.parse.medication_dict_to_bullet_str(med_status_dict_list_pv_[i])
        for i in range(n)
    ]

    med_status_and_evidence_ = [
        ev(
            snippet=dfe.iloc[i]["snippet"],
            bulleted_str=bulleted_str_list_pv_[i],
            llm=llm_verify,
        )
        for i in tqdm(range(n))
    ]
    med_status_dict_list_ev_ = [med_status_and_evidence_[i][0] for i in range(n)]
    med_evidence_dict_list_ev_ = [med_status_and_evidence_[i][1] for i in range(n)]

    # compute metrics
    med_status_results = {
        "original": med_status_dict_list_orig,
        "ov": med_status_dict_list_ov,
        "pv": med_status_dict_list_pv,
        "ev": med_status_dict_list_ev,
        "ov_pv": med_status_dict_list_pv_,
        "ov_pv_ev": med_status_dict_list_ev_,
    }
    for k in med_status_results.keys():
        mets_dict_single = clin.eval.calculate_metrics(
            med_status_results[k], dfe, verbose=False
        )
        for k_met in mets_dict_single.keys():
            r[k_met + "___" + k] = mets_dict_single[k_met]
        r["dict_" + k] = med_status_results[k]

    # print metrics
    logging.info(f'precision: {r["precision___original"]}')
    logging.info(f'recall: {r["recall___original"]}')
    logging.info(f'precision: {r["precision___ov_pv_ev"]}')
    logging.info(f'recall: {r["recall___ov_pv_ev"]}')

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")
