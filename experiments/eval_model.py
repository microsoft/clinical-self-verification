import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import joblib
import datasets
import pandas as pd
from tqdm import tqdm
import torch
import clin.config
import imodelsx.llm
import clin.parse
import clin.eval.eval
import clin.eval.med_status
import clin.eval.ebm
from clin.modules import med_status
from clin.modules import ebm
from clin.modules.med_status import extract, omission, prune, evidence, status
from clin.modules.ebm import extract, omission, prune, evidence
from imodelsx import cache_save_utils


# python experiments/01_eval_model.py --use_cache 0
def eval_med_status(r, args, df, nums, n):
    llm = imodelsx.llm.get_llm(args.checkpoint, seed=args.seed)

    # perform basic extraction
    extractor = med_status.extract.Extractor()
    r["extracted_strs"] = [
        extractor(i, df, nums, args.n_shots, llm, args.use_megaprompt)
        for i in tqdm(range(len(df)))
    ]

    extracted_strs_orig = r["extracted_strs"]
    med_status_dict_list_orig = [
        clin.parse.parse_response_medication_list(extracted_strs_orig[i])
        for i in range(n)
    ]
    med_status_results = {"original": med_status_dict_list_orig}

    if not args.use_megaprompt:
        # medications_dict_list = [
        #     clin.parse.parse_response_medication_list(r["extracted_strs"][i])
        #     for i in range(len(df))
        # ]

        # load llm for verification
        if args.checkpoint_verify is None:
            args.checkpoint_verify = args.checkpoint
        llm = imodelsx.llm.get_llm(
            args.checkpoint_verify, seed=args.seed, role=args.role_verify
        )

        ov = med_status.omission.OmissionVerifier()
        pv = med_status.prune.PruneVerifier()
        ev = med_status.evidence.EvidenceVerifier()
        sv = med_status.status.StatusVerifier()

        # apply individual verifiers ####################################
        # apply omission verifier
        med_status_dict_list_ov = [
            ov(df.iloc[i]["snippet"], bulleted_str=extracted_strs_orig[i], llm=llm)
            for i in tqdm(range(n))
        ]

        # apply prune verifier
        med_status_dict_list_pv = [
            pv(df.iloc[i]["snippet"], bulleted_str=extracted_strs_orig[i], llm=llm)
            for i in tqdm(range(n))
        ]

        # apply evidence verifier
        med_status_and_evidence = [
            ev(
                snippet=df.iloc[i]["snippet"],
                bulleted_str=extracted_strs_orig[i],
                llm=llm,
            )
            for i in tqdm(range(n))
        ]
        med_status_dict_list_ev = [med_status_and_evidence[i][0] for i in range(n)]
        med_evidence_dict_list_ev = [med_status_and_evidence[i][1] for i in range(n)]

        # apply sequential verifiers
        logging.info("sequential verifiers...")
        med_status_dict_list_ov_ = [
            ov(
                df.iloc[i]["snippet"],
                bulleted_str=extracted_strs_orig[i],
                llm=llm,
            )
            for i in tqdm(range(n))
        ]
        bulleted_str_list_ov_ = [
            clin.parse.medication_dict_to_bullet_str(med_status_dict_list_ov_[i])
            for i in range(n)
        ]

        med_status_dict_list_pv_ = [
            pv(
                df.iloc[i]["snippet"],
                bulleted_str=bulleted_str_list_ov_[i],
                llm=llm,
            )
            for i in tqdm(range(n))
        ]
        bulleted_str_list_pv_ = [
            clin.parse.medication_dict_to_bullet_str(med_status_dict_list_pv_[i])
            for i in range(n)
        ]

        med_status_and_evidence_ = [
            ev(
                snippet=df.iloc[i]["snippet"],
                bulleted_str=bulleted_str_list_pv_[i],
                llm=llm,
            )
            for i in tqdm(range(n))
        ]
        med_status_dict_list_ev_ = [med_status_and_evidence_[i][0] for i in range(n)]
        med_evidence_dict_list_ev_ = [med_status_and_evidence_[i][1] for i in range(n)]

        # status verifier
        logging.info("status verifier....")
        med_status_dict_list_sv = [
            sv(
                df.iloc[i]["snippet"],
                med_status_dict=med_status_dict_list_ev_[i],
                med_evidence_dict=med_evidence_dict_list_ev_[i],
                llm=llm,
            )
            for i in tqdm(range(n))
        ]

        # compute metrics and add to r
        med_status_results = {
            "original": med_status_dict_list_orig,
            "ov": med_status_dict_list_ov,
            "pv": med_status_dict_list_pv,
            "ev": med_status_dict_list_ev,
            "ov_pv": med_status_dict_list_pv_,
            "ov_pv_ev": med_status_dict_list_ev_,
            "sv": med_status_dict_list_sv,
        }
        r["dict_evidence_ov_pv_ev"] = med_evidence_dict_list_ev

    # compute eval
    for k in med_status_results.keys():
        mets_df = pd.DataFrame(
            [
                clin.eval.eval.calculate_precision_recall_from_lists(
                    *clin.eval.med_status.process_med_lists(
                        med_status_results[k][i], df.iloc[i]
                    )
                )
                for i in range(len(df))
            ]
        )
        mets_dict_single = clin.eval.eval.aggregate_precision_recall(mets_df)
        for k_met in mets_dict_single.keys():
            r[k_met + "___" + k] = mets_dict_single[k_met]
        r["dict_" + k] = med_status_results[k]
    return r


def eval_ebm(r, args, df, nums, n):
    extractor = ebm.extract.Extractor()
    ov = ebm.omission.OmissionVerifier()
    pv = ebm.prune.PruneVerifier()
    ev = ebm.evidence.EvidenceVerifier()

    llm = imodelsx.llm.get_llm(args.checkpoint, seed=args.seed)

    r["list_original"] = [
        extractor(i, df, nums, args.n_shots, llm, args.use_megaprompt)
        for i in tqdm(range(len(df)))
    ]
    if not args.use_megaprompt:
        if args.checkpoint_verify is None:
            args.checkpoint_verify = args.checkpoint
        llm = imodelsx.llm.get_llm(
            args.checkpoint_verify, seed=args.seed, role=args.role_verify
        )

        r["list_ov"] = [
            ov(df.iloc[i]["doc"], bullet_list=r["list_original"][i], llm=llm)
            for i in tqdm(range(n))
        ]

        r["list_pv"] = [
            pv(df.iloc[i]["doc"], bullet_list=r["list_original"][i], llm=llm)
            for i in tqdm(range(n))
        ]

        r["list_ov_pv"] = [
            pv(df.iloc[i]["doc"], bullet_list=r["list_ov"][i], llm=llm)
            for i in tqdm(range(n))
        ]

        r["dict_evidence_ov_pv_ev"] = [
            ev(df.iloc[i]["doc"], bullet_list=r["list_ov_pv"][i], llm=llm)
            for i in tqdm(range(n))
        ]
        r["list_ov_pv_ev"] = [
            list(r["dict_evidence_ov_pv_ev"][i].keys()) for i in range(n)
        ]

    # add metrics to r
    ks_list = [k for k in r.keys() if k.startswith("list_")]
    for k in ks_list:
        mets_dict_single = clin.eval.eval.aggregate_precision_recall(
            pd.DataFrame(
                [
                    clin.eval.eval.calculate_precision_recall_from_lists(
                        *clin.eval.ebm.process_ebm_lists(
                            r[k][i], df.iloc[i]["interventions"]
                        ),
                        verbose=False,
                    )
                    for i in range(len(df))
                ]
            )
        )
        for k_met in mets_dict_single.keys():
            r[k_met + "___" + k.replace("list_", "")] = mets_dict_single[k_met]
    return r


def get_data(args):
    if args.dataset_name in ["medication_status", "medication_attr", "coreference"]:
        dset = datasets.load_dataset("mitclinicalml/clinical-ie", args.dataset_name)
        df = pd.DataFrame.from_dict(dset["test"])
    elif args.dataset_name == "ebm":
        df = joblib.load(
            join(clin.config.PATH_REPO, "data", "ebm", "ebm_interventions_cleaned.pkl")
        )
        df = df.iloc[:100]
    else:
        raise Exception(f"dataset {args.dataset_name} not recognized")

    nums = np.arange(len(df))
    np.random.default_rng(seed=args.seed).shuffle(nums)
    n = len(df)
    return df, nums, n


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medication_status",
        help="name of dataset",
        choices=["medication_status", "ebm"],
    )

    # training misc args
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed (stratifies different llm regenerations, although openai doesn't actually seed)",
    )
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
    parser.add_argument(
        "--checkpoint_verify",
        type=str,  # choices=['gpt-4-0314', 'gpt-3.5-turbo', 'text-davinci-003',],
        default=None,  # if not specified, will default to the value of args.checkpoint
        help="name of llm checkpoint used for verification",
    )
    parser.add_argument(
        "--role_verify",
        type=str,
        default=None,
        help="role of llm checkpoint used for verification",
    )
    parser.add_argument(
        "--use_megaprompt",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to use megaprompt (runs extraction with a long prompt and does not run self-verification)",
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
    df, nums, n = get_data(args)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # evaluate
    if args.dataset_name == "medication_status":
        r = eval_med_status(r, args, df, nums, n)
    elif args.dataset_name == "ebm":
        r = eval_ebm(r, args, df, nums, n)

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")
