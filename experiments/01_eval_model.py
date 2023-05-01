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

import clin.llm
import clin.prompts
import cache_save_utils


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--dataset_name', type=str,
                        default='medication_status', help='name of dataset')
    parser.add_argument('--subsample_frac', type=float,
                        default=1, help='fraction of samples to use')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')

    # model args
    parser.add_argument('--checkpoint', type=str, choices=['gpt-4-0314'],
                        default='gpt-4-0314', help='name of llm checkpoint')


    # prompt args
    parser.add_argument('--n_shots', type=int, default=0, help='number of shots')
    return parser

def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)
    """
    parser.add_argument('--use_cache', type=int, default=1, choices=[0, 1],
                        help='whether to check for cache')
    return parser

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir)
    
    if args.use_cache and already_cached:
        logging.info(
            f'cached version exists! Successfully skipping :)\n\n\n')
        exit(0)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))
    logging.info(f'\n\n\tsaving to ' + save_dir_unique + '\n')

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # load text data
    # dataset: mitclinicalml/clinical-ie
    if args.dataset_name in ['medication_status', 'medication_attr', 'coreference']:
    # 3 splits here: 'medication_status', 'medication_attr', 'coreference
        dset = datasets.load_dataset('mitclinicalml/clinical-ie', args.dataset_name)
    else:
        dset = datasets.load_dataset(args.dataset_name)
    val = pd.DataFrame.from_dict(dset['validation'])
    test = pd.DataFrame.from_dict(dset['test'])
    df = pd.concat([val, test])

    

    # load model
    llm = clin.llm.get_llm('gpt-4-0314')



    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)


    # evaluate model
    resps = []
    nums = np.arange(len(df)).tolist()
    np.random.default_rng(seed=13).shuffle(nums)
    for i in tqdm(range(len(nums))):
        # print(i)
        if i - args.n_shots < 0:
            examples_nums_shot = nums[i - args.n_shots:] + nums[:i]
        else:
            examples_nums_shot = nums[i - args.n_shots: i]
        ex_num = nums[i]
        prompt = clin.prompts.get_multishot_prompt(df, examples_nums_shot, ex_num)
        # print('prompt', prompt)

        response = None
        while response is None:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    # {"role": "user", "content": "Where was it played?"}
                ]
                response = llm(messages)
            except:
                time.sleep(1)
        # if response is not None:
        response_text = response['choices'][0]['message']['content']
        resps.append(response_text)
    r['resps'] = resps


    # save results
    joblib.dump(r, join(save_dir_unique, 'results.pkl')) # caching requires that this is called results.pkl
    logging.info('Succesfully completed :)\n\n')
