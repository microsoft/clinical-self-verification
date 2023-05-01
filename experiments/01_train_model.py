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

import clin.model
import clin.data
import cache_save_utils


def fit_model(model, X_train, y_train, feature_names, r):
    # fit the model
    fit_parameters = inspect.signature(model.fit).parameters.keys()
    if 'feature_names' in fit_parameters and feature_names is not None:
        model.fit(X_train, y_train, feature_names=feature_names)
    else:
        model.fit(X_train, y_train)

    return r, model

def evaluate_model(model, X_train, X_cv, X_test, y_train, y_cv, y_test, r):
    """Evaluate model performance on each split
    """
    metrics = {
        'accuracy': accuracy_score,
    }
    for split_name, (X_, y_) in zip(['train', 'cv', 'test'], [(X_train, y_train), (X_cv, y_cv), (X_test, y_test)]):
        y_pred_ = model.predict(X_)
        for metric_name, metric_fn in metrics.items():
            r[f'{metric_name}_{split_name}'] = metric_fn(y_, y_pred_)
        
    return r

# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--dataset_name', type=str,
                        default='rotten_tomatoes', help='name of dataset')
    parser.add_argument('--subsample_frac', type=float,
                        default=1, help='fraction of samples to use')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')

    # model args
    parser.add_argument('--model_name', type=str, choices=['decision_tree', 'ridge'],
                        default='decision_tree', help='name of model')
    parser.add_argument('--alpha', type=float, default=1,
                        help='regularization strength')
    parser.add_argument('--max_depth', type=int,
                        default=2, help='max depth of tree')
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
    dset, dataset_key_text = project_name.data.load_huggingface_dataset(
        dataset_name=args.dataset_name, subsample_frac=args.subsample_frac)
    X_train, X_test, y_train, y_test, feature_names = project_name.data.convert_text_data_to_counts_array(
        dset, dataset_key_text)    

    # load tabular data
    # https://csinva.io/imodels/util/data_util.html#imodels.util.data_util.get_clean_dataset
    # X_train, X_test, y_train, y_test, feature_names = imodels.get_clean_dataset('compas_two_year_clean', data_source='imodels', test_size=0.33)


    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, random_state=args.seed)    

    # load model
    model = project_name.model.get_model(args)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)

    # fit
    r, model = fit_model(model, X_train, y_train, feature_names, r)
    
    # evaluate
    r = evaluate_model(model, X_train, X_cv, X_test, y_train, y_cv, y_test, r)

    # save results
    joblib.dump(r, join(save_dir_unique, 'results.pkl')) # caching requires that this is called results.pkl
    joblib.dump(model, join(save_dir_unique, 'model.pkl'))
    logging.info('Succesfully completed :)\n\n')
