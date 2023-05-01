import argparse
import sys
import os.path
from os.path import dirname, join
repo_dir = dirname(dirname(os.path.abspath(__file__)))


def get_main_args_list(fname='01_train_model.py'):
    """Returns main arguments from the argparser used by an experiments script
    """
    if fname.endswith('.py'):
        fname = fname[:-3]
    sys.path.append(join(repo_dir, 'experiments'))
    train_script = __import__(fname)
    args = train_script.add_main_args(argparse.ArgumentParser()).parse_args([])
    return list(vars(args).keys())

def fill_missing_args_with_default(df, fname='01_train_model.py'):
    """Returns main arguments from the argparser used by an experiments script
    """
    if fname.endswith('.py'):
        fname = fname[:-3]
    sys.path.append(join(repo_dir, 'experiments'))
    train_script = __import__(fname)
    parser = train_script.add_main_args(argparse.ArgumentParser())
    parser = train_script.add_computational_args(parser)
    args = parser.parse_args([])
    args_dict = vars(args)
    for k, v in args_dict.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna(v)
    return df