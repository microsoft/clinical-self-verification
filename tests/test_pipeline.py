import os
from os.path import dirname, join


def test_small_pipeline():
    repo_dir = dirname(dirname(os.path.abspath(__file__)))
    prefix = f'PYTHONPATH={join(repo_dir, "experiments")}'
    cmd = prefix + ' python ' + \
        os.path.join(repo_dir, 'experiments',
                     '01_train_model.py --use_cache 0 --subsample_frac 0.1')
    print(cmd)
    exit_value = os.system(cmd)
    assert exit_value == 0, 'default pipeline passed'


if __name__ == '__main__':
    test_small_pipeline()
