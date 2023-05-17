from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python experiments/eval_model.py --dataset_name ebm

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join(repo_dir, 'results')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    'n_shots': [1, 5], # [1, 5, 10]
    'checkpoint': ['text-davinci-003', 'gpt-4-0314'], # gpt-3.5-turbo, gpt-4-0314, text-davinci-003
    'dataset_name': ['ebm', 'medication_status'], # medication_status, ebm
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'eval_model.py'),
    actually_run=True,
)