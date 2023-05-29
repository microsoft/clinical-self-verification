from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python experiments/eval_model.py --dataset_name ebm

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1, 2, 3, 4, 5], # 1, 2, 3
    'save_dir': [join(repo_dir, 'results')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    # 'checkpoint': ['llama_65b', 'gpt-4-0314', 'text-davinci-003', 'chaoyi-wu/PMC_LLAMA_7B', 'gpt-3.5-turbo'], 
    'checkpoint': ['gpt-4-0314', 'text-davinci-003', 'gpt-3.5-turbo'], #, 'text-davinci-002',],
    # 'n_shots': [1, 5], # [1, 5]
    'n_shots': [1, 5],
    'dataset_name': ['ebm', 'medication_status'], # medication_status, ebm
    'use_megaprompt': [0], # 0, 1
    # 'role_verify': ['"You are an expert medical scribe."'], # only matters for chat models
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
    n_cpus=3,
    # shuffle=False,
    # reverse=True
)
