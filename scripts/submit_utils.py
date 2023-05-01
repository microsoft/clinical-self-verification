from typing import Any, Dict, List, Tuple

import itertools
import subprocess
import random
from functools import reduce
"""Handles utilities for job sweeps.
This file probably does not need to be edited.
"""


def run_args_list(
    args_list: List[Dict[str, Any]],
    cmd_python: str = 'python',
    script_name: str = '02_train_suffix.py',
    actually_run: bool = True,
    shuffle: bool = False,
    reverse: bool = False,
):
    """
    Params
    ------
    run_args_list
    cmd_python: str
        Command to run python
    script_name: str
        Name of script to run
    actually_run: bool
        Whether to actually run the script (otherwise just print the command)
    shuffle: bool
        Whether to shuffle the order of the script calls
    reverse: bool
        Whether to reverse the order of the script calls
    """
    # adjust order
    if shuffle:
        random.shuffle(args_list)
    if reverse:
        args_list = args_list[::-1]

    # loop over function calls
    for i in range(len(args_list)):
        param_str = cmd_python + ' ' + script_name + ' '
        for k, v in args_list[i].items():
            if isinstance(v, list):
                param_str += '--' + k + ' ' + ' '.join(v) + ' '
            else:
                param_str += '--' + k + ' ' + str(v) + ' '
        print(
            f'\n\n-------------------{i + 1}/{len(args_list)}--------------------\n', param_str)
        try:
            if actually_run:
                # os.system(param_str)
                sts = subprocess.Popen(param_str, shell=True).wait()
        except KeyboardInterrupt:
            print('Keyboard interrupt, exiting...')
            exit(0)
        except Exception as e:
            print(e)


def get_args_list(
    params_shared_dict: Dict[str, List],
    params_coupled_dict: Dict[Tuple[str], List[Tuple]],
) -> List[Dict[str, Any]]:
    _validate_arguments(params_shared_dict, params_coupled_dict)

    def combos_collapse(l: List[List[Dict]]) -> List[Dict]:
        # get param combos as List[Tuple[Dict]] then convert to List[Dict]
        return [
            # convert List[Dict[Tuple]] -> List[Dict]
            reduce(lambda a, b: {**a, **b}, dict_tup)
            # get param combos as List[Tuple[Dict]]
            for dict_tup in list(itertools.product(*l))
        ]

    # Shared params as List[List[Dict]]
    shared_combos_dict_list = combos_collapse(
        [[{k: v} for v in params_shared_dict[k]]
         for k in params_shared_dict.keys()]
    )

    # Coupled params as List[List[Dict]]]
    coupled_combos_dict_list = [[
        {k_tup[x]: v[i][x] for x in range(len(k_tup))}
        for i in range(len(v))]
        for k_tup, v in params_coupled_dict.items()
    ]
    if coupled_combos_dict_list == []:
        return shared_combos_dict_list

    # Combine each coupled List[Dict] with the shared List[Dict]
    combined_combos_dict_list = [
        combos_collapse(
            [coupled_combos_dict_list[i], shared_combos_dict_list])
        for i in range(len(coupled_combos_dict_list))
    ]
    args_list = sum(combined_combos_dict_list, [])
    return args_list


def _validate_arguments(
    params_shared_dict: Dict[str, List],
    params_coupled_dict: Dict[Tuple[str], List[Tuple]],
):
    for k, v in params_shared_dict.items():
        assert isinstance(
            k, str), f"params_shared_dict key {k} must be type list, got type {type(k)}"
        assert isinstance(
            v, list), f"params_shared_dict val {v} must be type list, got type {type(v)}"
    for k_tup, v_tup_list in params_coupled_dict.items():
        assert isinstance(
            k_tup, tuple), f"params_coupled_dict key {k_tup} must be type tuple, got type {type(k_tup)}"
        assert isinstance(
            v_tup_list, list), f"params_coupled_dict val {v_tup_list} must be type list, got type {type(v_tup_list)}"
        assert all([isinstance(x, str) for x in k_tup]
                   ), f"params_coupled_dict k {k_tup} must only contain strings"
        assert [len(
            k_tup) == x for x in v_tup_list], f"params_coupled_dict k and v must have same length but got {len(k_tup)} and {len(v_tup_list)} for {k_tup} and {v_tup_list} respectively"
        for k in k_tup:
            assert not k in params_shared_dict, f"params_coupled_dict key {k} should not be in params_shared_dict"


