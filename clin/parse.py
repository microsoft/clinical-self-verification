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
from typing import List, Dict, Tuple
from collections import defaultdict
import clin.prompts
import sklearn.metrics

def _drop_dosage(med: str) -> str:
        # if first word is numeric, can be okay (this case is for "6 mp")
        med_words = med.split()
        for j in range(1, len(med_words)):
            if med_words[j].isnumeric():
                return ' '.join(med_words[:j])
        return med


def parse_response_medication_list(s: str, with_status=True, lower=True) -> Dict[str, str]:
    """
    "Gatifloxacin" (initiated)
    - "Acyclovir" (prophylactic therapy through day 100)
    - "Bactrim" (active for PCP prophylaxis)
    - "Systemic steroids" (weaned)
    - "Cyclosporin" (discontinued)

    -> 

    {
        'gatifloxacin': 'initiated',
        'acyclovir': 'prophylactic therapy through day 100',
        'bactrim': 'active for PCP prophylaxis',
        'systemic steroids': 'weaned',
        'cyclosporin': 'discontinued',
    }

    Note: ends early if it encounters "None" in the list
    Params
    ------
    with_status: bool
        If true, input has no status in parentheses and outputs None as values for each key
        Inputs might not be in quotes
    """

    s = s.replace('- ', '')
    s_list = s.split('\n')
    med_status_dict = {}
    for i, s in enumerate(s_list):
        if s.lower().strip('-" ').startswith("non"):
            break

        if with_status:
            # find second occurence of "
            idx = s.find('"', s.find('"') + 1)
            medication = s[:idx].strip('"').strip(' "')
        else:
            medication = s.strip('"').strip(' "')

        # clean up
        if lower:
            medication = medication.lower()
        medication = _drop_dosage(medication)

        if with_status:
            status = s[idx + 1:].strip().strip('()').lower()
            med_status_dict[medication] = status
        else:
            med_status_dict[medication] = None
    return med_status_dict


def parse_response_medication_list_with_evidence(s: str, lower=True) -> Tuple[Dict[str, str]]:
    """
    "Percocet" (discontinued) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
    - "Gemzar" (active) "received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty"
    - "OxyContin IR" (active) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
    - "fentanyl" (active) "she was weaned off her PCA and started on a fentanyl patch"'''

    -> 

    (
        {
            'percocet': 'discontinued',
            'gemzar': 'active',
            'oxycontin ir': 'active',
            'fentanyl': 'active',
        },
        {
            'percocet': 'along with Percocet for breakthrough pain which was then switched to OxyContin IR',
            'gemzar': 'received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty',
            'oxycontin ir': 'along with Percocet for breakthrough pain which was then switched to OxyContin IR',
            'fentanyl': 'she was weaned off her PCA and started on a fentanyl patch',
        }
    )
    """

    s = s.replace('- ', '')
    s_list = s.split('\n')
    med_status_dict = {}
    med_evidence_dict = {}
    for i, s in enumerate(s_list):

        # find second occurence of "
        idx = s.find('"', s.find('"') + 1)
        medication = s[:idx].strip('"').strip(' "')
        if lower:
            medication = medication.lower()


        # clean up
        medication = _drop_dosage(medication)

        # find close paren
        idx2 = s.find(')', idx + 1)
        status = s[idx + 1: idx2].strip().strip('()').lower()

        # add evidence
        evidence = s[idx2 + 1:].strip().strip(' "').lower()
        med_status_dict[medication] = status
        med_evidence_dict[medication] = evidence

        # print(s, '**', s[:idx], '***', s[idx + 1:idx + 1 + idx2], '****', s[idx + 1 + idx2 + 1:])
    return med_status_dict, med_evidence_dict


def medication_dict_to_bullet_str(med_status_dict):
    '''
    Given a medication status dictionary, return a bulleted string
    
    Example
    -------
    {
        'Percocet': 'discontinued',
        'Gemzar': 'active',
        'Oxycontin IR': 'active',
        'Fentanyl': 'active',
    }

    ->

    "Percocet" (discontinued)
    - "Gemzar" (active)
    - "Oxycontin IR" (active)
    - "Fentanyl" (active)
    '''
    bulleted_str = ' ' + '\n- '.join([f'"{med}" ({status})' for med, status in med_status_dict.items()])
    return bulleted_str