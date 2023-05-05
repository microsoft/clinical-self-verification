import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
import joblib
from os.path import join
from clin.config import PATH_REPO

PROMPT_V1 = """'List all trial arms in the patient note that were missed in Extracted arms list. If no additional arms are found, return None\n'

### Patient Note ###
Safety, tolerability, and immunogenicity after 1 and 2 doses of zoster vaccine in healthy adults ≥60 years of age.

BACKGROUND Incidence and severity of herpes zoster (HZ) and postherpetic neuralgia increase with age, associated with age-related decrease in immunity to varicella-zoster virus (VZV). One dose of zoster vaccine (ZV) has demonstrated substantial protection against HZ; this study examined impact of a second dose of ZV.
METHODS Randomized, double-blind, multicenter study with 210 subjects ≥60 years old compared immunity and safety profiles after one and two doses of ZV, separated by 6 weeks, vs. placebo. Immunogenicity was evaluated using VZV interferon-gamma (IFN-γ) enzyme-linked immunospot (ELISPOT) assay and VZV glycoprotein enzyme-linked immunosorbent antibody (gpELISA) assay. Adverse experiences (AEs) were recorded on a standardized Vaccination Report Card.
RESULTS No serious vaccine-related AEs occurred. VZV IFN-γ ELISPOT geometric mean count (GMC) of spot-forming cells per 10(6) peripheral blood mononuclear cells increased in the ZV group from 16.9 prevaccination to 49.5 and 32.8 at 2 and 6 weeks postdose 1, respectively. Two weeks, 6 weeks and 6 months postdose 2, GMC was 44.3, 42.9, and 36.5, respectively. GMC in the placebo group did not change during the study. The peak ELISPOT response occurred ∼2 weeks after each ZV dose. The gpELISA geometric mean titers (GMTs) in the ZV group were higher than in the placebo group at 6 weeks after each dose. Correlation between the IFN-γ ELISPOT and gpELISA assays was poor.
CONCLUSIONS ZV was generally well-tolerated and immunogenic in adults ≥60 years old. A second dose of ZV was generally safe, but did not boost VZV-specific immunity beyond levels achieved postdose 1.

### Extracted trial arms ###
- zoster vaccine

### Missed trial arms ###
- placebo

### Patient Note ###
Effect of different rates of infusion of propofol for induction of anaesthesia in elderly patients.

The effect of changing the rate of infusion of propofol for induction of anaesthesia was studied in 60 elderly patients. Propofol was administered at 300, 600 or 1200 ml h-1 until loss of consciousness (as judged by loss of verbal contact with the patient) had been achieved. The duration of induction was significantly longer (P less than 0.001) with the slower infusion rates (104, 68 and 51 s), but the total dose used was significantly less (P less than 0.001) in these patients (1.2, 1.6 and 2.5 mg kg-1, respectively). The decrease in systolic and diastolic arterial pressure was significantly less in the 300-ml h-1 group at the end of induction and immediately after induction (P less than 0.01). The incidence of apnoea was also significantly less in the slower infusion group.

### Extracted trial arms ###
- propofol

### Missed trial arms ###
- None

### Patient Note ###
{snippet}

## Extracted trial arms ###
{bulleted_str}

### Missed trial arms ###
-"""


class OmissionVerifier:
    def __init__(self):
        self.prompt = PROMPT_V1

    def __call__(self, snippet, bulleted_str, llm, verbose=False) -> Tuple[Dict[str, str]]:
        prompt_ex = self.prompt.format(snippet=snippet, bulleted_str=bulleted_str)
        med_str_after_verification_and_deduplication = llm(prompt_ex)
        if verbose:
            print(prompt_ex, end='')
            print('<START>' + med_str_after_verification_and_deduplication + '<END>')
        med_status_dict  = clin.parse.parse_response_medication_list(med_str_after_verification_and_deduplication)
        med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str)
        return med_status_dict_init | med_status_dict

if __name__ == '__main__':
    ov = OmissionVerifier()
    llm = clin.llm.get_llm('text-davinci-003')
    dfv = joblib.load(join(PATH_REPO, 'data', 'ebm', 'ebm_interventions_cleaned.pkl')).iloc[100:]
    for i in range(len(dfv)):
        row = dfv.iloc[i]
        print(row['doc'])
        print(clin.parse.list_to_bullet_str(row['interventions']))
        print()