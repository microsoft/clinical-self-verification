from copy import deepcopy
import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm
import joblib
from os.path import join
from clin.config import PATH_REPO
import re

PROMPT_V1 = """Return each element in the Extracted trial arms which is not clearly a trial arm in the patient note, or is a duplicate.
If no misidentified trial arms are found, return "None".

# Patient Note
Safety, tolerability, and immunogenicity after 1 and 2 doses of zoster vaccine in healthy adults ≥60 years of age.

BACKGROUND Incidence and severity of herpes zoster (HZ) and postherpetic neuralgia increase with age, associated with age-related decrease in immunity to varicella-zoster virus (VZV). One dose of zoster vaccine (ZV) has demonstrated substantial protection against HZ; this study examined impact of a second dose of ZV.
METHODS Randomized, double-blind, multicenter study with 210 subjects ≥60 years old compared immunity and safety profiles after one and two doses of ZV, separated by 6 weeks, vs. placebo. Immunogenicity was evaluated using VZV interferon-gamma (IFN-γ) enzyme-linked immunospot (ELISPOT) assay and VZV glycoprotein enzyme-linked immunosorbent antibody (gpELISA) assay. Adverse experiences (AEs) were recorded on a standardized Vaccination Report Card.
RESULTS No serious vaccine-related AEs occurred. VZV IFN-γ ELISPOT geometric mean count (GMC) of spot-forming cells per 10(6) peripheral blood mononuclear cells increased in the ZV group from 16.9 prevaccination to 49.5 and 32.8 at 2 and 6 weeks postdose 1, respectively. Two weeks, 6 weeks and 6 months postdose 2, GMC was 44.3, 42.9, and 36.5, respectively. GMC in the placebo group did not change during the study. The peak ELISPOT response occurred ∼2 weeks after each ZV dose. The gpELISA geometric mean titers (GMTs) in the ZV group were higher than in the placebo group at 6 weeks after each dose. Correlation between the IFN-γ ELISPOT and gpELISA assays was poor.
CONCLUSIONS ZV was generally well-tolerated and immunogenic in adults ≥60 years old. A second dose of ZV was generally safe, but did not boost VZV-specific immunity beyond levels achieved postdose 1.

## Extracted trial arms
- zoster vaccine
- placebo

## Misidentified trial arms
- None

# Patient Note
Effect of different rates of infusion of propofol for induction of anaesthesia in elderly patients.

The effect of changing the rate of infusion of propofol for induction of anaesthesia was studied in 60 elderly patients. Propofol was administered at 300, 600 or 1200 ml h-1 until loss of consciousness (as judged by loss of verbal contact with the patient) had been achieved. The duration of induction was significantly longer (P less than 0.001) with the slower infusion rates (104, 68 and 51 s), but the total dose used was significantly less (P less than 0.001) in these patients (1.2, 1.6 and 2.5 mg kg-1, respectively). The decrease in systolic and diastolic arterial pressure was significantly less in the 300-ml h-1 group at the end of induction and immediately after induction (P less than 0.01). The incidence of apnoea was also significantly less in the slower infusion group.

## Extracted trial arms
- propofol
- ibuprofen

## Misidentified trial arms
- ibuprofen

# Patient Note
{snippet}

## Extracted trial arms
{bulleted_str}

### Misidentified trial arms
-"""

PROMPT_V2 = """Return each element in the Extracted interventions list which is not clearly a clinical trial arm in the patient note, or is a duplicate.
If no misidentified trial arms are found, return "None".

### Patient Note
Safety, tolerability, and immunogenicity after 1 and 2 doses of zoster vaccine in healthy adults ≥60 years of age.

BACKGROUND Incidence and severity of herpes zoster (HZ) and postherpetic neuralgia increase with age, associated with age-related decrease in immunity to varicella-zoster virus (VZV). One dose of zoster vaccine (ZV) has demonstrated substantial protection against HZ; this study examined impact of a second dose of ZV.
METHODS Randomized, double-blind, multicenter study with 210 subjects ≥60 years old compared immunity and safety profiles after one and two doses of ZV, separated by 6 weeks, vs. placebo. Immunogenicity was evaluated using VZV interferon-gamma (IFN-γ) enzyme-linked immunospot (ELISPOT) assay and VZV glycoprotein enzyme-linked immunosorbent antibody (gpELISA) assay. Adverse experiences (AEs) were recorded on a standardized Vaccination Report Card.
RESULTS No serious vaccine-related AEs occurred. VZV IFN-γ ELISPOT geometric mean count (GMC) of spot-forming cells per 10(6) peripheral blood mononuclear cells increased in the ZV group from 16.9 prevaccination to 49.5 and 32.8 at 2 and 6 weeks postdose 1, respectively. Two weeks, 6 weeks and 6 months postdose 2, GMC was 44.3, 42.9, and 36.5, respectively. GMC in the placebo group did not change during the study. The peak ELISPOT response occurred ∼2 weeks after each ZV dose. The gpELISA geometric mean titers (GMTs) in the ZV group were higher than in the placebo group at 6 weeks after each dose. Correlation between the IFN-γ ELISPOT and gpELISA assays was poor.
CONCLUSIONS ZV was generally well-tolerated and immunogenic in adults ≥60 years old. A second dose of ZV was generally safe, but did not boost VZV-specific immunity beyond levels achieved postdose 1.

### Extracted trial arms
- zoster vaccine
- placebo

### Misidentified trial arms
- None

### Patient Note
Effect of different rates of infusion of propofol for induction of anaesthesia in elderly patients.

The effect of changing the rate of infusion of propofol for induction of anaesthesia was studied in 60 elderly patients. Propofol was administered at 300, 600 or 1200 ml h-1 until loss of consciousness (as judged by loss of verbal contact with the patient) had been achieved. The duration of induction was significantly longer (P less than 0.001) with the slower infusion rates (104, 68 and 51 s), but the total dose used was significantly less (P less than 0.001) in these patients (1.2, 1.6 and 2.5 mg kg-1, respectively). The decrease in systolic and diastolic arterial pressure was significantly less in the 300-ml h-1 group at the end of induction and immediately after induction (P less than 0.01). The incidence of apnoea was also significantly less in the slower infusion group.

### Extracted trial arms
- propofol
- ibuprofen

### Misidentified trial arms
- ibuprofen

### Patient Note
{snippet}

### Extracted trial arms
{bulleted_str}

### Misidentified trial arms
-"""


def _prune_list_hardcoded(l):
    # remove parenthetical phrases
    l = [re.sub(r"\(.*?\)", "", x) for x in l]

    # don't keep anything with the word control
    l = [x for x in l if not "control" in x.lower()]

    # don't keep duplicates but preserve order
    l = list(dict.fromkeys(l))

    # strip each string
    l = [x.strip() for x in l]

    # remove empty strings
    l = [x for x in l if x and not x.lower().startswith('none')]

    # if any string is contained within another one, remove the longer one, but preserve the list order
    # l_s = sorted(l, key=len, reverse=False)
    # l_s = [
    # l_s[i] for i in range(len(l_s)) if not any([l_s[i] in y for y in l_s[i + 1 :]])
    # ]
    # l = [x for x in l if x in l_s]
    return l


PROMPT_CLEAN = """Return the list of clinical interventions removing the units from each element. If the intervention is a measurement and not actually a clinicial intervention, then don't return it.
    
### Interventions list
- zoster vaccine
- placebo

### Cleaned interventions list
- zoster vaccine
- placebo

### Interventions list
- 2,000 mg l⁻¹ clove solution
- 100 ml h-1 l⁻¹ clove solution
- 500 ml h-1 l⁻¹ clove solution

### Cleaned interventions list
- l⁻¹ clove solution
- l⁻¹ clove solution
- l⁻¹ clove solution
- l⁻¹ clove solution

### Interventions list
- 1 mg daily
- 100 mg twice daily
- placebo
- bupropion

### Cleaned interventions list
- placebo
- bupropion

### Interventions list
{bulleted_str}

##Cleaned interventions list
-"""


class PruneVerifier:
    def __init__(self):
        self.prompt = PROMPT_V2
        self.prompt_clean = PROMPT_CLEAN

    def __call__(
        self, snippet, bullet_list, llm, apply_cleaning_step=True, verbose=False
    ) -> List[str]:
        prompt_ex = self.prompt.format(
            snippet=snippet, bulleted_str=clin.parse.list_to_bullet_str(bullet_list)
        )
        bullet_str_extra = llm(prompt_ex)
        extra_list = clin.parse.bullet_str_to_list(bullet_str_extra)
        extra_list_lower = [x.lower() for x in extra_list]
        if verbose:
            print(prompt_ex, end="")
            print("<START>" + bullet_str_extra + "<END>")
            print(extra_list)
        bullet_list_pruned = [
            x for x in bullet_list if not x.lower() in extra_list_lower
        ]
        bullet_list_pruned = _prune_list_hardcoded(bullet_list_pruned)

        if apply_cleaning_step:
            prompt_clean = self.prompt_clean.format(
                bulleted_str=clin.parse.list_to_bullet_str(bullet_list_pruned)
            )
            bullet_str_clean = llm(prompt_clean)
            if "###" in bullet_str_clean:
                bullet_str_clean = bullet_str_clean.split("###")[0]
            bullet_list_clean = clin.parse.bullet_str_to_list(bullet_str_clean)
            return _prune_list_hardcoded(bullet_list_clean)
        else:
            return bullet_list_pruned


if __name__ == "__main__":
    v = PruneVerifier()
    llm = clin.llm.get_llm("text-davinci-003")
    arms_list = ["granisetron", "perphenazine", "banana"]
    snippet = "Anti-emetic efficacy of prophylactic granisetron compared with perphenazine for the prevention of post-operative vomiting in children.\n\nWe have compared the efficacy of granisetron with perphenazine in the prevention of vomiting after tonsillectomy with or without adenoidectomy in children. In a prospective, randomized, double-blind study, 90 paediatric patients, ASA I, aged 4-10 years, received granisetron 40 mg kg-1 or perphenazine 70 mg kg-1 (n = 45 each) intravenously immediately after an inhalation induction of anaesthesia. A standard general anaesthetic technique was employed throughout. A complete response, defined as no emesis with no need for another rescue antiemetic, during the first 3 h (0-3 h) after anesthesia was 87% with granisetron and 78% with perphenazine (P = 0.204). The corresponding incidence during the next 21 h (3-24 h) after anaesthesia was 87% and 62% (P = 0.007). No clinically serious adverse events were observed in any of the groups. We conclude that granisetron is a better anti-emetic than perphenazine for the long-term prevention of post-operative vomiting in children undergoing general anaesthesia for tonsillectomy."
    arms_new = v(snippet, arms_list, llm, verbose=True)
    print("arms_new", arms_new)
