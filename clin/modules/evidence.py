import numpy as np
from typing import Dict, List, Tuple
import clin.parse

NO_EVIDENCE = "no evidence"

######## Original examples with evidence ###############################
EX_WITH_EVIDENCE_0 = '''Patient Note
------------
_%#NAME#%_ tolerated his chemotherapy well with minimal nausea and no emesis. At the time of discharge, he was in no apparent distress and was afebrile. He went home with daily doses of 6 MP which they plan to crush, at home, to help swallowing. Also at the time of his discharge he was switched from dapsone to Bactrim, which was also to be crushed and mixed in with his food for PCP prophylaxis.

Extracted medications
---------------------
- "dapsone" (discontinued)
- "Bactrim" (active)
- "6 MP" (active)

Evidence
--------
- "dapsone" (discontinued) "he was switched from dapsone to Bactrim"
- "Bactrim" (active) "he was switched from dapsone to Bactrim"
- "6 MP" (active) "he went home with daily doses of 6 MP"'''

EX_WITH_EVIDENCE_1 = '''Patient Note
------------
On hospital day number three she was weaned off her PCA and started on a fentanyl patch 75 micrograms along with Percocet for breakthrough pain which was then switched to OxyContin IR as needed for breakthrough pain. She received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty. On hospital day number four she is stable and ready to be discharged to home with much improved pain control on her combination of fentanyl patch and oral OxyContin IR.

Extracted medications
---------------------
- "Percocet" (discontinued)
- "Gemzar" (active)
- "OxyContin IR" (active)
- "fentanyl" (active)

Evidence
--------
- "Percocet" (discontinued) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
- "Gemzar" (active) "received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty"
- "OxyContin IR" (active) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
- "fentanyl" (active) "she was weaned off her PCA and started on a fentanyl patch"'''

EX_WITH_EVIDENCE_2 = '''Patient Note
------------
The patient was transferred up to the floor, aspirated, and developed a pneumonia in her right lower and middle lobes. This was treated with a course of Timentin and started on a course of vancomycin. Sputum cultures did come back with MSSA and MRSA. The patient did complete a course of Timentin. This was discontinued. The patient had a positive sputum culture for MRSA on _%#MMDD2006#%_, and the vancomycin was continued.

Extracted medications
---------------------
- "Timentin" (discontinued)
- "vancomycin" (active)

Evidence
--------
- "Timentin" (discontinued) "patient did complete a course of Timentin. This was discontinued"
- "vancomycin" (active) "started on a course of vancomycin"'''

EX_WITH_EVIDENCE_3 = '''Patient Note
------------
5. Prozac 60 mg daily by mouth. 6. Regular insulin sliding scale as follows: 150 to 200 3 units, 201 to 250 6 units, 251 to 300 8 units, 301 to 351 10 units, 351 to 400 12 units, greater than 400 call the M.D. or NP. 7. Lantus insulin 6 units q.p.m. now being given at 1800. 8. Zosyn 3.375 gm IV q.6 h. which we will continue through the _%#DD#%_ then discontinue.

Extracted medications
---------------------
- "insulin" (active)
- "Prozac" (active)
- "Lantus insulin" (active)
- "Zosyn" (active)

Evidence
--------
- "insulin" (active) "Regular insulin sliding scale as follows: 150 to 200 3 units"
- "Prozac" (active) "Prozac 60 mg daily by mouth"
- "Lantus insulin" (active) "Lantus insulin 6 units q.p.m. now being given at 1800"
- "Zosyn" (active) "Zosyn 3.375 gm IV q.6 h."'''

EX_WITH_EVIDENCE_4 = f'''Patient Note
------------
Urinalysis and urine culture were negative. Chest x-ray revealed pleural effusion. She underwent a chest CT with PE protocol which demonstrated no PE but did have bilateral pleural effusions with underlying atelectasis. Antibiotics were changed to Cefotetan and this was discontinued prior to discharge as the patient has been afebrile and cultures negative.

Extracted medications
---------------------
- "Cefotetan" (discontinued)

Evidence
--------
- "Cefotetan" (discontinued) "Antibiotics were changed to Cefotetan and this was discontinued"'''


######## Original examples with added wrong medications ###############################
EX_WITH_EVIDENCE_ADDED_0 = f'''Patient Note
------------
_%#NAME#%_ tolerated his chemotherapy well with minimal nausea and no emesis. At the time of discharge, he was in no apparent distress and was afebrile. He went home with daily doses of 6 MP which they plan to crush, at home, to help swallowing. Also at the time of his discharge he was switched from dapsone to Bactrim, which was also to be crushed and mixed in with his food for PCP prophylaxis.

Extracted medications
---------------------
- "dapsone" (discontinued)
- "Bactrim" (active)
- "fentanyl" (active)
- "6 MP" (active)

Evidence
--------
- "dapsone" (discontinued) "he was switched from dapsone to Bactrim"
- "Bactrim" (active) "he was switched from dapsone to Bactrim"
- "fentanyl" (active) "{NO_EVIDENCE}"
- "6 MP" (active) "he went home with daily doses of 6 MP"'''

EX_WITH_EVIDENCE_ADDED_1 = f'''Patient Note
------------
On hospital day number three she was weaned off her PCA and started on a fentanyl patch 75 micrograms along with Percocet for breakthrough pain which was then switched to OxyContin IR as needed for breakthrough pain. She received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty. On hospital day number four she is stable and ready to be discharged to home with much improved pain control on her combination of fentanyl patch and oral OxyContin IR.

Extracted medications
---------------------
- "Percocet" (discontinued)
- "Gemzar" (active)
- "OxyContin IR" (active)
- "fentanyl" (active)

Evidence
--------
- "Percocet" (discontinued) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
- "Gemzar" (active) "received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty"
- "OxyContin IR" (active) "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
- "Tylenol" (active) "{NO_EVIDENCE}"
- "fentanyl" (active) "she was weaned off her PCA and started on a fentanyl patch"'''

EX_WITH_EVIDENCE_ADDED_2 = f'''Patient Note
------------
The patient was transferred up to the floor, aspirated, and developed a pneumonia in her right lower and middle lobes. This was treated with a course of Timentin and started on a course of vancomycin. Sputum cultures did come back with MSSA and MRSA. The patient did complete a course of Timentin. This was discontinued. The patient had a positive sputum culture for MRSA on _%#MMDD2006#%_, and the vancomycin was continued.

Extracted medications
---------------------
- "Timentin" (discontinued)
- "vancomycin" (active)
- "insulin" (discontinued)


Evidence
--------
- "Timentin" (discontinued) "This was treated with a course of Timentin"
- "vancomycin" (active) "started on a course of vancomycin"
- "insulin" (discontinued) "{NO_EVIDENCE}"'''

EX_WITH_EVIDENCE_ADDED_3 = f'''Patient Note
------------
5. Prozac 60 mg daily by mouth. 6. Regular insulin sliding scale as follows: 150 to 200 3 units, 201 to 250 6 units, 251 to 300 8 units, 301 to 351 10 units, 351 to 400 12 units, greater than 400 call the M.D. or NP. 7. Lantus insulin 6 units q.p.m. now being given at 1800. 8. Zosyn 3.375 gm IV q.6 h. which we will continue through the _%#DD#%_ then discontinue.

Extracted medications
---------------------
- "insulin" (active)
- "Percocet" (discontinued)
- "Prozac" (active)
- "Lantus insulin" (active)
- "Zosyn" (active)

Evidence
--------
- "insulin" (active) "Regular insulin sliding scale as follows: 150 to 200 3 units"
- "Percocet" (discontinued) "{NO_EVIDENCE}"
- "Prozac" (active) "Prozac 60 mg daily by mouth"
- "Lantus insulin" (active) "Lantus insulin 6 units q.p.m. now being given at 1800"
- "Zosyn" (active) "Zosyn 3.375 gm IV q.6 h."'''

EX_WITH_EVIDENCE_ADDED_4 = f'''Patient Note
------------
Urinalysis and urine culture were negative. Chest x-ray revealed pleural effusion. She underwent a chest CT with PE protocol which demonstrated no PE but did have bilateral pleural effusions with underlying atelectasis. Antibiotics were changed to Cefotetan and this was discontinued prior to discharge as the patient has been afebrile and cultures negative.

Extracted medications
---------------------
- "Cefotetan" (discontinued)
- "Prozac" (discontinued)

Evidence
--------
- "Cefotetan" (discontinued) "Antibiotics were changed to Cefotetan and this was discontinued"
- "Prozac" (discontinued) "{NO_EVIDENCE}"'''

EXS_POS = [EX_WITH_EVIDENCE_0, EX_WITH_EVIDENCE_1, EX_WITH_EVIDENCE_2, EX_WITH_EVIDENCE_3, EX_WITH_EVIDENCE_4]
EXS_NEG = [EX_WITH_EVIDENCE_ADDED_0, EX_WITH_EVIDENCE_ADDED_1, EX_WITH_EVIDENCE_ADDED_2, EX_WITH_EVIDENCE_ADDED_3, EX_WITH_EVIDENCE_ADDED_4]



PROMPT_V1 = f"""Verify whether each extracted medication is present in the patient note in a bulleted list.
If it is present, extract the span of text from the patient note as evidence. If it is not clearly present, write "{NO_EVIDENCE}". Write a bullet for every extracted medication."""

PROMPT_V2 = f"""Find the span of text which corresponds to each extracted medication and its status. If no evidence is found, write "{NO_EVIDENCE}". Write a bullet for every extracted medication."""


class EvidenceVerifier:
    def __init__(self, n_shots_pos=1, n_shots_neg=1):
        exs_pos = EXS_POS[: n_shots_pos]
        exs_neg = EXS_NEG[-n_shots_neg:]
        exs = exs_pos + exs_neg
        # print(exs.shape)
        # print(exs[0])
        self.prompt = PROMPT_V2 + '\n\n' + '\n\n\n'.join(exs)

    def __call__(self, snippet, bulleted_str, llm) -> Tuple[Dict[str, str]]:
        prompt_ex = self.prompt + '\n\n\n' + f'''Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Evidence
--------
-'''
        meds_with_evidence_str = llm(prompt_ex)
        med_status_dict, med_evidence_dict  = clin.parse.parse_response_medication_list_with_evidence(meds_with_evidence_str)
        # prune if no evidence
        med_status_dict = {
            k: v for k, v in med_status_dict.items()
            if not med_evidence_dict[k] == 'no evidence'
        }
        return med_status_dict, med_evidence_dict

if __name__ == '__main__':
    ev = EvidenceVerifier(n_shots_neg=2, n_shots_pos=0)
    print('\n\n\n', len(ev.prompt), '**********************************************')
    print(ev.prompt)

