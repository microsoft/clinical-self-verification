import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm

######## Original examples with evidence ###############################
EX_0_POS = f"""Patient Note
------------
_%#NAME#%_ tolerated his chemotherapy well with minimal nausea and no emesis. At the time of discharge, he was in no apparent distress and was afebrile. He went home with daily doses of 6 MP which they plan to crush, at home, to help swallowing. Also at the time of his discharge he was switched from dapsone to Bactrim, which was also to be crushed and mixed in with his food for PCP prophylaxis.

Original: "dapsone" (discontinued)
Evidence: "he was switched from dapsone to Bactrim"
Revised: "dapsone" (discontinued)

Original: "Bactrim" (active)
Evidence: "he was switched from dapsone to Bactrim"
Revised: "Bactrim" (active)

Original: "6 MP" (active)
Evidence: "he went home with daily doses of 6 MP"
Revised: "6 MP" (active)"""

EX_1_POS = """Patient Note
------------
On hospital day number three she was weaned off her PCA and started on a fentanyl patch 75 micrograms along with Percocet for breakthrough pain which was then switched to OxyContin IR as needed for breakthrough pain. She received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty. On hospital day number four she is stable and ready to be discharged to home with much improved pain control on her combination of fentanyl patch and oral OxyContin IR.

Original: "Percocet" (discontinued)
Evidence: "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
Revised: "Percocet" (discontinued)

Original: "Gemzar" (active)
Evidence: "received her Gemzar chemotherapy on _%#MMDD2003#%_ without difficulty"
Revised: "Gemzar" (active)

Original: "OxyContin IR" (active)
Evidence: "along with Percocet for breakthrough pain which was then switched to OxyContin IR"
Revised: "OxyContin IR" (active)

Original: "fentanyl" (active)
Evidence: "she was weaned off her PCA and started on a fentanyl patch"
Revised: "fentanyl" (active)"""

EX_0_ADDITION_TEXT = (
    "He does not want to take Celexa, so I put him back on Lexapro 2 mg p.o. q.d."
)
EX_0_ADDITION_LABEL = """Original: "Lexapro" (active)
Evidence: "I put him back on Lexapro 2 mg p.o. q.d."
Revised: "Lexapro" (active)

Original: "Celexa" (discontinued)
Evidence: "He does not want to take Celexa"
Revised: "Celexa" (neither)"""

EX_0_NEG = f"""Patient Note
------------
_%#NAME#%_ tolerated his chemotherapy well with minimal nausea and no emesis. At the time of discharge, he was in no apparent distress and was afebrile. Also at the time of his discharge he was switched from dapsone to Bactrim, which was also to be crushed and mixed in with his food for PCP prophylaxis. {EX_0_ADDITION_TEXT}

Original: "dapsone" (discontinued)
Evidence: "he was switched from dapsone to Bactrim"
Revised: "dapsone" (discontinued)

Original: "Bactrim" (active)
Evidence: "he was switched from dapsone to Bactrim"
Revised: "Bactrim" (active)

{EX_0_ADDITION_LABEL}"""

EX_2_NEG = """Patient Note
------------
The patient was transferred up to the floor, aspirated, and developed a pneumonia in her right lower and middle lobes. This was treated with a course of Timentin and started on a course of vancomycin. Sputum cultures did come back with MSSA and MRSA. The patient did complete a course of Timentin. This was discontinued. The patient had a positive sputum culture for MRSA on _%#MMDD2006#%_, and the vancomycin was continued.

Original: "Timentin" (active)
Evidence: "patient did complete a course of Timentin. This was discontinued"
Revised: "Timentin" (discontinued)

Original: "vancomycin" (active)
Evidence: "started on a course of vancomycin"
Revised: "vancomycin" (active)"""

EX_3_NEG = """Patient Note
------------
5. Prozac 60 mg daily by mouth. 6. Regular insulin sliding scale as follows: 150 to 200 3 units, 201 to 250 6 units, 251 to 300 8 units, 301 to 351 10 units, 351 to 400 12 units, greater than 400 call the M.D. or NP. 7. Lantus insulin 6 units q.p.m. now being given at 1800. 8. Zosyn 3.375 gm IV q.6 h. which we will continue through the _%#DD#%_ then discontinue.

Original: "insulin" (active)
Evidence: "Regular insulin sliding scale as follows: 150 to 200 3 units"
Revised: "insulin" (active)

Original: "Prozac" (active)
Evidence: "Prozac 60 mg daily by mouth"
Revised: "Prozac" (active)

Original: "Lantus insulin" (discontinued)
Evidence: "Lantus insulin 6 units q.p.m. now being given at 1800"
Revised: "Lantus insulin" (active)

Original: "Zosyn" (active)
Evidence: "Zosyn 3.375 gm IV q.6 h."
Revised: "Zosyn" (active)"""


EXS_POS = [EX_1_POS, EX_0_POS]
EXS_NEG = [EX_0_NEG, EX_2_NEG, EX_3_NEG]

PROMPT_V1 = f"""Check the status of each medication found in the patient note (active, discontinued, or neither). Use the patient note and the extracted evidence to revise the medication's status if necessary. If the medication status is not clearly active or discontinued, set it to "neither"."""
PROMPT_V2 = f"""Revise the status of each medication, based on the patient note and the extracted evidence. The status should be active, discontinued, or neither. If the medication status is not clearly active or discontinued, revise it to neither."""
PROMPT_V3 = f"""Revise the status of each medication, based on the patient note and the extracted evidence snippet. The status should be active, discontinued, or neither. If the evidence does not very clearly show that status is active or discontinued, revise it to neither."""
PROMPT_V4 = f"""Revise the status of each medication, based on the patient note and the extracted evidence snippet. The status should be active, discontinued, or neither. If the evidence does not show that status is clearly active or discontinued, revise it to neither."""
# + # 'Only change the status if the evidence clearly warrants a change.'


class StatusVerifier:
    def __init__(self, n_shots_pos=0, n_shots_neg=1):
        self.n_shots_pos = n_shots_pos
        self.n_shots_neg = n_shots_neg
        exs_pos = EXS_POS[: self.n_shots_pos]
        exs_neg = EXS_NEG[: self.n_shots_neg]
        exs = exs_pos + exs_neg
        self.prompt = PROMPT_V4 + "\n\n" + "\n\n\n".join(exs)

    def __call__(
        self,
        snippet,
        med_status_dict,
        med_evidence_dict,
        llm,
        verbose=False,
    ) -> Tuple[Dict[str, str]]:
        prompt_intro = (
            self.prompt
            + "\n\n\n"
            + f"""Patient Note
------------
{snippet}

"""
        )
        meds = list(med_status_dict.keys())
        med_status_dict_revised = {}
        for med in meds:
            prompt_ex = (
                prompt_intro
                + f"""Original: "{med}" ({med_status_dict[med]})
Evidence: "{med_evidence_dict[med]}"
Revised:"""
            )

            med_and_status_revised = llm(prompt_ex, stop="\n")
            status = clin.parse.parse_medication_and_status_to_status(
                med_and_status_revised
            )
            if verbose:
                print("PROMPT", repr(prompt_ex))
                print("RESP", repr(med_and_status_revised))
                # print("STATUS", status)
                if not status == med_status_dict[med]:
                    print('***status changed from', med_status_dict[med], 'to', status)
                print()
            

            # make new dict (don't revise if it already predicted neither, since that's kind of rare)
            if status in ["active", "discontinued", "neither"] and not med_status_dict[med] == 'neither':
                med_status_dict_revised[med] = status
            else:
                med_status_dict_revised[med] = med_status_dict[med]

        return med_status_dict_revised


if __name__ == "__main__":
    v = StatusVerifier()
    print("\n\n\n", len(v.prompt), "**********************************************")
    snippet = """I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d."""
    med_evidence_dict = {
        "ativan": "ativan 1 mg p.o. q. 8 hours and use ativan 1 mg iv q. 4 hours p.r.n.",
        "inderal la": "inderal la 60 mg p.o. q.d.",
        "celexa": "she does not want to take celexa",
        "lexapro": "lexapro 2 mg p.o. q.d.",
    }
    med_status_dict = {
        "ativan": "active",
        "inderal la": "active",
        "celexa": "discontinued",  # should be neither
        "lexapro": "discontinued",  # should be active
    }
    llm = clin.llm.get_llm("text-davinci-003")
    med_status_dict = v(snippet, med_status_dict, med_evidence_dict, llm, verbose=True)
    print(med_status_dict)
