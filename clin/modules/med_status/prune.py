import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import imodelsx.llm


PROMPT_V1 = """Return each element in the Potential Medications list which is not clearly a specific medication name.
Examples of elements which are not medication names are symptoms or procedures, such as "Infection", "Fever", "Biopsy", "Protocol", "Accu-Cheks", "I.V. Fluids", "Inhaler", or "Hypertension".
If no non-verified medications are found, return "None".

Potential Medications:
- "dapsone"
- "Bactrim"
- "6 MP"

Non-medications:
- "None"

Potential Medications:
- "Percocet"
- "Gemzar"
- "Accu-Chek"
- "Fever"

Non-medications:
- "Accu-Chek"
- "Fever"

Potential Medications:
- "Timentin" 
- "vancomycin"
- "IV"
- "sliding scale"

Non-medications:
- "IV"
- "sliding scale"

Potential Medications:
-{bulleted_str}

Non-medications:
-"""  # 0.927	0.909	0.945


class PruneVerifier:
    def __init__(self):
        self.prompt = PROMPT_V1

    def __call__(
        self, snippet, bulleted_str, llm, verbose=False, remove_len_2=False
    ) -> Tuple[Dict[str, str]]:
        med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str)
        bulleted_str_med_only = " " + "\n- ".join(
            [f'"{med}"' for med in med_status_dict_init.keys()]
        )
        prompt_ex = self.prompt.format(
            snippet=snippet, bulleted_str=bulleted_str_med_only
        )
        med_str_after_pruning = llm(prompt_ex)

        # remove meds that are in the returned list
        med_keys_lower = clin.parse.parse_response_medication_list(
            med_str_after_pruning, with_status=False, lower=True
        )
        med_status_dict = {
            med: med_status_dict_init[med]
            for med in med_status_dict_init
            if not med.lower() in med_keys_lower
        }

        # remove meds that are not in the snippet (this has no effect)
        med_status_dict = {
            med: med_status_dict[med]
            for med in med_status_dict
            if med.lower() in snippet.lower()
        }

        # remove meds that are too short
        if remove_len_2:
            med_status_dict = {
                med: med_status_dict[med] for med in med_status_dict if len(med) > 2
            }

        if verbose:
            print(prompt_ex, end="")
            print("<START>" + med_str_after_pruning + "<END>")
            print(
                "ks",
                med_status_dict_init.keys(),
                med_keys_lower,
                f"removed {len(med_status_dict_init) - len(med_status_dict)} keys",
            )
        return med_status_dict


if __name__ == "__main__":
    dp = PruneVerifier()
    print("\n\n\n", len(dp.prompt), "**********************************************")
    snippet = """I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d."""
    # bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)'''
    bulleted_str = """ "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)\n- "I.V. Fluids" (active)"""
    llm = imodelsx.llm.get_llm("text-davinci-003")
    med_status_dict = dp(
        snippet,
        bulleted_str,
        llm,
        verbose=True,
    )
    print(med_status_dict)
