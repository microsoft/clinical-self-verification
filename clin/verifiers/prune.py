import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm


PROMPT_V1 = """Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Verify whether each extracted medication is actually a medication name and remove any medications which are duplicates.
For example, "accu-checks", "ivf", and "pills" are not medications.
Return the verified medications as a bulleted list.
Keep the status of each medication the same as the original list and return the medication name in quotes.

Extracted medications after verification
----------------------------------------
-"""

PROMPT_V2 = """Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Verify whether each extracted medication is actually a medication name and not a symptom, procedure, or measurement. For example, "Accu-Check", "I.V. Fluids", "Hypertension", "pills" are not medications.
Return all verified medications as a bulleted list and only remove a medification if it is clearly not a medication name. Keep the status of each medication the same as the original list and return the medication name in quotes.

Extracted medications after verification
----------------------------------------
-"""

PROMPT_V3 = """Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Verify whether each extracted medication from the list is actually a medication name and return any non-verified medications as a bulleted list.
Examples of non-verified medications are symptoms, procedures, or measurement, such as "Accu-Check", "I.V. Fluids", "Hypertension", "pills".
If no non-verified medications are found, return "None".

Non-verified medications
------------------------
-"""

PROMPT_V4 = """Potential Medications:
-{bulleted_str}

Return each element in the above list which is not very clearly a medication name.
Examples of elements which are not medication names are symptoms or procedures, such as "Infection", "Fever", "Biopsy", "Protocol", "Accu-Cheks", "I.V. Fluids", "Inhaler", or "Hypertension".
If no non-verified medications are found, return "None".

Non-medications:
-"""


PROMPT_V5 = """Return each element in the Potential Medications list which is not very clearly a medication name.
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
-"""







class PruneVerifier:
    def __init__(self):
        self.prompt = PROMPT_V5

    def __call__(self, snippet, bulleted_str, llm, verbose=False, lower=True) -> Tuple[Dict[str, str]]:
        med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str, lower=False)
        bulleted_str_med_only = ' ' + '\n- '.join([f'"{med}"' for med in med_status_dict_init.keys()])
        prompt_ex = self.prompt.format(snippet=snippet, bulleted_str=bulleted_str_med_only)
        med_str_after_pruning = llm(prompt_ex)
        
        # remove meds that are in the returned list
        med_keys = clin.parse.parse_response_medication_list(med_str_after_pruning, with_status=False, lower=True)
        med_status_dict_init = {med.lower(): med_status_dict_init[med] for med in med_status_dict_init.keys()}
        med_status_dict = {
            med: med_status_dict_init[med] for med in med_status_dict_init
            if not med in med_keys
        }

        # remove meds that are not in the snippet (this has no effect)
        med_status_dict = {med: med_status_dict[med] for med in med_status_dict if med.lower() in snippet.lower()}

        if verbose:
            print(prompt_ex, end='')
            print('<START>' + med_str_after_pruning + '<END>')
            print('ks', med_status_dict_init.keys(), med_keys, f'removed {len(med_status_dict_init) - len(med_status_dict)} keys')
        return med_status_dict

if __name__ == '__main__':
    dp = PruneVerifier()
    print('\n\n\n', len(dp.prompt), '**********************************************')
    snippet = '''I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d.'''
    # bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)'''
    bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)\n- "I.V. Fluids" (active)'''
    llm = clin.llm.get_llm('text-davinci-003')
    med_status_dict = dp(snippet, bulleted_str, llm, verbose=True, )
    print(med_status_dict)
