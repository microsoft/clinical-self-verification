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

class PruneVerifier:
    def __init__(self):
        self.prompt = """Patient Note
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

    def __call__(self, snippet, bulleted_str, llm, verbose=False, lower=True) -> Tuple[Dict[str, str]]:
        # med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str, lower=False)
        # bulleted_str_med_only = ' ' + '\n- '.join([f'"{med}"' for med in med_status_dict.keys()])
        # print(bulleted_str_med_only)
        prompt_ex = self.prompt.format(snippet=snippet, bulleted_str=bulleted_str)
        med_str_after_verification_and_deduplication = llm(prompt_ex)
        if verbose:
            print(prompt_ex, end='')
            print('<START>' + med_str_after_verification_and_deduplication + '<END>')
        med_status_dict  = clin.parse.parse_response_medication_list(med_str_after_verification_and_deduplication, lower=lower)
        return med_status_dict

if __name__ == '__main__':
    dv = PruneVerifier`()
    print('\n\n\n', len(dv.prompt), '**********************************************')
    snippet = '''I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d.'''
    bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)'''
    llm = clin.llm.get_llm('text-davinci-003')
    med_status_dict = dv(snippet, bulleted_str, llm, verbose=True)
    print(med_status_dict)
