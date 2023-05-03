import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm

PROMPT_V1 = """You are an expert fact checker. Find any medications in the patient note that were omitted from the list below and add them to the list. Return the medications as a bulleted list. Return the medication name in quotes and the status in parentheses (active, discontinued, or neither).

Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Extracted medications, without any omissions
--------------------------------------------
-"""
 # note this outputted a whole list rather than an additional list

PROMPT_V1 = """You are an expert medical scribe and are reviewing a patient note.

Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

List all medication names in the patient note that were missed in the list above, with the medication name in quotes and the status in parentheses (active, discontinued, or neither). If no additional medications are found, write "None".

Additional extracted medications
--------------------------------
-"""

PROMPT_V2 = """Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

List all medications in the patient note that were missed in the list above, with the medication name in quotes and the status in parentheses (active, discontinued, or neither). If no additional medications are found, return "None".

Additional extracted medications
--------------------------------
-"""

PROMPT_V3 = """List all medication names in the patient note that were missed in Extracted medications list. Put the medication name in quotes and the status in parentheses (active, discontinued, or neither). If no additional medications are found, return "None".

Patient Note
------------
5. Prozac 60 mg daily by mouth. 6. Regular insulin sliding scale as follows: 150 to 200 3 units, 201 to 250 6 units, 251 to 300 8 units, 301 to 351 10 units, 351 to 400 12 units, greater than 400 call the M.D. or NP. 7. Lantus insulin 6 units q.p.m. now being given at 1800. 8. Zosyn 3.375 gm IV q.6 h. which we will continue through the _%#DD#%_ then discontinue.

Extracted medications
---------------------
- "insulin" (active)

Missed medications
------------------
- "Prozac" (active)
- "Lantus insulin" (active)
- "Zosyn" (active)

Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Missed medications
------------------
-"""

PROMPT_V4 = """List all medication names in the patient note that were missed in Extracted medications list. Put the medication name in quotes and the status in parentheses (active, discontinued, or neither). If no additional medications are found, return "None".

Patient Note
------------
5. Prozac 60 mg daily by mouth. 6. Regular insulin sliding scale as follows: 150 to 200 3 units, 201 to 250 6 units, 251 to 300 8 units, 301 to 351 10 units, 351 to 400 12 units, greater than 400 call the M.D. or NP. 7. Lantus insulin 6 units q.p.m. now being given at 1800. 8. Zosyn 3.375 gm IV q.6 h. which we will continue through the _%#DD#%_ then discontinue.

Extracted medications
---------------------
- "insulin" (active)

Missed medications
------------------
- "Prozac" (active)
- "Lantus insulin" (active)
- "Zosyn" (active)

Patient Note
------------
_%#NAME#%_ tolerated his chemotherapy well with minimal nausea and no emesis. At the time of discharge, he was in no apparent distress and was afebrile. He went home with daily doses of 6 MP which they plan to crush, at home, to help swallowing. Also at the time of his discharge he was switched from dapsone to Bactrim, which was also to be crushed and mixed in with his food for PCP prophylaxis.

Extracted medications
---------------------
- "dapsone" (discontinued)

Missed medications
------------------
- "Bactrim" (active)
- "6 MP" (active)

Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Missed medications
------------------
-"""



class OmissionVerifier:
    def __init__(self):
        self.prompt = PROMPT_V4

    def __call__(self, snippet, bulleted_str, llm, verbose=False, lower=True) -> Tuple[Dict[str, str]]:
        prompt_ex = self.prompt.format(snippet=snippet, bulleted_str=bulleted_str)
        med_str_after_verification_and_deduplication = llm(prompt_ex)
        # med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str, lower=lower)
        if verbose:
            print(prompt_ex, end='')
            print('<START>' + med_str_after_verification_and_deduplication + '<END>')
        med_status_dict  = clin.parse.parse_response_medication_list(med_str_after_verification_and_deduplication, lower=lower)
        med_status_dict_init = clin.parse.parse_response_medication_list(bulleted_str, lower=lower)
        return med_status_dict_init | med_status_dict

if __name__ == '__main__':
    ov = OmissionVerifier()
    print('\n\n\n', len(ov.prompt), '**********************************************')
    snippet = '''I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d.'''
    # bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)'''
    bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Lexapro" (active)'''
    llm = clin.llm.get_llm('text-davinci-003')
    med_status_dict = ov(snippet, bulleted_str, llm, verbose=True)
    print(med_status_dict)
