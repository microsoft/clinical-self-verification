import numpy as np
from typing import Dict, List, Tuple
import clin.parse
import clin.llm

class DeduplicateVerifier:
    def __init__(self):
        self.prompt = """Verify whether each extracted medication a medication and remove any medications which are duplicates. Return the verified, deduplicated medications as a bulleted list. Keep the status of each medication the same as the original list and return the medication name in quotes.

Patient Note
------------
{snippet}

Extracted medications
---------------------
-{bulleted_str}

Extracted medications after verification and deduplication
----------------------------------------------------------
-"""

    def __call__(self, snippet, bulleted_str, llm, verbose=False, lower=True) -> Tuple[Dict[str, str]]:
        prompt_ex = self.prompt.format(snippet=snippet, bulleted_str=bulleted_str)
        med_str_after_verification_and_deduplication = llm(prompt_ex)
        if verbose:
            print(prompt_ex, end='')
            print('<START>' + med_str_after_verification_and_deduplication + '<END>')
        med_status_dict  = clin.parse.parse_response_medication_list(med_str_after_verification_and_deduplication, lower=lower)
        return med_status_dict

if __name__ == '__main__':
    dv = DeduplicateVerifier()
    print('\n\n\n', len(dv.prompt), '**********************************************')
    snippet = '''I will recommend discontinuing the alcohol withdrawal protocol and start her on Ativan 1 mg p.o. q. 8 hours and use Ativan 1 mg IV q. 4 hours p.r.n. for agitation. I will also start her on Inderal LA 60 mg p.o. q.d. for essential tremors. She does not want to take Celexa, and I will put her back on Lexapro 2 mg p.o. q.d.'''
    bulleted_str = ''' "Ativan" (active)\n- "Inderal LA" (active)\n- "Celexa" (discontinued)\n- "Lexapro" (active)'''
    llm = clin.llm.get_llm('text-davinci-003')
    med_status_dict = dv(snippet, bulleted_str, llm, verbose=True)
    print(med_status_dict)
