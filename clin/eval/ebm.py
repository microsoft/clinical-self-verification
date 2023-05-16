from typing import List

def process_ebm_lists(
    pred_list: List[str],
    gt_list: List[str],
    verbose=False,
) -> List[bool]:
    # convert to lowercase
    pred_list = [pred.strip().lower() for pred in pred_list]
    gt_list = [gt.strip().lower() for gt in gt_list]

    if verbose:
        print("pred_list", pred_list)
        print("gt_list", gt_list)

    return pred_list, gt_list