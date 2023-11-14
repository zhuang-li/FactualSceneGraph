from ..utils import get_seg_list, space_out_symbols_in_graph


def eval_set_match(cand, refs):
    """
    Evaluate the set match score between source and reference phrases.

    :param src_phrases: Source phrases.
    :param ref_phrases: Reference phrases.
    :return: Calculated set match score.
    """

    cand_phrases = get_seg_list(cand)
    ref_phrases = get_seg_list(refs)

    return len(cand_phrases) == len(ref_phrases) and (sorted(cand_phrases) == sorted(ref_phrases))
