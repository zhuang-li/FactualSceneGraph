import numpy as np
from packaging import version
from ..utils import get_seg_list

def _get_graph_phrases(graph_str_list, type_dict):
    """
    Extract phrases from graph strings and classify their types.

    :param graph_str_list: List of graph strings.
    :param type_dict: Dictionary to store type information.
    :return: A list of unique phrases.
    """
    seg_list = get_seg_list(graph_str_list)
    new_pairs = set()

    for seg in seg_list:
        new_seg = [item.strip() for item in seg.split(',')]
        if len(new_seg) == 1:
            _handle_single_element(new_seg, new_pairs, type_dict)
        elif len(new_seg) == 2:
            _handle_two_elements(new_seg, new_pairs, type_dict)
        elif len(new_seg) >= 3:
            _handle_three_or_more_elements(new_seg, new_pairs, type_dict)

    return list(new_pairs)


def _handle_single_element(seg, pairs, type_dict):
    pairs.add(seg[0])
    type_dict[seg[0]] = "entity"


def _handle_two_elements(seg, pairs, type_dict):
    pairs.add(f"{seg[1]} {seg[0]}")
    type_dict[f"{seg[1]} {seg[0]}"] = "attribute"
    pairs.add(seg[0])
    type_dict[seg[0]] = "entity"


def _handle_three_or_more_elements(seg, pairs, type_dict):
    if seg[1] == 'is':
        pairs.add(f"{seg[2]} {seg[0]}")
        type_dict[f"{seg[2]} {seg[0]}"] = "attribute"
    else:
        phrase = f"{seg[0]} {' '.join(seg[1:-1])} {seg[-1]}"
        pairs.add(phrase)
        type_dict[phrase] = "relation"
        pairs.add(seg[-1])
        type_dict[seg[-1]] = "entity"

    pairs.add(seg[0])
    type_dict[seg[0]] = "entity"

def encode_phrases(text_encoder, all_cand_phrases, all_ref_phrases, batch_size):
    all_encoded_phrases = _normalize_features(
        text_encoder.encode(all_cand_phrases + all_ref_phrases, batch_size=batch_size)
    )
    encoded_cands, encoded_refs = np.split(all_encoded_phrases, [len(all_cand_phrases)])
    return encoded_cands, encoded_refs


def accumulate_phrases(candidates, references):
    all_cand_phrases = []
    all_ref_phrases = []
    cand_lengths = []
    ref_lengths = []
    type_dict = {}

    for cand, refs in zip(candidates, references):
        cand_phrases = _get_graph_phrases(cand, type_dict)
        ref_phrases = _get_graph_phrases(refs, type_dict)

        all_cand_phrases.extend(cand_phrases)
        all_ref_phrases.extend(ref_phrases)

        cand_lengths.append(len(cand_phrases))
        ref_lengths.append(len(ref_phrases))

    return all_cand_phrases, all_ref_phrases, cand_lengths, ref_lengths


def compute_scores(encoded_cands, encoded_refs, cand_lengths, ref_lengths):
    scores = []
    ref_start_idx = 0

    for cand_len, ref_len in zip(cand_lengths, ref_lengths):
        cand_feats = encoded_cands[:cand_len]
        encoded_cands = encoded_cands[cand_len:]  # Update the remaining candidate features

        ref_feats = encoded_refs[ref_start_idx:ref_start_idx + ref_len]
        ref_start_idx += ref_len

        all_sims = cand_feats.dot(ref_feats.T)
        score_per_phrase = np.max(all_sims, axis=1)
        scores.append(np.mean(score_per_phrase))

    return scores



def _normalize_features(features):
    """
    Normalize feature vectors.

    :param features: Feature vectors.
    :return: Normalized feature vectors.
    """
    return features / np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
