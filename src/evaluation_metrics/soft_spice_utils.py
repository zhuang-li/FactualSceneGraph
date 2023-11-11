import warnings

import numpy as np
import sklearn
from tqdm import tqdm

from soft_spice.utils import get_seg_list
from packaging import version

def get_graph_phrases(graph_str_list, type_dict):
    seg_list = get_seg_list(graph_str_list)
    #breakpoint()
    new_pairs = []
    for seg in seg_list:
        new_seg = [item.strip() for item in seg.split(',')]
        try:
            if len(new_seg) == 1 and len(seg_list) == 1:
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                continue

            if len(new_seg) == 2:
                new_pairs.append(new_seg[1] + " " + new_seg[0])
                type_dict[new_seg[1] + " " + new_seg[0]] = "attribute"
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                continue
            elif len(new_seg) == 3:
                if new_seg[1] == 'is':
                    new_pairs.append(new_seg[2] + " " + new_seg[0])
                    type_dict[new_seg[2] + " " + new_seg[0]] = "attribute"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                else:
                    new_pairs.append(new_seg[0] + " " + new_seg[1] + " " + new_seg[2])
                    type_dict[new_seg[0] + " " + new_seg[1] + " " + new_seg[2]] = "fact"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    new_pairs.append(new_seg[2])
                    type_dict[new_seg[2]] = "object"
            elif len(new_seg) > 3:
                new_pairs.append(new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1])
                type_dict[new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1]] = "fact"
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                new_pairs.append(new_seg[-1])
                type_dict[new_seg[-1]] = "object"
        except IndexError:
            print(seg_list)
            continue

    return list(set(new_pairs))

#src_phrases, ref_phrases, type_dict, cache, type_constraint=type_constraint, use_cache=use_cache
def eval_soft_spice(src_phrases, ref_phrases, batch_size, text_encoder, type_dict, cache, type_constraint, use_cache):

    phrases = src_phrases + ref_phrases
    #breakpoint()
    all_feats = text_encoder.encode(phrases, batch_size=batch_size)
    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        all_feats = sklearn.preprocessing.normalize(all_feats, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        all_feats = all_feats / np.sqrt(np.sum(all_feats ** 2, axis=1, keepdims=True))

    src_feats = all_feats[:len(src_phrases)]
    ref_feats = all_feats[len(src_phrases):]


    all_sims = src_feats.dot(ref_feats.transpose())
    #breakpoint()
    score_per_phrases = np.max(all_sims, axis=1)
    #breakpoint()

    return np.mean(score_per_phrases)