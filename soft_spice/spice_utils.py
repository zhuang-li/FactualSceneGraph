import numpy as np
from nltk.corpus import wordnet

from soft_spice.utils import get_seg_list


def eval_spice(spice_tuple, ref_tuple):
    count_tuple = 0


    num_ref = len(ref_tuple)
    num_pred = len(spice_tuple)
    check_ref = np.zeros((num_ref))
    check_pred = np.zeros((num_pred))

    ans = []
    for tup_id, tup in enumerate(ref_tuple):
        for spice_id, spice_tup in enumerate(spice_tuple):
            if check_pred[spice_id] == 0 and tup == spice_tup:
                ans.append(tup)
                check_ref[tup_id] = 1
                check_pred[spice_id] = 1
                count_tuple += 1
                break

    spice_wordnet = []

    for tup_id, tup in enumerate(spice_tuple):
        tup_syns = []
        if check_pred[tup_id] != 1:
            for word in tup:
                st = SemanticTuple(word)
                tup_syns.append(st.lemma_synset)

        spice_wordnet.append(tuple(tup_syns))

    for tup_id, tup in enumerate(ref_tuple):
        if check_ref[tup_id] == 1:
            continue
        tup_syns = []

        for word in tup:
            st = SemanticTuple(word)
            tup_syns.append(st.lemma_synset)

        for pred_id, pred in enumerate(spice_wordnet):
            if check_pred[pred_id] == 0 and similar(tup_syns, pred):
                count_tuple += 1
                check_ref[tup_id] = 1
                check_pred[pred_id] = 1
                break

    if num_pred == 0:
        p_score = 0
    else:
        p_score = count_tuple / float(num_pred)

    s_score = count_tuple / float(num_ref)

    if count_tuple == 0:
        sg_score = 0
    else:
        sg_score = 2 * p_score * s_score / (p_score + s_score)

    if sg_score > 1:
        # print ref_tuple
        # print spice_wordnet
        print(len(ref_tuple))
        print(len(spice_wordnet))
        print(p_score)
        print(s_score)
        print(sg_score)
        print("The F1 score is larger than 1.")
        exit()

    return sg_score

class SemanticTuple(object):

    def __init__(self, word):

        self.word = ' '.join(word.strip().lower().split())
        self.word_to_synset()


    def word_to_synset(self):

        lemma_synset = []
        word_split = self.word.split()
        if len(word_split) >= 2:
            self.word = "_".join(word_split)
        lemma_synset.append(self.word)

        for sys in wordnet.synsets(self.word):
            for l in sys.lemmas():
                lemma_synset.append(l.name())

        self.lemma_synset = set(lemma_synset)


def similar(tup_syns, pred):
    if len(tup_syns) != len(pred):
        return False
    else:
        for w_id in range(len(tup_syns)):
            # print "w_id:  ", w_id
            if len(tup_syns[w_id].intersection(pred[w_id])) == 0:
                return False
        return True





def get_graph_tuples(graph_str_list):
    seg_list = get_seg_list(graph_str_list)
    selected_obj_set = set()
    tuples = []
    for hyp in seg_list:
        lf_seg = [token.strip() for token in hyp.split(',')]
        if len(lf_seg) == 1:
            if not lf_seg[0] in selected_obj_set:
                tuples.append([lf_seg[0]])
                selected_obj_set.add(lf_seg[0])
        elif len(lf_seg) == 2:
            tuples.append((lf_seg[0], lf_seg[1]))
            if not lf_seg[0] in selected_obj_set:
                tuples.append([lf_seg[0]])
                selected_obj_set.add(lf_seg[0])
        elif len(lf_seg) == 3:
            if lf_seg[1] == 'is':
                tuples.append((lf_seg[0], lf_seg[2]))
                if not lf_seg[0] in selected_obj_set:
                    tuples.append([lf_seg[0]])
                    selected_obj_set.add(lf_seg[0])

            else:
                tuples.append((lf_seg[0], lf_seg[1], lf_seg[2]))
                if not lf_seg[0] in selected_obj_set:
                    tuples.append([lf_seg[0]])
                    selected_obj_set.add(lf_seg[0])
                if not lf_seg[2] in selected_obj_set:
                    tuples.append([lf_seg[2]])
                    selected_obj_set.add(lf_seg[2])

        elif len(lf_seg) > 3:
            tuples.append((lf_seg[0], " ".join(lf_seg[1:-1]), lf_seg[-1]))
            if not lf_seg[0] in selected_obj_set:
                tuples.append([lf_seg[0]])
                selected_obj_set.add(lf_seg[0])
            if not lf_seg[-1] in selected_obj_set:
                tuples.append([lf_seg[-1]])
                selected_obj_set.add(lf_seg[-1])

    return tuples


