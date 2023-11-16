from nltk.corpus import wordnet

from .synonym_dictionary import synonym_dictionary
from ..utils import get_seg_list

def eval_spice(cand, refs, merge_tuples_synonyms=False, synonym_match=True):
    """
    Evaluate the SPICE metric.

    :param cand_tuples: Candidate tuples from scene graph.
    :param ref_tuples: Reference tuples for comparison.
    :return: Calculated SPICE score.
    """
    cand_tuples = get_graph_tuples(cand, merge_tuples_synonyms)
    ref_tuples = get_graph_tuples(refs, merge_tuples_synonyms)

    return calculate_spice_score(cand_tuples, ref_tuples, synonym_match)

def calculate_spice_score(cand_tuples, ref_tuples, synonym_match):
    matched_cand_indices = set()
    matched_ref_indices = set()
    total_matches = 0

    # First pass: Exact matches
    for i, cand in enumerate(cand_tuples):
        for j, ref in enumerate(ref_tuples):
            if cand == ref and j not in matched_ref_indices:
                matched_cand_indices.add(i)
                matched_ref_indices.add(j)
                total_matches += 1
                break

    if synonym_match:
        # Second pass: WordNet-based similar matches (for unmatched candidates)
        for i, cand in enumerate(cand_tuples):
            if i not in matched_cand_indices:
                for j, ref in enumerate(ref_tuples):
                    if j not in matched_ref_indices and similar_to_any(cand, [ref]):
                        # print("Synonym match")
                        # print(cand)
                        # print(ref)
                        matched_ref_indices.add(j)
                        total_matches += 1
                        break
    # Calculate precision, recall, and F1 score
    precision = calculate_score(total_matches, len(cand_tuples))
    recall = calculate_score(total_matches, len(ref_tuples))

    assert precision <= 1 and recall <= 1, "Precision or recall is greater than 1, total_matches {0}, len(cand_tuples) {1}, len(ref_tuples) {2}".format(total_matches, len(cand_tuples), len(ref_tuples))

    return calculate_f1(precision, recall)

def similar_to_any(candidate, references):
    """
    Check if a candidate is similar to any reference tuples.

    :param candidate: The candidate tuple to compare.
    :param references: A list of reference tuples.
    :return: True if similar to any reference, False otherwise.
    """
    candidate_synsets = get_synsets(candidate)

    return any(are_synsets_similar(candidate_synsets, get_synsets(ref)) for ref in references)

def get_synsets(words):
    """
    Get synsets for a list of words.

    :param words: A list of words.
    :return: A set of synsets for the words.
    """
    return [word_to_synset(word) for word in words]


def word_to_synset(word):
    """
    Process a word into its synsets.

    :param word: The word to process.
    :return: A set of synsets for the word.
    """
    word = ' '.join(word.strip().lower().split())
    lemma_synset = set()

    # If the word consists of multiple parts, join them with an underscore
    word_split = word.split()
    if len(word_split) >= 2:
        word = "_".join(word_split)

    # # Add the word itself to the synset set
    # lemma_synset.add(word)
    #
    # # Add all synsets of the word to the set
    # for sys in wordnet.synsets(word):
    #     for lemma in sys.lemmas():
    #         lemma_synset.add(lemma.name())

    lemma_synset.update(synonym_dictionary.get_synsets(word))
    lemma_synset.update(synonym_dictionary.get_stem_synsets(word))

    return lemma_synset

def are_synsets_similar(synsets1, synsets2):
    """
    Determine if two lists of synsets have non-empty intersections for corresponding elements.

    :param synsets1: First list of synsets.
    :param synsets2: Second list of synsets.
    :return: True if all corresponding synsets have a non-empty intersection, False otherwise.
    """
    return len(synsets1) == len(synsets2) and all(s1.intersection(s2) for s1, s2 in zip(synsets1, synsets2))

def calculate_score(match_count, total_count):
    """
    Calculate precision or recall.

    :param match_count: The count of matched tuples.
    :param total_count: The total count of tuples.
    :return: The calculated score.
    """
    return match_count / total_count if total_count > 0 else 0

def calculate_f1(precision, recall):
    """
    Calculate the F1 score from precision and recall.

    :param precision: Precision value.
    :param recall: Recall value.
    :return: The calculated F1 score.
    """
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

def get_graph_tuples(graph_str_list, merge_tuples_synonyms=True):
    """
    Get tuples from a scene graph.
    """
    seg_list = get_seg_list(graph_str_list)
    selected_obj_set = set()
    tuples = []

    for hyp in seg_list:
        lf_seg = [token.strip() for token in hyp.split(',')]
        seg_len = len(lf_seg)

        # Handle segments based on their length
        if seg_len == 1:
            add_unique_tuple(lf_seg[0], tuples, selected_obj_set)
        elif seg_len >= 2:
            process_lf_segment(lf_seg, tuples, selected_obj_set, seg_len)

    if merge_tuples_synonyms:
        return merge_tuples_based_on_synonyms(sorted(tuples, key=tuple_sort_key))
    else:
        return sorted(tuples, key=tuple_sort_key)

def tuple_sort_key(t):
    # Join the words in the tuple into a single string
    return ' '.join(t)

def merge_tuples_based_on_synonyms(tuples):
    """
    Merge tuples that have synonyms in common.

    :param tuples: A list of tuples.
    :return: A list of merged tuples.
    """
    merged_tuples = []

    for t in tuples:
        if not similar_to_any(t, merged_tuples):
            merged_tuples.append(t)

    return merged_tuples

def add_unique_tuple(item, tuples, selected_obj_set):
    """
    Adds a tuple to the list if the item is not in the selected object set.
    """
    if item not in selected_obj_set:
        tuples.append([item])
        selected_obj_set.add(item)

def process_lf_segment(lf_seg, tuples, selected_obj_set, seg_len):
    """
    Processes a segment of length 2 or more and adds appropriate tuples.
    """
    # Construct the tuple string based on segment length
    if seg_len == 2 or (seg_len == 3 and lf_seg[1] == 'is'):
        tuple_str = lf_seg[0] + ' ' + lf_seg[-1]
        if tuple_str not in selected_obj_set:
            tuples.append((lf_seg[0], lf_seg[-1]))
            selected_obj_set.add(tuple_str)
        add_unique_tuple(lf_seg[0], tuples, selected_obj_set)

    elif seg_len == 3:
        tuple_str = ' '.join(lf_seg)
        if tuple_str not in selected_obj_set:
            tuples.append((lf_seg[0], lf_seg[1], lf_seg[2]))
            selected_obj_set.add(tuple_str)
        add_unique_tuple(lf_seg[0], tuples, selected_obj_set)
        add_unique_tuple(lf_seg[2], tuples, selected_obj_set)

    elif seg_len > 3:
        tuple_str = lf_seg[0] + ' ' + ' '.join(lf_seg[1:-1]) + ' ' + lf_seg[-1]
        if tuple_str not in selected_obj_set:
            tuples.append((lf_seg[0], " ".join(lf_seg[1:-1]), lf_seg[-1]))
            selected_obj_set.add(tuple_str)
        add_unique_tuple(lf_seg[0], tuples, selected_obj_set)
        add_unique_tuple(lf_seg[-1], tuples, selected_obj_set)



