import logging

from factual_scene_graph.evaluation.spice_evaluation import calculate_spice_score, eval_spice
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    cand = [('jean', 'be', 'blue'), ['jean'], ['blue'], ('man', 'wear', 'jeans'), ['man'], ['jeans']]
    refs = [('man', 'wear', 'jean'), ['man'], ['jean'], ('jean', 'be', 'blue'), ['blue']]
    print(calculate_spice_score(cand, refs, synonym_match=True))

    cand_graph = '( jean , is , blue ) , ( jean , is , blue )'

    ref_graphs = ['( jean , is , blue ) , ( jeans , is , blue )', '( jean , is , blue ) , ( water , is , blue )']

    print(eval_spice(cand_graph, ref_graphs, merge_tuples_synonyms=True, synonym_match=True))

