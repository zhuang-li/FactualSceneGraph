from factual_scene_graph.evaluation.spice_evaluation import calculate_spice_score

if __name__ == "__main__":
    cand = [('jean', 'be', 'blue'), ['jean'], ['blue'], ('man', 'wear', 'jeans'), ['man'], ['jeans']]
    refs = [('man', 'wear', 'jean'), ['man'], ['jean'], ('jean', 'be', 'blue'), ['blue']]
    print(calculate_spice_score(cand, refs))
