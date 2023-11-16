import json
import os
import pickle
import sys

import numpy as np
import scipy
import torch
from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser



def compute_correlation(input_json, tauvariant='c'):
    data = {}
    with open(input_json) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue

            candidate = ' '.join(human_judgement['caption'].split())
            candidates.append(candidate)

            ref = [' '.join(gt.split()) for gt in v['ground_truth']]
            refs.append(ref)
            human_scores.append(human_judgement['rating'])
    print('Loaded {} references and {} candidates'.format(len(refs), len(candidates)))
    assert len(candidates) == len(refs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device, lemmatize=False, lowercase=True)
    evaluator = Evaluator(parser=parser, text_encoder_checkpoint='all-MiniLM-L6-v2', device=device, lemmatize=True)

    # Paths for the pickle files
    cand_graphs_pickle = 'tests/test_data/cand_graphs.pkl'
    ref_graphs_pickle = 'tests/test_data/ref_graphs.pkl'

    # Function to save an object to a pickle file
    def save_to_pickle(obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)

    # Function to load an object from a pickle file
    def load_from_pickle(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    # Check if the cand_graphs pickle file exists
    if not os.path.exists(cand_graphs_pickle):
        spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(candidates, refs, method='spice',
                                                                   beam_size=1, batch_size=64, max_input_len=256,
                                                                   max_output_len=256, return_graphs=True)
        # Dump cand_graphs if the file doesn't exist
        save_to_pickle(cand_graphs, cand_graphs_pickle)
        save_to_pickle(ref_graphs, ref_graphs_pickle)
    else:
        # Load cand_graphs if the file exists
        cand_graphs = load_from_pickle(cand_graphs_pickle)
        ref_graphs = load_from_pickle(ref_graphs_pickle)
        spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='spice',
                                                                   beam_size=1, batch_size=64, max_input_len=256,
                                                                   max_output_len=256, return_graphs=False)




    assert len(spice_scores) == len(human_scores)
    print('SPICE score: ', sum(spice_scores) / len(spice_scores))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(spice_scores, human_scores, variant=tauvariant)[0]))


    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', batch_size=128)
    assert len(soft_spice_scores) == len(human_scores)
    print('Soft-SPICE score: ', sum(soft_spice_scores) / len(soft_spice_scores))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(soft_spice_scores, human_scores, variant=tauvariant)[0]))



if __name__ == '__main__':
    compute_correlation('tests/test_data/flickr8k.json', tauvariant='c')