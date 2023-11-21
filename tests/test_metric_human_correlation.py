import json
import os
import pickle
import sys

import numpy as np
import scipy
import torch
from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser


def collect_unique_captions(candidates, refs):
    """Collect unique captions from candidates and references."""
    caption_set = set(candidates)  # Add all candidates to the set
    for ref_list in refs:
        caption_set.update(ref_list)  # Add all elements from each list in refs
    return list(caption_set)

def parse_captions(captions, parser):
    """Parse captions and return a dictionary of results."""
    parse_results = parser.parse(captions, batch_size=64, max_input_len=256,
                                 max_output_len=256, beam_size=1, return_text=True)
    return {caption: result
            for caption, result in zip(captions, parse_results)}

def evaluate_graphs(candidates, refs, parse_dict, evaluator, return_graphs):
    """Evaluate the graphs and return the results."""
    cand_graphs = [parse_dict[cand] for cand in candidates]
    ref_graphs = [[parse_dict[ref_i] for ref_i in ref] for ref in refs]
    return evaluator.evaluate(cand_graphs, ref_graphs, method='spice',
                              beam_size=1, batch_size=64, max_input_len=256,
                              max_output_len=256, return_graphs=return_graphs)

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


    caption_list = collect_unique_captions(candidates, refs)
    parse_dict = parse_captions(caption_list, parser)

    # Evaluate with return_graphs=True
    spice_scores, cand_graphs, ref_graphs = evaluate_graphs(candidates, refs, parse_dict, evaluator, True)


    assert len(spice_scores) == len(human_scores)
    print('SPICE score: ', sum(spice_scores) / len(spice_scores))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(spice_scores, human_scores, variant=tauvariant)[0]))


    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', batch_size=128)
    assert len(soft_spice_scores) == len(human_scores)
    print('Soft-SPICE score: ', sum(soft_spice_scores) / len(soft_spice_scores))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(soft_spice_scores, human_scores, variant=tauvariant)[0]))



if __name__ == '__main__':
    compute_correlation('tests/test_data/flickr8k.json', tauvariant='c')