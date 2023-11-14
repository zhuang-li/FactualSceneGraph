import json
import os
import sys

import numpy as np
import scipy
import torch



sys.path.append('../')

from factual_scene_graph.evaluation.soft_spice import SoftSpiceScorer


def compute_spice_correlation(input_json, tauvariant='c'):
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
    soft_spice_scorer = SoftSpiceScorer(device=device, text_encoder_checkpoint='all-MiniLM-L6-v2', lemma=False,lowercase=True,max_output_length=256)

    score_list = soft_spice_scorer.soft_spice_score(candidates, refs, batch_size=64)
    assert len(score_list) == len(human_scores)
    print('Soft-SPICE score: ', sum(score_list) / len(score_list))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(score_list, human_scores, variant=tauvariant)[0]))

    score_list = soft_spice_scorer.spice_score(candidates, refs, batch_size=64)
    assert len(score_list) == len(human_scores)
    print('SPICE score: ', sum(score_list) / len(score_list))
    print('{} Tau-{}: {:.3f}'.format(tauvariant, tauvariant, 100*scipy.stats.kendalltau(score_list, human_scores, variant=tauvariant)[0]))


def main():
    compute_spice_correlation('flickr8k.json', tauvariant='c')
    compute_spice_correlation('crowdflower_flickr8k.json', tauvariant='c')





if __name__ == '__main__':
    main()