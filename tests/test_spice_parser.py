import pandas as pd
import torch

from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_scene_graph_parsing_on_random():
    evaluator = Evaluator(parser=None,text_encoder_checkpoint='all-MiniLM-L6-v2',  device=device,lemmatize=True)

    random_data_pd = pd.read_csv('data/factual_sg_id/random/test.csv')
    random_cand_captions = random_data_pd['caption'].tolist()


    caption_scene_dict = {}

    for line in open(file='tests/test_data/SPICE_parsing_outputs.txt', mode='r').readlines():
        caption_scene_dict[line.split('\t')[0].strip()] = line.split('\t')[1].strip()
    # print(caption_scene_dict)
    random_cand_graphs = []
    for caption in random_cand_captions:
        random_cand_graphs.append(caption_scene_dict[caption])

    random_data_graphs = [[scene] for scene in random_data_pd['scene_graph'].tolist()]
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(random_cand_graphs, random_data_graphs, method='spice', beam_size=1, batch_size=128, max_input_len=256, max_output_len=256, return_graphs=True)

    print('SPICE scores of SPICE Parser for random test set:')
    print(sum(spice_scores)/len(spice_scores))

    set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs,method='set_match', beam_size=1)

    print('Set Match scores of SPICE Parser for random test set:')
    print(sum(set_match_scores)/len(set_match_scores))

    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs,method='soft_spice', beam_size=1)

    print('Soft-SPICE scores of SPICE Parser for random test set:')
    print(sum(soft_spice_scores)/len(soft_spice_scores))



if __name__ == "__main__":
    #test_scene_graph_parsing()
    test_scene_graph_parsing_on_random()