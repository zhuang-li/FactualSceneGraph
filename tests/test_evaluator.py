import pandas as pd
import torch

from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

device = "cuda" if torch.cuda.is_available() else "cpu"
def test_scene_graph_parsing():

    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)
    evaluator = Evaluator(parser=parser, device='cuda:0')
    scores = evaluator.evaluate(["2 beautiful pigs are flying on the sky with 2 bags on their backs"],[['( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( bags , is , 2 ) , ( pigs , is , 2 ) , ( pigs , fly on , sky )']],method='spice', beam_size=1, max_output_len=128)
    print(scores)

def test_scene_graph_parsing_on_random():
    parser = SceneGraphParser('lizhuang144/flan-t5-large-VG-factual-sg-id', device=device,lemmatize=False)
    evaluator = Evaluator(parser=parser,text_encoder_checkpoint='all-MiniLM-L6-v2',  device='cuda:0',lemmatize=True)

    random_data_pd = pd.read_csv('data/factual_sg_id/random/test.csv')
    random_data_captions = random_data_pd['caption'].tolist()
    random_data_graphs = [[scene] for scene in random_data_pd['scene_graph'].tolist()]
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(random_data_captions, random_data_graphs, method='spice', beam_size=1, batch_size=128, max_input_len=256, max_output_len=256, return_graphs=True)

    print('SPICE scores for random test set:')
    print(sum(spice_scores)/len(spice_scores))

    set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs,method='set_match', beam_size=1)

    print('Set Match scores for random test set:')
    print(sum(set_match_scores)/len(set_match_scores))

    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs,method='soft_spice', beam_size=1)

    print('Soft-SPICE scores for random test set:')
    print(sum(soft_spice_scores)/len(soft_spice_scores))



if __name__ == "__main__":
    #test_scene_graph_parsing()
    test_scene_graph_parsing_on_random()