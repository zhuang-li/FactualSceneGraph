# %%
import spacy
import torch
import torch.nn as nn
import traceback

from clip import clip
from nltk import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from typing import List
import numpy as np

from soft_spice.soft_spice_utils import get_graph_phrases, eval_soft_spice
from soft_spice.spice_utils import get_graph_tuples, eval_spice
from soft_spice.utils import format_scene_graph


class SoftSpiceScorer:
    def __init__(self, device='cuda:0', max_input_length=64, max_output_length=64, beam_size = 1, lemma=False, lowercase=False, parser_checkpoint='lizhuang144/flan-t5-base-factual-sg', text_encoder_checkpoint=None, multi_modal_encoder_checkpoint=None):
        '''
        :param device: gpu or cpu
        :param max_length: max length of input text
        :param parser_checkpoint: model checkpoint for parser
        :param text_encoder_checkpoint: model checkpoint for text encoder
        :param multi_modal_encoder_checkpoint: model checkpoint for multi-modal encoder
        '''
        # the default models
        assert parser_checkpoint in ['lizhuang144/flan-t5-large-factual-sg','lizhuang144/flan-t5-base-factual-sg','lizhuang144/flan-t5-small-factual-sg',
                                     'lizhuang144/flan-t5-large-VG-factual-sg','lizhuang144/flan-t5-base-VG-factual-sg', 'lizhuang144/flan-t5-small-VG-factual-sg']
        assert text_encoder_checkpoint in [None, 'all-MiniLM-L6-v2', 'all-mpnet-base-v2']
        assert multi_modal_encoder_checkpoint in [None, 'ViT-B/32']

        # Set up parsing model
        self.device = device
        self.max_input_length = max_input_length
        self.parser_tokenizer = AutoTokenizer.from_pretrained(parser_checkpoint)
        self.parser = AutoModelForSeq2SeqLM.from_pretrained(parser_checkpoint)
        self.parser.eval()
        self.parser.to(device)

        self.lemma = lemma
        self.lowercase = lowercase
        if self.lemma:
            # please download the en_core_web_sm first
            self.lemmatizer = spacy.load("en_core_web_sm")



        self.max_output_length = max_output_length
        self.beam_size = beam_size

        if text_encoder_checkpoint is not None:
            # set up text encoder for the soft-spice
            self.text_encoder = SentenceTransformer(text_encoder_checkpoint)
            self.text_encoder.eval()
            self.text_encoder.to(device)

        if multi_modal_encoder_checkpoint is not None:
            # set up multi-modal encoder for soft-spice(img), now it only supports clip
            self.multi_modal_encoder, transform = clip.load(multi_modal_encoder_checkpoint, device=device, jit=False)
            self.multi_modal_encoder.eval()
            self.multi_modal_encoder.to(device)

    def parse(self, text_input,max_input_length=64, max_output_length=64, beam_size = 1,lowercase=False, lemma=False):
        '''
        :param text_input: one or a list of textual image descriptions
        :return: corresponding scene graphs of the input descriptions
        '''

        if isinstance(text_input, str):
            text_input = [text_input]

        if lowercase:
            text_input = [text.lower() for text in text_input]

        if lemma:
            text_input = [' '.join([token.lemma_ for token in self.lemmatizer(text)]) for text in text_input]

        #breakpoint()
        text_input = ['Generate Scene Graph: ' + text for text in text_input]
        with torch.no_grad():
            encoded_text = self.parser_tokenizer(
                text_input,
                max_length=max_input_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            text_tokens = encoded_text['input_ids'].to(self.device)
            text_mask = encoded_text['attention_mask'].to(self.device)

            generated_ids = self.parser.generate(
                text_tokens,
                attention_mask=text_mask,
                use_cache=True,
                decoder_start_token_id=self.parser_tokenizer.pad_token_id,
                num_beams=beam_size,
                max_length=max_output_length,
                early_stopping=True
            )

            # output to text
            output_text = self.parser_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_text = [format_scene_graph(text.replace('Generate Scene Graph:','').strip()) for text in output_text]
            return output_text

    def spice_score(self, cands, refs, batch_size=4):
        '''
        Reimplementation of SPICE score metric, which might result in slightly different scores than the original implementation.
        To reproduce the original score, please refer to: https://github.com/peteanderson80/SPICE

        :param: `cands` (list of str): candidate sentences
        :param: `refs` (list of str or list of list of str): reference sentences
        :param batch_size: (int) batch size to be processed
        :return: `spice_score`: a list of F1 spice scores for each candidate
        '''
        write_fp = open('spice_input.txt', 'w')
        score_list = []
        for i in tqdm(range(0, len(cands), batch_size)):
            src_list = cands[i: i + batch_size]
            ref_list_list = refs[i: i + batch_size]
            try:
                flatten_refs = [ref for ref_list in ref_list_list for ref in ref_list]
                merge_list = src_list + flatten_refs
                merge_scene_graphs = self.parse(merge_list, max_input_length=self.max_input_length, max_output_length=self.max_output_length, beam_size=self.beam_size, lemma=self.lemma, lowercase=self.lowercase)

                for src, scene_graph in zip(merge_list, merge_scene_graphs):
                    write_fp.write(src.strip()+'\t'+scene_graph.strip()+'\n')

                src_scene_graphs = merge_scene_graphs[:len(src_list)]
                merge_scene_graphs = merge_scene_graphs[len(src_list):]
                ref_scene_graphs = []
                #breakpoint()
                for idx, ref_list in enumerate(ref_list_list):
                    ref_scene_graphs.append(merge_scene_graphs[:len(ref_list)])
                    merge_scene_graphs = merge_scene_graphs[len(ref_list):]
                assert len(merge_scene_graphs) == 0

                for src_graph, ref_graphs in zip(src_scene_graphs, ref_scene_graphs):
                    src_tuple = get_graph_tuples(src_graph)
                    ref_tuples = get_graph_tuples(ref_graphs)
                    spice_score = eval_spice(src_tuple, ref_tuples)
                    score_list.append(spice_score)


            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {refs}')
                exit(0)
        return score_list

    def soft_spice_score(self, cands, refs, batch_size=4, type_constraint=False, use_cache=False):
        '''
        Reimplementation of SPICE score metric, which might result in slightly different scores than the original implementation.
        To reproduce the original score, please refer to: https://github.com/peteanderson80/SPICE

        :param: `cands` (list of str): candidate sentences
        :param: `refs` (list of str or list of list of str): reference sentences
        :param batch_size: (int) batch size to be processed
        :return: `spice_score`: a list of F1 spice scores for each candidate
        '''
        type_dict = {}
        cache = {}
        score_list = []
        for i in tqdm(range(0, len(cands), batch_size)):
            src_list = cands[i: i + batch_size]
            ref_list_list = refs[i: i + batch_size]
            try:
                flatten_refs = [ref for ref_list in ref_list_list for ref in ref_list]
                merge_list = src_list + flatten_refs
                merge_scene_graphs = self.parse(merge_list, max_input_length=self.max_input_length,
                                                max_output_length=self.max_output_length, beam_size=self.beam_size,
                                                lemma=self.lemma, lowercase=self.lowercase)


                src_scene_graphs = merge_scene_graphs[:len(src_list)]
                merge_scene_graphs = merge_scene_graphs[len(src_list):]
                ref_scene_graphs = []
                # breakpoint()
                for idx, ref_list in enumerate(ref_list_list):
                    ref_scene_graphs.append(merge_scene_graphs[:len(ref_list)])
                    merge_scene_graphs = merge_scene_graphs[len(ref_list):]
                assert len(merge_scene_graphs) == 0

                for src_graph, ref_graphs in zip(src_scene_graphs, ref_scene_graphs):
                    src_tuple = get_graph_phrases(src_graph,type_dict)
                    ref_tuples = get_graph_phrases(ref_graphs, type_dict)

                    spice_score = eval_soft_spice(src_tuple, ref_tuples, text_encoder=self.text_encoder, batch_size=batch_size, type_dict=type_dict, cache=cache, type_constraint=type_constraint, use_cache=use_cache)
                    score_list.append(spice_score)


            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {refs}')
                exit(0)
        return score_list

    def soft_spice_img_score(self, srcs, refs, batch_size=4, use_cache=False):
        pass

