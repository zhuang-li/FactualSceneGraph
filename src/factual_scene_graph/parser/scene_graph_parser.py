import torch
from nltk import WordNetLemmatizer
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

from ..utils import space_out_symbols_in_graph, clean_graph_string, remove_factual_chars


class SceneGraphParser:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        lemmatize: bool = False,
        lowercase: bool = False,
        parser_type: str = "default",
        refiner_checkpoint_path: Optional[str] = None,
    ) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device).eval()
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.parser_type = parser_type

        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        if parser_type in {"sentence_merge", "DiscoSG-Refiner"}:
            # Download required NLTK data if not already downloaded
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                try:
                    nltk.download('punkt_tab')
                except:
                    # Fallback to older punkt for older NLTK versions
                    nltk.download('punkt')

        if parser_type == "DiscoSG-Refiner":
            from discosg.parser.DualTaskSceneGraphParser import DualTaskSceneGraphParser

            self.refiner_model = DualTaskSceneGraphParser(
                model_path=refiner_checkpoint_path,
                device=device,
                lemmatize=lemmatize,
                lowercase=lowercase,
            )

    def _process_text(self, text: str) -> str:
        if self.lemmatize:
            tokens = text.split()
            text = " ".join(self.lemmatizer.lemmatize(t) for t in tokens)
        return text.lower() if self.lowercase else text

    def _merge_graphs(self, graph_texts):
        return clean_graph_string(" , ".join(graph_texts))


    # ------------------------------------------------------------------
    # sentence-level parsing then merge
    # ------------------------------------------------------------------
    def _sentence_merge_graph(
        self,
        descriptions,
        max_input_len,
        max_output_len,
        beam_size,
        filter_factual_chars,
        batch_size,
    ):
        all_sentences, sent2desc = [], []
        for d_idx, desc in enumerate(descriptions):
            for sent in nltk.sent_tokenize(desc):
                sent = sent.strip()
                if sent:
                    all_sentences.append(self._process_text(sent))
                    sent2desc.append(d_idx)

        if not all_sentences:
            return [""] * len(descriptions)

        sent_graphs = self._parse_batch(
            all_sentences,
            max_input_len,
            max_output_len,
            beam_size,
            filter_factual_chars,
            batch_size,
        )

        merged = [""] * len(descriptions)
        for s_idx, g in enumerate(sent_graphs):
            if g:
                d_idx = sent2desc[s_idx]
                merged[d_idx] = (
                    g
                    if not merged[d_idx]
                    else self._merge_graphs([merged[d_idx], g])
                )
        return merged


    def parse(
        self,
        descriptions,
        max_input_len: int = 64,
        max_output_len: int = 128,
        beam_size: int = 5,
        return_text: bool = False,
        filter_factual_chars: bool = False,
        batch_size: int = 32,
        task: str = "delete_before_insert",
        refinement_rounds: int = 2,
    ):
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # 1) DiscoSG-Refiner flow
        if self.parser_type == "DiscoSG-Refiner":
            # build initial graphs per description depending on sentence count
            single_idxs, multi_idxs = [], []
            for idx, desc in enumerate(descriptions):
                if len(nltk.sent_tokenize(desc.strip())) == 1:
                    single_idxs.append(idx)
                else:
                    multi_idxs.append(idx)

            # prepare list to hold initial graphs in original order
            current_graphs = [""] * len(descriptions)

            # process single-sentence descriptions in one batch
            if single_idxs:
                single_texts = [self._process_text(descriptions[i]) for i in single_idxs]
                single_graphs = self._parse_batch(
                    single_texts,
                    max_input_len,
                    max_output_len,
                    beam_size,
                    filter_factual_chars,
                    batch_size,
                )
                for idx, g in zip(single_idxs, single_graphs):
                    current_graphs[idx] = g

            # process multi-sentence descriptions through sentence_merge
            if multi_idxs:
                multi_descs = [descriptions[i] for i in multi_idxs]
                multi_graphs = self._sentence_merge_graph(
                    multi_descs,
                    max_input_len,
                    max_output_len,
                    beam_size,
                    filter_factual_chars,
                    batch_size,
                )
                for idx, g in zip(multi_idxs, multi_graphs):
                    current_graphs[idx] = g

                multi_descs = [descriptions[i] for i in multi_idxs]
                graph_to_fix = {d: g for d, g in zip(multi_descs, [current_graphs[i] for i in multi_idxs])}

                # --------------------------------------------------
                # Safety check: ensure our refiner's max_input_len is
                # sufficient for at least the shortest caption plus a
                # small buffer.  If not, refinement would error out in
                # discosg, so we warn the user and skip this phase.
                # --------------------------------------------------
                buffer_tokens = 20  # extra tokens for prompt overhead
                try:
                    token_lens = [
                        len(self.refiner_model.tokenizer.encode(d, add_special_tokens=False))
                        for d in multi_descs
                    ]
                    if token_lens and (min(token_lens) + buffer_tokens) > max_input_len:
                        import warnings
                        warnings.warn(
                            "Skipping DiscoSG refinement because max_input_len="
                            f"{max_input_len} is smaller than the shortest caption "
                            f"({min(token_lens)}) plus buffer ({buffer_tokens}). "
                            "Increase max_input_len or use shorter captions to enable refinement."
                        )
                        if return_text:
                            return current_graphs
                        return [self.graph_string_to_object(g) for g in current_graphs]
                except Exception:
                    # If tokenizer is unexpectedly missing just proceed without the check
                    pass

                # iterative refinement
                for _ in range(refinement_rounds):
                    outputs = self.refiner_model.parse(
                        descriptions=multi_descs,
                        graph_to_fix=graph_to_fix,
                        batch_size=batch_size,
                        task=task,
                        max_input_len=max_input_len,
                        max_output_len=max_output_len,
                        num_beams=beam_size,
                    )

                    del_dict = outputs.get("delete", {})
                    ins_dict = outputs.get("insert", {})
                    refined_dict = {}
                    for desc in multi_descs:
                        d_graph = del_dict.get(desc, "")
                        i_graph = ins_dict.get(desc, "")
                        if d_graph and i_graph:
                            refined_dict[desc] = self._merge_graphs([d_graph, i_graph])
                        elif i_graph:
                            refined_dict[desc] = i_graph
                        else:
                            refined_dict[desc] = d_graph

                    graph_to_fix = refined_dict
                    
                # write the final refined graphs back in the original order
                for desc, idx in zip(multi_descs, multi_idxs):
                    if desc in graph_to_fix:
                        current_graphs[idx] = graph_to_fix[desc]

            if return_text:
                return current_graphs
            return [self.graph_string_to_object(g) for g in current_graphs]

        # 2) sentence_merge flow
        if self.parser_type == "sentence_merge":
            merged_graphs = self._sentence_merge_graph(
                descriptions,
                max_input_len,
                max_output_len,
                beam_size,
                filter_factual_chars,
                batch_size,
            )
            if return_text:
                return merged_graphs
            return [self.graph_string_to_object(g) for g in merged_graphs]

        # 3) default flow
        processed = [self._process_text(d) for d in descriptions]
        processed = self._parse_batch(
            processed,
            max_input_len,
            max_output_len,
            beam_size,
            filter_factual_chars,
            batch_size,
        )
        if return_text:
            return processed
        return [self.graph_string_to_object(t) for t in processed]


    def _parse_batch(self, descriptions, max_input_len, max_output_len, beam_size, filter_factual_chars, batch_size):
        """Helper method to parse a batch of descriptions"""
        all_formatted_texts = []
        
        # Process descriptions in batches
        for i in tqdm(range(0, len(descriptions), batch_size)):
            batch_descriptions = descriptions[i:i + batch_size]
            prompt_texts = [
                'Generate Scene Graph: ' + d.strip() for d in batch_descriptions
            ]

            # Compute the true max length in this mini-batch (in tokens) to avoid
            # padding everything up to the global `max_input_len` when not needed.
            with self.tokenizer.as_target_tokenizer():
                token_lens = [
                    len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompt_texts
                ]
            local_max_len = min(max_input_len, max(token_lens))

            with torch.no_grad():
                encoded_inputs = self.tokenizer(
                    prompt_texts,
                    max_length=local_max_len,
                    truncation=True,
                    padding='longest',  # pad only to longest in this batch
                    return_tensors='pt',
                )
                tokens = encoded_inputs['input_ids'].to(self.device)
                attention_masks = encoded_inputs['attention_mask'].to(self.device)

                early_stopping = beam_size > 1

                generated_ids = self.model.generate(
                    tokens,
                    attention_mask=attention_masks,
                    use_cache=True,
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                    num_beams=beam_size,
                    max_length=max_output_len,
                    early_stopping=early_stopping,
                    num_return_sequences=1,
                )

                # Decoding the output
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=True)

                if filter_factual_chars:
                    generated_texts = [remove_factual_chars(text) for text in generated_texts]

                formatted_texts = [clean_graph_string(
                    space_out_symbols_in_graph(text.replace('Generate Scene Graph:', '').strip())) for text in
                                   generated_texts]
                all_formatted_texts.extend(formatted_texts)
        
        return all_formatted_texts

    def graph_string_to_object(self, graph_text):
        graph = {'entities': [], 'relations': []}
        entity_map = {}  # Entity name to index mapping

        # Process each relation in the description
        relation_strs = graph_text.strip().split(') ,')
        for relation_str in relation_strs:
            relation_str = relation_str.strip().strip('()')
            parts = [part.strip() for part in relation_str.split(',')]

            if len(parts) != 3 and len(relation_strs) > 1:
                continue  # Skip malformed relations
            elif len(parts) != 3 and len(relation_strs) == 1:
                self._get_or_create_entity_index(parts[0], graph, entity_map)
            else:
                subject, relationship, object_ = parts

                subject_index = self._get_or_create_entity_index(subject, graph, entity_map)

                if relationship == 'is':
                    if object_.isdigit():  # Quantity
                        graph['entities'][subject_index]['quantity'] = object_
                    else:  # Attribute
                        graph['entities'][subject_index]['attributes'].add(object_)
                else:
                    object_index = self._get_or_create_entity_index(object_, graph, entity_map)
                    # Add relation
                    graph['relations'].append({'subject': subject_index, 'relation': relationship, 'object': object_index})

        return graph

    def _get_or_create_entity_index(self, entity_name, graph, entity_map):
        if entity_name not in entity_map:
            new_index = len(graph['entities'])
            graph['entities'].append({'head': entity_name, 'quantity': '', 'attributes': set()})
            entity_map[entity_name] = new_index
        else:
            new_index = entity_map[entity_name]

        return new_index


# unit test main


