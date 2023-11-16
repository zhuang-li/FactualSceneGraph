import torch
from nltk import WordNetLemmatizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..utils import space_out_symbols_in_graph, clean_graph_string


class SceneGraphParser:
    def __init__(self, checkpoint_path, device='cuda:0', lemmatize=False, lowercase=False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device).eval()
        self.lemmatize = lemmatize
        self.lowercase = lowercase

        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def _process_text(self, text):
        """
        Perform text processing: lemmatization and optionally converting to lowercase.

        :param text: A string containing the text to be processed.
        :return: Processed text as a string.
        """

        if self.lemmatize:
            # Lemmatize each word in the text
            tokens = text.split(' ')
            text = ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])



        if self.lowercase:
            text = text.lower()

        return text

    def parse(self, descriptions, max_input_len=64, max_output_len=128, beam_size=5, return_text=False, batch_size=32):
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # Apply text processing (lemmatization and lowercase) to each description
        processed_descriptions = [self._process_text(desc) for desc in descriptions]

        # Initialize results list
        all_formatted_texts = []

        # Process descriptions in batches
        for i in tqdm(range(0, len(processed_descriptions), batch_size)):
            batch_descriptions = processed_descriptions[i:i + batch_size]
            prompt_texts = ['Generate Scene Graph: ' + desc for desc in batch_descriptions]

            with torch.no_grad():
                encoded_inputs = self.tokenizer(
                    prompt_texts,
                    max_length=max_input_len,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
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
                formatted_texts = [clean_graph_string(
                    space_out_symbols_in_graph(text.replace('Generate Scene Graph:', '').strip())) for text in
                                   generated_texts]
                all_formatted_texts.extend(formatted_texts)

        if return_text:
            return all_formatted_texts
        else:
            return [self.graph_string_to_object(text) for text in all_formatted_texts]

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


