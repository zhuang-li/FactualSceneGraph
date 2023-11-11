import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import space_out_symbols_in_graph, remove_redundant_facts, tprint


class SceneGraphParser:
    def __init__(self, checkpoint_path, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def parse(self, descriptions, max_input_len=64,max_output_len=128, beam_size=5, return_text=False):
        # Parsing logic
        prompt_texts = ['Generate Scene Graph: ' + desc for desc in descriptions]
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

            early_stopping = True
            if beam_size == 1:
                early_stopping = False

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
            formatted_texts = [remove_redundant_facts(space_out_symbols_in_graph(text.replace('Generate Scene Graph:', '').strip())) for text in
                               generated_texts]

            if return_text:
                return formatted_texts
            else:
                return [self.parse_scene_graph(text) for text in formatted_texts]

    def parse_scene_graph(self, scene_description):
        graph = {'entities': [], 'relations': []}
        entity_map = {}  # Entity name to index mapping

        # Process each relation in the description
        relation_strs = scene_description.strip().split(') ,')
        for relation_str in relation_strs:
            relation_str = relation_str.strip().strip('()')
            parts = [part.strip() for part in relation_str.split(',')]

            if len(parts) != 3:
                continue  # Skip malformed relations

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
def test_space_out_symbols_in_graph():
    # Test normal case
    assert space_out_symbols_in_graph("(bench,is,wooden)") == "( bench , is , wooden )"
    # Test with extra spaces
    assert space_out_symbols_in_graph(" (bench ,is,wooden) ") == "( bench , is , wooden )"
    # Test with multiple relations
    assert space_out_symbols_in_graph("(bench,is,wooden),(bench,v:faces,sea)") == "( bench , is , wooden ) , ( bench , v:faces , sea )"
    # Test empty string
    assert space_out_symbols_in_graph("") == ""
    print("All tests for space_out_symbols_in_graph passed!")

def test_scene_graph_parser():
    # Assuming SceneGraphParser is correctly instantiated as `parser`
    # Test normal input
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
    text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"],return_text=True)

    print(text_graph[0])
    graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs",
                              "a blue sky"], return_text=False)
    print(graph_obj[0])
    tprint(graph_obj[0])


if __name__ == "__main__":
    test_space_out_symbols_in_graph()
    test_scene_graph_parser()

