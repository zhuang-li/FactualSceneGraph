from importlib import resources

from factual_scene_graph.evaluation.synonym_dictionary import SynonymDictionary


with resources.open_text('factual_scene_graph.evaluation.resources', 'english.exceptions') as f_exc, \
     resources.open_text('factual_scene_graph.evaluation.resources', 'english.synsets') as f_syn:
    synonym_dictionary = SynonymDictionary(f_exc.name, f_syn.name)
print(synonym_dictionary.get_stem_synsets('write_down'))