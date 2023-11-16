import logging
from importlib import resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SynonymDictionary:
    def __init__(self, exc_file_path, syn_file_path):
        self.word_to_bases = {}
        self.word_to_synsets = {}
        self.set_to_relations = {}

        # logging.info("Reading synonym dictionary..")
        # Reading exception file
        with open(exc_file_path, 'r', encoding='utf-8') as file:
            for base in file:
                base = base.strip()
                forms = next(file).strip().split()
                for form in forms:
                    self.word_to_bases.setdefault(form, []).append(base)

        # Reading synset file
        with open(syn_file_path, 'r', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                sets = set(map(int, next(file).strip().split()))
                self.word_to_synsets[word] = sets

        # logging.info("Reading synonym dictionary.. Done")
    def get_synsets(self, word):
        return self.word_to_synsets.get(word, set())

    def get_stem_synsets(self, word):
        bases = self.word_to_bases.get(word)
        if bases:
            sets = set()
            for base in bases:
                sets.update(self.get_synsets(base))
            return sets
        return self.get_synsets(self.morph(word))


    def morph(self, word):
        sufx = [
            "s", "ses", "xes", "zes", "ches", "shes", "men", "ies",  # Noun suffixes
            "s", "ies", "es", "es", "ed", "ed", "ing", "ing",        # Verb suffixes
            "er", "est", "er", "est"                                 # Adjective suffixes
        ]

        addr = [
            "", "s", "x", "z", "ch", "sh", "man", "y",               # Noun endings
            "", "y", "e", "", "e", "", "e", "",                      # Verb endings
            "", "", "e", "e"                                         # Adjective endings
        ]

        if word.endswith("ful"):
            base = word[:-3]  # Remove 'ful'
            if base in self.word_to_synsets:
                return base
            return word

        if word.endswith("ss") or len(word) <= 2:
            return word

        for sufx, addr in zip(sufx, addr):
            if word.endswith(sufx):
                base = word[:-len(sufx)] + addr
                if base in self.word_to_synsets:
                    return base

        return word


with resources.open_text('factual_scene_graph.evaluation.resources', 'english.exceptions') as f_exc, \
        resources.open_text('factual_scene_graph.evaluation.resources', 'english.synsets') as f_syn:
    synonym_dictionary = SynonymDictionary(f_exc.name,
                                            f_syn.name)


