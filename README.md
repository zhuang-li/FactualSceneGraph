# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

This repository contains the code and dataset for the paper [FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing](https://arxiv.org/pdf/2305.17497.pdf) (ACL 2023).

<p align="center">
  <img src="logo/monash_logo.png" height="80" />
  <img src="logo/adobe_logo.png" height="80" />
  <img src="logo/wuhan_logo.png" height="80" />
</p>

## Dataset

The FACTUAL Scene Graph dataset includes 40,369 instances with lemmatized predicates/relations.

### FACTUAL Scene Graph dataset:

- **Storage**: `data/factual_sg/factual_sg.csv`
- **From Huggingface**: `load_dataset('lizhuang144/FACTUAL_Scene_Graph')`

**Splits**:
- Random Split:
  - Train: `data/factual_sg/random/train.csv`
  - Test: `data/factual_sg/random/test.csv`
  - Dev: `data/factual_sg/random/dev.csv`
- Length Split:
  - Train: `data/factual_sg/length/train.csv`
  - Test: `data/factual_sg/length/test.csv`
  - Dev: `data/factual_sg/length/dev.csv`

**Data Fields**:

- `image_id`: The ID of the image in Visual Genome.
- `region_id`: The ID of the region in Visual Genome.
- `caption`: The caption of the image region.
- `scene_graph`: The scene graph of the image region and caption.

**Related Resources**: [Visual Genome](https://huggingface.co/datasets/visual_genome)

### FACTUAL-MR dataset:

- TODO: Add cleaned FACTUAL-MR dataset.

### VG Scene Graph dataset:

- **From Huggingface**: `load_dataset('lizhuang144/VG_scene_graph_clean')`
- **Details**: Cleaned to exclude empty instances; includes 2.9 million instances.

### FACTUAL Scene Graph dataset with identifiers:

- **From Huggingface**: `load_dataset('lizhuang144/FACTUAL_Scene_Graph_ID')`
- **Enhancements**: Contains verb identifiers, passive voice indicators, and node indexes.

## Scene Graph Parsing Models Performance

### Simplified Model Training Without Node Indexes and Passive Identifiers

The following table shows the performance comparison of various scene graph parsing models. Notably, the original SPICE parser exhibits lower performance relative to more recent models.

| Model | Set Match | SPICE | Model Weight |
|-------|-----------|-------|--------------|
| SPICE Parser | 13.00 | 56.15 | [modified-SPICE-score](https://github.com/yychai74/modified-SPICE-score) |
| Flan-T5-large | 80.17 | 92.64 | [flan-t5-large-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-factual-sg) |
| Flan-T5-base | 80.70 | 92.72 | [flan-t5-base-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-factual-sg) |
| Flan-T5-small | 77.72 | 91.67 | [flan-t5-small-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-factual-sg) |
| (pre) Flan-T5-large | 81.30 | 93.17 | [flan-t5-large-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg) |
| (pre) Flan-T5-base | 81.50 | 93.33 | [flan-t5-base-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg) |
| (pre) Flan-T5-small | 79.77 | 92.76 | [flan-t5-small-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg) |

The prefix "(pre)" indicates models that were pre-trained on the VG scene graph dataset before being fine-tuned on the FACTUAL dataset. The outdated SPICE parser, despite its historical significance, shows a Set Match rate of only 13% and a SPICE score of 56.15, which is significantly lower than the more recent Flan-T5 models fine-tuned on FACTUAL data.

> **Note**: In the training of these models, we have removed the node index, meaning that different nodes with identical names will not be distinguished by their indexes. Furthermore, passive identifiers, such as 'p:', are excluded, and verbs and prepositions have been merged. This format, while losing some information from the FACTUAL-MR dataset, remains compatible with the Visual Genome scene graphs and can be effectively used in downstream scene graph tasks.



**Usage Example**:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")

text = tokenizer(
    "Generate Scene Graph: 2 pigs are flying on the sky with 2 bags on their backs",
    max_length=200,
    return_tensors="pt",
    truncation=True
)

generated_ids = model.generate(
    text["input_ids"],
    attention_mask=text["attention_mask"],
    use_cache=True,
    decoder_start_token_id=tokenizer.pad_token_id,
    num_beams=1,
    max_length=200,
    early_stopping=True
)

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# Output: `( pigs, is, 2), (bags, on back of, pigs), (bags, is, 2), (pigs, fly on, sky )`
```

Note here 'is' is referred to as 'has_attribute'.

### With node indexes and verb identifiers

This type of scene graph parsing generates a detailed scene graph, complete with verb identifiers and node indexes, based on the input text provided. To illustrate, the sentence "a monkey is sitting next to another monkey" would be parsed into the scene graph "( monkey, v:sit next to, monkey:1 )". In this graph, the prefix "v:" specifies that "sit" functions as the verb, while the suffix ":1" denotes that the second "monkey" is distinct from the first.

Similarly, the sentence "a car is parked on the ground" would be translated into the scene graph "( car, pv:park, on, ground )". In this case, the prefix "pv:" indicates that "park" is a passive verb. We highlight passive verbs as the order of nodes in the scene graphs is very important.

These improvements in scene graph parsing offer several benefits over the original scene graphs used in the Visual Genome. For instance, in the original VG scene graph, nodes with identical names couldn't be differentiated. Our enhanced parsing approach solves this issue by adding indexes to uniquely identify nodes with the same name. Additionally, we annotate each predicate with the corresponding verb and its tense. These added features enrich the scene graph with more fine-grained information, enhancing its utility for downstream tasks.

|  | Set Match | SPICE |Model Weight|
| -------- | -------- | -------- |-------- |
| (pretrain + fine-tune) Flan-T5-large    | 80.57   | 92.97   | [lizhuang144/flan-t5-large-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg-id) |
| (pretrain + fine-tune) Flan-T5-base    | 80.70   | 92.89   | [lizhuang144/flan-t5-base-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg-id) |
| (pretrain + fine-tune) Flan-T5-small    | 78.38   | 91.93   | [lizhuang144/flan-t5-small-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg-id) |

## FACTUAL-MR Scene Graph Parsing Model

## Soft-SPICE

## Citation

To cite this work, please use the following Bibtex entry:

```
@inproceedings{li-etal-2023-factual,
    title = "{FACTUAL}: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing",
    author = "Li, Zhuang  and
      Chai, Yuyang  and
      Zhuo, Terry Yue  and
      Qu, Lizhen  and
      Haffari, Gholamreza  and
      Li, Fei  and
      Ji, Donghong  and
      Tran, Quan Hung",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.398",
    pages = "6377--6390",
    abstract = "Textual scene graph parsing has become increasingly important in various vision-language applications, including image caption evaluation and image retrieval. However, existing scene graph parsers that convert image captions into scene graphs often suffer from two types of errors. First, the generated scene graphs fail to capture the true semantics of the captions or the corresponding images, resulting in a lack of faithfulness. Second, the generated scene graphs have high inconsistency, with the same semantics represented by different annotations.To address these challenges, we propose a novel dataset, which involves re-annotating the captions in Visual Genome (VG) using a new intermediate representation called FACTUAL-MR. FACTUAL-MR can be directly converted into faithful and consistent scene graph annotations. Our experimental results clearly demonstrate that the parser trained on our dataset outperforms existing approaches in terms of faithfulness and consistency. This improvement leads to a significant performance boost in both image caption evaluation and zero-shot image retrieval tasks. Furthermore, we introduce a novel metric for measuring scene graph similarity, which, when combined with the improved scene graph parser, achieves state-of-the-art (SOTA) results on multiple benchmark datasets for the aforementioned tasks.",
}
```


