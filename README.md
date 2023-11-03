# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

This repository contains the code and dataset for the paper [FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing](https://arxiv.org/pdf/2305.17497.pdf) (ACL 2023).


<p align="center" float="left">
  <img src="logo/monash_logo.png" height="80" />
  <img src="logo/adobe_logo.png" height="80" />
  <img src="logo/wuhan_logo.png" height="80" />
</p>

## Dataset
### FACTUAL Scene Graph dataset:
FACTUAL Scene Graph dataset includes 40,369 instances. The scene graphs are converted from FACTUAL-MRs with the words in the predicates/relations of the graphs lemmatized. As we mentioned in the paper, there are several ways to convert the FACTUAL-MRs into scene graphs. Here is the collective way.
```angular2html
data/factual_sg/factual_sg.csv
```
or load from huggingface data hub
```angular2html
sg_dataset = load_dataset('lizhuang144/FACTUAL_Scene_Graph')
```
The train, dev and test splits in our experiments are in the following files:

Random Split:
```angular2html
data/factual_sg/random/train.csv
data/factual_sg/random/test.csv
data/factual_sg/random/dev.csv
```
Length Split:
```angular2html
data/factual_sg/length/train.csv
data/factual_sg/length/test.csv
data/factual_sg/length/dev.csv
```

#### Data Fields
```angular2html
image_id: the id of image in Visual Genome
region_id: the id of region in Visual Genome
caption: the caption of the image region
scene_graph: the scene graph of the image region and caption
```
Please refer to [Visual Genome](https://huggingface.co/datasets/visual_genome) to find the images and regions given the image and region ids.

### FACTUAL-MR dataset:
FACTUAL-MR dataset is annotated with FACTUAL-MR instead of a conventional scene graph for ease of annotation.
todo: add cleaned FACTUAL-MR dataset

### VG Scene Graph dataset:
load from huggingface data hub
```angular2html
sg_dataset = load_dataset('lizhuang144/VG_scene_graph_clean')
```
We clean the VG dataset so it does not include empty instances. The dataset includes 2.9 million instances.

### FACTUAL Scene Graph dataset with identifiers:
load from huggingface data hub
```angular2html
sg_dataset = load_dataset('lizhuang144/FACTUAL_Scene_Graph_ID')
```
The data instances contain identifiers that help determine whether a word in a predicate is a verb, as well as whether it is in the passive tense. Additionally, indexes of nodes are included to distinguish between nodes that share the same name.
## FACTUAL Scene Graph Parsing Model

The flan-t5 models are trained and evaluated on the training and test sets of the Random split. The SPICE parser is evaluated on the test set of the Random split. (pre-train + fine-tune) means the parser is pre-trained on the 3 million VG instances and fine-tuned on the FACTUAL dataset.

### Scene Graph Parsers without node indexes and passive identifiers

|  | Set Match | SPICE |Model Weight|
| -------- | -------- | -------- |-------- |
| SPICE Parser   | 13.00 | 56.15   |[modified-SPICE-score](https://github.com/yychai74/modified-SPICE-score)|
| Flan-T5-large   | 80.17   | 92.64   |[lizhuang144/flan-t5-large-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-factual-sg)|
| Flan-T5-base    | 80.70   | 92.72   | [lizhuang144/flan-t5-base-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-factual-sg) |
| Flan-T5-small    | 77.72   | 91.67   | [lizhuang144/flan-t5-small-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-factual-sg) |
| (pretrain + fine-tune) Flan-T5-large    | 81.30   | 93.17   | [lizhuang144/flan-t5-large-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg) |
| (pretrain + fine-tune) Flan-T5-base    | 81.50   | 93.33   | [lizhuang144/flan-t5-base-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg) |
| (pretrain + fine-tune) Flan-T5-small    | 79.77   | 92.76   | [lizhuang144/flan-t5-small-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg) |

As the table demonstrates, the predominant [SPICE parser](https://panderson.me/images/SPICE.pdf)—created 7 years ago and widely integrated into numerous research projects and practical applications—falls dramatically short in performance metrics. With only a 13% set match rate and a SPICE score of just 56.15, it's clear that this tool is far from optimal. Let's move forward by adopting a more robust and efficient parser!

!!!**Note** that we removed the node index in the dataset when training these models, so the different nodes with the same names won't be distinguished by their indexes. The identifier of passive (i.e. 'p:') has also been removed. The verbs and prepositions are concatenated together as well. Such a format loses some information from FACTUAL-MR while the format is compatible with the scene graphs in Visual Genome such that it can be applied to the downstream tasks of scene graphs.

Usage Example:

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")
text = tokenizer("Generate Scene Graph: 2 pigs are flying on the sky with 2 bags on their backs", max_length=200, return_tensors="pt", truncation=True)
generated_ids = model.generate(
    text["input_ids"],
    attention_mask=text["attention_mask"],
    use_cache=True,
    decoder_start_token_id=tokenizer.pad_token_id,
    num_beams=1,
    max_length=200,
    early_stopping=True
)
tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

`( pigs, is, 2), (bags, on back of, pigs), (bags, is, 2), (pigs, fly on, sky )`

```

Note here 'is' is referred to as 'has_attribute'.

### Scene Graph Parsers with node indexes and verb identifiers
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


