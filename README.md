# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

This repository contains the code and dataset for the paper [FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing](https://arxiv.org/pdf/2305.17497.pdf) (ACL 2023).

## Dataset
### FACTUAL Scene Graph dataset:
FACTUAL Scene Graph dataset includes 40,369 instances. The scene graphs are converted from FACTUAL-MRs with the words in the graphs lemmatized. As we mentioned in the paper, there are several ways to convert the FACTUAL-MRs into scene graphs. Here is the collective way.
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

## FACTUAL Scene Graph Parsing Model

flan-t5 models trained on scene graph dataset in the Random split:

### Scene Graph Parsers without node indexes and passive identifiers

|  | Set Match | SPICE |HF Model Weight|
| -------- | -------- | -------- |-------- |
| Flan-T5-large   | 80.17   | 92.64   |[lizhuang144/flan-t5-large-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-factual-sg)|
| Flan-T5-base    | 80.70   | 92.72   | [lizhuang144/flan-t5-base-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-factual-sg) |
| Flan-T5-small    | 77.72   | 91.67   | [lizhuang144/flan-t5-small-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-factual-sg) |
| (pretrain + fine-tune) Flan-T5-large    | 81.30   | 93.17   | [lizhuang144/flan-t5-large-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg) |
| (pretrain + fine-tune) Flan-T5-base    | 81.50   | 93.33   | [lizhuang144/flan-t5-base-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg) |
| (pretrain + fine-tune) Flan-T5-small    | 79.77   | 92.76   | [lizhuang144/flan-t5-small-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg) |


!!!**Note** that we removed the node index in the dataset when training these models, so the different nodes with the same names won't be distinguished by their indexes. The identifier of passive (i.e. 'p:') has also been removed. The verb and prepositions are concatenated together as well. Such a format loses some information from FACTUAL-MR while the format is compatible with the scene graphs in Visual Genome such that it can be applied to the downstream tasks of scene graphs.

Usage Example:

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-factual-sg")
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-factual-sg")
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

### Scene Graph Parsers with node indexes and passive identifiers

% to do: train the models with dataset annotated with scene graphs with node indexes and passive identifiers

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


