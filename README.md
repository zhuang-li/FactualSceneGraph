# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

This repository contains the code and dataset for the paper [FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing](https://arxiv.org/pdf/2305.17497.pdf) (ACL 2023).

## Dataset
### FACTUAL Scene Graph dataset:
FACTUAL Scene Graph dataset includes 40,369 instances.
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
FACTUAL-MR dataset is annotated with FACTUAL-MR instead of conventional scene graph for the ease of annotation.
todo: add cleaned FACTUAL-MR dataset

### VG Scene Graph dataset:
load from huggingface data hub
```angular2html
sg_dataset = load_dataset('lizhuang144/VG_scene_graph_clean')
```
We clean the VG dataset so it does not include empty instances. The dataset includes 2.9 million instances.

## FACTUAL Scene Graph Parsing Model

## FACTUAL-MR Scene Graph Parsing Model

## Soft-SPICE

## Citation
```
@article{li2023factual,
  title={FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing},
  author={Li, Zhuang and Chai, Yuyang and Yue, Terry Zhuo and Qu, Lizhen and Haffari, Gholamreza and Li, Fei and Ji, Donghong and Tran, Quan Hung},
  journal={arXiv preprint arXiv:2305.17497},
  year={2023}
}
```


