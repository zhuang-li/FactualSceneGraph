# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

Welcome to the official repository for the ACL 2023 paper:  
[**FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing**](https://arxiv.org/pdf/2305.17497.pdf). Here, you'll find both the code and dataset associated with our research.

<div align="center">
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/monash_logo.png" alt="Monash University Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/adobe_logo.png" alt="Adobe Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/wuhan_logo.png" alt="Wuhan University Logo" height="80" />
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2305.17497"><img src="https://img.shields.io/badge/arXiv-2305.17497-b31b1b.svg"></a>
    <a href="https://pypi.org/project/FactualSceneGraph/"><img src="https://img.shields.io/pypi/v/FactualSceneGraph?color=g"></a>
    <a href="https://pepy.tech/project/FactualSceneGraph"><img src="https://static.pepy.tech/badge/FactualSceneGraph"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
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

**Related Resources**: Please find the details of images and regions from [Visual Genome](https://huggingface.co/datasets/visual_genome) given their corresponding IDs.

### FACTUAL-MR dataset:

- `data/factual_mr/factual_mr.csv`
- `data/factual_mr/meta.json`: the metadata for mapping the abbreviations of quantifiers in `factual_mr.csv` to their complete names.

### VG Scene Graph dataset:

- **From Huggingface**: `load_dataset('lizhuang144/VG_scene_graph_clean')`
- **Details**: Cleaned to exclude empty instances; includes 2.9 million instances.

### FACTUAL Scene Graph dataset with identifiers:

- **From Huggingface**: `load_dataset('lizhuang144/FACTUAL_Scene_Graph_ID')`
- **Enhancements**: Contains verb identifiers, passive voice indicators, and node indexes.

## Scene Graph Parsing Models

### Simplified Model Training Without Node Indexes and Passive Identifiers

The following table shows the performance comparison of various scene graph parsing models. Notably, the original SPICE parser performs worse than our more recent models.

#### Performance Metrics Explained:
- **SPICE F-score**: A metric that measures the similarity between candidate and reference scene graph representations derived from captions. It assesses the quality of scene graph parsing by evaluating how well the parser's output matches the ground truth graph in terms of propositional content.
- **Exact Set Match**: Adapted from the methodology described by [Yu et al., 2019](https://aclanthology.org/P19-1443/), this metric evaluates the parser's accuracy by verifying whether the strings of parsed facts match the ground truth facts, without considering the ordering of those facts. This adaptation is a stringent accuracy measure, necessitating an exact correspondence between the candidate and ground truth facts.

> **Note**: It is important to note that in the original work of Yu et al., 2019, the metric was applied to SQL clauses, whereas in our context, it has been tailored to assess scene graph facts.


| Model | Set Match | SPICE | Soft-SPICE | Model Weight |
|-------|-----------|-------|------------|--------------|
| SPICE/Stanford Parser | 19.30 | 64.77 | 92.60     | [modified-SPICE-score](https://github.com/yychai74/modified-SPICE-score) |
| (pre) Flan-T5-large | 81.63     | 93.20 | 98.75      | [flan-t5-large-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg) |
| (pre) Flan-T5-base | 81.37     | 93.27  | 98.83      | [flan-t5-base-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg) |
| (pre) Flan-T5-small | 78.18     | 92.26 | 98.67      | [flan-t5-small-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg) |

The prefix "(pre)" indicates models that were pre-trained on the VG scene graph dataset before being fine-tuned on the FACTUAL dataset. The outdated SPICE parser, despite its historical significance, shows a Set Match rate of only 19.30% and a SPICE score of 64.77, which is significantly lower than the more recent Flan-T5 models fine-tuned on FACTUAL data.

> **Note**:
> 1. **Model Training Adjustments**: In training these models, the node index has been removed. This means that different nodes with identical names are not distinguished by their indexes. Additionally, passive identifiers such as 'p:' are excluded, and verbs and prepositions have been merged. While this format loses some information from the FACTUAL-MR dataset, it remains compatible with the Visual Genome scene graphs and is effectively usable for downstream scene graph tasks.
> 2. **SPICE Parser Performance**: The performance of the SPICE Parser in the table above differs significantly from the original results reported in our paper. This is because the parser is based on dependency parsing. To ensure a fair comparison, we have aligned its parsing outputs with the ground truth generated by research on dependency parsing-based scene graph parsing (See [Scene Graph Parsing as Dependency Parsing](https://arxiv.org/abs/1803.09189)). As a result, our comparison in our paper was more aligned with their findings. However, in the table above, we recompare the SPICE Parser outputs with the ground truth from our dataset and show a new result. Please see ``tests/test_spice_parser.py`` to replicate the SPICE results.



### Enhanced Scene Graph Parsing with Node Indexes and Verb Identifiers

Enhanced scene graph parsing includes detailed annotations such as verb identifiers and node indexes, which offer a more nuanced understanding of the relationships within the input text. For example:

- The sentence "A monkey is sitting next to another monkey" is parsed as:
  `( monkey, v:sit next to, monkey:1 )`
  Here, "v:" indicates a verb, and ":1" differentiates the second "monkey" as a unique entity.

- For "A car is parked on the ground", the scene graph is:
  `( car, pv:park on, ground )`
  The "pv:" prefix highlights "park" as a passive verb, underscoring the significance of node order in the graph.

This advanced parsing technique offers substantial enhancements over the original Visual Genome (VG) scene graphs by:

- **Uniquely Identifying Similar Entities**: Assigning indexes to nodes with the same name allows for clear differentiation between identical entities.
- **Detailing Predicates**: Annotating each predicate with the specific verb and its tense provides richer contextual information.

Such improvements are invaluable for complex downstream tasks, as they facilitate a deeper semantic understanding of the scenes.

#### Model Performance with Advanced Parsing:

| Model | Set Match | SPICE | Soft-SPICE |Model Weight |
|-------|-----------|-------|--------------|--------------|
| (pre) Flan-T5-large | 81.03 | 93.00 | 98.66 |[flan-t5-large-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg-id) |
| (pre) Flan-T5-base | 81.37 | 93.29 | 98.76 |[flan-t5-base-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg-id) |
| (pre) Flan-T5-small | 79.64 | 92.40 | 98.53 |[flan-t5-small-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg-id) |

The acronym (pre) stands for models that were pre-trained on VG and then fine-tuned on FACTUAL, indicating a two-phase learning process that enhances model performance.

### Usage Example

This section demonstrates how to use our models for scene graph parsing. We provide two examples: a basic usage with our pre-trained model and a more advanced usage with the `SceneGraphParser` class.

#### Basic Usage

First, install the necessary package:

```sh
pip install FactualSceneGraph
```

Then, you can use our pre-trained model as follows:

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
# Output: `( pigs , is , 2 ) , ( bags , on back of , pigs ), ( bags , is , 2 ) , ( pigs , fly on , sky )`
```
Note: In this example, the predicate 'is' is referred to as 'has_attribute'.

Advanced Usage with ``SceneGraphParser``
For a more advanced parsing, utilize the ``SceneGraphParser`` class:

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"], beam_size=1, return_text=True)
graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs"], beam_size=5, return_text=False,max_output_len=128)

print(text_graph[0])
# Output: ( pigs , is , 2 ) , ( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( pigs , fly on , sky ) , ( bags , is , 2 )

from sng_parser.utils import tprint
tprint(graph_obj[0])
```
This will produce a formatted scene graph output:
```
Entities:
+----------+------------+------------------+
| Entity   | Quantity   | Attributes       |
|----------+------------+------------------|
| pigs     | 2          | beautiful,strong |
| bags     | 2          |                  |
| sky      |            |                  |
+----------+------------+------------------+
Relations:
+-----------+------------+----------+
| Subject   | Relation   | Object   |
|-----------+------------+----------|
| pigs      | fly on     | sky      |
| bags      | on back of | pigs     |
+-----------+------------+----------+
```

## A Comprehensive Toolkit for Scene Graph Parsing Evaluation

This package provides implementations for evaluating scene graphs using SPICE, SoftSPICE, and Set Match metrics. These evaluations can be performed on various inputs, including captions and scene graphs in both list and nested list formats.

### Supported Input Formats

- `(list of candidate_captions, list of list reference_captions)`
- `(list of candidate_captions, list of list reference_graphs)`
- `(list of candidate_graphs, list of list reference_graphs)`

### Installation

```sh
pip install FactualSceneGraph
```

### Usage

Below are examples demonstrating how to use the evaluation methods provided in this package.

#### Example 1: Testing Scene Graph Parsing

This example demonstrates evaluating a single scene graph using the SPICE method.

```python
import pandas as pd
import torch
from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

def test_scene_graph_parsing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)
    evaluator = Evaluator(parser=parser, device=device)

    scores = evaluator.evaluate(
        ["2 beautiful pigs are flying on the sky with 2 bags on their backs"],
        [['( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( bags , is , 2 ) , ( pigs , is , 2 ) , ( pigs , fly on , sky )']],
        method='spice',
        beam_size=1,
        max_output_len=128
    )
    print(scores)

# Uncomment to run the example
# test_scene_graph_parsing()
```

#### Example 2: Testing Scene Graph Parsing on the Test Set of FACTUAL Random Split

This example demonstrates evaluating a dataset of scene graphs using SPICE, Set Match, and SoftSPICE methods.

```python
import pandas as pd
import torch
from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

def test_scene_graph_parsing_on_random():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device, lemmatize=False)
    evaluator = Evaluator(parser=parser, text_encoder_checkpoint='all-MiniLM-L6-v2', device=device, lemmatize=True)

    random_data_pd = pd.read_csv('data/factual_sg/random/test.csv')
    random_data_captions = random_data_pd['caption'].tolist()
    random_data_graphs = [[scene] for scene in random_data_pd['scene_graph'].tolist()]

    # Evaluating using SPICE
    spice_scores, cand_graphs, ref_graphs = evaluator.evaluate(
        random_data_captions, 
        random_data_graphs, 
        method='spice', 
        beam_size=1, 
        batch_size=128, 
        max_input_len=256, 
        max_output_len=256, 
        return_graphs=True
    )
    print('SPICE scores for random test set:', sum(spice_scores)/len(spice_scores))

    # Evaluating using Set Match
    set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='set_match', beam_size=1)
    print('Set Match scores for random test set:', sum(set_match_scores)/len(set_match_scores))

    # Evaluating using Soft-SPICE
    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', beam_size=1)
    print('Soft-SPICE scores for random test set:', sum(soft_spice_scores)/len(soft_spice_scores))

# Uncomment to run the example
# test_scene_graph_parsing_on_random()
```

### Human Correlation Performance on the Flickr8k Dataset

In our study, we evaluated the correlation between various metrics and human judgment in image caption generation on the Flickr8k dataset using Kendall’s tau. This comparison helps in understanding how well each metric aligns with human perception.

#### Results

Below is a table showing the Tau-c correlation values for different models:

| Model            | Tau-c |
|------------------|-------|
| SPICE(Official-Original) | 44.77 |
| SPICE(Official-Factual) | 45.13 |
| SPICE(Ours-Factual)        | 45.25 |
| Soft-SPICE       | 54.20  |
| RefCLIPScore     | 53.00 |
| BERTScore        | 36.71 |

#### SPICE Implementations

This section provides an overview of the different SPICE implementations used in our project.

- **1. SPICE(Official-Original)**:

  - Uses the original parser from the [Modified SPICE Score](https://github.com/yychai74/modified-SPICE-score) repository.
  - Follows the official SPICE implementation as provided in the repository.
  - Employs the original parser to process the input and generate the SPICE score.

- **2. SPICE(Official-Factual)**:

  - Follows the official SPICE implementation from the [Modified SPICE Score](https://github.com/yychai74/modified-SPICE-score) repository.
  - Uses the `lizhuang144/flan-t5-base-VG-factual-sg` checkpoint as the parser instead of the original parser.

- **3. SPICE(Ours-Factual)**:

  - Our own SPICE implementation, denoted by the "Ours" prefix.
  - Utilizes the `lizhuang144/flan-t5-base-VG-factual-sg` checkpoint as the parser.
  - Updated with an improved synonym-matching dictionary, resulting in closer alignment with the official SPICE synonym-matching version.
  - The update, now the default setting in SPICE(Ours-Factual), shows a stronger correlation with human judgment than the official SPICE version.
  - Recommended for better performance in relevant applications.

- **4. Soft-SPICE**:

  - A variant of the SPICE score that incorporates a soft matching mechanism.
  - Uses the `lizhuang144/flan-t5-base-VG-factual-sg` checkpoint as the parser.
  - The default text encoder is `all-MiniLM-L6-v2` from the `SentenceTransformer` library.
  - Aims to provide a more flexible and nuanced evaluation of the generated text by considering soft matches between the reference and the generated content.

These SPICE implementations offer various options for evaluating the quality of the generated text, each with its own characteristics and parser choices. The "Official" implementations follow the original SPICE repository, while our implementation (SPICE(Ours-Factual)) introduces improvements and updates for enhanced performance.


#### Replicating the Results

To replicate the human correlation results for Our SPICE and Soft-SPICE, please refer to the script located at `tests/test_metric_human_correlation.py`. This script provides a straightforward way to validate our findings.



## Citation

If you find the paper or the accompanying code beneficial, please acknowledge our work in your own research. Please use the following BibTeX entry for citation:

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
}
```

## Acknowledgments

This project has been developed with the use of code from the [SceneGraphParser](https://github.com/vacancy/SceneGraphParser) repository by [Jiayuan Mao](https://github.com/vacancy). We gratefully acknowledge their pioneering work and contributions to the open-source community.



