# [ACL 2023 Findings] FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

<p align="center">
  <strong>Faithful and Consistent Textual Scene Graph Parsing</strong><br/>
  Official repository for the ACL 2023 Findings paper, with code, datasets, pretrained models, and evaluation tools.
</p>

<p align="center">
  <a href="https://aclanthology.org/2023.findings-acl.398">
    <img src="https://img.shields.io/badge/Paper-ACL%202023%20Findings-blue.svg" alt="FACTUAL">
  </a>
  <a href="https://arxiv.org/abs/2305.17497">
    <img src="https://img.shields.io/badge/arXiv-2305.17497-b31b1b.svg" alt="arXiv 2305.17497">
  </a>
  <a href="https://arxiv.org/abs/2506.15583">
    <img src="https://img.shields.io/badge/Paper-DiscoSG--Refiner-purple.svg" alt="DiscoSG-Refiner">
  </a>
  <a href="https://pypi.org/project/FactualSceneGraph/">
    <img src="https://img.shields.io/pypi/v/FactualSceneGraph?color=green" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/projects/FactualSceneGraph">
    <img src="https://static.pepy.tech/badge/FactualSceneGraph" alt="Downloads">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  </a>
</p>

<p align="center">
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/monash_logo.png" alt="Monash University Logo" height="72" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/adobe_logo.png" alt="Adobe Logo" height="72" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/wuhan_logo.png" alt="Wuhan University Logo" height="72" />
</p>

---

## Overview

FACTUAL is a benchmark and toolkit for **faithful**, **consistent**, and **practical** textual scene graph parsing.
It provides:

- pretrained parsers for converting text into scene graphs,
- benchmark datasets for training and evaluation,
- evaluation tools including **SPICE**, **Soft-SPICE**, and **Set Match**,
- support for both **single-sentence** and **multi-sentence** scene graph parsing.

This repository now also includes **discourse-level multi-sentence parsing** with two options:

| Mode | Best for | Description |
|---|---|---|
| `default` | Single sentences | Standard one-pass parsing |
| `sentence_merge` | Multi-sentence descriptions | Split → parse → merge → deduplicate |
| `DiscoSG-Refiner` | Highest-quality multi-sentence parsing | Sentence merging followed by iterative refinement |

---

## Highlights

- **40,369** FACTUAL scene graph instances with lemmatized predicates
- **2.9M** cleaned Visual Genome scene graph instances for pretraining
- **Pretrained Flan-T5 models** in multiple sizes
- **Advanced discourse-level parsing** for long, multi-sentence descriptions
- **Unified evaluation toolkit** for scene graph parsing and caption evaluation
- **Supports CPU, CUDA, and Apple Silicon (MPS)**

---

## Installation

```bash
pip install FactualSceneGraph
```

### For DiscoSG-Refiner

The `discosg` dependency is installed automatically. If installation fails or you want the latest version:

```bash
pip install --upgrade FactualSceneGraph
```

For development or the newest GitHub version:

```bash
pip install git+https://github.com/zhuang-li/FACTUAL.git
```

### Notes

- **Python**: 3.8+
- **Apple Silicon**: use `device='mps'`
- **CUDA**: install a CUDA-enabled PyTorch build before installing this package

### Main Dependencies

- Core: `torch`, `transformers`, `nltk`, `spacy`
- Refinement: `discosg`, `peft`, `huggingface-hub`
- Evaluation: `sentence-transformers`, `pandas`, `numpy`

---

## Quick Start

### 1) Minimal example

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
result = parser.parse(
    ["2 beautiful pigs are flying on the sky with 2 bags on their backs"],
    beam_size=1,
    return_text=True
)

print(result[0])
# ( pigs , is , 2 ) , ( pigs , is , beautiful ) , ( bags , on back of , pigs ) ,
# ( pigs , fly on , sky ) , ( bags , is , 2 )
```

### 2) Raw Transformers usage

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

> In this output format, the predicate `is` corresponds to an attribute-style relation.

---

## Multi-Sentence Parsing

Modern VLMs often generate long, rich descriptions instead of single-sentence captions. FACTUAL now supports multi-sentence scene graph parsing in two ways.

### Option A — `sentence_merge`

Use this for efficient parsing of long descriptions, detailed captions, or short stories.

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg',
    parser_type='sentence_merge',
    device='cpu'
)

descriptions = [
    """The image captures a serene scene in a park. A gravel path, dappled with sunlight
    filtering through the tall trees on either side, winds its way towards a white bridge.""",

    """A bustling urban scene unfolds in the city center. People walk along the sidewalks
    carrying shopping bags. Cars and buses navigate through the busy streets.""",
]

results = parser.parse(
    descriptions,
    beam_size=5,
    batch_size=8,
    return_text=True
)
```

**How it works**

1. Split text into sentences with NLTK
2. Parse each sentence efficiently in batches
3. Merge graphs automatically
4. Deduplicate repeated entities and relations

### Option B — `DiscoSG-Refiner`

Use this when you want the highest-quality multi-sentence scene graphs.

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg',
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='cuda'
)

result = parser.parse(
    ["""The image captures a bustling urban scene, likely in a European city.
    The setting appears to be a pedestrian-friendly square or plaza.
    There are numerous people of various ages and attire walking around."""],
    task='delete_before_insert',
    refinement_rounds=2,
    max_input_len=1024,
    max_output_len=512,
    batch_size=2,
    beam_size=5,
    return_text=True
)
```

**Why use it**

- iterative multi-round refinement,
- smart single-sentence vs multi-sentence handling,
- graceful fallback when the input is too long,
- strong performance for research and high-precision applications.

### Which one should I choose?

| Parser type | Recommended use | Strength |
|---|---|---|
| `default` | Simple captions or short text | Fastest single-pass parser |
| `sentence_merge` | Long descriptions and multi-sentence text | Fast, stable, efficient |
| `DiscoSG-Refiner` | Research and highest-quality parsing | Best quality through iterative refinement |

### Example: mixed batch with automatic refinement

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg',
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='mps'
)

mixed_descriptions = [
    "A red car is parked.",
    """The dog runs in the park. The park has green grass and tall trees.
    Children are playing on the swings.""",
    "The cat sleeps on the sofa.",
    """The restaurant is busy tonight. Waiters serve food to customers at tables.
    The kitchen staff prepares fresh meals. Soft music plays in the background."""
]

results = parser.parse(
    mixed_descriptions,
    max_input_len=1024,
    max_output_len=512,
    beam_size=5,
    batch_size=4,
    refinement_rounds=2,
    task='delete_before_insert',
    return_text=True
)
```

### Recommended settings

- `max_input_len=512~1024` for long descriptions
- `refinement_rounds=2~3` for a good quality/speed balance
- `device='mps'` for Apple Silicon, `device='cuda'` for NVIDIA GPUs
- tune `batch_size` based on memory budget

### Safety behavior

The parser checks whether `max_input_len` is sufficient before running refinement. If the input is too long, refinement is skipped automatically and the parser falls back to sentence merging.

```python
parser.parse(["Very long description..."], beam_size=5, max_input_len=64)
# Warning: Skipping DiscoSG refinement because max_input_len is too small.
```

---

## Datasets

### FACTUAL Scene Graph Dataset

The FACTUAL Scene Graph dataset contains **40,369** instances with lemmatized predicates/relations.

**Storage**

- Local: `data/factual_sg/factual_sg.csv`
- Hugging Face: `load_dataset('lizhuang144/FACTUAL_Scene_Graph')`

**Splits**

| Split type | Train | Dev | Test |
|---|---|---|---|
| Random | `data/factual_sg/random/train.csv` | `data/factual_sg/random/dev.csv` | `data/factual_sg/random/test.csv` |
| Length | `data/factual_sg/length/train.csv` | `data/factual_sg/length/dev.csv` | `data/factual_sg/length/test.csv` |

**Fields**

- `image_id`: Visual Genome image ID
- `region_id`: Visual Genome region ID
- `caption`: region caption
- `scene_graph`: scene graph for the region caption

**Related resource**

- Visual Genome images/regions: `load_dataset('visual_genome')`

### FACTUAL-MR

- Data: `data/factual_mr/factual_mr.csv`
- Metadata: `data/factual_mr/meta.json`

The metadata maps quantifier abbreviations in `factual_mr.csv` to their full names.

### Visual Genome Scene Graph Dataset (Cleaned)

- Hugging Face: `load_dataset('lizhuang144/VG_scene_graph_clean')`
- Size: **2.9 million** instances
- Preprocessed to remove empty instances

### FACTUAL Scene Graph Dataset with Identifiers

- Hugging Face: `load_dataset('lizhuang144/FACTUAL_Scene_Graph_ID')`
- Includes:
  - verb identifiers,
  - passive voice indicators,
  - node indexes.

---

## Pretrained Models

### A. Models without node indexes / passive identifiers

These models are the most directly compatible with common Visual Genome style scene graphs.

| Model | Set Match | SPICE | Soft-SPICE | Weight |
|---|---:|---:|---:|---|
| SPICE / Stanford Parser | 19.30 | 64.77 | 92.60 | [modified-SPICE-score](https://github.com/yychai74/modified-SPICE-score) |
| (pre) Flan-T5-large | 81.63 | 93.20 | 98.75 | [flan-t5-large-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg) |
| (pre) Flan-T5-base | 81.37 | 93.27 | 98.83 | [flan-t5-base-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg) |
| (pre) Flan-T5-small | 78.18 | 92.26 | 98.67 | [flan-t5-small-VG-factual-sg](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg) |

**Notes**

1. `(pre)` means the model was pretrained on Visual Genome scene graphs and then fine-tuned on FACTUAL.
2. These models remove node indexes and passive markers such as `pv:`.
3. Verbs and prepositions are merged for compatibility with downstream scene graph tasks.
4. The SPICE baseline shown here is re-evaluated against FACTUAL ground truth, which differs from the comparison protocol used in the original paper. See `tests/test_spice_parser.py` for replication.

### B. Models with node indexes and verb identifiers

These models keep richer structural information, which is useful for advanced downstream tasks.

**Examples**

- `A monkey is sitting next to another monkey`  
  → `( monkey, v:sit next to, monkey:1 )`
- `A car is parked on the ground`  
  → `( car, pv:park on, ground )`

This format helps:

- distinguish repeated entities,
- represent passive constructions,
- preserve more fine-grained predicate information.

| Model | Set Match | SPICE | Soft-SPICE | Weight |
|---|---:|---:|---:|---|
| (pre) Flan-T5-large | 81.03 | 93.00 | 98.66 | [flan-t5-large-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg-id) |
| (pre) Flan-T5-base | 81.37 | 93.29 | 98.76 | [flan-t5-base-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg-id) |
| (pre) Flan-T5-small | 79.64 | 92.40 | 98.53 | [flan-t5-small-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg-id) |

---

## Advanced Usage

### Parse to graph objects instead of text

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from factual_scene_graph.utils import tprint

parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
graph_obj = parser.parse(
    ["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs"],
    beam_size=5,
    return_text=False,
    max_output_len=128
)

tprint(graph_obj[0])
```

Example output:

```text
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

### Full configuration example

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser(
    checkpoint_path='lizhuang144/flan-t5-base-VG-factual-sg',
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='cuda',
    lemmatize=False,
    lowercase=False
)

result = parser.parse(
    descriptions=["Your multi-sentence text 1 here...", "Your multi-sentence text 2 here..."],
    max_input_len=1024,
    max_output_len=512,
    beam_size=5,
    refinement_rounds=2,
    task='delete_before_insert',
    batch_size=32,
    return_text=True
)
```

### Available parser types

```python
parser_types = [
    'default',
    'sentence_merge',
    'DiscoSG-Refiner'
]
```

### Available refinement tasks

```python
tasks = [
    'delete_before_insert',
    'insert_delete',
    'insert',
    'delete'
]
```

---

## Evaluation Toolkit

This package includes a complete evaluation toolkit for scene graph parsing and caption quality analysis.

### Supported metrics

- **SPICE**
- **Soft-SPICE**
- **Set Match**

### Supported input formats

- `(list of candidate_captions, list of list reference_captions)`
- `(list of candidate_captions, list of list reference_graphs)`
- `(list of candidate_graphs, list of list reference_graphs)`

### Metric definitions

- **SPICE F-score**: measures semantic overlap between candidate and reference scene graphs.
- **Exact Set Match**: checks whether candidate facts exactly match reference facts, ignoring order.
- **Soft-SPICE**: uses soft matching for more flexible comparison between graphs.

> The Exact Set Match protocol is adapted from Yu et al. (2019), where the original metric was applied to SQL clauses. Here it is adapted for scene graph facts.

### Example 1 — evaluate a single example

```python
import torch
from factual_scene_graph.evaluation.evaluator import Evaluator
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

def test_scene_graph_parsing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)
    evaluator = Evaluator(parser=parser, device=device)

    scores = evaluator.evaluate(
        ["2 beautiful pigs are flying on the sky with 2 bags on their backs"],
        [["( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( bags , is , 2 ) , ( pigs , is , 2 ) , ( pigs , fly on , sky )"]],
        method='spice',
        beam_size=1,
        max_output_len=128
    )
    print(scores)
```

### Example 2 — evaluate FACTUAL random split test set

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
    print('SPICE scores for random test set:', sum(spice_scores) / len(spice_scores))

    set_match_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='set_match', beam_size=1)
    print('Set Match scores for random test set:', sum(set_match_scores) / len(set_match_scores))

    soft_spice_scores = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice', beam_size=1)
    print('Soft-SPICE scores for random test set:', sum(soft_spice_scores) / len(soft_spice_scores))
```

---

## Human Correlation on Flickr8k

We evaluated how well different metrics correlate with human judgment using **Kendall's tau-c**.

| Model | Tau-c |
|---|---:|
| SPICE (Official-Original) | 44.77 |
| SPICE (Official-Factual) | 45.13 |
| SPICE (Ours-Factual) | 45.25 |
| Soft-SPICE | 54.20 |
| RefCLIPScore | 53.00 |
| BERTScore | 36.71 |

### SPICE variants in this repository

1. **SPICE (Official-Original)**  
   Uses the original parser from the [Modified SPICE Score](https://github.com/yychai74/modified-SPICE-score) repository.

2. **SPICE (Official-Factual)**  
   Uses the official SPICE implementation, but replaces the parser with `lizhuang144/flan-t5-base-VG-factual-sg`.

3. **SPICE (Ours-Factual)**  
   Our implementation with an improved synonym-matching dictionary. This is the default configuration and is recommended for better alignment with human judgment.

4. **Soft-SPICE**  
   A soft-matching extension using sentence embeddings. The default encoder is `all-MiniLM-L6-v2`.

For replication, see: `tests/test_metric_human_correlation.py`

---

## Citation

If this repository helps your research, please cite:

```bibtex
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

@inproceedings{lin-etal-2025-discosg,
    title = "{D}isco{SG}: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement",
    author = "Lin, Shaoqing  and
      Teng, Chong  and
      Li, Fei  and
      Ji, Donghong  and
      Qu, Lizhen  and
      Li, Zhuang",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.398/",
    doi = "10.18653/v1/2025.emnlp-main.398",
    pages = "7848--7873",
    ISBN = "979-8-89176-332-6",
}
```

---

## Acknowledgments

This project builds on ideas and code from [SceneGraphParser](https://github.com/vacancy/SceneGraphParser) by [Jiayuan Mao](https://github.com/vacancy). We gratefully acknowledge their pioneering contribution to open-source scene graph parsing.
