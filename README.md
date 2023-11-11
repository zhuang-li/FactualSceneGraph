# FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

Welcome to the official repository for the ACL 2023 paper:  
[**FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing**](https://arxiv.org/pdf/2305.17497.pdf). Here, you'll find both the code and dataset associated with our research.

<div align="center">
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/monash_logo.png" alt="Monash University Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/adobe_logo.png" alt="Adobe Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/wuhan_logo.png" alt="Wuhan University Logo" height="80" />
</div>

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

The following table shows the performance comparison of various scene graph parsing models. Notably, the original SPICE parser performs worse than our more recent models.

#### Performance Metrics Explained:
- **SPICE F-score**: A metric that measures the similarity between candidate and reference scene graph representations derived from captions. It assesses the quality of scene graph parsing by evaluating how well the parser's output matches the ground truth graph in terms of propositional content.
- **Exact Set Match**: Adapted from the methodology described by [Yu et al., 2019](https://aclanthology.org/P19-1443/), this metric evaluates the parser's accuracy by verifying whether the strings of parsed facts match the ground truth facts, without considering the ordering of those facts. This adaptation is a stringent accuracy measure, necessitating an exact correspondence between the candidate and ground truth facts.

> **Note**: It is important to note that in the original work of Yu et al., 2019, the metric was applied to SQL clauses, whereas in our context, it has been tailored to assess scene graph facts.


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

| Model | Set Match | SPICE | Model Weight |
|-------|-----------|-------|--------------|
| (pre) Flan-T5-large | 80.57 | 92.97 | [flan-t5-large-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-large-VG-factual-sg-id) |
| (pre) Flan-T5-base | 80.90 | 92.99 | [flan-t5-base-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg-id) |
| (pre) Flan-T5-small | 78.38 | 91.93 | [flan-t5-small-VG-factual-sg-id](https://huggingface.co/lizhuang144/flan-t5-small-VG-factual-sg-id) |

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
from sng_parser.scene_graph_parser import SceneGraphParser

parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"], beam_size=1, return_text=True)
graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs"], beam_size=1, return_text=False,max_output_len=128)

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
| pigs     | 2          | strong,beautiful |
| sky      |            |                  |
| bags     |            |                  |
+----------+------------+------------------+
Relations:
+-----------+------------+----------+
| Subject   | Relation   | Object   |
|-----------+------------+----------|
| pigs      | fly on     | sky      |
| bags      | on back of | pigs     |
+-----------+------------+----------+
```

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


