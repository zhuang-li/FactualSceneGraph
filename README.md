# [ACL 2023 Findings] FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing

Welcome to the official repository for the ACL 2023 Findings paper:  
[**FACTUAL: A Benchmark for Faithful and Consistent Textual Scene Graph Parsing**](https://arxiv.org/pdf/2305.17497.pdf). Here, you'll find both the code and dataset associated with our research.

<div align="center">
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/monash_logo.png" alt="Monash University Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/adobe_logo.png" alt="Adobe Logo" height="80" />
  <img src="https://github.com/zhuang-li/FACTUAL/blob/main/logo/wuhan_logo.png" alt="Wuhan University Logo" height="80" />
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2305.17497"><img src="https://img.shields.io/badge/arXiv-2305.17497-b31b1b.svg"></a>
    <a href="https://pypi.org/project/FactualSceneGraph/"><img src="https://img.shields.io/pypi/v/FactualSceneGraph?color=g"></a>
    <a href="https://pepy.tech/projects/FactualSceneGraph"><img src="https://static.pepy.tech/badge/FactualSceneGraph"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

## 🆕 New Feature: Multi-Sentence Scene Graph Parsing

> **✨ Now supports parsing complex, multi-sentence descriptions with two powerful approaches!**  
> - Use `parser_type='sentence_merge'` for efficient multi-sentence parsing with automatic merging
> - Use `parser_type='DiscoSG-Refiner'` for advanced multi-sentence parsing with iterative refinement

**Key Benefits:**
- 🔄 **Automatic sentence segmentation** using NLTK
- ⚡ **Efficient batch processing** for optimal performance  
- 🧹 **Smart deduplication** of entities and relations
- 🔗 **Maintains relationships** across sentence boundaries
- 📝 **Perfect for complex descriptions** like image captions, stories, or detailed scene descriptions
- 🚀 **Advanced refinement** available with DiscoSG-Refiner for state-of-the-art quality

**Quick Example:**
```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# Basic multi-sentence parser
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', parser_type='sentence_merge')

# Advanced multi-sentence parser with refinement
parser_advanced = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', 
                                  parser_type='DiscoSG-Refiner',
                                  refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only')

# Parse complex description
result = parser.parse(["""The cat sits on a mat. The mat is red and soft. 
                         A dog runs nearby."""])
```

[👀 See full documentation below](#multi-sentence-parsing-approaches)

### 🚀 New Feature: Advanced Multi-Sentence Parsing with Multi-Round Refinement with DiscoSG-Refiner

> **✨ Introducing state-of-the-art scene graph refinement for multi-sentence descriptions!**  
> Use `parser_type='DiscoSG-Refiner'` to leverage the powerful DiscoSG-Refiner model for iterative scene graph improvement through multi-round refinement.

**Key Features:**
- 🎯 **Multi-round iterative refinement** for enhanced accuracy on multi-sentence text
- 🔍 **Smart processing** - single-sentence descriptions use basic parsing, multi-sentence descriptions get full refinement
- 🛡️ **Automatic safety checks** - warns and skips refinement when input length exceeds limits

**Quick Example:**
```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# DiscoSG-Refiner parser with multi-round refinement for multi-sentence text
parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg', 
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='mps'  # or 'cuda', 'cpu'
)

# Single sentence: uses basic parsing (no refinement needed)
single_result = parser.parse(["One dog is running."])

# Multi-sentence: gets full refinement treatment
multi_result = parser.parse([
    "One dog is running. Another dog is flying and blue."
], 
max_input_len=1024,  # Important: ensure sufficient length for refinement
max_output_len=512,
beam_size=5, 
refinement_rounds=2,  # Number of refinement iterations
return_text=True
)
```

**Advanced Configuration:**
```python
# Custom refinement with different task types
parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg',
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='cuda'
)

# Parse with specific task and multiple rounds
result = parser.parse([
    """The image captures a bustling urban scene, likely in a European city. 
    The setting appears to be a pedestrian-friendly square or plaza. 
    There are numerous people of various ages and attire walking around."""
],
task='delete_before_insert',  # or 'insert_delete', 'insert', 'delete'
refinement_rounds=2,
max_input_len=1024,
max_output_len=512,
beam_size=5,
return_text=True
)
```

**Safety Features:**
- **Automatic Length Checking**: The parser automatically checks if your `max_input_len` is sufficient for the input text. If not, it issues a warning and skips refinement to prevent errors.
- **Graceful Fallback**: When refinement is skipped, you still get high-quality results from the base sentence-merge parsing.

[👀 See full documentation below](#advanced-multi-round-refinement-with-discosG-refiner)

---

## Installation

```sh
pip install FactualSceneGraph
```

**For DiscoSG-Refiner functionality:**
The DiscoSG-Refiner feature requires the `discosg` package, which is automatically installed as a dependency. However, if you encounter installation issues:

```sh
# If you encounter issues with discosg installation
pip install --upgrade FactualSceneGraph

# For development/latest features
pip install git+https://github.com/zhuang-li/FACTUAL.git
```

**Troubleshooting:**
- **Apple Silicon (M1/M2) Users**: The package supports MPS acceleration. Use `device='mps'` for optimal performance.
- **CUDA Users**: Ensure PyTorch with CUDA support is installed before installing FactualSceneGraph.
- **Python Compatibility**: Requires Python 3.8+ for full functionality.

**Dependencies:**
- Core: `torch`, `transformers`, `nltk`, `spacy`
- Refinement: `discosg`, `peft`, `huggingface-hub`
- Evaluation: `sentence-transformers`, `pandas`, `numpy`

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

## Usage Examples

This section demonstrates how to use our models for scene graph parsing. We provide examples for basic usage, advanced usage with the `SceneGraphParser` class, and the **new sentence merge functionality** for multi-sentence text processing.

### Basic Usage

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

### Advanced Usage with SceneGraphParser

For more advanced parsing, utilize the `SceneGraphParser` class:

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# Default parser for single sentences or simple text
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
text_graph = parser.parse(["2 beautiful pigs are flying on the sky with 2 bags on their backs"], 
                         beam_size=1, return_text=True)
graph_obj = parser.parse(["2 beautiful and strong pigs are flying on the sky with 2 bags on their backs"], 
                        beam_size=5, return_text=False, max_output_len=128)

print(text_graph[0])
# Output: ( pigs , is , 2 ) , ( pigs , is , beautiful ) , ( bags , on back of , pigs ) , ( pigs , fly on , sky ) , ( bags , is , 2 )

from factual_scene_graph.utils import tprint
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

### 🌟 Multi-Sentence Scene Graph Parsing

**NEW:** Parse complex, multi-sentence descriptions with two powerful approaches designed specifically for multi-sentence text:

#### **Approach 1: Sentence Merge (`sentence_merge`)**
Efficient and reliable multi-sentence parsing with automatic merging:

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# Sentence merge parser for multi-sentence text
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', 
                         device='cpu', 
                         parser_type='sentence_merge')

# Batch processing multiple multi-sentence descriptions
descriptions = [
    """The image captures a serene scene in a park. A gravel path, dappled with sunlight 
    filtering through the tall trees on either side, winds its way towards a white bridge.""",
    
    """A bustling urban scene unfolds in the city center. People walk along the sidewalks 
    carrying shopping bags. Cars and buses navigate through the busy streets.""",
    
    """The kitchen scene shows a chef preparing dinner. Fresh vegetables are arranged on 
    the counter. Steam rises from the cooking pots on the stove."""
]

# Batch processing - all descriptions processed efficiently together
results = parser.parse(descriptions, 
                      beam_size=5, 
                      batch_size=8,  # Process multiple descriptions in batches
                      return_text=True)

for i, result in enumerate(results):
    print(f"Description {i+1} scene graph:")
    print(result)
    print("-" * 50)
```

#### **Approach 2: DiscoSG-Refiner (`DiscoSG-Refiner`)**
Advanced multi-sentence parsing with state-of-the-art iterative refinement:

```python
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# DiscoSG-Refiner parser for advanced multi-sentence refinement
parser = SceneGraphParser(
    'lizhuang144/flan-t5-base-VG-factual-sg', 
    parser_type='DiscoSG-Refiner',
    refiner_checkpoint_path='sqlinn/DiscoSG-Refiner-Large-t5-only',
    device='mps'  # Supports 'cpu', 'cuda', 'mps'
)

# Mixed batch: single and multi-sentence descriptions
mixed_descriptions = [
    "A red car is parked.",  # Single sentence - basic parsing
    
    """The dog runs in the park. The park has green grass and tall trees. 
    Children are playing on the swings.""",  # Multi-sentence - gets refinement
    
    "The cat sleeps on the sofa.",  # Single sentence - basic parsing
    
    """The restaurant is busy tonight. Waiters serve food to customers at tables. 
    The kitchen staff prepares fresh meals. Soft music plays in the background."""  # Multi-sentence - gets refinement
]

# Batch processing with automatic smart detection
results = parser.parse(mixed_descriptions,
                      max_input_len=1024,
                      max_output_len=512,
                      beam_size=5,
                      batch_size=4,  # Process 4 descriptions at once
                      refinement_rounds=2,  # Only applied to multi-sentence descriptions
                      task='delete_before_insert',
                      return_text=True)

for i, (desc, result) in enumerate(zip(mixed_descriptions, results)):
    sentence_count = len(desc.split('.')) - 1  # Rough sentence count
    processing_type = "Multi-round refinement" if sentence_count > 1 else "Basic parsing"
    print(f"Description {i+1} ({processing_type}):")
    print(f"Input: {desc[:50]}...")
    print(f"Scene graph: {result}")
    print("-" * 70)
```

**How Multi-Sentence Parsing Works:**

**Sentence Merge Process:**
1. **Sentence Tokenization**: Uses NLTK to split text into individual sentences
2. **Batch Processing**: All sentences processed efficiently in batches
3. **Graph Merging**: Results merged and deduplicated automatically
4. **Fast & Reliable**: Optimized for speed and consistency

**DiscoSG-Refiner Process:**
1. **Smart Detection**: Automatically detects single vs. multi-sentence descriptions
2. **Efficient Processing**: Single sentences use basic parsing (no refinement overhead)
3. **Multi-Round Refinement**: Multi-sentence descriptions get iterative improvement:
   - Initial parsing via sentence merge
   - Multiple rounds of delete-before-insert refinement
   - Each round enhances scene graph quality
4. **Safety Checks**: Validates input length compatibility
5. **Graceful Fallback**: Skips refinement with warning if input too long

**When to Use Which Approach:**

| Parser Type | Best For | Advantages | Use Cases |
|-------------|----------|------------|-----------|
| `sentence_merge` | Multi-sentence text | Fast, efficient, reliable | Complex descriptions, stories, detailed captions |
| `DiscoSG-Refiner` | Multi-sentence text requiring highest quality | State-of-the-art accuracy, iterative improvement | Research, high-quality applications, complex scenes |

**Benefits:**
- **Handles Complex Text**: Both approaches designed specifically for multi-sentence descriptions
- **Maintains Relationships**: Preserves connections across sentence boundaries
- **Automatic Deduplication**: Removes repeated entities and relations
- **Efficient Batch Processing**: Optimized for GPU utilization
- **Smart Processing** (DiscoSG-Refiner): Only applies expensive refinement where needed

**Configuration Options:**

```python
# Available parser types
parser_types = [
    'default',           # Single-pass parsing (for single sentences)
    'sentence_merge',    # Multi-sentence parsing with merging
    'DiscoSG-Refiner'   # Advanced multi-sentence parsing with refinement
]

# Available refinement tasks
tasks = [
    'delete_before_insert',  # Delete unwanted triples, then insert new ones (default)
    'insert_delete',         # Insert new triples, then delete unwanted ones
    'insert',               # Only insert new triples
    'delete'                # Only delete unwanted triples
]

# Example with all options
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
    max_input_len=1024,      # Token limit for input
    max_output_len=512,      # Token limit for output
    beam_size=5,             # Beam search width
    refinement_rounds=2,     # Number of refinement iterations
    task='delete_before_insert',  # Refinement strategy
    batch_size=32,           # Batch size for processing
    return_text=True         # Return text or parsed objects
)
```

**Performance Tips:**
- **Input Length**: Set `max_input_len` to at least 512-1024 for complex descriptions
- **Refinement Rounds**: 2-3 rounds typically provide the best quality/speed trade-off
- **Device Selection**: Use 'mps' for Apple Silicon, 'cuda' for NVIDIA GPUs, 'cpu' for compatibility
- **Batch Size**: Adjust based on your GPU memory. You cannot set batch size too large as if the token length is too long!!!

**Error Handling:**
The parser includes automatic safety checks:
```python
# This will show a warning and skip refinement if input is too long
parser.parse(["Very long description..."], beam_size=5, max_input_len=64)  # Too small
# Warning: Skipping DiscoSG refinement because max_input_len=64 is smaller than 
# the shortest caption (129) plus buffer (20). Increase max_input_len or use 
# shorter captions to enable refinement.
```

**Advanced Features:**
- **Higher Quality**: Multi-round refinement significantly improves scene graph accuracy
- **Intelligent Processing**: Only applies expensive refinement where it's needed (multi-sentence text)
- **Robust**: Built-in safety checks prevent runtime errors
- **Flexible**: Supports various refinement strategies and configurations
- **Compatible**: Works with existing FACTUAL models and evaluation metrics
- **Batch Processing**: Both approaches support efficient batch processing for multiple descriptions

## A Comprehensive Toolkit for Scene Graph Parsing Evaluation

This package provides implementations for evaluating scene graphs using SPICE, SoftSPICE, and Set Match metrics. These evaluations can be performed on various inputs, including captions and scene graphs in both list and nested list formats.

### Supported Input Formats

- `(list of candidate_captions, list of list reference_captions)`
- `(list of candidate_captions, list of list reference_graphs)`
- `(list of candidate_graphs, list of list reference_graphs)`

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

In our study, we evaluated the correlation between various metrics and human judgment in image caption generation on the Flickr8k dataset using Kendall's tau. This comparison helps in understanding how well each metric aligns with human perception.

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



