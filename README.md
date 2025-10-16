# CEMTM: Contextual Embedding-based Topic Model

<p align="center">
  <br>
  <a href="https://arxiv.org/abs/2509.11465"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
</p>


A novel multimodal topic modeling framework that leverages Vision-Language Models (VLMs) to discover coherent topics from documents containing both text and images.

<img width="1220" height="964" alt="image" src="https://github.com/user-attachments/assets/c2626f97-ecd4-4290-8ed8-8bff50fe973a" />

## Abstract

We introduce CEMTM, a context-enhanced multimodal topic model designed to infer coherent and interpretable topic structures from both short and long documents containing text and images. CEMTM builds on fine-tuned large vision language models (LVLMs) to obtain contextualized embeddings, and employs a distributional attention mechanism to weight token-level contributions to topic inference. A reconstruction objective aligns topic-based representations with the document embedding, encouraging semantic consistency across modalities. Unlike existing approaches, CEMTM can process multiple images per document without repeated encoding and maintains interpretability through explicit word-topic and document-topic distributions. Extensive experiments on six multimodal benchmarks show that CEMTM consistently outperforms unimodal and multimodal baselines, achieving a remarkable average LLM score of 2.61. Further analysis shows its effectiveness in downstream few-shot retrieval and its ability to capture visually grounded semantics in complex domains such as scientific articles.

## Overview

CEMTM (Contextual Embedding-based Topic Model) addresses the limitations of traditional topic models by incorporating multimodal information through contextualized embeddings. This approach:

- **Leverages VLMs**: Uses pretrained Vision-Language Models to extract rich contextual embeddings from text-image pairs
- **Token-level Analysis**: Learns importance weights for individual tokens to identify the most relevant content for topic discovery
- **Multimodal Integration**: Seamlessly combines textual and visual information in a unified embedding space
- **Contextual Understanding**: Captures semantic relationships that traditional bag-of-words approaches miss
- **Flexible Architecture**: Supports various VLM backbones and can be adapted to different multimodal datasets

## Key Contributions

1. **Novel Architecture**: Introduction of an importance network that learns to weight tokens based on their relevance to topic discovery
2. **Multimodal Topic Discovery**: First approach to use contextualized embeddings from VLMs for joint text-image topic modeling
3. **Comprehensive Evaluation**: Extensive evaluation on multiple datasets with both automatic metrics and human evaluation
4. **Superior Performance**: Demonstrates improved topic coherence and diversity compared to traditional and neural topic models

## Architecture

CEMTM consists of four key components working together to extract topics from multimodal documents:

### 1. **VLM2Vec Module**
- **Purpose**: Extracts contextualized embeddings from text-image pairs
- **Implementation**: Wrapper around Vision-Language Models
- **Output**: 
  - Contextual token embeddings: `H ∈ ℝ^(N×D)` where N is sequence length, D is embedding dimension
  - Document-level embedding: `e_d ∈ ℝ^D`

### 2. **Topic Encoder**
- **Purpose**: Maps contextual embeddings to topic space
- **Function**: `t_i = softmax(W_t h_i)` where `t_i ∈ ℝ^K` (K topics)
- **Output**: Token-level topic distributions

### 3. **Importance Network**
- **Purpose**: Learns which tokens are most relevant for topic discovery
- **Architecture**: Multi-layer Transformer encoder
- **Function**: Outputs `μ` and `σ²` parameters for importance distribution
- **Sampling**: `α_i ~ N(μ_i, σ_i²)` followed by `β_i = softmax(α)` for importance weights

### 4. **Reconstruction Module**
- **Purpose**: Ensures learned representations preserve document semantics
- **Function**: `e_d' = f_dec(∑_i β_i t_i)` where `f_dec` is a linear decoder
- **Loss**: Minimizes reconstruction error between `e_d` and `e_d'`

### Mathematical Formulation

The model optimizes the following objective:
```
L = L_rec + λ_ent L_ent + λ_kl L_kl
```

Where:
- **L_rec**: Reconstruction loss `||e_d - e_d'||²`
- **L_ent**: Entropy regularization `∑_i β_i log β_i`
- **L_kl**: KL divergence between learned and prior importance distributions

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

### Additional Setup
1. **NLTK Data**: Required for text preprocessing
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

2. **Word Embeddings** (for evaluation metrics):
   Download GloVe embeddings for the WE (Word Embedding) metric:
   ```bash
   mkdir -p data/embeddings
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip -d data/embeddings/
   ```

3. **OpenAI API Key** (optional, for LLM evaluation):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Supported Datasets

CEMTM has been evaluated on multiple multimodal datasets:

### Text-Image Datasets
- **MS-COCO**: Image captioning dataset with rich visual-textual associations
- **WikiWeb2M**: Large-scale Wikipedia articles with associated images
- **SPIQA**: Scientific paper abstracts with figures and diagrams
- **TQA**: Textbook question-answering with educational diagrams
- **VIST**: Visual storytelling dataset with image sequences and narratives

### Meme and Social Media Datasets  
- **FHM (Facebook Hateful Memes)**: Multimodal content analysis
- **T4SA**: Twitter sentiment analysis with images

Each dataset loader handles the specific format and preprocessing requirements automatically.

## Dataset Setup

Choose one of the supported datasets and set up the data directory:

### Option 1: MS-COCO (Recommended for testing)
```bash
mkdir -p data/corpus/annotations data/corpus/train2014
# Download MS-COCO 2014 train images and annotations
# Place captions_train2014.json in data/corpus/annotations/
# Place images in data/corpus/train2014/
```

### Option 2: Other Supported Datasets
- **WikiWeb2M**: Place `wikiweb2m.jsonl` and image folders in `data/corpus/`
- **SPIQA**: Place `metadata.json` and `images/` folder in `data/corpus/`
- **TQA**: Place `tqa.json` and `images/` folder in `data/corpus/`
- **FHM**: Place `hateful_memes.jsonl` and `img/` folder in `data/corpus/`
- **T4SA**: Place `t4sa.csv` and `images/` folder in `data/corpus/`
- **VIST**: Place `stories.json` and `images/` folder in `data/corpus/`

## Configuration

## Key Hyperparameters

Understanding the main configuration parameters:

### Model Architecture
```yaml
model:
  d_model: 4096              # VLM embedding dimension (depends on chosen VLM)
  n_topics: 50               # Number of topics to discover
  transformer_layers: 2      # Layers in importance network
  transformer_heads: 8       # Multi-head attention heads
  dropout: 0.1              # Regularization
```

### Training Configuration
```yaml
training:
  batch_size: 4             # Adjust based on GPU memory
  num_epochs: 20            # Training epochs
  gradient_clip: 1.0        # Gradient clipping for stability
  device: cuda              # Use GPU for faster training
```

### Loss Weights
```yaml
loss:
  lambda_entropy: 0.01      # Entropy regularization weight
  lambda_kl: 0.1           # KL divergence weight
```

### Optimization
```yaml
optimizer:
  name: adamw
  lr: 5e-5                 # Learning rate
  weight_decay: 0.01       # L2 regularization

scheduler:
  name: linear             # Learning rate schedule
  warmup_steps: 500        # Warmup period
```

## Usage

### Training

Train the CEMTM model:

```bash
python scripts/train_cemtm.py --config config/config.yaml
```

Training outputs:
- Model checkpoints saved to `outputs/`
- Training logs in `logs/`

### Evaluation

Evaluate trained model on topic quality metrics:

```bash
python scripts/evaluate_topics.py --config config/config.yaml
```

### Vocabulary Quality Check

After training, you can verify the quality of the generated vocabulary using the vocabulary checker tool:

```bash
python scripts/check_vocabulary.py --vocab_path outputs/vocabulary.json
```

This tool provides:
- **Vocabulary Statistics**: Total size, word length distribution
- **Quality Checks**: Detects duplicates, suspicious short words, numeric tokens
- **Sample Words**: Shows most and least frequent words
- **Recommendations**: Suggests improvements based on vocabulary characteristics

**When to Use:**
- After training to verify vocabulary was built correctly
- If you're getting low NPMI scores (may indicate vocabulary issues)
- When tuning `vocab_size` or `min_word_freq` parameters
- To ensure text preprocessing is working properly

**Configuration Parameters:**
You can adjust vocabulary generation in `config/config.yaml`:
```yaml
data:
  vocab_size: 2000         # Maximum vocabulary size
  min_word_freq: 5         # Minimum frequency threshold (filters rare words)
```

**Tips:**
- **Low NPMI scores?** Increase `min_word_freq` to filter more rare words (e.g., 10-20)
- **Too small vocabulary?** Decrease `min_word_freq` or increase `vocab_size`
- **Too many noisy words?** Improve text preprocessing or increase `min_word_freq`

## Evaluation Metrics

CEMTM provides comprehensive evaluation through multiple metrics:

### Automatic Metrics
- **Topic Diversity (TD)**: Measures the proportion of unique words across all topics
  - Higher values indicate more diverse topics
  - Range: [0, 1], where 1 means all topic words are unique

- **Inverse Rank-Biased Overlap (I-RBO)**: Measures topic distinctiveness
  - Computes 1 - average RBO across all topic pairs
  - Higher values indicate more distinct topics

- **Normalized Pointwise Mutual Information (NPMI)**: Measures topic coherence
  - Uses co-occurrence statistics from the corpus
  - Range: [-1, 1], where higher values indicate more coherent topics

- **Word Embedding Score (WE)**: Semantic coherence based on word embeddings
  - Computes average cosine similarity between topic words
  - Requires pretrained word embeddings (GloVe/Word2Vec)

### LLM-based Evaluation
- **LLM Score**: GPT-based topic quality assessment
  - Uses OpenAI API to rate topic coherence on a 1-3 scale
  - Provides human-like evaluation of topic interpretability

### Clustering Metrics (when ground truth available)
- **Purity**: Measures cluster homogeneity
- **Adjusted Rand Index (ARI)**: Similarity between predicted and true clusters
- **Normalized Mutual Information (NMI)**: Information-theoretic clustering metric

## Directory Structure

```
CEMTM/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── dataset.py           # Dataset loaders
│   └── preprocessing.py     # Text/image preprocessing
├── evaluation/
│   ├── llm_api.py          # OpenAI API interface
│   ├── metrics.py          # Evaluation metrics
│   └── topic_extraction.py # Topic analysis utilities
├── losses/
│   └── losses.py           # Loss functions
├── model/
│   ├── cemtm.py           # Main CEMTM model
│   ├── encoder.py         # Topic encoder
│   ├── importance_net.py  # Importance network
│   └── vlm2vec.py        # Vision-Language Model wrapper
├── scripts/
│   ├── evaluate_topics.py # Evaluation script
│   └── train_cemtm.py    # Training script
├── trainer/
│   ├── trainer.py        # Training logic
│   └── utils.py         # Training utilities
└── requirements.txt     # Dependencies
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size` in config.yaml
   - Use `device: cpu` for CPU-only training

2. **Import Errors**:
   - Ensure you're running from the CEMTM root directory
   - Install all requirements: `pip install -r requirements.txt`

3. **Dataset Loading Errors**:
   - Check dataset path in config.yaml
   - Ensure dataset files are in correct format and location

4. **Evaluation Errors**:
   - Download word embeddings for WE metric
   - Set OPENAI_API_KEY for LLM evaluation

## Computational Requirements
- **GPU Memory**: 8-12GB VRAM recommended for batch_size=4
- **Training Time**: 2-4 hours on RTX 3080 for 20 epochs on MS-COCO subset
- **Inference**: Real-time topic extraction for new documents

## Advantages over Traditional Methods
- **Contextual Understanding**: Captures semantic relationships beyond word co-occurrence
- **Multimodal Integration**: Leverages both text and visual information
- **Better Coherence**: Produces more interpretable and coherent topics
- **Scalability**: Efficient processing of large multimodal datasets

## Related Work

CEMTM builds upon and extends several lines of research:
- **Topic Modeling**: LDA, Neural Topic Models, BERTopic
- **Vision-Language Models**: CLIP, LLaVA, BLIP
- **Multimodal Learning**: Cross-modal attention, multimodal transformers
- **Contextualized Embeddings**: BERT, RoBERTa, contextual topic models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please:
- Open a GitHub issue for bug reports or feature requests
- Contact the authors: [amirhossein.abaskohi@gmail.com]
- Check the paper for theoretical details and experimental results

## Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC).  
Ce projet a été financé par le Conseil de recherches en sciences naturelles et en génie du Canada (CRSNG).

## Citation
```
@inproceedings{
  abaskohi2025cemtm,
  title={{CEMTM}: Contextual Embedding-based Multimodal Topic Modeling},
  author={Amirhossein Abaskohi and Raymond Li and Chuyuan Li and Shafiq Joty and Giuseppe Carenini},
  booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  url={https://openreview.net/forum?id=VRH4rYFe0v}
}
```
