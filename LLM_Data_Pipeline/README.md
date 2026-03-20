# LLM Data Pipeline Lab

## Overview

This lab demonstrates how to build an end-to-end data pipeline for language models using the AG News dataset. The pipeline covers dataset loading, exploratory data analysis (EDA), tokenization, sequence batching, and DataLoader creation for both language modeling and text classification tasks.

## What This Lab Does

1. **Load & Explore Data**: Fetches the AG News dataset (120,000 news articles across 4 categories: World, Sports, Business, Sci/Tech)
2. **Perform EDA**: Analyzes label distribution, text lengths, and samples from each category
3. **Tokenize**: Converts raw text into token IDs using the DistilGPT-2 tokenizer
4. **Create Training Sequences**: Groups tokens into fixed-length blocks (256 tokens) for language model pretraining
5. **Build DataLoaders**: Creates PyTorch DataLoaders for both:
   - **Language Modeling**: Sequences for causal language model training
   - **Text Classification**: Padded/truncated sequences with preserved labels

## Prerequisites

- Python 3.11+
- pip (Python package manager)

## Setup Instructions


### Install Dependencies

```bash
pip install datasets transformers torch jupyter
```

This installs:
- `datasets`: For loading HuggingFace datasets
- `transformers`: For tokenizers and model utilities
- `torch`: PyTorch for tensor operations
- `jupyter`: To run the notebook

### Select the Virtual Environment in VS Code

1. Open `Lab1.ipynb` in VS Code
2. Click on the Python version indicator (bottom right)
3. Select the virtual environment kernel (should show `venv` in the path)

### Run the Notebook

Click **"Run All"** or run cells individually using the play button.

## Lab Structure

### Cell 1: Installation (Commented Out)
```python
# !pip install transformers datasets torch
```
- This is commented out since dependencies are already installed via step 2

### Cell 2: Imports
Imports all required libraries: `datasets`, `transformers`, `torch`, and utilities

### Cell 3-4: Dataset Loading
- Loads AG News train split (120,000 examples)
- Displays dataset size and feature structure

### Cell 5: Exploratory Data Analysis
- Shows sample text from each category
- Displays label distribution (balanced: 25% each)
- Analyzes text length statistics (min, max, mean)

### Cell 6: Tokenization
- Initializes DistilGPT-2 tokenizer (vocab size: 50,257)
- Tokenizes all examples in batches for efficiency
- Outputs `input_ids` and `attention_mask` for each example

### Cell 7-8: Language Modeling Data Preparation
- Groups tokens into fixed-length sequences (block_size=256)
- Creates 24,361 training sequences from 120,000 examples
- Handles token concatenation across multiple articles

### Cell 9-10: Language Modeling DataLoader
- Creates a PyTorch DataLoader with batch size 8
- Uses causal LM setup: labels = input_ids (predict next token)
- Produces 3,046 total batches

### Cell 11-12: Text Classification DataLoader
- Alternative DataLoader that preserves original labels
- Pads/truncates sequences to block_size=256
- Maintains label information for classification tasks
- Outputs: `input_ids`, `attention_mask`, `labels`

## Expected Output

When you run all cells, you should see:

```
Number of examples in dataset: 120000
Features: {'text': Value('string'), 'label': ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech'])}

=== Label distribution ===
  World: 30000 examples (25.0%)
  Sports: 30000 examples (25.0%)
  Business: 30000 examples (25.0%)
  Sci/Tech: 30000 examples (25.0%)

=== Text length (chars) ===
  Min:    100
  Max:    1012
  Mean:   236.5

Vocab size: 50257
LM training sequences (block_size=256): 24361
Total batches in DataLoader: 3046
```


## Notes

- The notebook uses the AG News dataset instead of WikiText-2 for a more practical, real-world example
- DistilGPT-2 is used instead of GPT-2 for lighter memory footprint