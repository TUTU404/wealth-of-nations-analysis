# Narrative Structure Analysis Pipeline

A robust Python pipeline for extracting and analyzing narrative structures from English text using semantic role labeling (SRL) and pronoun coreference resolution.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the analysis:**
   ```bash
   jupyter notebook tutorial_english.ipynb
   ```

## Core Workflow

The pipeline processes text through these stages:

1. **Data Loading** - Import and preprocess text corpus
2. **SVO Extraction** - Extract Subject-Verb-Object triplets using semantic role labeling
3. **Pronoun Resolution** - Filter pronouns and resolve coreferences using spaCy
4. **Entity Mining** - Identify and categorize named entities
5. **Analysis & Visualization** - Cluster narrative patterns and generate network graphs

## Project Structure

```
relatio-analysis/
├── tutorial_english.ipynb    # Main analysis notebook
├── requirement.txt          # Dependencies
├── data_process.py          # Data processing utilities
├── data/                    # Input data
│   └── sentences_dataset.csv
└── output/                  # Generated results
```

## Key Features

- **Advanced Pronoun Filtering**: Comprehensive filtering of personal, relative, and demonstrative pronouns
- **Coreference Resolution**: Automatic resolution of pronouns to their referents using spaCy
- **High-Quality SVO Extraction**: Clean semantic triplets for narrative analysis
- **Scalable Processing**: Batch processing for large corpora
- **Export Options**: JSON, pickle, and visualization outputs

## Configuration

Key parameters in the notebook:
- `batch_size`: Processing batch size (adjust for memory constraints)
- Pronoun filtering settings
- Clustering algorithms and parameters
- Visualization options

## Requirements

- Python 3.7+
- See `requirement.txt` for package dependencies
- SpaCy English model (`en_core_web_sm`)

## Notes for NLP Newcomers

This pipeline uses established NLP techniques:
- **Semantic Role Labeling (SRL)**: Identifies who did what to whom in sentences
- **Coreference Resolution**: Links pronouns to the entities they refer to
- **Named Entity Recognition**: Identifies people, places, organizations in text

The notebook includes detailed comments explaining each step and outputs sample results for verification.
