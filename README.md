# Large Language Models Project

This repository contains resources and notebooks for working with large language models.

## Setup

### Virtual Environment

1. Navigate to the project directory:
   ```
   cd <base>/large-language-models
   ```

2. Create the conda environment:
   ```
   conda env create --file deploy/conda/linux_py312.yml
   ```

3. Activate the environment:
   ```
   conda activate llm
   ```

4. To update the environment file (if necessary):
   ```
   conda env export --name llm > deploy/conda/linux_py312.yml
   ```

### Trained Model Downloads

1. Install megacmd based on your operating system from [https://mega.io/cmd](https://mega.io/cmd).

2. For Ubuntu 24.04:
   ```
   wget https://mega.nz/linux/repo/xUbuntu_24.04/amd64/megacmd-xUbuntu_24.04_amd64.deb && sudo apt install "$PWD/megacmd-xUbuntu_24.04_amd64.deb"
   ```

3. Download the trained models:
   ```
   mega-get https://mega.nz/folder/GNwjiCxR#bQtpQ8HMZ9jgoB1deKOTxA
   mega-get https://mega.nz/folder/nBAXVDaa#Iu-PvhWUDHSDd78HvEleTA
   mega-get https://mega.nz/folder/mUoGSTzR#7LQo8MLe_dz_zTG6nxdFTA
   mega-get https://mega.nz/folder/GVpXxITD#9YqNR_uhUyxqsDI-KUMr0w
   ```

## Notebooks

### In-context Learning
**File**: `In-context-learning.ipynb`

This notebook explores the influence of input text on model output. It focuses on prompt engineering techniques, comparing zero-shot, one-shot, and few-shot inferences to enhance Large Language Model outputs.

### Instruction Fine-tuning
**File**: `Instruction-fine-tuning.ipynb`

This notebook demonstrates fine-tuning the FLAN-T5 model from Hugging Face for improved dialogue summarization. It covers:
- Full fine-tuning
- Evaluation using ROUGE metrics
- Parameter Efficient Fine-Tuning (PEFT)
- Comparison of performance metrics

### Reinforcement Learning Fine-tuning
**File**: `Reinforcement-learning-fine-tuning.ipynb`

This notebook focuses on fine-tuning a FLAN-T5 model to generate less toxic content using:
- Meta AI's hate speech reward model (a binary classifier predicting "not hate" or "hate")
- Proximal Policy Optimization (PPO) for reducing model toxicity

## BERT vs. FLAN-T5

| Feature | BERT | FLAN-T5 |
|---------|------|---------|
| Architecture | Encoder-only | Encoder-decoder |
| Pre-training | Masked Language Modeling and Next Sentence Prediction | Text-to-Text Transfer Transformer (T5) |
| Fine-tuning | Task-specific fine-tuning required | Instruction-tuned, can handle multiple tasks without task-specific fine-tuning |
| Input/Output | Fixed-length input, typically used for classification and token-level tasks | Variable-length input and output, suitable for a wide range of NLP tasks |
| Multilingual Support | Available in multilingual versions | Inherently supports multiple languages |
| Size | Various sizes, typically smaller than T5 models | Generally larger, with various sizes available |
| Instruction Following | Not designed for direct instruction following | Specifically trained to follow natural language instructions |

FLAN-T5 is an advancement over BERT, offering more flexibility in task handling and better performance on a wider range of NLP tasks without requiring task-specific fine-tuning.