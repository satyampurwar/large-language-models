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