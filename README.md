# Generative AI with Large Language Models

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
- Proximal Policy Optimization (PPO) for reducing model toxicity.

### RAG Implementation
**File**: `RAG-Implementation.ipynb`

This notebook demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using Python and LangChain libraries. The notebook guides users through the process of:
- Ingesting documents from the web
- Splitting text into manageable chunks
- Generating vector embeddings for semantic search
- Storing and retrieving information using a vector database
- Leveraging a large language model (LLM) to generate context-aware answers to user queries

By following this notebook, one will learn how to build an AI system that combines the power of search and generative models, enabling accurate, up-to-date, and contextually relevant responses based on your own data sources. This approach is ideal for building document Q&A systems, knowledge assistants, and enterprise chatbots, and serves as a practical introduction to modern RAG architectures.

### AI-Agent Demonstration
**File**: `AI-Agent-Demonstration.ipynb`

This notebook focuses on LangGraph-based code which implements a structured workflow for automated job description analysis using FLAN-T5. The notebook demonstrates two core functionalities:
- Role-Specific Job Analysis Pipeline
   - Creates a LangGraph workflow with four sequential components: role classification, skill extraction, experience detection, and summary generation
   - Uses in-context learning patterns with few-shot examples for each processing stage
- Instruction-Tuned Model Deployment
   - Leverages FLAN-T5-base's text-to-text architecture for multiple NLP tasks without retraining

The code aligns with documented approach how to use LangGraph for workflow/pipeline orchestration.

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

## Infrastructure Decision-making
Here is a table summarizing key information about storage and training memory required for large language models based on model size and parameters:

| Aspect | Details |
|--------|---------|
| Model Size | - Typically measured in number of parameters (e.g. 175B for GPT-3) |
| Parameters | - Each parameter is usually a 32-bit float (4 bytes) |
| Storage | - 1B parameters ≈ 4GB storage |
| Training Memory | - Model parameters: 4 bytes per parameter |
|  | - Adam optimizer states: 8 bytes per parameter |
|  | - Gradients: 4 bytes per parameter  |
|  | - Activations/temp memory: ~8 bytes per parameter |
|  | - Total: ~24 bytes per parameter |
| Example | - 1B parameter model: |
|  | - 4GB to store |
|  | - ~24GB GPU RAM to train |
| Quantization | - FP16: 2 bytes per parameter |
|  | - INT8: 1 byte per parameter |
|  | - Reduces storage and memory requirements |
| PEFT Methods | - LoRA |
|  | - Train small number of parameters (e.g. <1%) |
|  | - Drastically reduce memory/storage needs |

The exact numbers can vary based on model architecture, training approach and optimizations used.

**Example**: Here's a table summarizing the storage and training memory requirements for **FLAN-T5-base (250M parameters)**, which has been used as the base model in the notebooks referred above:

| Data Type | Model Size | Inference VRAM | Training VRAM (using Adam) |
|-----------|------------|----------------|----------------------------|
| float32   | 850.31 MB  | 94.12 MB       | 3.32 GB                   |
| float16/bfloat16 | 425.15 MB | 47.06 MB | 1.66 GB                   |
| int8      | 212.58 MB  | 23.53 MB       | 850.31 MB                 |
| int4      | 106.29 MB  | 11.77 MB       | 425.15 MB                 |