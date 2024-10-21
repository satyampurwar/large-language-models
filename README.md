## Virtual Environment
 - `cd <base>/large-language-models`
 - `conda env create --file deploy/conda/linux_py312.yml`
 - `conda activate llm`
## Step by Step Learning
 - **In-context-learning.ipynb**: This file explores the influence of input text on model output and involves practicing prompt engineering through comparisons of zero-shot, one-shot, and few-shot inferences to enhance Large Language Model outputs.
 - **Instruction-fine-tuning.ipynb**: This file focuses on fine-tuning the FLAN-T5 model from Hugging Face for improved dialogue summarization, exploring full fine-tuning and evaluating results with ROUGE metrics, followed by Parameter Efficient Fine-Tuning (PEFT) to demonstrate its advantages despite slightly lower performance metrics.