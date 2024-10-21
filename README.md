## Virtual Environment
 - `cd <base>/large-language-models`
 - `conda env create --file deploy/conda/linux_py312.yml`
 - `conda activate llm`
## Trained Model Downloads
 - Refer this website (https://mega.io/cmd) to install megacmd based on the operating system.
 - For Ubuntu 24.04: `wget https://mega.nz/linux/repo/xUbuntu_24.04/amd64/megacmd-xUbuntu_24.04_amd64.deb && sudo apt install "$PWD/megacmd-xUbuntu_24.04_amd64.deb"`
 - `mega-get https://mega.nz/folder/GNwjiCxR#bQtpQ8HMZ9jgoB1deKOTxA`
 - `mega-get https://mega.nz/folder/nBAXVDaa#Iu-PvhWUDHSDd78HvEleTA`
 - `mega-get https://mega.nz/folder/mUoGSTzR#7LQo8MLe_dz_zTG6nxdFTA`
 - `mega-get https://mega.nz/folder/GVpXxITD#9YqNR_uhUyxqsDI-KUMr0w`
## Step by Step Learning (based on Inferences)
 - **In-context-learning.ipynb**: This file explores the influence of input text on model output and involves practicing prompt engineering through comparisons of zero-shot, one-shot, and few-shot inferences to enhance Large Language Model outputs.
 - **Instruction-fine-tuning.ipynb**: This file focuses on fine-tuning the FLAN-T5 model from Hugging Face for improved dialogue summarization, exploring full fine-tuning and evaluating results with ROUGE metrics, followed by Parameter Efficient Fine-Tuning (PEFT) to demonstrate its advantages with slightly lower performance metrics.
 - **Reinforcement-learning-fine-tuning.ipynb**: This file focuses on fine-tuning a FLAN-T5 model to generate less toxic content using Meta AI's hate speech reward model, a binary classifier that predicts "not hate" or "hate." Proximal Policy Optimization (PPO) will be used for this fine-tuning to reduce the model's toxicity.