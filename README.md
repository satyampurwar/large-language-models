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