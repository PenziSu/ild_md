# ILD_MD Repo

## Setup Instructions
First, you'll need to create a Conda environment and install these packages:

```Bash
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
Faiss is a bit tricky. If you can't install it with pip or conda, you'll need to use the method recommended by the official Conda team:
```

```Bash
conda install conda-forge::faiss
```

## Model Selection
You have to pick a model that can use tools, otherwise you'll run into errors.

## Changing Ollama Server and LLM Model
Open config.py and tweak these settings:

```Python

OLLAMA_MODEL_NAME = "Elixpo/LlamaMedicine:latest"
OLLAMA_API_BASE = "http://192.168.1.100:11434/v1"
EXTRACTOR_OLLAMA_MODEL_NAME = "Elixpo/LlamaMedicine:latest" # For the Langchain-based
```