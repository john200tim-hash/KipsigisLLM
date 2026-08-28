# KipsigisLLM

A modular, scalable, and open-source foundation for building and training a language model for the Kipsigis language.

## Licensing and Data Policy

* **Code & Architecture:** All pipeline scripts, tokenization utilities, and architectural code are open-source and available under the [MIT License](LICENSE).
* **Model Weights:** The pre-trained model weights (`checkpoints/kipsigis_model.pt`) are provided for research and deployment use.
* **Training Data:** The proprietary text corpora, source PDFs, and customized training datasets (`data/`) used to train this model are **private and strictly proprietary**. They are excluded from this repository and will not be published or distributed. Users wishing to train the model must supply their own training corpora.

## Local Setup & Usage

This project uses a modular Python structure and comes with batch scripts that handle virtual environment setup and execution automatically.

### 1. Training the Model
To start training the Kipsigis LLM:
1. Place your text corpus inside `data/raw/` or `data/processed/`.
2. Double-click `run_train.bat`.
   * *This will automatically create a Python virtual environment (`venv`), install dependencies from `requirements.txt`, process the data, and begin training based on `configs/train_config.yaml`.*

### 2. Running Inference
To generate text using the trained weights:
1. Double-click `run_inference.bat`.
   * *This will activate the virtual environment and load the model weights from `checkpoints/kipsigis_model.pt` to generate outputs based on the prompts in `inference.py`.*

### Configuration
You can adjust model hyperparameters (epochs, learning rate, batch size, etc.) by editing `configs/train_config.yaml`.
