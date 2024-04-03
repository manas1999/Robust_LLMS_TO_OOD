### README for Fine-Tuning and Testing Language Models

This repository contains scripts for fine-tuning and testing large language models with options for full fine-tuning, LoRA, and QLoRA adaptation methods. It supports generating synthetic data for training and provides functionality for testing the fine-tuned models with custom prompts.


#### Setup

```bash
pip install transformers torch numpy datasets
pip install git+https://github.com/path/to/peft.git # Assuming peft is not available on PyPI and 
```

#### Usage

The main script is designed to be used from the command line with several options for customization:

- `--model_name`: The name of the model you want to fine-tune or test.
- `--finetune`: Specify the fine-tuning method (`lora`, `qlora`, or `full_finetune`).
- `--infer`: Enable this flag to test the model directly without fine-tuning.
- `--test_data`: Path to the test data file (default: "test_dataset.txt").
- `--prompt`: Prompt string for testing the model in inference mode.
- `--data`: Choose the data source for fine-tuning: `synthetic` data or an external `dataset`.

**Example Command**:

```bash
python main.py --model_name="chargoddard/Yi-34B-Llama" --finetune="lora" --data="synthetic"

python main.py --model_name nomic-ai/gpt4all-j  --infer --prompt "Hello , How are you ?"

```

#### Fine-Tuning Methods

- **Full Fine-Tuning**: Updates all parameters of the model.
- **LoRA**: Low-Rank Adaptation focuses on updating only a small set of parameters, reducing the computational cost.
- **QLoRA**: A quantized version of LoRA that further reduces the model size and computational requirements.

#### Data Generation

The script can generate synthetic data if the `--data` option is set to `synthetic`. This data is used for fine-tuning when no external dataset is provided.