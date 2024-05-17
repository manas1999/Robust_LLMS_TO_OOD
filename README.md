### README : Robust LLMs that work in OOD settings


This repository contains scripts for 
fine-tuning 
Inference from differennt public models like GPT-2 , LLAMA etc 
testing LLMs 
FineTuning : full fine-tuning, LoRA, and QLoRA adaptation methods. 
generating synthetic data

#### Setup

```bash
pip install transformers torch numpy datasets
pip install git+https://github.com/path/to/peft.git 
pip install langchain
pip install openai==0.28
pip install -U langchain-community  
pip install -U sentence-transformers    
```

#### Usage

The main script is designed to be used from the command line with several options for customization:

- `--model_name`: The name of the model you want to fine-tune or test.
- `--finetune`: Specify the fine-tuning method (`lora`, `qlora`, or `full_finetune`).
- `--infer`: Enable this flag to test the model directly without fine-tuning.
- `--test_data`: Path to the test data file (default: "test_dataset.txt").
- `--prompt`: Prompt string for testing the model in inference mode.
- `--data`: Choose the data source for fine-tuning: `synthetic` data or an external `dataset`.

set your together api key in the local environment using this command
```
export TOGETHER_API_KEY=your_api_key
```
set your wandb key in the local environment using this command
```
export WAND_KEY=your_wandb_key
```

**Example Command**:

```bash
python main.py --model_name="chargoddard/Yi-34B-Llama" --finetune="lora" --data="synthetic"
python main.py --model_name="chargoddard/Yi-34B-Llama" --infer "finetune" "tell me a jock ?"
python main.py --model_name="chargoddard/Yi-34B-Llama" --test_data PATH_TEST_FILE



python main.py   --infer langchain "Which is the hottest planet?"
python main.py   --infer gpt "Which is the hottest planet?"
python main.py   --infer hf "Which is the hottest planet?"


#FLIPKART Data
python main.py --flipkart Clean # to clean the data and save it to the disk use only once 
python main.py --finetune qlora
#YELP DATA CLEAN
python main.py --yelp Clean 
#imdb DATA CLEAN
python main.py --imdb Clean 

#PLOT EMBADDINGS:
python main.py --plot_embaddings ag

#FIne tune roberta on Yelp dataset
python main.py --finetune='finetune_roberta' --dataset='yelp'

```

#### Fine-Tuning Methods

- **Full Fine-Tuning**: Updates all parameters of the model.
- **LoRA**: Low-Rank Adaptation focuses on updating only a small set of parameters, reducing the computational cost.
- **QLoRA**: A quantized version of LoRA that further reduces the model size and computational requirements.
- **finetune-roberta**: full finetuning roberta model.

#### Data Generation

The script can generate synthetic data if the `--data` option is set to `synthetic`. This data is used for fine-tuning when no external dataset is provided.