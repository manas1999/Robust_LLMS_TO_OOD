import numpy as np
import torch
import pandas as pd
import argparse
from Prompts.Zero_shot_Prompting import run_sentiment_analysis_on_all_datasets
from Prompts.COT_Prompting import run_CoT_on_all_datasets
from Prompts.dataset_generation import rewrite_reviews
from scripts.fine_tune import finetune_roberta,finetune_t5,finetune_gpt2
from scripts.other_tuning import fine_tune_lora,fine_tune_with_qlora,full_finetune
from data.dataLoader import data_loader


def main():
    parser = argparse.ArgumentParser(description='Fine-tune or test a model on a given dataset.')
    parser.add_argument('--model_name', type=str, help='Model name to use', default="bert-base-uncased")
    parser.add_argument('--train_data', type=str, help='Path to the training data file')
    parser.add_argument('--test_data', type=str, help='Path to the test data file')
    parser.add_argument('--finetune', type=str, choices=['lora', 'qlora', 'full_finetune','finetune_roberta','finetune_t5','finetune_gpt2'], help='Choose the fine-tuning method: LoRA, qlora, or full fine-tune')
    parser.add_argument('--infer', action='store_true', help='Flag to run inference on the model')
    parser.add_argument('--plot_embeddings', action='store_true', help='Flag to plot embeddings')
    parser.add_argument('--dataset', type=str, help='dataset to load')
    parser.add_argument('--prompt_type', type=str, choices= ['zero_shot_prompt','k_shot_prompt','CoT','rewrite_reviews'])

    args = parser.parse_args()

     ## for prompting
    if args.prompt_type == 'zero_shot_prompt':
        run_sentiment_analysis_on_all_datasets("llama_70b")
        return 
    elif args.prompt_type == 'CoT':
        run_CoT_on_all_datasets("llama_70b")
        return
    elif args.prompt_type == 'rewrite_reviews':
         rewrite_reviews("sst5","llama_70b")
         return
    
    #Load the dataset 
    if args.dataset:
            data = data_loader.generic_data_loader(args.dataset)

        # select the method to finetune on
    if args.finetune == 'lora':
        fine_tune_lora("train_dataset.txt", args.model_name)
        return
    elif args.finetune == 'qlora':
        fine_tune_with_qlora("train_dataset.txt", args.model_name)
        return
    elif args.finetune == "full_finetune":
        full_finetune("train_dataset.txt", args.model_name)
        return
    elif args.finetune == "finetune_roberta":
         finetune_roberta(data)
         return
    elif args.finetune == "finetune_t5":
         finetune_t5(data)
         return
    elif args.finetune == "finetune_gpt2":
         finetune_gpt2(data)
         return

if __name__ == "__main__":
    main()
