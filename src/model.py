import numpy as np
import torch
import pandas as pd
import argparse
from Prompts.Zero_shot_Prompting import run_sentiment_analysis_on_all_datasets,k_shot_run_sentiment_analysis_on_all_datasets
from Prompts.kshot_Prompting import k_shot_run_sentiment_analysis_on_all_datasets_kshot
from Prompts.ICR import run_reformulation_on_all_datasets
from Prompts.COT_Prompting import run_CoT_on_all_datasets
from Prompts.dataset_generation import rewrite_reviews
from scripts.fine_tune import finetune_roberta,finetune_t5,finetune_gpt2
from scripts.other_tuning import fine_tune_lora,fine_tune_with_qlora,full_finetune
from data.dataLoader import data_loader
from data.subSampling import subsample_and_save
from scripts.Evaluation import evaluate
from Prompts.Explanation import explanation_sentiment_analysis_on_all_datasets
from Prompts.abstinence import abstain_sentiment_analysis_on_all_datasets



def main():
    parser = argparse.ArgumentParser(description='Fine-tune or test a model on a given dataset.')
    parser.add_argument('--model_name', type=str, help='Model name to use', default="bert-base-uncased")
    parser.add_argument('--train_data', type=str, help='Path to the training data file')
    parser.add_argument('--test_data', type=str, help='Path to the test data file')
    parser.add_argument('--finetune', type=str, choices=['lora', 'qlora', 'full_finetune','finetune_roberta','finetune_t5','finetune_gpt2'], help='Choose the fine-tuning method: LoRA, qlora, or full fine-tune')
    parser.add_argument('--infer', action='store_true', help='Flag to run inference on the model')
    parser.add_argument('--plot_embeddings', action='store_true', help='Flag to plot embeddings')
    parser.add_argument('--dataset', type=str, help='dataset to load')
    parser.add_argument('--prompt_type', type=str, choices= ['zero_shot_prompt','k_shot_prompt','CoT','rewrite_reviews','explanation','abstain','k_shot_prompt_with_samples','reformulation'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--subSample', type=str)
    parser.add_argument('--evaluate',type=str)

    args = parser.parse_args()

    print(args.evaluate)
    if args.evaluate :
         evaluate(args.evaluate)
         return
     ## for prompting
    
    if args.prompt_type == 'zero_shot_prompt':
        run_sentiment_analysis_on_all_datasets(args.model_name)
        return 
    elif args.prompt_type == 'k_shot_prompt_with_samples':
        k_shot_run_sentiment_analysis_on_all_datasets_kshot(args.model_name)
        return
    elif args.prompt_type == 'CoT':
        run_CoT_on_all_datasets(args.model_name)
        return
    elif args.prompt_type == 'rewrite_reviews':
         rewrite_reviews("sst5","llama_70b")
         return
    elif args.prompt_type == "k_shot_prompt":
         k_shot_run_sentiment_analysis_on_all_datasets(args.model_name)
         return
    elif args.prompt_type == 'explanation':
        explanation_sentiment_analysis_on_all_datasets(args.model_name)
        return
    elif args.prompt_type == 'abstain':
        abstain_sentiment_analysis_on_all_datasets(args.model_name)
        return
    elif args.prompt_type == 'reformulation':
        run_reformulation_on_all_datasets(args.model_name,args.model_name)
        return
    
        
        
         
    
    if args.subSample:
        total_samples, samples_per_label, subsample_file_path = subsample_and_save(args.subSample)
        print(f"Total samples in subsample: {total_samples}")
        print(f"Samples per label:\n{samples_per_label}")
        print(f"Subsampled data saved to: {subsample_file_path}")
        return 
    
    #Load the dataset 
    if args.dataset:
            _ , data = data_loader.generic_data_loader(args.dataset)

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
         finetune_roberta(data,args.batch_size)
         return
    elif args.finetune == "finetune_t5":
         finetune_t5(data,args.batch_size)
         return
    elif args.finetune == "finetune_gpt2":
         finetune_gpt2(data,args.batch_size)
         return

if __name__ == "__main__":
    main()
