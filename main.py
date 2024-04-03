import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_synthetic_data import generate_synthetic_data, data_to_text
from inference import test_model,infer
import argparse 
from inference import infer_langchain,infer_hf,infer,infer_gpt

from fineTune import full_finetune,fine_tune_lora,fine_tune_with_qlora


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fine-tune or test a model on a given dataset.')
    parser.add_argument('--model_name', type=str, help='Model name to use', default="chargoddard/Yi-34B-Llama")
    parser.add_argument('--finetune', type=str, choices=['lora', 'qlora', 'full_finetune'], help='Choose the fine-tuning method: LoRA, qlora, or full fine-tune')
    parser.add_argument('--infer', nargs=2, metavar=('MODEL_TYPE', 'PROMPT'), help='Specify the inference model and the prompt string for testing the model')
    parser.add_argument('--test_data', type=str, help='Path to the test data file', default="test_dataset.txt")
    parser.add_argument('--prompt', type=str, help='Prompt string for testing the model', default="")
    parser.add_argument('--data', type=str, choices=['synthetic', 'dataset'], help='Choose the data source for fine-tuning: synthetic data or external dataset')


    args = parser.parse_args()

    #Test ChatGPT-4
    if args.infer[0] == 'gpt':
        infer_gpt(args.infer[1])
    elif args.infer[0] == 'langchain':
        infer_langchain(args.infer[1])
    elif args.infer[0] == 'hf':
        infer_hf(args.infer[1])


    #Load the model

    api_access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=api_access_token,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=api_access_token)
        


    if args.infer[0] == 'finetune':
        infer(model, tokenizer, args.infer[1])
    else:
        if args.data_source == 'synthetic':
    # Generate synthetic data
            X_train, y_train, X_test_ood, y_test_ood = generate_synthetic_data(n_samples=1000, n_features=10, shift_intensity=5, sigma_noise=1)
            train_texts = data_to_text(X_train, y_train)
            train_texts_str = "\n".join(train_texts)
            test_texts_ood = data_to_text(X_test_ood, y_test_ood)
            test_texts_ood_str = "\n".join(test_texts_ood)
            train_dataset_path = "train_dataset.txt"
            with open("train_dataset.txt", "w") as f:
                f.write(train_texts_str)
        else:
            if not args.data_source:
                raise ValueError("Dataset path must be provided if data_source is 'dataset'")
            train_dataset_path = args.data_source
        

        
        # select the method to finetune on
        if args.finetune == 'lora':
            fine_tune_lora("train_dataset.txt", model,tokenizer)
        elif args.finetune == 'qlora':
            fine_tune_with_qlora("train_dataset.txt", model,tokenizer)
        elif args.finetune == "full_finetune":
            full_finetune("train_dataset.txt", model,tokenizer)

        test_model(test_texts_ood, y_test_ood)

if __name__ == "__main__":
    main()
