import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,BertForSequenceClassification
# from generate_synthetic_data import generate_synthetic_data, data_to_text
# from inference import test_model, infer
import argparse
from scripts.fine_tune import finetune_roberta,finetune_t5,finetune_gpt2
# from inference import infer_langchain, infer_hf, infer, infer_gpt
# 
from data.dataLoader import data_loader

'''# Function to load the tokenizer and model
from scripts.fine_tune import fine_tune_lora, fine_tune_with_qlora, full_finetune, finetune_roberta
from scripts.gen_data import generate_synthetic_data, data_to_text
from scripts.infer import infer_gpt, infer_langchain, infer_hf, test_model, infer'''


def get_model_and_tokenizer(model_name='bert-base-uncased', num_labels=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune or test a model on a given dataset.')
    parser.add_argument('--model_name', type=str, help='Model name to use', default="bert-base-uncased")
    parser.add_argument('--train_data', type=str, help='Path to the training data file')
    parser.add_argument('--test_data', type=str, help='Path to the test data file')
    parser.add_argument('--finetune', type=str, choices=['lora', 'qlora', 'full_finetune','finetune_roberta','finetune_t5','finetune_gpt2'], help='Choose the fine-tuning method: LoRA, qlora, or full fine-tune')
    parser.add_argument('--infer', action='store_true', help='Flag to run inference on the model')
    parser.add_argument('--plot_embeddings', action='store_true', help='Flag to plot embeddings')
    parser.add_argument('--dataset', type=str, help='dataset to load')



    args = parser.parse_args()

    #Load the dataset 
    if args.dataset:
            data = data_loader.generic_data_loader(args.dataset)

        # select the method to finetune on
    if args.finetune == 'lora':
        fine_tune_lora("train_dataset.txt", args.model_name, tokenizer)
        return
    elif args.finetune == 'qlora':
        fine_tune_with_qlora("train_dataset.txt", args.model_name)
        return
    elif args.finetune == "full_finetune":
        full_finetune("train_dataset.txt", args.model_name, tokenizer)
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



    
    # plot embeddings
    if args.plot_embaddings:
        embeddings_plot.plot_embeddings()
        return

    # Test ChatGPT-4
    if args.infer:
        if args.infer[0] == 'gpt':
            infer_gpt(args.infer[1])
            return
        elif args.infer[0] == 'langchain':
            infer_langchain(args.infer[1])
            return
        elif args.infer[0] == 'hf':
            infer_hf(args.infer[1])
            return


    # Load the tokenizer and model
    model, tokenizer = get_model_and_tokenizer(args.model_name)


    # test_model(test_texts_ood, y_test_ood)

    # Load the model
    api_access_token = ""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=api_access_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=api_access_token)

    if args.infer[0] == 'finetune':
        infer(model, tokenizer, args.infer[1])
    else:
        if args.data_source == 'synthetic':
            # Generate synthetic data
            x_train, y_train, x_test_ood, y_test_ood = generate_synthetic_data(n_samples=1000, n_features=10,
                                                                               shift_intensity=5, sigma_noise=1)
            train_texts = data_to_text(x_train, y_train)
            train_texts_str = "\n".join(train_texts)
            test_texts_ood = data_to_text(x_test_ood, y_test_ood)
            test_texts_ood_str = "\n".join(test_texts_ood)
            train_dataset_path = "../data/processed/train_dataset.txt"
            with open(train_dataset_path, "w") as f:
                f.write(train_texts_str)
        else:
            if not args.data_source:
                raise ValueError("Dataset path must be provided if data_source is 'dataset'")
            train_dataset_path = args.data_source


if __name__ == "__main__":
    main()
