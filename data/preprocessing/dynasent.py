import os
import pandas as pd
import torch
import random
import numpy as np
import re
from datasets import load_dataset, concatenate_datasets

def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_data(dataset, path, split):
    header = ["Text", "Label"]
    df = pd.DataFrame(dataset, columns=header)
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{split}.tsv", sep="\t", index=False, header=header)

def text_process(example):
    example["text"] = re.sub("\t", " ", example["text"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example

label_mapping = {"negative": 0, "positive": 1, "neutral": 2}
def label_process(example):
    example["label"] = label_mapping[example["label"]]
    return example

def preprocess_dynasent():
    set_seed(0)
    # Specify the dataset configuration
    config_name_r1 = 'dynabench.dynasent.r1.all'
    config_name_r2 = 'dynabench.dynasent.r2.all'

    # Load the dataset with the specified configuration
    dataset_r1 = load_dataset("dynabench/dynasent", config_name_r1)
    dataset_r2 = load_dataset("dynabench/dynasent", config_name_r2)

    # Rename columns and process labels and text
    processed_datasets = {}
    splits = ['train', 'test']
    for config, config_name in zip([dataset_r1, dataset_r2], [config_name_r1, config_name_r2]):
        for split in splits:
            processed_datasets[f"{config_name}_{split}"] = config[split].rename_column("sentence", "text").rename_column("gold_label", "label")
            processed_datasets[f"{config_name}_{split}"] = processed_datasets[f"{config_name}_{split}"].map(text_process).map(label_process)

    # Concatenate train and test sets from both rounds and shuffle
    train = concatenate_datasets([processed_datasets[f"{config_name_r1}_train"], processed_datasets[f"{config_name_r2}_train"]]).shuffle(seed=0)
    test = concatenate_datasets([processed_datasets[f"{config_name_r1}_test"], processed_datasets[f"{config_name_r2}_test"]]).shuffle(seed=0)

    # Prepare datasets for saving
    train_dataset = [{"Text": data['text'], "Label": data['label']} for data in train]
    test_dataset = [{"Text": data['text'], "Label": data['label']} for data in test]

    # Save the processed data
    save_data(train_dataset, "./data/processed/dynasent", "train")
    save_data(test_dataset, "./data/processed/dynasent", "test")


