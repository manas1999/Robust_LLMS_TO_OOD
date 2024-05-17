import os
import pandas as pd
import torch
import random
import numpy as np
import re
from datasets import load_dataset, DatasetDict

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

label_mapping = {0:0, 4:1, 2:2}  
def label_process(example):
    example["label"] = label_mapping[example["label"]]
    return example
def print_dataset_info(train_data, test_data):
    print("Number of train samples:", len(train_data))
    print("Number of test samples:", len(test_data))
    train_label_counts = np.unique([data['Label'] for data in train_data], return_counts=True)
    test_label_counts = np.unique([data['Label'] for data in test_data], return_counts=True)
    print("Labels in train set:", dict(zip(train_label_counts[0], train_label_counts[1])))
    print("Labels in test set:", dict(zip(test_label_counts[0], test_label_counts[1])))

def preprocess_yelp():
    set_seed(0)
    dataset = load_dataset("yelp_review_full")
    
    print(dataset['train'].column_names)  

    dataset = dataset.filter(lambda example: example["label"] in [0, 2, 4])
    dataset = dataset.map(text_process).map(label_process).shuffle(seed=0)

    if isinstance(dataset, DatasetDict):
        train = dataset['train']
        test = dataset['test']
    else:
        train_test_split = dataset.train_test_split(test_size=0.1)
        train = train_test_split['train']
        test = train_test_split['test']

    max_length = 50000
    label_count = {0:0, 1:0, 2:0}

    train_dataset = []
    for data in train:
        if label_count[data["label"]] < max_length:
            train_dataset.append({"Text": data['text'], "Label": data['label']})
            label_count[data["label"]] += 1

    test_dataset = [{"Text": data['text'], "Label": data['label']} for data in test]

    save_data(train_dataset, "./data/processed/yelp_full", "train")
    save_data(test_dataset, "./data/processed/yelp_full", "test")
    print_dataset_info(train_dataset, test_dataset)