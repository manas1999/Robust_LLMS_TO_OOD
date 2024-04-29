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

label_mapping = {1:0, 2:0, 3:2, 4:2, 5:1}  # Adjust according to actual dataset label distribution
def label_process(example):
    if example["label"] in label_mapping:
        example["label"] = label_mapping[example["label"]]
    return example

def preprocess_yelp():
    set_seed(0)
    # Load dataset from Hugging Face
    dataset = load_dataset("yelp_review_full")
    
    # Check the column names
    print(dataset['train'].column_names)  # This will help confirm the correct label column name

    # Assuming the label column is correctly named 'label'
    dataset = dataset.map(lambda example: {'text': example['text'], 'label': example['label']})
    dataset = dataset.filter(lambda example: example['label'] in [1, 3, 5])  # Adjust as needed
    dataset = dataset.map(text_process).map(label_process).shuffle(seed=0)

    # Split into train and test
    if isinstance(dataset, DatasetDict):
        train = dataset['train']
        test = dataset['test']
    else:
        train_test_split = dataset.train_test_split(test_size=0.1)
        train = train_test_split['train']
        test = train_test_split['test']

    # Prepare datasets for saving
    train_dataset = [{"Text": data['text'], "Label": data['label']} for data in train]
    test_dataset = [{"Text": data['text'], "Label": data['label']} for data in test]

    # Save the processed data
    save_data(train_dataset, "./data/processed/SentimentAnalysis/yelp", "train")
    save_data(test_dataset, "./data/processed/SentimentAnalysis/yelp", "test")


