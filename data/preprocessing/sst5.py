from datasets import load_dataset
import numpy as np
import pandas as pd
import re
import torch
import random
import os

def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def save_data(dataset, path, split):
    if len(dataset[0]) == 2:
        header = ["Text", "Label"]
    elif len(dataset[0]) == 3:
        header = ["Premise", "Hypothesis", "Label"]
    df = pd.DataFrame(dataset, columns=header)
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{split}.tsv", sep="\t", index=False, header=header)

def text_process(example):
    # Ensure the correct field is being processed
    example["text"] = re.sub("\t", " ", example["sentence"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example

def label_process(example):
    # Assuming you are classifying into three categories based on the label value
    if example["label"] < 0.2:
        example["label"] = 0
    elif example["label"] >= 0.8:
        example["label"] = 1
    elif example["label"] >= 0.4 and example["label"] < 0.6:
        example["label"] = 2
    else:
        example["label"] = -1
    return example

def process_sst5_dataset():
    set_seed(0)
    dataset = load_dataset("sst", split='train')
    
    # Rename 'sentence' to 'text' to keep consistency in processing
    dataset = dataset.map(lambda example: {'text': example['sentence'], 'label': example['label']})
    
    # Process text and labels
    dataset = dataset.map(text_process)
    dataset = dataset.map(label_process)
    
    # Filter out unwanted labels
    dataset = dataset.filter(lambda example: example["label"] != -1)
    
    # Optionally shuffle the dataset
    dataset = dataset.shuffle(seed=0)
    
    # Split the data (you may adjust the split ratio as needed)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train = train_test_split['train']
    test = train_test_split['test']

    # Prepare datasets for saving
    train_dataset = [(example['text'], example['label']) for example in train]
    test_dataset = [(example['text'], example['label']) for example in test]
    
    # Save the datasets
    save_data(train_dataset, "./data/processed/sst5", "train")
    save_data(test_dataset, "./data/processed/sst5", "test")

