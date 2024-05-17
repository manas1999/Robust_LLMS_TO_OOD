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
    example["text"] = re.sub("\t", " ", example["sentence"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example

def label_process(example):
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
    
    dataset = dataset.map(lambda example: {'text': example['sentence'], 'label': example['label']})
    
    dataset = dataset.map(text_process)
    dataset = dataset.map(label_process)
    
    dataset = dataset.filter(lambda example: example["label"] != -1)
    
    dataset = dataset.shuffle(seed=0)
    
    train_test_split = dataset.train_test_split(test_size=0.1)
    train = train_test_split['train']
    test = train_test_split['test']

    train_dataset = [(example['text'], example['label']) for example in train]
    test_dataset = [(example['text'], example['label']) for example in test]
    
    save_data(train_dataset, "./data/processed/sst5", "train")
    save_data(test_dataset, "./data/processed/sst5", "test")

