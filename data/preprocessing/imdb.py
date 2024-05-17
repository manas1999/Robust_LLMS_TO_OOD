import os
import pandas as pd
import torch
import random
import numpy as np
import re
from datasets import load_dataset

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

def preprocess_imdb():
    set_seed(0)
    dataset = load_dataset("stanfordnlp/imdb")

    dataset = dataset.map(text_process).shuffle(seed=0)

    train = dataset['train']
    test = dataset['test']

    labels = np.array(train['label'])
    max_length = max(
        np.sum(labels == 0),
        np.sum(labels == 1)
    )
    print("max length:", max_length)

    train_dataset = []
    label_count = {0: 0, 1: 0}
    for data in train:
        if label_count[data['label']] < max_length:
            train_dataset.append({'Text': data['text'], 'Label': data['label']})
            label_count[data['label']] += 1

    test_dataset = [{'Text': data['text'], 'Label': data['label']} for data in test]

    save_data(train_dataset, "./data/processed/imdb", "train")
    save_data(test_dataset, "./data/processed/imdb", "test")


