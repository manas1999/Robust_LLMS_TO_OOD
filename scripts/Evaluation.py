from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
import os

def load_preprocessed_data(file_path):
    try:
        data = pd.read_csv(file_path, sep="\t")
        data['Text'] = data['Text'].astype(str)
        data['Label'] = data['Label'].astype(int)
        data = shuffle(data, random_state=42)
        return Dataset.from_pandas(data)
    except Exception as e:
        print(f"Failed to load or process the data from {file_path}. Error: {e}")
        return None

def tokenize_function(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs['labels'] = torch.tensor(examples['Label'])
    return tokenized_inputs

def evaluate_model(model, data, tokenizer, device):
    model.eval()
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    predictions, labels = [], []

    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data, batch_size=256):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            label = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(label.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    return {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }

def evaluate(model):
    model_name  = model
    model_path = f'../outputs/models/{model}'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        
        "amazon": "./data/processed/amazon_subsample/test.tsv",
        "sst5": "./data/processed/sst5_subsample/test.tsv",
        "semeval": "./data/processed/semeval_subsample/test.tsv",
        "dynasent": "./data/processed/dynasent_subsample/test.tsv"
    }

    results_df = pd.DataFrame(columns=["Dataset", "Accuracy", "F1 Score", "Precision", "Recall"])

    for name, path in datasets.items():
        test_data = load_preprocessed_data(path)
        if test_data is not None:
            tokenized_test_data = test_data.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
            result = evaluate_model(model, tokenized_test_data, tokenizer, device)
            result["Dataset"] = name
            results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

    results_df.to_csv(f"./results/{model_name}_finetuned_results.csv", index=False)
    print(f"Saved the evaluation results ./results/{model_name}_finetuned_results.csv ")
