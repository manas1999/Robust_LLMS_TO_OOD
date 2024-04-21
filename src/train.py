import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertForSequenceClassification

# from model import get_model_and_tokenizer

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(file_path, labels_path, masks_path):
    # Load the data
    inputs = pd.read_csv(file_path).values.astype(np.int64)
    labels = pd.read_csv(labels_path).values.flatten().astype(np.int64)
    masks = pd.read_csv(masks_path).values.astype(np.int64)

    # Convert to torch tensors
    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)

    return TensorDataset(inputs, masks, labels)


def create_data_loader(data_set, batch_size=16):
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def calculate_accuracy(predictions, true_labels):
    pred_labels = np.argmax(predictions, axis=1)
    correct = np.sum(pred_labels == true_labels)
    return correct / len(true_labels)


def get_model_and_tokenizer(model_name='bert-base-uncased', num_labels=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def main():
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    model.to(device)

    # Load training and validation data
    train_dataset = load_data('../data/tokenized/imdb/train_tokenized_input_ids.csv',
                              '../data/processed/imdb/train_labels.csv',
                              '../data/tokenized/imdb/train_tokenized_attention_masks.csv')
    validation_dataset = load_data('../data/tokenized/imdb/validation_tokenized_input_ids.csv',
                                   '../data/processed/imdb/validation_labels.csv',
                                   '../data/tokenized/imdb/validation_tokenized_attention_masks.csv')

    train_loader = create_data_loader(train_dataset)
    validation_loader = create_data_loader(validation_dataset)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

    # Training loop
    model.train()
    for epoch in range(3):  # number of epochs
        total_train_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch
            model.zero_grad()
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            total_train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f'Epoch {epoch + 1}, Training Loss: {total_train_loss / len(train_loader)}')

        # Validation loop
        model.eval()
        total_eval_accuracy, total_eval_f1 = 0, 0
        total_true_labels = []
        total_predictions = []
        for batch in validation_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch
            with torch.no_grad():
                outputs = model(inputs, attention_mask=masks, labels=labels)
                logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += calculate_accuracy(np.argmax(logits, axis=1), label_ids)
            total_true_labels.extend(label_ids)
            total_predictions.extend(np.argmax(logits, axis=1))

        # Calculate the total F1 score
        total_eval_f1 = f1_score(total_true_labels, total_predictions, average='weighted')
        print(f'Epoch {epoch + 1}, Validation Accuracy: {total_eval_accuracy / len(validation_loader)}')
        print(f'Epoch {epoch + 1}, Validation F1 Score: {total_eval_f1}')

    # Save the fine-tuned model
    model_save_path = '../outputs/models/imdb_model.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    main()
