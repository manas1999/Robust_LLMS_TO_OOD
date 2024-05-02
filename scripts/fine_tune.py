from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset,AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import logging
import time
import wandb
from transformers.integrations import WandbCallback
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import default_data_collator


def finetune_roberta(dataset):
    wandb.login(key="5035d804b450ae72d3a317de6ddde7e467aab080")
    wandb.init(project="Robust_LLM", name = "Roberta_Finetuning")
    # Setup the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer from pretrained
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Tokenize dataset function adjusted for batch processing
    print("Tokenizing dataset...")
    
    def tokenize_function(batch):
        # Use 'text' as per your dataset schema
        texts = [text if text is not None else "" for text in batch['Text']]
        # Apply the updated label mapping
        labels = [ label for label in batch['Label']]
        # Tokenize and add labels
        tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    # Split dataset into training and evaluation
    dataset_dict = dataset.train_test_split(test_size=0.1)

    # Apply tokenization to training and evaluation datasets
    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)
    
    # Load the model and move it to the correct device
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
    
    # Define training arguments
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/roberta/results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),  # Enabling this only if CUDA is available
    )

    # Create and configure the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    # Start the training process
    trainer.train()

    # Save the model and tokenizer
    print("Saving the model...")
    save_path = '../outputs/models/roberta_amazon'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return model


# Define the finetune function for T5 with Weights & Biases
def finetune_t5(dataset):
    # Initialize Weights & Biases
    wandb.login(key="5035d804b450ae72d3a317de6ddde7e467aab080")
    wandb.init(project="Robust_LLM", name="T5_Finetuning")

    # Setup the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer from pretrained
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # Tokenize dataset function for T5
    print("Tokenizing dataset...")

    def tokenize_function(batch):
        # Convert texts and labels into appropriate input-output sequences for T5
        texts = [text if text is not None else "" for text in batch['Text']]
        label_texts = ["positive" if label == 1 else "negative" if label == 0 else "neutral" for label in batch['Label']]

        inputs = [f"classify: {text}" for text in texts]
        outputs = [f"{label}" for label in label_texts]
        
        # Tokenize inputs
        tokenized_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Tokenize labels (outputs)
        tokenized_labels = tokenizer(
            outputs,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        # Attach labels to inputs for training
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    # Split dataset into training and evaluation
    dataset_dict = dataset.train_test_split(test_size=0.1)

    # Apply tokenization to training and evaluation datasets
    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    # Load the T5 model and move it to the correct device
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    # Define training arguments with Weights & Biases callback
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/t5/results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        gradient_clip_val=1.0,
        fp16=torch.cuda.is_available(),
    )

    # Create and configure the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    # Start the training process
    trainer.train()

    # Save the model and tokenizer
    print("Saving the model...")
    save_path = '../outputs/models/t5_sentiment'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model


# Define the fine-tune function for GPT-2 with Weights & Biases
def finetune_gpt2(dataset):
    # Initialize Weights & Biases
    wandb.login(key="5035d804b450ae72d3a317de6ddde7e467aab080")
    wandb.init(project="Robust_LLM", name="GPT2_Finetuning")

    # Setup the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer from pretrained
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to EOS to avoid issues with GPT-2

    # Tokenization function for GPT-2
    print("Tokenizing dataset...")

    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        label_texts = ["positive" if label == 1 else "negative" if label == 0 else "neutral" for label in batch['Label']]

        # Prepare input-output text for GPT-2
        input_texts = [f"Text: {text} | Sentiment:" for text in texts]
        output_texts = [f" {label}" for label in label_texts]

        # Tokenize the inputs and outputs
        tokenized_inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokenized_labels = tokenizer(
            output_texts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        if tokenized_labels["input_ids"].nelement() == 0:
            print("Warning: Empty labels encountered")

        # Append labels to inputs for GPT-2
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    # Split dataset into training and evaluation
    dataset_dict = dataset.train_test_split(test_size=0.1)

    # Apply tokenization to training and evaluation datasets
    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    # Load the GPT-2 model and move it to the correct device
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Define training arguments with Weights & Biases callback
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/gpt2/results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # Create and configure the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=default_data_collator,  # Ensures proper batching
        callbacks=[WandbCallback()],
    )

    # Start the training process
    trainer.train()

    # Save the model and tokenizer
    print("Saving the model...")
    save_path = '../outputs/models/gpt2_sentiment'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model


