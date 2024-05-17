from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset,AutoModelForSequenceClassification, BertForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import logging


def fine_tune_lora(dataset_path, model_name, output_dir="./finetuned_model", epochs=3):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

def fine_tune_with_qlora(dataset_path, model_id, output_dir="./qlora_finetuned", epochs=3):
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], truncation=True,max_length=512)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting `pad_token` to `eos_token`:", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token

    reviews = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_reviews.csv')
    labels =  pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_labels.csv')

    data = Dataset.from_pandas(pd.concat([reviews, labels], axis=1))
    print(print(data.column_names))
    
    tokenized_data = data.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=3, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="SEQUENCE_CLASSIFICATION"
    )
    model = get_peft_model(model, config) 
    model.to(device)
  
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f'{output_dir}/logs',  
        logging_steps=10,  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    model.save_pretrained('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/qlora_finetuned')
    tokenizer.save_pretrained('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/qlora_finetuned')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}


def full_finetune(dataset_path, model, output_dir="./full_finetuned", epochs=3):
    model = BertForSequenceClassification.from_pretrained(model, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = load_dataset('text', data_files={'train': dataset_path})
    
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,
    )    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    model.config.use_cache = False 
    trainer.train()

