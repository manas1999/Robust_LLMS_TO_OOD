from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset,AutoModelForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import logging
import time
from accelerate import DataLoaderConfiguration


def fine_tune_lora(dataset_path, model,tokenizer, output_dir="./finetuned_model", epochs=3):

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

# Updated function for fine-tuning with QLoRA
def fine_tune_with_qlora(dataset_path, model_id, output_dir="./qlora_finetuned", epochs=3):
    

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], truncation=True,max_length=512)

    # Explicitly trust the remote code by setting trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting `pad_token` to `eos_token`:", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    reviews = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_reviews.csv')
    labels =  pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_labels.csv')

    data = Dataset.from_pandas(pd.concat([reviews, labels], axis=1))
    print(print(data.column_names))
    
    tokenized_data = data.map(tokenize_function, batched=True)
    
    # Load the model and apply QLoRA configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=3, trust_remote_code=True)
    # Enable gradient checkpointing and prepare for k-bit training
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
        logging_dir=f'{output_dir}/logs',  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
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
    # Save the fine-tuned model and tokenizer

    model.save_pretrained('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/qlora_finetuned')
    tokenizer.save_pretrained('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/qlora_finetuned')
    # Define accuracy metric function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    



def full_finetune(dataset_path, model, tokenizer, output_dir="./full_finetuned", epochs=3):
    # Load the dataset
    dataset = load_dataset('text', data_files={'train': dataset_path})
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Create a data collator that dynamically pads the inputs received, as well as the labels.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    model.config.use_cache = False 
    # Start fine-tuning           
    trainer.train()


def finetune_roberta(dataset):
    
    start_time = time.time()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Tokenize data
    print("Tokenizing dataset...")
    tokenize_start_time = time.time()
    # Adjust the tokenization function to properly handle batched data
    
    def tokenize_function(batch):
        cleaned_reviews = [review if review is not None else "" for review in batch['cleaned_review']]
        # Use the correct column name for labels
        tokenized_inputs = tokenizer(cleaned_reviews, padding="max_length", truncation=True, max_length=512)
        #tokenized_inputs['labels'] = batch['Sentiment']  # Correct column name
        tokenized_inputs['labels'] = [1 if label == 'positive' else 0 for label in batch['Sentiment']]

        return tokenized_inputs


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenize_end_time = time.time()
    print("Tokenization completed in {:.2f} seconds".format(tokenize_end_time - tokenize_start_time))
    
    # Load the model
    print("Loading model...")
    model_loading_start_time = time.time()
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model_loading_end_time = time.time()
    print("Model loaded in {:.2f} seconds".format(model_loading_end_time - model_loading_start_time))
    
    data_loader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
    # Define training arguments
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training
    print("Starting training...")
    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    print("Training completed in {:.2f} seconds".format(training_end_time - training_start_time))

    # Save the model
    print("Saving the model...")
    save_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/roberta/yelp_roberta_finetuned'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    total_time = time.time() - start_time
    print("Total process completed in {:.2f} seconds".format(total_time))

    return model




