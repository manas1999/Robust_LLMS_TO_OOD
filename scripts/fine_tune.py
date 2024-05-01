from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset,AutoModelForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import logging
import time

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

    # Setup the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer from pretrained
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    unique_labels = dataset['train']['Label'].unique()
    num_labels = len(unique_labels)
    print(f"Detected {num_labels} unique labels.")
    
    # Tokenize dataset function adjusted for batch processing
    print("Tokenizing dataset...")
    tokenize_start_time = time.time()
    
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
    

    tokenize_end_time = time.time()
    print(f"Tokenization completed in {tokenize_end_time - tokenize_start_time:.2f} seconds")
    
    # Load the model and move it to the correct device
    print("Loading model...")
    model_loading_start_time = time.time()
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels).to(device)
    model_loading_end_time = time.time()
    print(f"Model loaded in {model_loading_end_time - model_loading_start_time:.2f} seconds")
    
    # Define training arguments
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/roberta/results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
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
        eval_dataset=tokenized_eval_dataset
    )

    # Start the training process
    print("Starting training...")
    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    print(f"Training completed in {training_end_time - training_start_time:.2f} seconds")

    # Save the model and tokenizer
    print("Saving the model...")
    save_path = '../outputs/models/roberta_yelp'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    total_time = time.time() - start_time
    print(f"Total process completed in {total_time:.2f} seconds")

    return model





