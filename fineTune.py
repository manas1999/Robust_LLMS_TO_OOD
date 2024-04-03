from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset
from peft import LoraConfig, get_peft_model



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
    # Explicitly trust the remote code by setting trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load the model and apply QLoRA configuration
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)  
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Load the dataset
    from datasets import load_dataset
    data = load_dataset('text', data_files={'train': dataset_path})
    tokenized_datasets = data.map(tokenize_function, batched=True)
    
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
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
        train_dataset=tokenized_datasets["train"],
    )
    
    trainer.train()



def full_finetune(dataset_path, model, tokenizer, output_dir="./full_finetuned", epochs=3):
    # Load the dataset
    dataset = load_dataset('text', data_files={'train': dataset_path})
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
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
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )
    
    # Start fine-tuning
    trainer.train()
