from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,TextDataset,AutoModelForSequenceClassification
from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments,GPT2LMHeadModel, GPT2Tokenizer, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import wandb
from transformers.integrations import WandbCallback
from datasets import load_dataset, DatasetDict
import os


def finetune_roberta(dataset,batch_size):
    wandb.login(key= os.getenv("WAND_KEY"))
    wandb.init(project="Robust_LLM", name = "Roberta_Finetuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    print("Tokenizing dataset...")
    
    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        labels = [ label for label in batch['Label']]
        tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    dataset_dict = dataset.train_test_split(test_size=0.1)
    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/roberta/results',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    trainer.train()

    print("Saving the model...")
    save_path = '../outputs/models/roberta_amazon'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    artifact = wandb.Artifact('roberta_model', type='model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)
    return model

def finetune_t5(dataset,batch_size):
    wandb.login(key=os.getenv("WAND_KEY"))
    wandb.init(project="Robust_LLM", name="T5_Finetuning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    print("Tokenizing dataset...")

    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        label_texts = ["positive" if label == 1 else "negative" if label == 0 else "neutral" for label in batch['Label']]

        inputs = [f"classify: {text}" for text in texts]
        outputs = [f"{label}" for label in label_texts]
        
        tokenized_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokenized_labels = tokenizer(
            outputs,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    dataset_dict = dataset.train_test_split(test_size=0.1)

    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/t5/results',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    trainer.train()

    print("Saving the model...")
    save_path = '../outputs/models/t5_sentiment'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    artifact = wandb.Artifact('roberta_model', type='model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)

    return model


def finetune_gpt2(dataset,batch_size):
    wandb.login(key=os.getenv("WAND_KEY"))
    wandb.init(project="Robust_LLM", name="GPT2_Finetuning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  

    print("Tokenizing dataset...")

    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        label_texts = ["positive" if label == 1 else "negative" if label == 0 else "neutral" for label in batch['Label']]

        input_texts = [f"Text: {text} | Sentiment:" for text in texts]
        output_texts = [f" {label}" for label in label_texts]

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

        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    dataset_dict = dataset.train_test_split(test_size=0.1)

    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/gpt2/results',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=default_data_collator, 
        callbacks=[WandbCallback()],
    )

    trainer.train()

    print("Saving the model...")
    save_path = '../outputs/models/gpt2_sentiment'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    artifact = wandb.Artifact('roberta_model', type='model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)

    return model


def finetune_bert(dataset, batch_size):
    wandb.login(key=os.getenv("WAND_KEY"))
    wandb.init(project="Robust_LLM", name="BERT_Finetuning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print("Tokenizing dataset...")

    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        labels = [label for label in batch['Label']]
        
        tokenized_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokenized_inputs['labels'] = torch.tensor(labels)  
        return tokenized_inputs

    dataset_dict = dataset.train_test_split(test_size=0.1)

    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3 
    ).to(device)

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/bert/results',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    trainer.train()

    print("Saving the model...")
    save_path = '../outputs/models/bert_finetuning'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    artifact = wandb.Artifact('bert_model', type='model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)

    return model

def finetune_deberta(dataset, batch_size):
    wandb.login(key=os.getenv("WAND_KEY"))
    wandb.init(project="Robust_LLM", name="DeBERTa_Finetuning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    print("Tokenizing dataset...")

    def tokenize_function(batch):
        texts = [text if text is not None else "" for text in batch['Text']]
        labels = [label for label in batch['Label']]
        
        tokenized_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokenized_inputs['labels'] = torch.tensor(labels) 
        return tokenized_inputs

    dataset_dict = dataset.train_test_split(test_size=0.1)

    tokenized_train_dataset = dataset_dict['train'].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset_dict['test'].map(tokenize_function, batched=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-v3-base',
        num_labels=3  
    ).to(device)

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir='../outputs/deberta/results',
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[WandbCallback()],
    )

    trainer.train()

    print("Saving the model...")
    save_path = '../outputs/models/deberta_finetuning'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    artifact = wandb.Artifact('deberta_model', type='model')
    artifact.add_dir(save_path)
    wandb.log_artifact(artifact)

    return model