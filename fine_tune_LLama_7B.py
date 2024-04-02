import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model 

def generate_synthetic_data(n_samples, n_features, shift_intensity=0, sigma_noise=1):
    cov_matrix = np.eye(n_features)
    X_train = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov_matrix, size=n_samples)
    beta = np.random.normal(0, 1, size=(n_features, 1))
    noise_train = np.random.normal(0, sigma_noise, size=(n_samples, 1))
    y_train = X_train @ beta + noise_train
    
    mean_shift = np.full(n_features, shift_intensity)
    X_test = np.random.multivariate_normal(mean=mean_shift, cov=cov_matrix, size=n_samples)
    noise_test = np.random.normal(0, sigma_noise, size=(n_samples, 1))
    y_test = X_test @ beta + noise_test
    
    return X_train, y_train, X_test, y_test

def data_to_text(X, y):
    texts = []
    for features, target in zip(X, y):
        text = f"Given the features: {' '.join([f'x_{i+1}={feature}' for i, feature in enumerate(features)])}, the output value is {target[0]:.2f}."
        texts.append(text)
    return texts

def fine_tune_llama(dataset_path, model_name="chargoddard/Yi-34B-Llama", output_dir="./llama_finetuned", epochs=3):
    api_access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_access_token,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_access_token)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

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
def fine_tune_with_qlora(dataset_path, model_id="chargoddard/Yi-34B-Llama", output_dir="./qlora_finetuned", epochs=3):
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

def test_model(test_texts_ood, y_test_ood, model_name="./llama_finetuned"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cuda")

    def tokenize_texts(texts):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    test_dataset = tokenize_texts(test_texts_ood)
    test_dataset = {k: v.to("cuda") for k, v in test_dataset.items()}

    predicted_values = []
    for i in range(len(test_texts_ood)):
        input_ids = test_dataset["input_ids"][i:i+1]
        attention_mask = test_dataset["attention_mask"][i:i+1]

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Here, you need to extract the numeric prediction from generated_text
        # Implement this extraction based on your model's output format
        predicted_value = float(generated_text.split()[-1])  # Placeholder extraction
        predicted_values.append(predicted_value)

    predicted_values = np.array(predicted_values)
    true_values = np.squeeze(y_test_ood)

    mae = np.mean(np.abs(predicted_values - true_values))
    print(f"Mean Absolute Error on Test Data: {mae}")

def main():
    X_train, y_train, X_test_ood, y_test_ood = generate_synthetic_data(n_samples=1000, n_features=10, shift_intensity=5, sigma_noise=1)
    train_texts = data_to_text(X_train, y_train)
    test_texts_ood = data_to_text(X_test_ood, y_test_ood)

    train_texts_str = "\n".join(train_texts)
    test_texts_ood_str = "\n".join(test_texts_ood)
    with open("train_dataset.txt", "w") as f:
        f.write(train_texts_str)
    
    fine_tune_with_qlora("train_dataset.txt")
    test_model(test_texts_ood, y_test_ood)

if __name__ == "__main__":
    main()
