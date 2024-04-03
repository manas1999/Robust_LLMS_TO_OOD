import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_model(test_texts_ood, y_test_ood, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    def tokenize_texts(texts):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    test_dataset = tokenize_texts(test_texts_ood)
    test_dataset = {k: v.to(device) for k, v in test_dataset.items()}

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

def infer(model, tokenizer , prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Loaded the model\n")
     # Check if tokenizer has a pad_token if not, set it to eos_token this is to remove the warning 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    encoded_input = tokenizer.encode_plus(prompt, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)#,pad_token_id=pad_token_id)
    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("Generated Text:", decoded_output)
