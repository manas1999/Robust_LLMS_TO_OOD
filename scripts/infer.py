import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from langchain_openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv


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
        predicted_value = float(generated_text.split()[-1])  
        predicted_values.append(predicted_value)

    predicted_values = np.array(predicted_values)
    true_values = np.squeeze(y_test_ood)

    mae = np.mean(np.abs(predicted_values - true_values))
    print(f"Mean Absolute Error on Test Data: {mae}")

def infer(model, tokenizer , prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Loaded the model\n")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    encoded_input = tokenizer.encode_plus(prompt, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)    
    print("Generated Text:", decoded_output)

def infer_gpt(msg):
    load_dotenv()
    openai.api_key = ''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "user", "content": msg}
            ]
        )
        print(response['choices'][0]['message']['content'].strip())  
    except Exception as e:
        print(f"An error occurred: {e}")


def infer_langchain(prompt):
    llm = OpenAI(api_key="")
    response = llm(prompt)
    print(response)


def infer_hf(prompt):
    hf_api_key = ""
    generator = pipeline(model="gpt2", use_auth_token=hf_api_key)
    response = generator(prompt, max_length=50)
    print(response[0]['generated_text'])
