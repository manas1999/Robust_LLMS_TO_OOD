import os
import pandas as pd
import numpy as np
import time
import requests

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = '76d1f3828a741254dd8bbd827864a95196c1845ff2dca061114700f6cd895952'
model_name = "llama_70b"
model_type = '70B'

def inference(json):
    res = requests.post(endpoint, json=json, headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}"
    })
    try:
        res.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        response_headers = res.headers
        prediction = res.json()['output']['choices'][0]['text']
        remaining_requests = response_headers['x-ratelimit-remaining']
        remaining_seconds = response_headers['x-ratelimit-reset']
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        prediction = "HTTP error from API"
        remaining_requests = "N/A"
        remaining_seconds = "N/A"
    except Exception as e:
        print(f"An error occurred: {e}")
        prediction = "Error from API"
        remaining_requests = "N/A"
        remaining_seconds = "N/A"

    return prediction, remaining_requests, remaining_seconds

def process_batch(data_batch, batch_num):
    prediction_df_rows = []
    for index, row in data_batch.iterrows():
        prompt = row['zero_shot_prompt']
        json = {
            'model': 'meta-llama/Llama-2-70b-chat-hf',  
            'prompt': "",
            'request_type': 'language-model-inference',
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
            'negative_prompt': '',
            'messages': [
                {
                    'content': prompt,
                    'role': 'user'
                }
            ],
            'prompt_format_string': '<human>: {prompt}\n'
        }
        predicted_label, remaining_requests, remaining_seconds = inference(json)
        print("Prompt:", prompt)
        print("Predicted Label:", predicted_label)
        print('-' * 50)
        prediction_df_rows.append({'prompt': prompt, 'predicted_label': predicted_label, 'actual_label': row['Sentiment']})
        time.sleep(1)  # Rate limit handling

    if len(prediction_df_rows) == 0:
        print("No data processed for batch", batch_num)
    else:
        prediction_csv = pd.DataFrame(prediction_df_rows)
        csv_file_path = f'Predictions/{model_type}/predictions_batch_{model_name}_{batch_num}.csv'
        prediction_csv.to_csv(csv_file_path, index=False)
        return csv_file_path

if __name__ == '__main__':
    data = pd.read_csv('Zero_shot/flipkart_prompts_sentiment.csv')
    # data = data.head(100)  # For testing purposes, remove this line for full processing

    chunk_size = 10000
    num_chunks = len(data) // chunk_size + 1

    batch_files = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        print(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx}-{end_idx})...")
        
        data_chunk = data[start_idx:end_idx]
        file_path = process_batch(data_chunk, i+1)
        if file_path:
            print("data chunk shape:", data_chunk.shape)
            batch_files.append(file_path)
        else:
            print(f"No data processed for batch {i+1}")
          
    valid_batch_files = [file for file in batch_files if os.path.exists(file)]
    if valid_batch_files:
        final_df = pd.concat([pd.read_csv(file) for file in valid_batch_files], ignore_index=True)
        final_df.to_csv(f'Predictions/{model_type}/final_predictions_{model_name}.csv', index=False)
    else:
        print("No valid CSV files to concatenate.")

   
## Calculate accuracy of the model 
data_final = pd.read_csv(f'Predictions/{model_type}/final_predictions_{model_name}.csv')
data_final['predicted_label'] = data_final['predicted_label'].str.lower().str.strip()
data_final['actual_label'] = data_final['actual_label'].str.lower().str.strip()
data_final['Match'] = np.where(data_final['predicted_label'] == data_final['actual_label'],1,0)
print("data_final['Match'].sum()", data_final['Match'].sum())
print("data_final.shape[0]", data_final.shape[0])
print("match dataset", data_final)
accuracy = data_final['Match'].sum()/data_final.shape[0]
print("accuracy of the model:" , accuracy)