import pandas as pd
import numpy as np
from data.dataLoader import data_loader
import time
import os
import requests

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

def inference(json, retries=3):
    for attempt in range(retries):
        try:
            res = requests.post(endpoint, json=json, headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"})
            res.raise_for_status()
            response_headers = res.headers
            prediction = res.json()['output']['choices'][0]['text']
            remaining_requests = response_headers.get('x-ratelimit-remaining', 'N/A')
            reset_time = int(response_headers.get('x-ratelimit-reset', 60))
            
            if remaining_requests == '0':
                print(f"Sleeping until rate limit resets in {reset_time} seconds.")
                time.sleep(reset_time + 0.5)

            return prediction, remaining_requests, reset_time
        except requests.exceptions.HTTPError as err:
            if res.status_code == 429:
                reset_time = int(response_headers.get('x-ratelimit-reset', 60))
                print(f"Rate limit exceeded. Retrying in {reset_time} seconds.")
                time.sleep(reset_time + 0.5)
            else:
                print(f"HTTP error occurred: {err}")
                return "HTTP error from API", "N/A", "N/A"
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Error from API", "N/A", "N/A"

    return "Failed after retries", "N/A", "N/A"

def process_batch(data_batch, model):
    prediction_df_rows = []
    for index, row in data_batch.iterrows():
        json = {
            'model': model,
            'prompt': row['zero_shot_prompt'],
            'request_type': 'language-model-inference',
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
        }
        predicted_text, remaining_requests, reset_time = inference(json)
        prediction_df_rows.append({
            'original_text': row['Text'], 
            'rewritten_text': predicted_text
        })
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} reviews.")


    prediction_data = pd.DataFrame(prediction_df_rows)
    return prediction_data

def rewrite_reviews(dataset_name, model_name):
    dataset, _ = data_loader.generic_data_loader(dataset_name)
    data = dataset.to_pandas().sample(n=1000, random_state=1)
    rewrite_prompt = (
        "Given the text of a review, rewrite the review to standardize its style, tone, and format. The rewritten review should be neutral in tone, concise, and should focus on the essential aspects of the product or service reviewed without personal anecdotes. Ensure that the language used is formal and consistent, suitable for a generic review platform. Adjust any colloquialisms, overly casual or overly formal language to these standards."
    )
    data['zero_shot_prompt'] = data['Text'].apply(lambda x: f"{rewrite_prompt}\nReview: {x}\nRewritten Review:")
    
    rewritten_data = process_batch(data, model_map[model_name])
    rewritten_data.to_csv(f'./rewritten_datasets/{dataset_name}_rewritten.csv', index=False)
    print(f"Rewritten data saved to {dataset_name}_rewritten.csv")

def run_rewriting_on_datasets():
    datasets = ['sst5','amazon']
    for dataset in datasets:
        rewrite_reviews(dataset, 'llama_70b')
        print(f"Completed rewriting for {dataset}.")

model_map = {
    "llama_70b": 'meta-llama/Llama-2-70b-chat-hf'
}


