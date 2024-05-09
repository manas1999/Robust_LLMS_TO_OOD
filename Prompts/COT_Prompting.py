import pandas as pd
import numpy as np
import requests
import time
from data.dataLoader import data_loader

# API configuration
endpoint = "https://api.together.xyz/inference"
TOGETHER_API_KEY = "76d1f3828a741254dd8bbd827864a95196c1845ff2dca061114700f6cd895952"
model_map = {
    "gemma_2b": "google/gemma-2b-it",
    "phi_2": "microsoft/phi-2",
    "llama_2b_it": "togethercomputer/Llama-2-7B-32K-Instruct",
    "Mistral": "mistralai/Mistral-7B-v0.1",  # not instruct
    "Gemma": "google/gemma-7b",  # not instruct
    "llama_70b": 'meta-llama/Llama-2-70b-chat-hf'
}

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
        prompt = row['CoT_prompt']
        json = {
            'model': model,
            'prompt': "",
            'request_type': 'language-model-inference',
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
            'negative_prompt': '',
            'messages': [{'content': prompt, 'role': 'user'}],
            'prompt_format_string': '<human>: {prompt}\n'
        }
        prediction, remaining_requests, remaining_seconds = inference(json)
        prediction_df_rows.append({'prompt': prompt, 'prediction': prediction, 'actual_label': row['actual_label']})
        time.sleep(1)  # Rate limit handling

    prediction_data = pd.DataFrame(prediction_df_rows)
    return prediction_data

def format_prediction(text):
    reason = ""
    parts = text.split('Explanation:')
    sentiment_part = parts[0]
    if len(parts) > 1:
        reason = parts[1].strip()
    sentiment = sentiment_part.split('Sentiment:')[-1].strip()
    return sentiment, reason

def CoT(dataset_name, model_name):
    _, test_dataset = data_loader.generic_data_loader(dataset_name)
    data = test_dataset.to_pandas()

    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    data['actual_label'] = data['Label'].map(label_map)

    prompt = ("Analyze the sentiment of the below review. Provide your sentiment prediction "
              "(positive or negative or neutral) and a concise explanation for your judgment "
              "in only one sentence. Expected Output Format:\nSentiment: <prediction>\nExplanation: <explanation>")
    data['CoT_prompt'] = data.apply(lambda x: f"Prompt: {prompt}\nReview : {x['Text']}", axis=1)

    prediction_data = process_batch(data, model_map[model_name])
    prediction_data[['predicted_label', 'reason']] = prediction_data['prediction'].apply(format_prediction).apply(pd.Series)

    prediction_data['predicted_label'] = prediction_data['predicted_label'].str.lower().str.strip()
    prediction_data['actual_label'] = prediction_data['actual_label'].str.lower().str.strip()
    prediction_data['Match'] = np.where(prediction_data['predicted_label'] == prediction_data['actual_label'], 1, 0)
    accuracy = prediction_data['Match'].sum() / prediction_data.shape[0]
    print(f"Accuracy of the model on {dataset_name}: {accuracy:.2%}")

    # Save the results including the reason for the prediction
    results_path = f'./Prompts/results/COT_detailed_results_{dataset_name}.csv'
    prediction_data.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return accuracy

def run_CoT_on_all_datasets(model_name):
    datasets = ['amazon_subsample', 'dynasent_subsample', 'sst5_subsample', 'semeval_subsample']
    results = []
    
    for dataset in datasets:
        accuracy = CoT(dataset, model_name)
        results.append({'Dataset': dataset, 'Accuracy': accuracy})
        print(f"Completed {dataset} with accuracy: {accuracy:.2%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('./Prompts/results/COT_summary_results.csv', index=False)
    print("Overall results saved to COT_summary_results.csv")
    
    return results_df
