import pandas as pd
import numpy as np
import time
import requests
import re

# Assuming data_loader is correctly imported and configured
from data.dataLoader import data_loader

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = '76d1f3828a741254dd8bbd827864a95196c1845ff2dca061114700f6cd895952'
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
        prompt = row['zero_shot_prompt']
        json = {
            'model': model,
            'prompt': "",
            'request_type': 'language-model-inference',
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
            'negative_prompt': '',
            'messages': [{'content': prompt, 'role': 'user'}],
            'prompt_format_string': '<human>: {prompt}\n'
        }
        predicted_label, remaining_requests, remaining_seconds = inference(json)
        #print(prompt)
        #print(predicted_label)
        prediction_df_rows.append({'prompt': prompt, 'predicted_label': predicted_label, 'actual_label': row['actual_label']})
        time.sleep(1)  # Rate limit handling

    prediction_data = pd.DataFrame(prediction_df_rows)
    return prediction_data

def main_abstain_function(dataset_name, model_name):
    _, test_dataset = data_loader.generic_data_loader(dataset_name)
    
    # Downsample the dataset
    data = test_dataset.to_pandas()
    #data = data.head(1)
    #data = pd.DataFrame([{'Text': " asonfoin safnioie fainfoiqen   ", 'Label': 'Neutral'}])
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    data['actual_label'] = data['Label'].map(label_map)
    
    # Creating the prompt for each sample in the dataset
    prompt = ('''For sentiment analysis: Your task is to analyze the sentiment of the given input text and provide a single word indicating the sentiment as either 'positive', 'negative', or 'neutral'. Consider the overall tone, context, and language used in the text to determine the sentiment accurately. If the text presents multiple sentiments, select the one that best represents the overall feeling conveyed by the author. 

Your response should be:
- 'positive' if the sentiment is clearly positive,
- 'negative' if the sentiment is clearly negative,
- 'neutral' if the sentiment is clearly neutral,
- '#Abstain' if the sentiment is mixed, unclear, or if you are unsure about the dominant sentiment.

Please ensure that your analysis is sensitive to different styles and languages of the input text, and be prepared to #Abstain if a clear sentiment cannot be determined.'''
)
    data['zero_shot_prompt'] = data.apply(lambda x: f"Prompt: {prompt}\nText: {x['Text']}\nSentiment:", axis=1)
    
    # Process the batch using the model
    prediction_data = process_batch(data, model_map[model_name])
    
    # Extract sentiment or handle abstain
    def extract_sentiment(text):
        # Look for specific sentiments or '#Abstain'
        pattern = re.compile(r"^\s*(positive|negative|neutral|#abstain|Positive|Negitive|Neutral|#Abstain)\s*$", re.IGNORECASE)
        match = pattern.search(text.strip())
        if match:
            return match.group(1).strip().lower()
        # If no direct match, look for '#Abstain' followed by an explanation
        if "abstain" or 'Abstain'  in text:
            return "Abstain"
        return None
    
     # Apply the extraction function to get predicted sentiment
    prediction_data['predicted_label'] = prediction_data['predicted_label'].apply(extract_sentiment)
    #print(prediction_data['predicted_label'])
    
    # Here we treat 'abstain' as being correct if the actual sentiment is unclear
    prediction_data['Match'] = prediction_data.apply(lambda x: 1 if (x['predicted_label'] == x['actual_label'] or x['predicted_label'] == 'abstain' or x['predicted_label'] == 'Abstain') else 0, axis=1)
    
    accuracy = prediction_data['Match'].sum() / len(prediction_data)
    
    print(f"Accuracy of the model on {dataset_name} (considering 'abstain' as correct): {accuracy:.2%}")
    
    
    # Save the results
    results_path = f'./Prompts/results/sentiment_analysis_results_with_abstainance_{dataset_name}.csv'
    prediction_data.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return accuracy, prediction_data


def abstain_sentiment_analysis_on_all_datasets(model_name):
    datasets = ['amazon_subsample', 'dynasent_subsample', 'sst5_subsample', 'semeval_subsample']
    results = []
    
    for dataset in datasets:
        accuracy, prediction_data = main_abstain_function(dataset, model_name)
        results.append({'Dataset': dataset, 'Accuracy': accuracy})
        print(f"Completed {dataset} with accuracy: {accuracy:.2%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('./Prompts/results/abstainance_sentiment_analysis_overall_results.csv', index=False)
    print("Overall results saved to abstainance_sentiment_analysis_overall_results.csv")
    
    return results_df