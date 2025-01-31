import pandas as pd
import numpy as np
import time
import os
import requests
from data.dataLoader import data_loader

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
model_map = { 
    "gemma_2b": {"model": "google/gemma-2b-it", "prompt_format_string": "<human>: {prompt}\n<bot>:", "type":"chat"},
    "llama_70b": {"model": 'meta-llama/Llama-2-70b-chat-hf', "prompt_format_string": "[INST]  {prompt}\n [/INST]", "type":"chat"},
    "llama_8b_it": {"model": 'meta-llama/Llama-3-8b-chat-hf', "prompt_format_string": "<human>: {prompt}\n<bot>:", "type":"chat"}}

def inference(json, retries=3):
    response_headers = {}  
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
            if 'res' in locals():
                response_headers = res.headers
                if res.status_code == 429:
                    reset_time = int(response_headers.get('x-ratelimit-reset', 60))
                    print(f"Rate limit exceeded. Retrying in {reset_time} seconds.")
                    time.sleep(reset_time + 0.5)
                else:
                    print(f"HTTP error occurred: {err}")
                    return "HTTP error from API", "N/A", "N/A"
            else:
                print(f"HTTP error occurred before response was received: {err}")
                return "HTTP error from API", "N/A", "N/A"
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Error from API", "N/A", "N/A"

    return "Failed after retries", "N/A", "N/A"

def process_batch(data_batch, model_map, model_name):
    prediction_df_rows = []
    for index, row in data_batch.iterrows():
        prompt = row['zero_shot_prompt']
        json = {
            'model': model_map[model_name]['model'],
            'prompt': "",
            'request_type': 'language-model-inference',
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
            'negative_prompt': '',
            "type": model_map[model_name]['type'],
            'messages': [{'content': prompt, 'role': 'user'}],
            'prompt_format_string': model_map[model_name]['prompt_format_string']
        }
        predicted_label, remaining_requests, remaining_seconds = inference(json)
        prediction_df_rows.append({'prompt': prompt, 'predicted_label': predicted_label, 'actual_label': row['actual_label']})
<<<<<<< Updated upstream
        time.sleep(1)  
=======
        # Print progress after every 100 data points
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} data points.")
        time.sleep(1)  # Rate limit handling
>>>>>>> Stashed changes

    prediction_data = pd.DataFrame(prediction_df_rows)
    return prediction_data

def reformulate_inputs(data_batch, model_name, id_examples):
    id_examples_samples = id_examples['Text'].sample(n=20, random_state=5).tolist()
    id_examples_str = "\n".join(id_examples_samples)
    reformulated_inputs = []
    for _, row in data_batch.iterrows():
        input_text = row['Text']
        prompt = f"I have few examples that follow a particular style and now I want you to paraphrase a given input text such that it matches the style of the provided examples. Here are the examples: \n\n{id_examples_str}\nNow paraphrase {input_text} and provide only the paraphrased text without any additional information. Note: it is very important that you only provide the final output without any additional comments or remarks."
        json = {
            'model': model_map[model_name]['model'],
            'prompt': "",
            'request_type': 'language-model-inference',
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'repetition_penalty': 1,
            'negative_prompt': '',
            "type": model_map[model_name]['type'],
            'messages': [{'content': prompt, 'role': 'user'}],
            'prompt_format_string': model_map[model_name]['prompt_format_string']
        }
        reformulated_input, remaining_requests, remaining_seconds = inference(json)
        reformulated_inputs.append({'original_input': input_text, 'reformulated_input': reformulated_input})
    reformulated_data = pd.DataFrame(reformulated_inputs)
    return reformulated_data


def get_accuracy_with_reformulated_inputs(reformulated_data, model_name):
    prompt = ("For sentiment analysis: Your task is to perform a sentiment analysis on a given input text and "
              "provide a single word indicating whether the sentiment is positive, negative, or neutral. The input text "
              "may contain any language or style of writing. Please ensure that your analysis takes into account the overall "
              "tone and context of the text. Your response should be concise and clear, providing a single word that accurately "
              "reflects the sentiment of the input text. If there are multiple sentiments present in the text, please choose "
              "the one that best represents the overall feeling conveyed by the author. Please note that your analysis should "
              "take into account all relevant factors, such as tone, language use, and content. Your response should also be "
              "flexible enough to allow for various types of input texts.")
    reformulated_data['zero_shot_prompt'] = reformulated_data.apply(lambda x: f"Prompt: {prompt}\nsummary : {x['reformulated_input']}\nSentiment:", axis=1)
    prediction_data = process_batch(reformulated_data, model_map, model_name)
    prediction_data['predicted_label'] = prediction_data['predicted_label'].str.lower().str.strip()
    prediction_data['actual_label'] = prediction_data['actual_label'].str.lower().str.strip()
    prediction_data['Match'] = np.where(prediction_data['predicted_label'] == prediction_data['actual_label'], 1, 0)
    prediction_data['rewritten'] = reformulated_data['reformulated_input']
    prediction_data['original_input'] = reformulated_data['original_input']
    print("prediction_data cols:", prediction_data.columns)
    print("prediction_data head:", prediction_data.head(2))
    accuracy = prediction_data['Match'].sum() / prediction_data.shape[0]
    print(f"Accuracy of the model on reformulated inputs: {accuracy:.2%}")

    return accuracy, prediction_data

def main_reformulation_function(dataset_name, model_name_L1, model_name_L2, data_id):
    _, test_dataset_ood = data_loader.generic_data_loader(dataset_name)
    train_dataset_id, _ = data_loader.generic_data_loader(data_id)
    
    data_ood = test_dataset_ood.to_pandas()
    data_id = train_dataset_id.to_pandas()
    
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    data_ood['actual_label'] = data_ood['Label'].map(label_map)
    data_id['actual_label'] = data_id['Label'].map(label_map)
    
    reformulated_data = reformulate_inputs(data_ood, model_name_L2, data_id)
    reformulated_data['actual_label'] = data_ood['actual_label']

    accuracy, prediction_data = get_accuracy_with_reformulated_inputs(reformulated_data, model_name_L1 ) 

    results_path = f'./Prompts/results/{model_name_L1}_reformulated_results_{dataset_name}.csv'
    prediction_data.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return accuracy, prediction_data

def run_reformulation_on_all_datasets(model_name_L1, model_name_L2):
    datasets = ['amazon_subsample','dynasent_subsample', 'sst5_subsample', 'semeval_subsample']
    results = []
    
    for dataset in datasets:
        accuracy, prediction_data = main_reformulation_function(dataset, model_name_L1, model_name_L2, "amazon")
        results.append({'Dataset': dataset, 'Accuracy': accuracy})
        print(f"Completed {dataset} with accuracy: {accuracy:.2%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./Prompts/results/{model_name_L1}_{model_name_L2}_reformulation_overall_results.csv', index=False)
    print(f"Overall results saved to ./Prompts/results/{model_name_L1}_{model_name_L2}_reformulation_overall_results.csv")
    
    return results_df
