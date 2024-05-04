import pandas as pd
import numpy as np
from data.dataLoader import data_loader
import time
import requests

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = '76d1f3828a741254dd8bbd827864a95196c1845ff2dca061114700f6cd895952'
model_map = {"gemma_2b": "google/gemma-2b-it",
             "phi_2":"microsoft/phi-2",
             "llama_2b_it":"togethercomputer/Llama-2-7B-32K-Instruct",
             "Mistral":"mistralai/Mistral-7B-v0.1", ## not instruct
             "Gemma":"google/gemma-7b",  ## not instruct
            #  "Mistral_70B":"",
             "llama_70b":'meta-llama/Llama-2-70b-chat-hf'
             }


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

def process_batch(data_batch, model):
        prediction_df_rows = []
        for index, row in data_batch.iterrows():
            prompt = row['zero_shot_prompt']
            json = {
                 ## need to change the parameters
                'model': model,  
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
                ## change this complsory 
                'prompt_format_string': '<human>: {prompt}\n'
            }
            predicted_label, remaining_requests, remaining_seconds = inference(json)
            print("Prompt:", prompt)
            print("Predicted Label:", predicted_label)
            print('-' * 50)
            prediction_df_rows.append({'prompt': prompt, 'predicted_label': predicted_label, 'actual_label': row['actual_label']})
            time.sleep(1)  # Rate limit handling

        prediction_data = pd.DataFrame(prediction_df_rows)
        return prediction_data
        



def main_zero_shot_fucntion(dataset_name, model_name):
    ## loading the dataset 
    _, test_dataset = data_loader.generic_data_loader(dataset_name)

    ## converting to pandas dataframe 
    data = test_dataset.to_pandas()

    ## changing labels column format 
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    data['actual_label'] = data['Label'].map(label_map)

    ## creating prompt column in the dataframe 
    prompt = "For sentiment analysis: Your task is to perform a sentiment analysis on a given input text and provide a single word indicating whether the sentiment is positive, negative, or neutral. The input text may contain any language or style of writing. Please ensure that your analysis takes into account the overall tone and context of the text.Your response should be concise and clear, providing a single word that accurately reflects the sentiment of the input text. If there are multiple sentiments present in the text, please choose the one that best represents the overall feeling conveyed by the author.Please note that your analysis should take into account all relevant factors, such as tone, language use, and content. Your response should also be flexible enough to allow for various types of input texts."
    data['zero_shot_prompt'] = data.apply(lambda x: f"Prompt: {prompt}\nsummary : {x['Text']}\nSentiment:", axis=1) #Review : {x['Review']}
    

    prediction_data = process_batch(data, model_map[model_name])
    prediction_data['predicted_label'] = prediction_data['predicted_label'].str.lower().str.strip()
    prediction_data['actual_label'] = prediction_data['actual_label'].str.lower().str.strip()
    prediction_data['Match'] = np.where(prediction_data['predicted_label'] == prediction_data['actual_label'],1,0)
    print("prediction_data['Match'].sum()", prediction_data['Match'].sum())
    print("data_final.shape[0]", prediction_data.shape[0])
    accuracy = prediction_data['Match'].sum()/prediction_data.shape[0]
    print("accuracy of the model:" , accuracy)

    