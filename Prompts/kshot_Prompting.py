import pandas as pd
import numpy as np
import time
import requests
from collections import defaultdict

# Assuming data_loader is correctly imported and configured
from data.dataLoader import data_loader

endpoint = 'https://api.together.xyz/inference'
TOGETHER_API_KEY = '9b3a40a05619e5f1992938cc171800fe2140055652aa4524fc942ae4e1f89ff7'
model_map = {
    "gemma_2b": "google/gemma-2b-it",
    "phi_2": "microsoft/phi-2",
    "llama_2b_it": "togethercomputer/Llama-2-7B-32K-Instruct",
    "Mistral": "mistralai/Mistral-7B-v0.1",  # not instruct
    "Gemma": "google/gemma-7b",  # not instruct
    "llama_70b": 'meta-llama/Llama-2-70b-chat-hf',
    "llama_8b_it":'meta-llama/Llama-3-8b-chat-hf'
}

def inference(json, retries=3):
    response_headers = {}  # Ensure response_headers is always defined
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
        prediction_df_rows.append({'prompt': prompt, 'predicted_label': predicted_label, 'actual_label': row['actual_label']})
        time.sleep(1)  # Rate limit handling

    prediction_data = pd.DataFrame(prediction_df_rows)
    return prediction_data


def main_K_shot_function(dataset_name, model_name):
    _, test_dataset = data_loader.generic_data_loader(dataset_name)
    
    # Downsample the dataset
    data = test_dataset.to_pandas()
    
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    data['actual_label'] = data['Label'].map(label_map)

    prompt = ("For sentiment analysis: Your task is to perform a sentiment analysis on a given input text and "
              "provide a single word indicating whether the sentiment is positive, negative, or neutral. The input text "
              "may contain any language or style of writing. Please ensure that your analysis takes into account the overall "
              "tone and context of the text. Your response should be concise and clear, providing a single word that accurately "
              "reflects the sentiment of the input text. If there are multiple sentiments present in the text, please choose "
              "the one that best represents the overall feeling conveyed by the author. Please note that your analysis should "
              "take into account all relevant factors, such as tone, language use, and content. Your response should also be "
              "flexible enough to allow for various types of input texts.")
    data['zero_shot_prompt'] = data.apply(lambda x: f"Prompt: {prompt}\nsummary : {x['Text']}\nSentiment:", axis=1)

    text_and_sentiment = [
        ("One of my favorites", "positive"),
        ("Another I-cannot-figure-this-out book by John. Loved it,", "positive"),
        ("This is a nice shoe at a price, but one reason I bought it was that it pictured what appeared to be an adjustable strap across the top. Since I have a narrow foot, I always look for this feature because shoe companies seem to manufacture for medium width. Unfortunately, this one was for show only, so the shoe, while wearable, does not meet my five-star standard. It would have been very easy to make the strap so that it could have been taken up a bit for those of us with narrow feet.", "neutral"),
        ("I like the thick ones I get from RiteAid. They do the job. I'll use this cause I got em, after that it's back to RiteAid.", "neutral"),
        ("Very tiny I had to use a magnifying glass justcto be able to read what the stamp said.. Very disappointed.. A $1.00 stamp sold at a higher price just for the name..","negative"),
        ("Doesn't stick correctly and is not properly fitted for iPhone. I wouldn't recommend this product.","negative")
    ]
    def generate_specific_string(text_and_sentiment):
        specific_string = ""
        for text, sentiment in text_and_sentiment:
            specific_string += f"Prompt: {prompt}\nsummary : {text}\nSentiment: {sentiment}\n"
        return specific_string.strip()
    specific_string = generate_specific_string(text_and_sentiment)

    data['k_shot_prompt'] = specific_string + "\n" + data['zero_shot_prompt']
    prediction_data = process_batch(data, model_map[model_name])
    prediction_data['predicted_label'] = prediction_data['predicted_label'].str.lower().str.strip()
    prediction_data['actual_label'] = prediction_data['actual_label'].str.lower().str.strip()
    prediction_data['Match'] = np.where(prediction_data['predicted_label'] == prediction_data['actual_label'], 1, 0)

    # Calculate accuracy
    accuracy = prediction_data['Match'].sum() / prediction_data.shape[0]
    print(f"Accuracy of the model on {dataset_name}: {accuracy:.2%}")

    # Save the results
    results_path = f'./Prompts/results/{model_name}_results_6_k_shot_samples_{dataset_name}.csv'
    prediction_data.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return accuracy, prediction_data


def k_shot_run_sentiment_analysis_on_all_datasets_kshot(model_name):
    datasets = ['amazon_subsample', 'dynasent_subsample', 'sst5_subsample', 'semeval_subsample']
    results = []
    
    for dataset in datasets:
        accuracy, prediction_data = main_K_shot_function(dataset, model_name)
        exit
        results.append({'Dataset': dataset, 'Accuracy': accuracy})
        print(f"Completed {dataset} with accuracy: {accuracy:.2%}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'./Prompts/results/{model_name}_k_shot_overall_results_6_k_shot_samples.csv', index=False)
    print("Overall results saved to k_shot_sentiment_analysis_overall_results_6_k_shot_samples.csv")
    
    return results_df

