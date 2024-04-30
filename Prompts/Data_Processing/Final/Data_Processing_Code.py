import pandas as pd
from datasets import Dataset
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove Emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def flipkart_datacleaner_save():

    df = pd.read_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_product_reviews.csv', encoding='latin1')
    df['cleaned_review'] = df['Review'].apply(clean_text)
    

    df['zero_shot_prompt'] = df.apply(lambda x: f"Prompt: Given this summary, is the sentiment positive, negative, or neutral? Your answer must be in a single word which should be either positive or negative or neutral only.\nsummary : {x['Summary']}\nSentiment:", axis=1) #Review : {x['Review']}
    print("df columns", df.columns)
    df = df[['Review', 'cleaned_review', 'Summary', 'Sentiment', 'zero_shot_prompt']]

    # save the cleaned to a new CSV file
    df.to_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipcart_processed_data.csv', index=False)

    df['zero_shot_prompt'].to_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_reviews.csv', index=False)
    df['Sentiment'].to_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_labels.csv', index=False)
    df['cleaned_review'].to_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/cleaned_reviews.csv', index=False)
    df_prompt_review = df[['zero_shot_prompt','Sentiment']]
    df_prompt_review.to_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Zero_shot/flipkart_prompts_sentiment.csv', index=False)
    # print(df_prompt_review['Sentiment'].unique())
    
    return

def data_loader():
    reviews = pd.read_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_reviews.csv', encoding='latin1')
    labels =  pd.read_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_labels.csv', encoding='latin1')
    prompts = pd.read_csv('/Users/priyayarrabolu/Desktop/Priya/Spring_24/685/Project/Datasets/Prompts/Data_Processing/flipkart/flipkart_prompts.csv', encoding='latin1')
    return Dataset.from_pandas(pd.concat([prompts, labels], axis=1))


    
flipkart_datacleaner_save()
dataset = data_loader()






