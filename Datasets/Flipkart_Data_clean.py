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

    df = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_product_reviews.csv')
    df['cleaned_review'] = df['Review'].apply(clean_text)
    

    df['input_text'] = df.apply(lambda x: f"Prompt: Given this review, is the sentiment positive or negative?\nReview : {x['Review']}\nsummary : {x['Summary']}\nSentiment:", axis=1)
    


    # Select the combined text with sentiment for fine-tuning
    #input_data_for_finetuning = df['input_text'].tolist()

    # save the cleaned to a new CSV file
    df['input_text'].to_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_prompts.csv', index=False)
    df['Sentiment'].to_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_labels.csv', index=False)
    df['cleaned_review'].to_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/cleaned_reviews.csv', index=False)
    
    return

def data_loader():
    reviews = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_reviews.csv', index=False)
    labels =  pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_labels.csv', index=False)
    prompts = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/flipkart_prompts.csv', index=False)
    return Dataset.from_pandas(pd.concat([prompts, labels], axis=1))


def yelp_data_loader():
    reviews = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_cleaned.csv')
    
    labels = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_sentiment.csv')
    
    dataset = pd.concat([reviews, labels], axis=1)
    return Dataset.from_pandas(dataset)


def yelp_data_processor():
    chunk_size = 10000  # Adjust chunk size based on your system's memory
    review_json_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_academic_dataset_review.json'
    business_json_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_academic_dataset_business.json'
    
    # Create an iterator to process data in chunks
    review_iter = pd.read_json(review_json_path, lines=True, chunksize=chunk_size)
    business_df = pd.read_json(business_json_path, lines=True)
    
    # We assume 'business_id' is a common key
    business_df.set_index('business_id', inplace=True)

    # Prepare CSV file paths
    prompts_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_prompts.csv'
    sentiment_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_sentiment.csv'
    cleaned_review_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_cleaned.csv'
    
    # Initialize CSV files, writing headers
    with open(prompts_csv_path, 'w') as f:
        f.write("prompts\n")
    with open(sentiment_csv_path, 'w') as f:
        f.write("Sentiment\n")
    with open(cleaned_review_csv_path, 'w') as f:
        f.write("cleaned_review\n")

    for df in review_iter:
        print("Processing a new chunk of reviews...")

        # Merge with business data
        df = df.join(business_df, on='business_id', rsuffix='_business')
        
        # Clean the reviews
        df['cleaned_review'] = df['text'].apply(clean_text)
        
        # Label sentiment based on rating
        df['Sentiment'] = df['stars_business'].apply(lambda x: 'positive' if x > 2.5 else 'negative')
        
        # Create input text for training
        df['prompts'] = df.apply(lambda x: f"Prompt: Given this review, is the sentiment positive or negative?\nReview: {x['cleaned_review']}\nSentiment:", axis=1)
        
        # Save to CSV in chunks
        df['prompts'].to_csv(prompts_csv_path, mode='a', header=False, index=False)
        df['Sentiment'].to_csv(sentiment_csv_path, mode='a', header=False, index=False)
        df['cleaned_review'].to_csv(cleaned_review_csv_path, mode='a', header=False, index=False)

    print("All data processed.")

def imdb_data_processor():
    # Adjust the file path to your IMDb dataset location
    file_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_movie_reviews.csv'
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Clean the review texts
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Create input text for sentiment analysis in prompt format
    df['input_text'] = df.apply(lambda x: f"Prompt: Given this movie review, is the sentiment positive or negative?\nReview: {x['cleaned_review']}\nSentiment:", axis=1)

    # File paths for saving output
    prompts_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_prompts.csv'
    labels_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_labels.csv'
    cleaned_reviews_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_cleaned_reviews.csv'

    # Save the cleaned reviews, prompts, and labels to CSV files
    df['input_text'].to_csv(prompts_csv_path, index=False)
    df['sentiment'].to_csv(labels_csv_path, index=False)
    df['cleaned_review'].to_csv(cleaned_reviews_csv_path, index=False)
    
    print("IMDb data processing complete and saved to CSV files.")

def process_financial_data():
    # Path to the financial dataset CSV file
    file_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_data.csv'
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Clean the text data
    df['cleaned_text'] = df['Sentiment'].apply(clean_text)
    
    # Create input text for sentiment analysis in prompt format
    df['input_text'] = df.apply(lambda x: f"Prompt: Given this financial statement, is the sentiment positive, negative, or neutral?\nText: {x['cleaned_text']}\nSentiment:", axis=1)

    # Output file paths
    prompts_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_prompts.csv'
    labels_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_labels.csv'
    cleaned_texts_csv_path = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_cleaned_texts.csv'

    # Save processed data to CSV files
    df['input_text'].to_csv(prompts_csv_path, index=False)
    df['Sentiment'].to_csv(labels_csv_path, index=False)
    df['cleaned_text'].to_csv(cleaned_texts_csv_path, index=False)
    
    print("Financial sentiment data processed and saved.")



