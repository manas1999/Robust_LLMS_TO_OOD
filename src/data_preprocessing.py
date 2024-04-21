import pandas as pd
import re
from pathlib import Path

from sklearn.model_selection import train_test_split


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

def encode_labels(label):
    mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
    return mapping[label]

def data_processor(file_path, prompt_template, output_dir, split_data=False):
    # Read the dataset
    df = pd.read_csv(file_path)

    # getting headers
    header1 = df.columns[0]
    header2 = df.columns[1]
    df = df.rename(columns={header2: 'sentiment'})

    # Convert 'positive'/'neutral'/'negative' to 0,1,2
    df['sentiment'] = df['sentiment'].apply(encode_labels)

    # Clean the text data
    df['cleaned_text'] = df[header1].apply(clean_text)

    # Create input text for sentiment analysis in prompt format
    df['input_text'] = df.apply(lambda x: prompt_template.format(x['cleaned_text']), axis=1)

    if split_data:
        # Split the data into training and validation sets
        train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

        # Output paths for saving
        train_prompts_path = Path(output_dir) / 'train_prompts.csv'
        train_labels_path = Path(output_dir) / 'train_labels.csv'
        train_cleaned_text_path = Path(output_dir) / 'train_cleaned_text.csv'

        validation_prompts_path = Path(output_dir) / 'validation_prompts.csv'
        validation_labels_path = Path(output_dir) / 'validation_labels.csv'
        validation_cleaned_text_path = Path(output_dir) / 'validation_cleaned_text.csv'

        # Save the train set
        train_df['input_text'].to_csv(train_prompts_path, index=False)
        train_df['sentiment'].to_csv(train_labels_path, index=False)
        train_df['cleaned_text'].to_csv(train_cleaned_text_path, index=False)

        # Save the validation set
        validation_df['input_text'].to_csv(validation_prompts_path, index=False)
        validation_df['sentiment'].to_csv(validation_labels_path, index=False)
        validation_df['cleaned_text'].to_csv(validation_cleaned_text_path, index=False)

        print(f"Training and validation data saved in {output_dir}.")
    else:
        # Paths for saving output
        prompts_path = Path(output_dir) / 'prompts.csv'
        labels_path = Path(output_dir) / 'labels.csv'
        cleaned_text_path = Path(output_dir) / 'cleaned_text.csv'

        # Save the full set
        df['input_text'].to_csv(prompts_path, index=False)
        df['sentiment'].to_csv(labels_path, index=False)
        df['cleaned_text'].to_csv(cleaned_text_path, index=False)

        print(f"Full data set saved in {output_dir}.")


if __name__ == "__main__":
    # IMDB DATA
    imdb_prompt_template = "Prompt: Based on the movie review provided below, is the sentiment positive or negative?\n" \
                           "Review: {}\n" \
                           "Sentiment:"
    imdb_file_path = '../data/raw/imdb_movie_reviews.csv'  # Update this path
    process_imdb_output_dir = '../data/processed/imdb'  # Update this path

    # FINANCIAL DATA
    financial_prompt_template = "Prompt: Considering this financial statement, would you classify the sentiment as positive, negative, or neutral?\n" \
                                "Text: {}\n" \
                                "Sentiment:"
    financial_file_path = '../data/raw/financial_dataset.csv'  # Update this path
    process_financial_output_dir = '../data/processed/financial'  # Update this path

    # Call data_processor with split_data as True for IMDb dataset to split it
    data_processor(imdb_file_path, imdb_prompt_template, process_imdb_output_dir, split_data=True)

    # For financial data, we may not split it if it's only used for testing
    data_processor(financial_file_path, financial_prompt_template, process_financial_output_dir, split_data=False)
