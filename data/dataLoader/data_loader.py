import pandas as pd
from datasets import Dataset


def yelp_data_loader():
    reviews = pd.read_csv('./data/raw/yelp_reviews_cleaned.csv')

    labels = pd.read_csv('./data/raw/yelp_reviews_sentiment.csv')

    dataset = pd.concat([reviews, labels], axis=1)
    return Dataset.from_pandas(dataset)

def generic_data_loader(dataset_name):
    # Assuming the file structure follows the format shown in the screenshot
    test_file_path = f'./data/processed/{dataset_name}/test.tsv'
    train_file_path = f'./data/processed/{dataset_name}/train.tsv'
    
    # Read the TSV files
    test_df = pd.read_csv(test_file_path, sep='\t')
    train_df = pd.read_csv(train_file_path, sep='\t')
    
    # Concatenate test and train data frames if necessary
    # Note: You may need to adjust this if test and train need to be kept separate
    dataset = pd.concat([train_df, test_df]).reset_index(drop=True)
    
    # Convert the pandas dataframe to a HuggingFace dataset
    hf_dataset = Dataset.from_pandas(dataset)
    return hf_dataset
