import pandas as pd
from datasets import Dataset


def yelp_data_loader():
    reviews = pd.read_csv('./data/raw/yelp_reviews_cleaned.csv')

    labels = pd.read_csv('./data/raw/yelp_reviews_sentiment.csv')

    dataset = pd.concat([reviews, labels], axis=1)
    return Dataset.from_pandas(dataset)