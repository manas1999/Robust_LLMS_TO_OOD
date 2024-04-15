from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from datasets import load_dataset
import time

def plot_embeddings():
    start_time = time.time()

    # Load the pretrained RoBERTa model
    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    end_time = time.time()
    print("Loaded the model in {:.2f} seconds".format(end_time - start_time))

    # Load and preprocess the dataset
    df = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/cleaned_reviews.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df))
    df_yelp = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_cleaned.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df_yelp))
    df_imdb = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_cleaned_reviews.csv', nrows=1000)
    df_finance = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_cleaned_texts.csv', nrows=1000)

    # Load SST-2 dataset
    sst2_dataset = load_dataset("stanfordnlp/sst2")
    sst2_sentences = [item['sentence'] for item in sst2_dataset['train']][:1000]  # Adjust size as needed

    
    sentences_dataset1 = df['cleaned_review'].astype(str).tolist()
    sentences_dataset2 = df_yelp['cleaned_review'].astype(str).tolist()
    sentences_dataset3 = df_imdb['cleaned_review'].astype(str).tolist()
    sentences_dataset4 = df_finance['cleaned_text'].astype(str).tolist()

    # Generate embeddings
    embeddings_dataset1 = model.encode(sentences_dataset1, show_progress_bar=True)
    embeddings_dataset2 = model.encode(sentences_dataset2, show_progress_bar=True)
    embeddings_sst2 = model.encode(sst2_sentences, show_progress_bar=True)
    embeddings_imdb = model.encode(sentences_dataset3, show_progress_bar=True)
    embeddings_finance = model.encode(sentences_dataset4, show_progress_bar=True)

    # Reduce the dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d_dataset1 = tsne.fit_transform(embeddings_dataset1)
    embeddings_2d_dataset2 = tsne.fit_transform(embeddings_dataset2)
    embeddings_2d_sst2 = tsne.fit_transform(embeddings_sst2)
    embeddings_2d_imdb = tsne.fit_transform(embeddings_imdb)
    embeddings_2d_finance = tsne.fit_transform(embeddings_finance)


    '''# Clustering Dataset 1 to find centers
    kmeans = KMeans(n_clusters=1, random_state=0).fit(embeddings_2d_dataset1)
    centers = kmeans.cluster_centers_

    # Calculate distances from each point in dataset 2 to the nearest center in dataset 1
    distances = np.min([np.linalg.norm(embeddings_2d_dataset2 - center, axis=1) for center in centers], axis=0)

    # Find indices of the 1000 farthest points
    farthest_indices = np.argsort(-distances)[:1000]
    farthest_points = embeddings_2d_dataset2[farthest_indices]'''

    # Plot the embeddings
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d_sst2[:, 0], embeddings_2d_sst2[:, 1], c='red', label='SST-2')
    plt.scatter(embeddings_2d_dataset1[:, 0], embeddings_2d_dataset1[:, 1], c='green', alpha=0.5, label='FLipkart')
    plt.scatter(embeddings_2d_dataset2[:, 0], embeddings_2d_dataset2[:, 1], c='blue', alpha=0.5, label='Yelp')
    plt.scatter(embeddings_2d_imdb[:, 0], embeddings_2d_imdb[:, 1], c='black', alpha=0.5, label='imdb')
    plt.scatter(embeddings_2d_finance[:, 0], embeddings_2d_finance[:, 1], c='orange', alpha=0.5, label='finance')
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Cluster Centers')
    #plt.scatter(farthest_points[:, 0], farthest_points[:, 1], c='blue', label='1000 Farthest Points from Dataset 2')
    plt.legend()
    plt.show()

    total_time = time.time() - start_time
    print("Total time taken: {:.2f} seconds".format(total_time))


