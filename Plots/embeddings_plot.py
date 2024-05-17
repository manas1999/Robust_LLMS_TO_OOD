from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def plot_combined_gmm_embeddings(embeddings1, embeddings2, title):
    colors = ['green', 'blue']  
    labels = ['Flipkart Reviews', 'Yelp Reviews']
    
    plt.figure(figsize=(12, 10))

    for i, embeddings in enumerate([embeddings1, embeddings2]):
        gmm = GaussianMixture(n_components=1, random_state=0)
        gmm.fit(embeddings)

        plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[i], label=labels[i], alpha=0.5)

        x = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), 100)
        y = np.linspace(embeddings[:, 1].min(), embeddings[:, 1].max(), 100)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -gmm.score_samples(XX)
        Z = Z.reshape(X.shape)

        plt.contour(X, Y, Z, levels=14, colors=[colors[i]], linewidths=1.2, alpha=0.6)

        mean = gmm.means_[0]
        variances = gmm.covariances_[0]
        plt.annotate(f'Mean: {mean.round(2)}\nVariance: {variances.round(2)}', 
                     xy=mean, xytext=(mean[0] + 5, mean[1] + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=9, color=colors[i])

    plt.title(title)
    plt.legend()
    plt.show()


def plot_embeddings():
    start_time = time.time()
    model_directory = 'roberta-base'
    model_directory = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/Roberta_yelp/roberta'
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    end_time = time.time()
    print("Loaded the model in {:.2f} seconds".format(end_time - start_time))
    
    def load_centroid(file_path):
        with open(file_path, 'r') as file:
            data = file.read().replace(',', ' ').replace('[', ' ').replace(']', ' ')
            numbers = data.split()
            centroid_array = np.array([float(num.strip()) for num in numbers], dtype=np.float32)
            return centroid_array

    def get_embeddings(sentences, batch_size=32):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = model.roberta(**encoded_input, return_dict=True)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    centroid_yelp = load_centroid('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Centroid/centroid.txt')
    centroid_yelp = centroid_yelp.reshape(1, -1) 


    df = pd.read_csv('../Datasets/cleaned_reviews.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df))
    df_yelp = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_cleaned.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df_yelp))
    
    sentences_dataset1 = df['cleaned_review'].astype(str).tolist()
    sentences_dataset2 = df_yelp['cleaned_review'].astype(str).tolist()

    embeddings_dataset1 = get_embeddings(df['cleaned_review'].astype(str).tolist())
    embeddings_dataset2 = get_embeddings(df_yelp['cleaned_review'].astype(str).tolist())

    combined_embeddings = np.vstack([embeddings_dataset1, embeddings_dataset2, centroid_yelp])
    
    tsne = TSNE(n_components=2, random_state=0)
    all_embeddings_2d = tsne.fit_transform(combined_embeddings)

    centroid_2d_yelp = all_embeddings_2d[-1]
    embeddings_2d_dataset1 = all_embeddings_2d[:len(embeddings_dataset1)]
    embeddings_2d_dataset2 = all_embeddings_2d[len(embeddings_dataset1):-1]


    plot_combined_gmm_embeddings(embeddings_2d_dataset1, embeddings_2d_dataset2, "GMM for Flipkart and Yelp Reviews")

    total_time = time.time() - start_time
    print("Total time taken: {:.2f} seconds".format(total_time))

