from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
#from GMM_Plots import plot_gmm_and_embeddings
from sklearn.mixture import GaussianMixture


def plot_combined_gmm_embeddings(embeddings1, embeddings2, title):
    # Define colors and labels
    colors = ['green', 'blue']  # Different colors for each dataset
    labels = ['Flipkart Reviews', 'Yelp Reviews']
    
    # Create a figure
    plt.figure(figsize=(12, 10))

    # Fit GMM to both datasets and plot
    for i, embeddings in enumerate([embeddings1, embeddings2]):
        gmm = GaussianMixture(n_components=1, random_state=0)
        gmm.fit(embeddings)

        # Scatter plot of the dataset
        plt.scatter(embeddings[:, 0], embeddings[:, 1], color=colors[i], label=labels[i], alpha=0.5)

        # Generate a grid of points for GMM contours
        x = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), 100)
        y = np.linspace(embeddings[:, 1].min(), embeddings[:, 1].max(), 100)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -gmm.score_samples(XX)
        Z = Z.reshape(X.shape)

        # Plot GMM contours
        plt.contour(X, Y, Z, levels=14, colors=[colors[i]], linewidths=1.2, alpha=0.6)

        # Annotate mean and variance
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
    #Load the already finetuned model 
    model_directory = 'roberta-base'
    model_directory = '/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/FineTuned_Models/Roberta_yelp/roberta'
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Load the pretrained RoBERTa model
    #model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    end_time = time.time()
    print("Loaded the model in {:.2f} seconds".format(end_time - start_time))
    
    #print the centroid
    # Function to load centroid from the file
    def load_centroid(file_path):
        with open(file_path, 'r') as file:
            # Read data and remove unwanted characters thoroughly
            data = file.read().replace(',', ' ').replace('[', ' ').replace(']', ' ')
            # Split into numbers based on whitespace and convert each to float
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

    # Load the centroid
    centroid_yelp = load_centroid('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Centroid/centroid.txt')
    centroid_yelp = centroid_yelp.reshape(1, -1) 


    # Load and preprocess the dataset
    df = pd.read_csv('../Datasets/cleaned_reviews.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df))
    df_yelp = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/yelp_reviews_cleaned.csv', nrows=1000)
    print("Number of rows in the dataset:", len(df_yelp))
    #df_imdb = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/imdb_cleaned_reviews.csv', nrows=1000)
    #df_finance = pd.read_csv('/Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/Datasets/financial_cleaned_texts.csv', nrows=1000)

    # Load SST-2 dataset
    #sst2_dataset = load_dataset("stanfordnlp/sst2")
    #sst2_sentences = [item['sentence'] for item in sst2_dataset['train']][:1000]  # Adjust size as needed

    
    sentences_dataset1 = df['cleaned_review'].astype(str).tolist()
    sentences_dataset2 = df_yelp['cleaned_review'].astype(str).tolist()
    #sentences_dataset3 = df_imdb['cleaned_review'].astype(str).tolist()
    #sentences_dataset4 = df_finance['cleaned_text'].astype(str).tolist()


        # Generate embeddings
    embeddings_dataset1 = get_embeddings(df['cleaned_review'].astype(str).tolist())
    embeddings_dataset2 = get_embeddings(df_yelp['cleaned_review'].astype(str).tolist())
    #embeddings_sst2 = get_embeddings(sst2_sentences)
    #embeddings_imdb = get_embeddings(df_imdb['cleaned_review'].astype(str).tolist())
    #embeddings_finance = get_embeddings(df_finance['cleaned_text'].astype(str).tolist())
    

    

    # Generate embeddings
    '''embeddings_dataset1 = model.encode(sentences_dataset1, show_progress_bar=True)
    embeddings_dataset2 = model.encode(sentences_dataset2, show_progress_bar=True)
    embeddings_sst2 = model.encode(sst2_sentences, show_progress_bar=True)
    embeddings_imdb = model.encode(sentences_dataset3, show_progress_bar=True)
    embeddings_finance = model.encode(sentences_dataset4, show_progress_bar=True)'''




    combined_embeddings = np.vstack([embeddings_dataset1, embeddings_dataset2, centroid_yelp])
    
    # Reduce the dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=0)
    #embeddings_2d_dataset1 = tsne.fit_transform(embeddings_dataset1)
    #embeddings_2d_dataset2 = tsne.fit_transform(embeddings_dataset2)
    #centroid_2d_yelp = tsne.fit_transform(centroid_yelp)
    all_embeddings_2d = tsne.fit_transform(combined_embeddings)
    #embeddings_2d_sst2 = tsne.fit_transform(embeddings_sst2)
    #embeddings_2d_imdb = tsne.fit_transform(embeddings_imdb)
    #embeddings_2d_finance = tsne.fit_transform(embeddings_finance)

    centroid_2d_yelp = all_embeddings_2d[-1]
    embeddings_2d_dataset1 = all_embeddings_2d[:len(embeddings_dataset1)]
    embeddings_2d_dataset2 = all_embeddings_2d[len(embeddings_dataset1):-1]



    '''# Clustering Dataset 1 to find centers
    kmeans = KMeans(n_clusters=1, random_state=0).fit(embeddings_2d_dataset1)
    centers = kmeans.cluster_centers_

    # Calculate distances from each point in dataset 2 to the nearest center in dataset 1
    distances = np.min([np.linalg.norm(embeddings_2d_dataset2 - center, axis=1) for center in centers], axis=0)

    # Find indices of the 1000 farthest points
    farthest_indices = np.argsort(-distances)[:1000]
    farthest_points = embeddings_2d_dataset2[farthest_indices]'''

    plot_combined_gmm_embeddings(embeddings_2d_dataset1, embeddings_2d_dataset2, "GMM for Flipkart and Yelp Reviews")
    # Plot the embeddings
    '''plt.figure(figsize=(12, 10))
    #plt.scatter(embeddings_2d_sst2[:, 0], embeddings_2d_sst2[:, 1], c='red', label='SST-2')
    plt.scatter(embeddings_2d_dataset1[:, 0], embeddings_2d_dataset1[:, 1], c='green', alpha=0.5, label='FLipkart')
    plt.scatter(embeddings_2d_dataset2[:, 0], embeddings_2d_dataset2[:, 1], c='blue', alpha=0.5, label='Yelp')
    plt.scatter(centroid_2d_yelp[0], centroid_2d_yelp[1], c='red', alpha=0.5, label='Center_yelp')
    #plt.scatter(embeddings_2d_imdb[:, 0], embeddings_2d_imdb[:, 1], c='black', alpha=0.5, label='imdb')
    #plt.scatter(embeddings_2d_finance[:, 0], embeddings_2d_finance[:, 1], c='orange', alpha=0.5, label='finance')
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Cluster Centers')
    #plt.scatter(farthest_points[:, 0], farthest_points[:, 1], c='blue', label='1000 Farthest Points from Dataset 2')
    plt.legend()
    plt.show()'''

    total_time = time.time() - start_time
    print("Total time taken: {:.2f} seconds".format(total_time))

