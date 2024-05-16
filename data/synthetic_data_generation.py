import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def categorize_y_values(y_values):
    # Calculate the 33rd and 66th percentiles
    thresholds = np.percentile(y_values, [33, 66])
    categories = np.digitize(y_values, bins=thresholds)
    return categories

def generate_synthetic_data(num_samples, d, mu_range=(0.2, 1), cov_sigma = 1, sigma = 1):
    synthetic_data_id = {"id_text" : [], "id_label" : []}
    synthetic_data_ood_25 = {"id_text" : [], "id_label" : []}
    synthetic_data_ood_50 = {"id_text" : [], "id_label" : []}

    all_y = []
    all_y_25 = []
    all_y_50 = []

    for _ in range(num_samples):
        cov = np.eye(d) * cov_sigma
        x = np.random.multivariate_normal(mean=[0]*d, cov=cov)
        beta = np.random.multivariate_normal(mean=[0]*d, cov=cov)
        y = np.dot(beta, x) + np.random.normal(0, sigma)
        synthetic_data_id['id_text'].append(" ".join(map(str, x)))
        synthetic_data_id['id_label'].append(y)
        all_y.append(y)
        
        # 25% distribution shift 
        mu_25 = [0.25]*d
        xj_25 = np.random.multivariate_normal(mean = mu_25, cov = cov)
        beta_j_25 = np.random.multivariate_normal(mean = mu_25, cov = cov)
        yj_25 = np.dot(beta_j_25, xj_25) + np.random.normal(0, sigma)
        synthetic_data_ood_25['id_text'].append(" ".join(map(str, xj_25)))
        synthetic_data_ood_25['id_label'].append(yj_25)
        all_y_25.append(yj_25)

        # 50% distribution shift 
        mu_50 = [0.5]*d
        xj_50 = np.random.multivariate_normal(mean = mu_50, cov = cov)
        beta_j_50 = np.random.multivariate_normal(mean = mu_50, cov = cov)
        yj_50 = np.dot(beta_j_50, xj_50) + np.random.normal(0, sigma)
        synthetic_data_ood_50['id_text'].append(" ".join(map(str, xj_50)))
        synthetic_data_ood_50['id_label'].append(yj_50)
        all_y_50.append(yj_50)
    
    all_y = np.array(all_y)
    all_y_25 = np.array(all_y_25)
    all_y_50 = np.array(all_y_50)

    synthetic_data_id['id_label'] = categorize_y_values(all_y)
    synthetic_data_ood_25['id_label'] = categorize_y_values(all_y_25)
    synthetic_data_ood_50['id_label'] = categorize_y_values(all_y_50)

    synthetic_data_id_df = pd.DataFrame(synthetic_data_id)
    synthetic_data_ood_25_df = pd.DataFrame(synthetic_data_ood_25)
    synthetic_data_ood_50_df = pd.DataFrame(synthetic_data_ood_50)
        
    return synthetic_data_id_df, synthetic_data_ood_25_df, synthetic_data_ood_50_df

def main_synthetic_data():
    synthetic_data_id_df, synthetic_data_ood_25_df, synthetic_data_ood_50_df = generate_synthetic_data(10000, 10)

    ## splitting dataset to test and train 
    synthetic_data_id_df_train, synthetic_data_id_df_test = train_test_split(synthetic_data_id_df, test_size=0.2, random_state=42)
    synthetic_data_ood_25_df_train, synthetic_data_ood_25_df_test = train_test_split(synthetic_data_ood_25_df, test_size=0.2, random_state=42)
    synthetic_data_ood_50_df_train, synthetic_data_ood_50_df_test = train_test_split(synthetic_data_ood_50_df, test_size=0.2, random_state=42)


    ## exporting train dataset
    synthetic_data_id_df_train.to_csv("./data/processed/synthetic_data_id/train.tsv", sep= '\t', index= False)
    synthetic_data_ood_25_df_train.to_csv("./data/processed/synthetic_data_ood_25/train.tsv", sep= '\t', index= False)
    synthetic_data_ood_50_df_train.to_csv("./data/processed/synthetic_data_ood_50/train.tsv", sep= '\t', index= False)

    ## exporting test dataset
    synthetic_data_id_df_test.to_csv("./data/processed/synthetic_data_id/test.tsv", sep= '\t', index= False)
    synthetic_data_ood_25_df_test.to_csv("./data/processed/synthetic_data_ood_25/test.tsv", sep= '\t', index= False)
    synthetic_data_ood_50_df_test.to_csv("./data/processed/synthetic_data_ood_50/test.tsv", sep= '\t', index= False)

    return 


