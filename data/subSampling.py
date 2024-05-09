import pandas as pd
import numpy as np
from datasets import Dataset
from data.dataLoader import data_loader
def subsample_and_save(dataset_name):
    # Load the full training dataset using your loading convention
    train_df,test_df  = data_loader.generic_data_loader(dataset_name)
    
    # Convert Hugging Face Dataset to Pandas DataFrame
    train_df = train_df.to_pandas()
    test_df = test_df.to_pandas()
    
    # Step 1: Filter sentences with word count between 20 and 70
    test_df['word_count'] = test_df['Text'].apply(lambda x: len(x.split()))
    filtered_df = test_df
    
    # Step 2: Subsample the data
    sampled_dfs = []
    for label in [1, 0, 2]:
        label_df = filtered_df[filtered_df['Label'] == label]
        if len(label_df) >= 300:
            sampled_dfs.append(label_df.sample(n=1000, random_state=42))
        else:
            print(f"Not enough samples in label {label}. Only {len(label_df)} available.")
    
    # Concatenate all the subsamples into one dataframe
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Step 3: Save the subset to a TSV file in the structured path
    subsample_file_path = f'./data/processed/{dataset_name}_subsample/test.tsv'
    final_sample.drop(columns=['word_count'], inplace=True)
    final_sample.to_csv(subsample_file_path, sep='\t', index=False)
    
    # Step 4: Provide statistics
    total_samples = len(final_sample)
    samples_per_label = final_sample['Label'].value_counts()
    
    return total_samples, samples_per_label, subsample_file_path

# Usage Example
total_samples, samples_per_label, subsample_file_path = subsample_and_save('amazon')
print(f"Total samples in subsample: {total_samples}")
print(f"Samples per label:\n{samples_per_label}")
print(f"Subsampled data saved to: {subsample_file_path}")


