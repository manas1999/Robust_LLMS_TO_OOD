import pandas as pd
from data.dataLoader import data_loader

def subsample_and_save(dataset_name):
    train_df,test_df  = data_loader.generic_data_loader(dataset_name)
    
    train_df = train_df.to_pandas()
    test_df = test_df.to_pandas()
    
    filtered_df = test_df
    
    sampled_dfs = []
    for label in [1, 0, 2]:
        label_df = filtered_df[filtered_df['Label'] == label]
        if len(label_df) >= 300:
            sampled_dfs.append(label_df.sample(n=300, random_state=42))
        else:
            print(f"Not enough samples in label {label}. Only {len(label_df)} available.")
    
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    subsample_file_path = f'./data/processed/{dataset_name}_subsample/test.tsv'
    final_sample.to_csv(subsample_file_path, sep='\t', index=False)
    total_samples = len(final_sample)
    samples_per_label = final_sample['Label'].value_counts()
    
    return total_samples, samples_per_label, subsample_file_path