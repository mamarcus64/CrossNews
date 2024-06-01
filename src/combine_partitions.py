import pandas as pd
import os
import csv

partition_folder = 'verification_data/train/partitions'

file_stem = 'CrossNews_Article_Article_short'
original_df = pd.read_csv('verification_data/train/CrossNews_Article_Article.csv')

# file_stem = 'CrossNews_Article_Tweet_short'
# original_df = pd.read_csv('verification_data/train/CrossNews_Article_Tweet.csv')

starts = []

dfs = []
for file in os.listdir(partition_folder):
    if file.startswith(file_stem):
        start_index = int(file.split('_')[-2])
        starts.append(start_index)

for i in range(0, len(original_df), 100):
    partition_path = os.path.join(partition_folder, f'{file_stem}_{i}_{i+100}.csv')
    partition_df = pd.read_csv(partition_path)
    dfs.append(partition_df)
    
stacked_df = pd.concat(dfs, axis=0)
stacked_df.reset_index(drop=True, inplace=True)

print(len(stacked_df), len(original_df))
assert len(stacked_df) == len(original_df)

stacked_df.to_csv(os.path.join('verification_data/train', f'{file_stem}.csv'), index=False, quoting=csv.QUOTE_ALL)