from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import pandas as pd
import csv

old_file = 'verification_data/train/CrossNews_Article_Tweet.csv'
new_file = 'verification_data/train/CrossNews_Article_Tweet_short.csv'

def shorten_text(text, threshold=500):
    if len(text) <= threshold:
        return text
    shortened_text = ''
    sentences = sent_tokenize(text)
    for sentence in sentences:
        shortened_text += ' ' + sentence
        if len(shortened_text) > threshold:
            break
    return shortened_text    
    

old_df = pd.read_csv(old_file)

for i in tqdm(range(len(old_df))):
    row = old_df.iloc[i]
    if 'genre' in row:
        text0 = shorten_text(row['text0']) if row['genre0'] == 'Article' else row['text0']
        text1 = shorten_text(row['text1']) if row['genre1'] == 'Article' else row['text1']
    else:
        text0 = shorten_text(row['text0'])
        text1 = shorten_text(row['text1'])
    old_df.at[i, 'text0'] = text0
    old_df.at[i, 'text1'] = text1
    
old_df.to_csv(new_file, index=False, quoting=csv.QUOTE_ALL)




# from data_preprocessing.ersatz_shortening import shorten_df
# import sys
# import os
# from pathlib import Path
# import time
# import random

# # time.sleep(random.randint(1, 30))

# input_file = sys.argv[1]
# output_file = sys.argv[2]

# start_row = 0
# max_row = 100000

# partition_length = 100
# for start_row in range(0, max_row, partition_length):
#     end_row = start_row + partition_length
    
#     claim_file = output_file.replace('.csv', f'_{start_row}_{end_row}.csv').replace('/train/', '/train/partitions/')
#     if not os.path.exists(f'verification_data/claims/{Path(claim_file).stem}.txt'):
#         print(f'------------ STARTING {output_file} {start_row} {end_row} ------------------')
#         shorten_df(input_file, output_file, start_row, end_row)

# """

# salloc
# conda activate AuthorID_copy
# python src/shorten.py verification_data/test/CrossNews_Article_Tweet.csv verification_data/test/CrossNews_Article_Tweet_short.csv

# salloc
# conda activate AuthorID_copy
# python src/shorten.py verification_data/test/CrossNews_Article_Article.csv verification_data/test/CrossNews_Article_Article_short.csv

# salloc -c 16
# conda activate AuthorID_copy
# python src/shorten.py verification_data/train/CrossNews_Article_Tweet.csv verification_data/train/CrossNews_Article_Tweet_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Tweet.csv verification_data/train/CrossNews_Article_Tweet_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Tweet.csv verification_data/train/CrossNews_Article_Tweet_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Tweet.csv verification_data/train/CrossNews_Article_Tweet_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Tweet.csv verification_data/train/CrossNews_Article_Tweet_short.csv &


# python src/shorten.py verification_data/train/CrossNews_Article_Article.csv verification_data/train/CrossNews_Article_Article_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Article.csv verification_data/train/CrossNews_Article_Article_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Article.csv verification_data/train/CrossNews_Article_Article_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Article.csv verification_data/train/CrossNews_Article_Article_short.csv &
# python src/shorten.py verification_data/train/CrossNews_Article_Article.csv verification_data/train/CrossNews_Article_Article_short.csv &

# """