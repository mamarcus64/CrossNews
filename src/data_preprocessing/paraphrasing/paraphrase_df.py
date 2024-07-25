import pandas as pd
from data_preprocessing.paraphrasing import tweet_paraphrase
from tqdm import tqdm


MODEL_ID = None

# Function to apply paraphrasing based on genre
def apply_paraphrasing(row):
    row['old_text0'] = row['text0']
    row['old_text1'] = row['text1']
    if 'genre0' not in row or row['genre0'] == 'Article':
        row['text0'] = tweet_paraphrase.paraphrase(row['text0'], MODEL_ID)
    if 'genre1' not in row or row['genre1'] == 'Article':
        row['text1'] = tweet_paraphrase.paraphrase(row['text1'], MODEL_ID)    
    return row

def paraphrase_df(df, model_id, start_row=None, end_row=None):
    global MODEL_ID
    MODEL_ID = model_id
    if start_row is None:
        start_row = 0
    if end_row is None:
        end_row = len(df)
    
    tqdm.pandas(desc='Paraphrasing rows...')
    paraphrase_df = df[start_row:end_row].progress_apply(apply_paraphrasing, axis=1)
    
    return paraphrase_df
