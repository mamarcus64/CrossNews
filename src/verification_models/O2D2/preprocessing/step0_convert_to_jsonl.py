import uuid
import json
import pandas as pd
import os
import random

def pair_file_to_jsonl(df, out_folder, out_name):
    pairs, labels = [], []
    
    for i in range(len(df)):
        row = df.iloc[i]
        id = str(uuid.uuid4())
        
        same = True if row['label'] == 1 else False
        text0, text1 = row['text0'], row['text1']
        auth0, auth1 = row['author0'], row['author1']
        
        labels.append(json.dumps({
            'id': id,
            'same': same,
            'authors': [auth0, auth1],
        }))
        
        if 'genre0' not in row:
            pairs.append(json.dumps({
                'id': id,
                'fandoms': ['A', 'B'] if random.random() < 0.5 else ['B', 'A'],
                'pair': [text0, text1],
            }))
        else:
            pairs.append(json.dumps({
                'id': id,
                'fandoms': [row['genre0'], row['genre1']],
                'pair': [text0, text1],
            }))
        
    
    with open(os.path.join(out_folder, f'{out_name}_pairs.jsonl'), 'w', encoding='utf-8') as pair_out:
        for pair in pairs:
            pair_out.write(f'{pair}\n')
            
    with open(os.path.join(out_folder, f'{out_name}_labels.jsonl'), 'w', encoding='utf-8') as label_out:
        for label in labels:
            label_out.write(f'{label}\n')
            
            
def run(train_file, folder, seed=2345):
    random.seed(seed)
    pair_file_to_jsonl(train_file, folder, 'train')