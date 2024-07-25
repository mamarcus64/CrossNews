import random
import pandas as pd
from tqdm import tqdm
import csv

def preprocess_data(df, save_name):
    
    id_to_info = {}
    for _, row in df.iterrows():
        
        text0, text1 = row['text0'], row['text1']
        author0, author1 = row['author0'], row['author1']
        id0, id1 = row['id0'], row['id1']
        if 'genre0' in row:
            genre0, genre1 = row['genre0'], row['genre1']
        else:
            genre0, genre1 = 'A', 'A'
            
        if id0 not in id_to_info:
            id_to_info[id0] = {
                'text': text0,
                'author': author0,
                'genre': genre0
            } 
        if id1 not in id_to_info:
            id_to_info[id1] = {
                'text': text1,
                'author': author1,
                'genre': genre1
            } 
            
    doc_id_to_positive_id = {}
    doc_id_to_negative_id = {}
    
    triplets = []
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        id0, id1 = row['id0'], row['id1']
        if row['label'] == 1:
            if id0 in doc_id_to_negative_id:
                triplets.append([id0, doc_id_to_negative_id[id0], id1])
                del doc_id_to_negative_id[id0]
                doc_id_to_positive_id[id1] = id0
            elif id1 in doc_id_to_negative_id:
                triplets.append([id1, doc_id_to_negative_id[id1], id0])
                del doc_id_to_negative_id[id1]
                doc_id_to_positive_id[id0] = id1
            else:
                doc_id_to_positive_id[id0] = id1
                doc_id_to_positive_id[id1] = id0
        if row['label'] == 0:
            if id0 in doc_id_to_positive_id:
                triplets.append([id0, doc_id_to_positive_id[id0], id1])
                del doc_id_to_positive_id[id0]
                doc_id_to_negative_id[id1] = id0
            elif id1 in doc_id_to_positive_id:
                triplets.append([id1, doc_id_to_positive_id[id1], id0])
                del doc_id_to_positive_id[id1]
                doc_id_to_negative_id[id0] = id1
            else:
                doc_id_to_negative_id[id0] = id1
                doc_id_to_negative_id[id1] = id0
    
    anchor_ids = list(doc_id_to_positive_id.keys())
    random.shuffle(anchor_ids)
    
    conv_id = 0
    
    rows = []
    
    for anchor_id, pos_id, neg_id in triplets:
        
        anchor_info = id_to_info[anchor_id]
        anchor_text = anchor_info['text'].replace('\t', ' ')
        anchor_author = anchor_info['author']
        anchor_genre = anchor_info['genre']
        
        pos_info = id_to_info[pos_id]
        pos_text = pos_info['text'].replace('\t', ' ')
        pos_author = pos_info['author']
        pos_genre = pos_info['genre']
        
        neg_info = id_to_info[neg_id]
        neg_text = neg_info['text'].replace('\t', ' ')
        neg_author = neg_info['author']
        neg_genre = neg_info['genre']
        
        anchor_conv = pos_conv = neg_conv = conv_id
        conv_id += 1
        
        if random.random() < 0.5:
            # anchor = U1 author (same_label = 1)
            rows.append([anchor_text, pos_text, neg_text,
                         anchor_id, pos_id, neg_id,
                         anchor_author, pos_author, neg_author,
                         anchor_conv, pos_conv, neg_conv,
                         anchor_genre, pos_genre, neg_genre, 1])
        else:
            # anchor = U2 author (same_label = 0)
            rows.append([anchor_text, neg_text, pos_text,
                         anchor_id, neg_id, pos_id,
                         anchor_author, neg_author, pos_author,
                         anchor_conv, neg_conv, pos_conv,
                         anchor_genre, neg_genre, pos_genre, 0])
            
    pd.DataFrame(rows, columns=['Anchor (A)', 'Utterance 1 (U1)', 'Utterance 2 (U2)',
                                'Utterance ID A', 'ID U1', 'ID U2',
                                'Author A', 'Author U1', 'Author U2',
                                'Conversation ID A', 'Conversation ID U1', 'Conversation ID U2',
                                'Subreddit A', 'Subreddit U1', 'Subreddit U2', 'Same Author Label'
                                ]).to_csv(save_name, index=False, sep='\t', quoting=csv.QUOTE_ALL)
        