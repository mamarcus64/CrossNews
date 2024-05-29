import os
import pandas as pd
import csv
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

from embedding_models.embedding_model import EmbeddingModel
import embedding_models.PART.train_part as train_part

class LUAR(EmbeddingModel):
    pass

import json
import time
from datetime import datetime
import os
import random

class LuarProcessor:
    def __init__(self, df, model_folder, out_name):
        self.df = df
        self.model_folder = model_folder
        self.out_name = out_name
        
    def to_jsonl(self, file_loc, author_list):
        with open(file_loc, 'w') as out:
            for author in author_list:
                out.write(json.dumps(author) + '\n')
            
    def get_doc_tuple(self, text, genre_number, id):
        symbols = text
        return (
            symbols,
            len(symbols),
            genre_number,
            0,
            id,
        )
 
    def to_author_dict(self, author_id, all_tuples):
        num_posts = len(all_tuples)
        author_dict = {
            'author_id': author_id,
            'num_posts': [num_posts],
            'document_ids': [x[4] for x in all_tuples],
            'action_type': [x[2] for x in all_tuples],
            'lens': [x[1] for x in all_tuples],
            'hour': [x[3] for x in all_tuples],
            'syms': [x[0] for x in all_tuples],
        }
        return author_dict
    
    def process(self):
        
        text_author_pairs = []
        seen_texts = {}
        genre_count = 0
        genre_to_genre_id = {}
        for _, row in self.df.iterrows():
            text0, text1 = row['text0'], row['text1']
            if 'genre0' in row:
                genre0, genre1 = row['genre0'], row['genre1']
                if genre0 not in genre_to_genre_id:
                    genre_to_genre_id[genre0] = genre_count
                    genre_count += 1
                if genre1 not in genre_to_genre_id:
                    genre_to_genre_id[genre1] = genre_count
                    genre_count += 1
                genre0 = genre_to_genre_id[genre0]
                genre1 = genre_to_genre_id[genre1]
            else:
                genre0, genre1 = 0, 0
            if text0 not in seen_texts:
                text_author_pairs.append((text0, row['author0'], genre0))
                seen_texts[text0] = row['author0']
            if text1 not in seen_texts:
                text_author_pairs.append((text1, row['author1'], genre1))
                seen_texts[text1] = row['author1']
        
        author_to_author_id = {}
        author_id_to_tuples = {}
        author_id = 0
        doc_id = 0
        for text, author, genre in text_author_pairs:
            if author not in author_to_author_id:
                author_to_author_id[author] = author_id
                author_id += 1
                author_id_to_tuples[author_to_author_id[author]] = []
            author_id_to_tuples[author_to_author_id[author]].append(self.get_doc_tuple(text, genre, doc_id))
            doc_id += 1
            
        json.dump(author_to_author_id, open(os.path.join(self.model_folder, f'{self.out_name}_authors.json'), 'w'), indent=4)
        
        rows = []
        for author, docs in author_id_to_tuples.items():
            rows.append(self.to_author_dict(author, docs))
            
        self.to_jsonl(os.path.join(self.model_folder, f'{self.out_name}.jsonl'), rows)