import os
import pandas as pd
import csv
from tqdm import tqdm
import json
import time
from datetime import datetime
import os
import random

import numpy as np
import pytorch_lightning as pt
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from tqdm import tqdm

from embedding_models.LUAR.models.transformer import Transformer
from embedding_models.embedding_model import EmbeddingModel

class LUAR(EmbeddingModel):

    def get_model_name(self):
        return 'luar'
    
    def preprocess_verification_data(self, df, save_name):
        processor = LuarProcessor(df, self.model_folder, save_name)
        processor.process()
    
    def get_embeddings(self, texts):
        pass
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        # just for compatability purposes
        if args.train:
            self.preprocess_verification_data(self.eval_df, 'test_queries')
            self.preprocess_verification_data(self.eval_df, 'test_targets')
        self.parameter_set = SimpleNamespace(**self.parameter_set)
        self.parameter_set.dataset_name = self.model_folder
        self.params = self.parameter_set
        self.experiment_dir = os.path.join(self.model_folder, 'saved_model')
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.model = Transformer(self.parameter_set)
        self.batch_size = self.parameter_set.batch_size
        
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.experiment_dir, 
            save_top_k=-1, 
        )

    def load_model(self, load_folder):
        
        if not hasattr(self, 'loaded'):
            self.loaded = True
            # get the latest checkpoint
            version = "version_0" if self.params.version is None else self.params.version
            checkpoint_folder = os.path.join(load_folder, self.params.log_dirname, version, 'checkpoints')
            if os.path.exists(checkpoint_folder):
                checkpoint_files = os.listdir(checkpoint_folder)
                if len(checkpoint_files) > 0:
                    assert len(checkpoint_files) == 1
                    checkpoint_file = checkpoint_files[0]
                    epochs_and_steps = checkpoint_file.replace('epoch=', '').replace('.ckpt', '').split('-step=')
                    epoch_num = int(epochs_and_steps[0])
                    step_num = int(epochs_and_steps[1])
                    
                    resume_from_checkpoint = os.path.join(checkpoint_folder, checkpoint_file)
                    print("Checkpoint: {}".format(resume_from_checkpoint))

                    checkpoint = torch.load(resume_from_checkpoint)
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def train(self):
        logger = TensorBoardLogger(self.experiment_dir, name=self.params.log_dirname, version=self.params.version)
        trainer = pt.Trainer(
            default_root_dir=self.experiment_dir, 
            max_epochs=self.params.num_epoch,
            logger=logger,
            enable_checkpointing=self.checkpoint_callback,
            gpus=self.params.gpus, 
            strategy='dp' if self.params.gpus > 1 else None, 
            precision=self.params.precision,
        )
        
        trainer.fit(self.model)
        
    def get_embeddings(self, texts):
        self.load_model(self.model_folder)
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        
        embeddings = []
        
        # for text in tqdm(texts):
        #     tokens = tokenizer([text], padding='max_length', truncation=True, max_length=self.parameter_set.token_max_length, return_tensors='pt')
        #     input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        #     input_ids = input_ids.unsqueeze(0).unsqueeze(0)
        #     attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        #     embedding = self.model([input_ids, attention_mask])[0]
        #     embeddings.append(embedding[0,:].tolist())
            
        embeddings = []
        batch_texts = []
        for i, text in tqdm(enumerate(texts), total=len(texts)):
            batch_texts.append(text)
            if i == len(texts) - 1 or len(batch_texts) == self.parameter_set.batch_size:
                tokens = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=self.parameter_set.token_max_length, return_tensors='pt')
                input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
                input_ids = input_ids.unsqueeze(1).unsqueeze(1)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                embedding = self.model([input_ids, attention_mask])[0]
                for j in range(embedding.shape[0]):
                    embeddings.append(embedding[j,:].tolist())
                batch_texts = []
        
        return embeddings
    
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
        seen_ids = []
        genre_count = 0
        genre_to_genre_id = {}
        for _, row in self.df.iterrows():
            text0, text1 = row['text0'], row['text1']
            id0, id1 = row['id0'], row['id1']
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
            if id0 not in seen_ids:
                text_author_pairs.append((text0, row['author0'], genre0))
                seen_ids.append(id0)
            if id1 not in seen_ids:
                text_author_pairs.append((text1, row['author1'], genre1))
                seen_ids.append(id1)
        
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
        
"""

salloc -c 16 -G a40

date
dataset="CrossNews_mini.csv"
model="luar"
conda activate luar
cd /nethome/mma81/storage/CrossNews

python src/train_embedding.py \
--model ${model} \
--train \
--train_file verification_data/train/${dataset} \
--parameter_sets default
date
exit

"""