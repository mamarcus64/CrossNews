import os
import pandas as pd
import csv
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import torch

from embedding_models.embedding_model import EmbeddingModel
import embedding_models.PART.train_part as train_part
from embedding_models.PART.model import ContrastiveLSTMHead

from tqdm import tqdm


class PART(EmbeddingModel):
    
    def get_model_name(self):
        return 'part'
    
    def split_data(self, eid, text, CHUNK_SIZE=512):
        input_ids = self.tokenizer(text).input_ids
        effective_chunk_size = CHUNK_SIZE
        if len(input_ids) <= CHUNK_SIZE:
            effective_chunk_size = len(input_ids)//2
        chunked = [input_ids[chunk: chunk + effective_chunk_size] for chunk in range(0, len(input_ids), effective_chunk_size)]
        decoded_chunked = self.tokenizer.batch_decode(chunked)
        all_chunks = []
        for i in range(len(chunked)):
            all_chunks.append([eid, chunked[i], decoded_chunked[i]])
        return all_chunks
    
    def preprocess_verification_data(self, df, save_name):
        text_author_pairs = []
        seen_ids = []
        for _, row in df.iterrows():
            text0, text1 = row['text0'], row['text1']
            id0, id1 = row['id0'], row['id1']
            if id0 not in seen_ids:
                text_author_pairs.append((text0, row['author0']))
                seen_ids.append(id0)
            if id1 not in seen_ids:
                text_author_pairs.append((text1, row['author1']))
                seen_ids.append(id1)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.parameter_set['transformer'])
            
        rows = []
        for text, author in text_author_pairs:
            rows.extend(self.split_data(author, text))
            
        processed_df = pd.DataFrame(rows, columns=['id', 'pretokenized_text', 'decoded_text'])
        processed_df.to_csv(os.path.join(self.model_folder, f'{save_name}.csv'), index=False, quoting=csv.QUOTE_ALL, lineterminator='\n')
        
    def train(self):
        train_file = os.path.join(self.model_folder, 'train.csv')
        val_file = os.path.join(self.model_folder, 'val.csv')
        train_dataset_name = Path(self.args.train_file).stem
        train_part.run(train_file, val_file, self.parameter_set, self.model_folder, train_dataset_name)
    
    def get_embeddings(self, texts):
        tokenizer = AutoTokenizer.from_pretrained(self.parameter_set['transformer'])
        checkpoint_folder = os.path.join(self.model_folder, 'model_checkpoint')
        checkpoint_file = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
        model = ContrastiveLSTMHead.load_from_checkpoint(checkpoint_path=checkpoint_file)
        
        # embeddings = []
        
        # for text in tqdm(texts):
        #     tokens = tokenizer([text], return_tensors='pt', truncation=True, padding=True, max_length=512)
        #     with torch.no_grad():
        #         embedding = model(tokens.input_ids, tokens.attention_mask)
        #     embeddings.append(embedding[0,:].tolist())
            
        embeddings = []
        batch_texts = []
        
        for i, text in tqdm(enumerate(texts), total=len(texts)):
            batch_texts.append(text)
            if i == len(texts) - 1 or len(batch_texts) == self.parameter_set['batch_size']:
                tokens = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    embedding = model(tokens.input_ids, tokens.attention_mask)
                for j in range(embedding.shape[0]):
                    embeddings.append(embedding[j,:].tolist())
                batch_texts = []
        
        return embeddings
    
"""

salloc -c 16 -G a40

date
dataset="CrossNews_mini.csv"
model="part"
conda activate part
cd /nethome/mma81/storage/CrossNews

python src/train_embedding.py \
--model ${model} \
--train \
--train_file verification_data/train/${dataset} \
--parameter_sets testing
date
exit

"""