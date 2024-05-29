import os
import pandas as pd
import csv
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

from embedding_models.embedding_model import EmbeddingModel
import embedding_models.PART.train_part as train_part

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
        seen_texts = {}
        for _, row in df.iterrows():
            text0, text1 = row['text0'], row['text1']
            if text0 not in seen_texts:
                text_author_pairs.append((text0, row['author0']))
                seen_texts[text0] = row['author0']
            if text1 not in seen_texts:
                text_author_pairs.append((text1, row['author1']))
                seen_texts[text1] = row['author1']
            
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
    
    def load_model(self, load_folder):
        pass
    
    def get_embeddings(self, texts):
        pass
    
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