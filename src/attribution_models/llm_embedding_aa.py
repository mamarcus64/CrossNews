import random
import os
import json
from types import SimpleNamespace
from scipy.spatial.distance import cosine
import pdb
from pathlib import Path
from tqdm import tqdm
import numpy as np

from attribution_models.attribution_model import AttributionModel

class LLM_Embedding_AA(AttributionModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        
        self.train_embedding_loc = parameter_set['train_embedding_loc']
        self.test_embedding_loc = parameter_set['test_embedding_loc']
        
        self.id_to_embedding = json.load(open(self.train_embedding_loc, 'r'))
        print('loaded train')
        self.id_to_embedding.update(json.load(open(self.test_embedding_loc, 'r')))
        print('loaded test')
    
    def get_model_name(self):
        return self.parameter_set['name']
    
    def train_internal(self, params):
        pass
    
    def save_model(self, folder):
        pass
                
    def load_model(self, folder):
        pass
    
    def evaluate_internal(self, query_df, target_df, df_name=None):
        
        author_ids = sorted(list(set(target_df['author'])))
        query_embeddings = []
                
        for author in author_ids:
            author_doc_ids = list(query_df[query_df['author'] == author]['id'])
            # import pdb; pdb.set_trace()
            query_embeddings.append(np.array([self.id_to_embedding[str(id)] for id in author_doc_ids]).mean(axis=0).tolist())
            
        all_distances = []
        
        for i in tqdm(list(range(len(target_df)))):
            row = target_df.iloc[i]
            target_embedding = self.id_to_embedding[str(row['id'])]
            
            distances = []
            for query in query_embeddings:
                distances.append(-round(cosine(query, target_embedding), 4))
                
            all_distances.append(distances)
            
        return all_distances
