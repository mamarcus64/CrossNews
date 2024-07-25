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

class PART_AA(AttributionModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        if hasattr(args, 'load') and args.load:
            part_args = {
                'model': 'part',
                'train': False,
                'load': True,
                'load_folder': parameter_set['model_folder'],
                'parameter_sets': [parameter_set['parameter_set']],
                'evaluation_metric': 'F1'
            }
        elif hasattr(args, 'train') and args.train:
            part_args = {
                'model': 'part',
                'load': False,
                'train': True,
                'save_folder': args.save_folder,
                'train_file': args.query_file,
                'seed': 1234,
                'eval_ratio': 0.2,
                'parameter_sets': [parameter_set['parameter_set']],
                'evaluation_metric': 'F1'
            }
        args.query_file = 'CrossNews_Article.csv'
        self.id_to_embedding = json.load(open(f'gold_embeddings/part/{Path(args.query_file).stem}.json', 'r'))
    
    def get_model_name(self):
        return 'part_aa'
    
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
