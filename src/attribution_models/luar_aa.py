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
from embedding_models.luar import LUAR

class LUAR_AA(AttributionModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        if hasattr(args, 'load') and args.load:
            luar_args = {
                'model': 'luar',
                'train': False,
                'load': True,
                'load_folder': parameter_set['model_folder'],
                'parameter_sets': [parameter_set['parameter_set']],
                'evaluation_metric': 'F1'
            }
        elif hasattr(args, 'train') and args.train:
            luar_args = {
                'model': 'luar',
                'load': False,
                'train': True,
                'save_folder': args.save_folder,
                'train_file': args.query_file,
                'seed': 1234,
                'eval_ratio': 0.2,
                'parameter_sets': [parameter_set['parameter_set']],
                'evaluation_metric': 'F1'
            }
        
        luar_parameter_set = json.load(open('src/model_parameters/luar.json', 'r'))[parameter_set['parameter_set']]
        
        # self.luar_model = LUAR(SimpleNamespace(**luar_args), luar_parameter_set)
        
        # if hasattr(args, 'train') and args.train:
            # self.luar_model.train()
       
        # if 'embedding_folder' in parameter_set:
            # self.embedding_folder = parameter_set['embedding_folder']
        # args.query_file = 'CrossNews_Article.csv'
        if 'article' in args.query_file.lower():
            args.query_file = 'CrossNews_Article'
        if 'tweet' in args.query_file.lower():
            args.query_file = 'CrossNews_Tweet'
        
        
        self.id_to_embedding = json.load(open(f'gold_embeddings/luar/{Path(args.query_file).stem}.json', 'r'))
    
    def get_model_name(self):
        return 'luar_aa'
    
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
