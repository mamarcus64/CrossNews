import random
import os
import json
from types import SimpleNamespace
from scipy.spatial.distance import cosine
import pdb

from verification_models.verification_model import VerificationModel
from embedding_models.luar import LUAR

class LUAR_AV(VerificationModel):
    
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
                'train_file': args.train_file,
                'seed': 1234,
                'eval_ratio': 0.2,
                'parameter_sets': [parameter_set['parameter_set']],
                'evaluation_metric': 'F1'
            }
        
        luar_parameter_set = json.load(open('src/model_parameters/luar.json', 'r'))[parameter_set['parameter_set']]
        
        self.luar_model = LUAR(SimpleNamespace(**luar_args), luar_parameter_set)
        
        if hasattr(args, 'train') and args.train:
            self.luar_model.train()
        elif hasattr(args, 'load') and args.load:
            self.part_model.load_model(args.load_folder)
       
        if 'threshold' in parameter_set:
            self.threshold = parameter_set['threshold']
        if 'embedding_folder' in parameter_set:
            self.embedding_folder = parameter_set['embedding_folder']
    
    def get_model_name(self):
        return 'luar_av'
    
    def get_distances(self, df, df_name=None):
        if not hasattr(self, 'embedding_folder'):
            id_to_doc = {}
            for _, row in df.iterrows():
                id_to_doc[row['id0']] = row['text0']
                id_to_doc[row['id1']] = row['text1']
                
            ids = list(id_to_doc.keys())
            texts = [id_to_doc[id] for id in ids]
            embeddings = self.luar_model.get_embeddings(texts)
            id_to_embedding = {id: embedding for id, embedding in zip(ids, embeddings)}
        else:
            id_to_embedding = json.load(open(os.path.join(self.embedding_folder, f'{df_name}.json'), 'r'))
            ks = list(id_to_embedding.keys())
            for k in ks:
                id_to_embedding[int(k)] = id_to_embedding[k]
        
        if df_name:
            embedding_folder = os.path.join(self.model_folder, 'embeddings')
            os.makedirs(embedding_folder, exist_ok=True)
            embedding_path = os.path.join(embedding_folder, f'{df_name}.json')
            json.dump(id_to_embedding, open(embedding_path, 'w'), indent=4)
        
        distances = []
        
        for _, row in df.iterrows():
            embed0 = id_to_embedding[row['id0']]
            embed1 = id_to_embedding[row['id1']]
            distances.append(cosine(embed0, embed1))
        
        return distances
    
    def train_internal(self, params):
        eval_distances = self.get_distances(self.eval_df)
        eval_labels = self.eval_df['label'].tolist()
        
        # now, find best score threshold on eval dataset
        highest_accuracy, best_threshold = 0, 0
        granularity = 1000
        for threshold in range(granularity):
            threshold /= float(granularity)
            predictions = [0 if score >= threshold else 1 for score in eval_distances]
            accuracy = sum([1 if pred == label else 0 for pred, label in zip(predictions, eval_labels)]) / len(eval_labels)
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_threshold = threshold
        
        print(f'Optimal cosine distance threshold of {best_threshold} on eval dataset ({len(eval_distances)} total documents).')
        self.threshold = best_threshold
    
    def save_model(self, folder):
        pass
                
    def load_model(self, folder):
        pass
    
    def evaluate_internal(self, df, df_name=None):
        distances = self.get_distances(df, df_name=df_name)
        labels = df['label'].tolist()
        return [0.5 - (distance - self.threshold) / (1 - self.threshold) * 0.5 if distance >= self.threshold 
                else 0.5 + (self.threshold - distance) / (self.threshold) * 0.5 for distance in distances], labels
