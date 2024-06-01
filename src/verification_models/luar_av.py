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
        part_args = {
            'model': 'luar',
            'train': False,
            'load': True,
            'load_folder': parameter_set['model_folder'],
            'parameter_sets': [parameter_set['parameter_set']],
            'evaluation_metric': 'F1'
        }
        
        luar_parameter_set = json.load(open('src/model_parameters/luar.json', 'r'))[parameter_set['parameter_set']]
        
        self.part_model = LUAR(SimpleNamespace(**part_args), luar_parameter_set)
    
    def get_model_name(self):
        return 'luar_av'
    
    def get_distances(self, df, df_name=None):
        id_to_doc = {}
        for _, row in df.iterrows():
            id_to_doc[row['id0']] = row['text0']
            id_to_doc[row['id1']] = row['text1']
            
        ids = list(id_to_doc.keys())
        texts = [id_to_doc[id] for id in ids]
        embeddings = self.part_model.get_embeddings(texts)
        id_to_embedding = {id: embedding for id, embedding in zip(ids, embeddings)}
        
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

"""
salloc -c 16 -G a40

date
dataset="CrossNews_mini.csv"
model="luar_av"
conda activate luar
cd /nethome/mma81/storage/CrossNews

python src/run_verification.py \
--model ${model} \
--train \
--train_file verification_data/train/${dataset} \
--parameter_sets default \
--test \
--test_files verification_data/test/CrossNews_Tweet_Tweet.csv \
verification_data/test/CrossNews_Article_Article.csv \
verification_data/test/CrossNews_Article_Tweet.csv

date
exit

"""
