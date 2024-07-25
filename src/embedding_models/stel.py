import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pdb


from embedding_models.embedding_model import EmbeddingModel
from embedding_models.STEL.preprocess_data import preprocess_data
from embedding_models.STEL.utility.neural_trainer import SentenceBertFineTuner
from embedding_models.STEL.utility.training_const import TRIPLET_LOSS, TRIPLET_EVALUATOR

class STEL(EmbeddingModel):
    
    def preprocess_verification_data(self, df, save_name):
        preprocess_data(df, os.path.join(self.model_folder, f'{save_name}.tsv'))
    
    def get_model_name(self):
        return 'stel'
    
    def train(self):
        train_path = os.path.join(self.model_folder, 'train.tsv')
        dev_path = os.path.join(self.model_folder, 'val.tsv')
        tuner = SentenceBertFineTuner(model_path="roberta-base",
                                      train_filename=train_path,
                                      dev_filename=dev_path,
                                      loss=TRIPLET_LOSS,
                                      evaluation_type=TRIPLET_EVALUATOR,
                                      save_folder=self.model_folder,
                                      save_every=self.parameter_set['save_every'])
        self.model_path = tuner.train(epochs=self.parameter_set['epochs'], batch_size=self.parameter_set['batch_size'])
        self.model = tuner.model
        
    
    def get_embeddings(self, texts):
        
        # load model if not loaded
        if not hasattr(self, 'model'):
            largest_checkpoint_num = -1
            for folder in os.listdir(self.model_folder):
                if folder.startswith('checkpoint'):
                    largest_checkpoint_num = max(largest_checkpoint_num, int(folder.split('-')[-1]))
            checkpoint_folder = os.path.join(self.model_folder, f'checkpoint-{largest_checkpoint_num}')
            print(f'Loading checkpoint {checkpoint_folder}...')
            self.model = SentenceTransformer(checkpoint_folder)
            self.model.max_seq_length = 512
        
        embeddings = []
        indices = list(range(0, len(texts), self.parameter_set['batch_size']))
        for start in tqdm(indices):
            end = min(start + self.parameter_set['batch_size'], len(texts))
            batch_embeddings = self.model.encode(texts[start:end])
            for i in range(batch_embeddings.shape[0]):
                embeddings.append([float(x) for x in list(batch_embeddings[i,:])])
        return embeddings

"""

python src/train_embedding.py --model stel --train --train_file verification_data/train/CrossNews_mini.csv --parameter_sets default


python src/train_embedding.py --model stel --load --load_folder models/stel/CrossNews_mini/06-03-21-20-54-gthqba --parameter_sets default \
    --test --test_files verification_data/test/CrossNews_Article_Article.csv verification_data/test/CrossNews_Article_Tweet.csv



"""