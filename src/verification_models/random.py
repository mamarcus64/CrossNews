import random
import os

from verification_models.verification_model import VerificationModel

class Random(VerificationModel):
    
    def get_model_name(self):
        return 'random'
                
    def train_internal(self, params):
        self.seed = params['seed']
        random.seed(self.seed)
    
    def save_model(self, folder):
        with open(os.path.join(folder, 'seed.txt'), 'w') as out:
            out.write(str(self.seed))
                
    def load_model(self, folder):
        with open(os.path.join(folder, 'seed.txt'), 'r') as seed_file:
            self.seed = int(seed_file.readlines()[0])
            random.seed(self.seed)
    
    def evaluate_internal(self, df, df_name=None):
        labels = df['label'].tolist()
        return [random.random() for _ in range(len(labels))], labels
    
