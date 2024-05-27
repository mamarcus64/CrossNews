# Code based on this repository: https://github.com/boenninghoff/pan_2020_2021_authorship_verification
from pathlib import Path

from verification_models.verification_model import VerificationModel
from verification_models.O2D2.preprocessing import step0_convert_to_jsonl, \
step1_parse_and_split, step2_preprocess, step3_count, step4_make_vocabularies, \
step5_sample_pairs_cal, step6_sample_pairs_val, step7_sample_pairs_dev, step8_make_test_pairs
from verification_models.O2D2.training_adhominem import train_adhominem
from verification_models.O2D2.training_o2d2 import train_o2d2
from verification_models.O2D2.inference import run_inference

class O2D2(VerificationModel):
    
    def __init__(self, args, parameter_set):
        args.eval_ratio = 0 # o2d2 uses its own evaluation setup in the code
        super().__init__(args, parameter_set)
        
        if hasattr(self, 'train_df') and self.train:
            step0_convert_to_jsonl.run(self.train_df, self.model_folder)
            step1_parse_and_split.run(self.model_folder)
            step2_preprocess.run(self.model_folder)
            step3_count.run(self.model_folder)
            step4_make_vocabularies.run(self.model_folder)
            step5_sample_pairs_cal.run(self.model_folder)
            step6_sample_pairs_val.run(self.model_folder)
            step7_sample_pairs_dev.run(self.model_folder)
        if args.test_files is not None:
            step8_make_test_pairs.run(self.model_folder, args.test_files)
    
    def get_model_name(self):
        return 'O2D2'
                
    def train_internal(self, params):
        train_adhominem.run(self.model_folder, params['adhominem_arguments'])
        train_o2d2.run(self.model_folder, params["o2d2_arguments"])
        
    
    def save_model(self, folder):
        # this code already saves the models internally
        pass
                
    def load_model(self, folder):
        # additionally, evaluate_internal loads correctly as long as the model folder is set correctly.
        pass
    
    def evaluate_internal(self, df, df_name=None):
        file_stem = Path(df_name).stem
        return run_inference.run(self.model_folder, file_stem)
