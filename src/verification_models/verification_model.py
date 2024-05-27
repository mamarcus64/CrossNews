from abc import ABC, abstractmethod
import pandas as pd
import os
import json
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import evaluate_scores
from fileutils import create_model_folder

class VerificationModel(ABC):
    
    def __init__(self, args, parameter_set):
        
        self.parameter_set = parameter_set
        self.args = args
        
        if args.load:
            self.model_folder = args.load_folder
            self.load_model(args.load_folder)
        elif args.train:
            self.model_folder = create_model_folder(self.get_model_name(), args.train_file, args, parameter_set)
        if args.train:
            self.train_df = pd.read_csv(args.train_file)
            if args.eval_ratio > 0:
                self.train_df, self.eval_df = train_test_split(self.train_df, test_size=args.eval_ratio, random_state=args.seed)
        if args.test:
            self.test_dfs = {}
            for file in args.test_files:
                file_stem = Path(file).stem
                self.test_dfs[file_stem] = pd.read_csv(file)
                
    @abstractmethod
    def get_model_name(self) -> str:
        """Returns model name

        Returns:
            str: model name, i.e. "ngram"
        """
        pass
    
    @abstractmethod
    def train_internal(self, params) -> None:
        """Trains a model. To be used in conjunction with save_model, i.e. by setting instance variables to save the model.
        """
        pass
    
    def train(self):
        """Trains the model on a specific variation and saves the model.
        """
        self.train_internal(self.parameter_set)
        self.save_model(self.model_folder)
    
    @abstractmethod
    def save_model(self, folder):
        """Saves the model to the specific folder. To be used after train_internal(), which should set the instance variables to be saved.

        Args:
            folder: Location of the file(s) to be saved.
        """
        pass
    
    @abstractmethod
    def load_model(self, folder):
        """Loads the model from an output of save_model().

        Args:
            folder: Location of the file(s) to be loaded.
        """
        pass
    
    @abstractmethod
    def evaluate_internal(self, df, df_name=None) -> Tuple[List[float], List[int]]:
        """Evalutes the trained model on the df.

        Args:
            df: dataframe to evaluate on.

        Returns:
            Tuple[List[float], List[int]]: Tuple of (predictions, labels), where predictions is a float score from 0 to 1 and labels is either 0 or 1.
        """
        pass
        
        
    def evaluate(self, df, df_name=None):
        """Evaluates the model on a specific dataset.
        """
        predictions, labels = self.evaluate_internal(df, df_name=df_name)
        result = evaluate_scores(predictions, labels)
        return result, predictions, labels
        
    def test_and_save(self):
        """Tests on all test datasets and saves the results.
        """
        
        predictions_folder = os.path.join(self.model_folder, 'predictions')
        os.makedirs(predictions_folder, exist_ok=True)
        results_dict = {}
        
        for test_name, test_df in self.test_dfs.items():
            scores, predictions, labels = self.evaluate(test_df, df_name=test_name)
            results_dict[test_name] = scores
            with open(os.path.join(predictions_folder, f'{test_name}.csv'), 'w') as predictions_file:
                json.dump({'predictions': predictions, 'labels': labels}, predictions_file, indent=4)
        
        with open(os.path.join(self.model_folder, 'test_results.json'), 'w') as results_file:
            json.dump(results_dict, results_file, indent=4)