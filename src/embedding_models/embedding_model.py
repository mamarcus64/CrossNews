from abc import ABC, abstractmethod
import pandas as pd
import os
import json
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import evaluate_scores
from fileutils import create_model_folder

class EmbeddingModel(ABC):
    
    def __init__(self, args, parameter_set):
        
        self.parameter_set = parameter_set
        self.args = args
        
        if args.load:
            self.model_folder = args.load_folder
        elif args.train:
            self.model_folder = create_model_folder(self.get_model_name(), args.train_file, args, parameter_set, base_folder=args.save_folder)
            self.train_df = pd.read_csv(args.train_file)
            if args.eval_ratio > 0:
                self.train_df, self.eval_df = train_test_split(self.train_df, test_size=args.eval_ratio, random_state=args.seed)
            self.preprocess_verification_data(self.train_df, 'train')
            if hasattr(self, 'eval_df'):
                self.preprocess_verification_data(self.eval_df, 'val')
                
                
    @abstractmethod
    def preprocess_verification_data(self, df, save_name):
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def get_embeddings(self, texts):
        pass