from abc import ABC, abstractmethod
import pandas as pd
import os
import json
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from pathlib import Path
import pdb

from utils import evaluate_attribution_scores
from fileutils import create_model_folder

class AttributionModel(ABC):
    
    def __init__(self, args, parameter_set):
        
        self.parameter_set = parameter_set
        self.args = args
        
        self.query_df = pd.read_csv(args.query_file)
        if args.test:
            self.target_df = pd.read_csv(args.target_file)
            assert set(self.target_df['author']) == (set(self.query_df['author']))
        
        author_list = set(self.query_df['author'])
            
        self.author_to_author_id = {author: i for i, author in enumerate(author_list)}
        
        def update_authors(row, author_to_author_id):
            row['author'] = author_to_author_id[row['author']]
            return row
        
        if hasattr(self, 'query_df'):
            self.query_df = self.query_df.apply(lambda row: update_authors(row, self.author_to_author_id), axis=1)
        if hasattr(self, 'target_df'):
            self.target_df = self.target_df.apply(lambda row: update_authors(row, self.author_to_author_id), axis=1)
            
        if args.load:
            self.model_folder = args.load_folder
            self.load_model(args.load_folder)
        elif args.train:
            self.model_folder = create_model_folder(self.get_model_name(), args.query_file, args, parameter_set, base_folder=args.save_folder)
                
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
    def evaluate_internal(self, query_df, target_df, df_name=None):
        # returns List of Lists of size text_data X num_authors, corresponding to probabilities of each author
        pass
        
        
    def evaluate(self, df_name=None):
        """Evaluates the model on a specific dataset.
        """
        all_scores = self.evaluate_internal(self.query_df, self.target_df, df_name=df_name)
        
        if type(all_scores) == tuple: # logging responses for LLMs
            all_scores, all_responses = all_scores
        else:
            all_scores, all_responses = all_scores, None
        
        id_to_author = {i: author for author, i in self.author_to_author_id.items()}
        author_ids = sorted(list(id_to_author.keys()))
        author_list = [id_to_author[i] for i in author_ids] # not id's, but original names
        
        results = []
        
        for i, scores in enumerate(all_scores):
            row = self.target_df.iloc[i]
            if all_responses is not None:
                results.append({
                    'id': str(row['id']),
                    'genre': row['genre'],
                    'label': id_to_author[row['author']],
                    'prediction': id_to_author[scores.index(max(scores))],
                    'rank': sorted(scores, reverse=True).index(scores[row['author']]) + 1,
                    'response': all_responses[i],
                    'scores': scores
                })
            else:
                results.append({
                    'id': str(row['id']),
                    'genre': row['genre'],
                    'label': id_to_author[row['author']],
                    'prediction': id_to_author[scores.index(max(scores))],
                    'rank': sorted(scores, reverse=True).index(scores[row['author']]) + 1,
                    'scores': scores
                })
        
        return results, author_list
        
    def test_and_save(self):
        """Tests on all test datasets and saves the results.
        """
        
        results, author_list = self.evaluate()
        
        scores = {}
        scores['Overall'] = evaluate_attribution_scores(results)
        scores['Article'] = evaluate_attribution_scores(results, lambda x: x['genre'] == 'Article')
        scores['Tweet'] = evaluate_attribution_scores(results, lambda x: x['genre'] == 'Tweet')
        
        with open(os.path.join(self.model_folder, 'test_results.json'), 'w') as results_file:
            json.dump(scores, results_file, indent=4)
        with open(os.path.join(self.model_folder, 'predictions.json'), 'w') as predictions_file:
            json.dump({'predictions': results, 'author_list': author_list}, predictions_file, indent=4)
        
        # predictions_folder = os.path.join(self.model_folder, 'predictions')
        # os.makedirs(predictions_folder, exist_ok=True)
        # results_dict = {}
        
        # for test_name, test_df in self.test_dfs.items():
            # scores, predictions, labels = self.evaluate(test_df, df_name=test_name)
            # results_dict[test_name] = scores
            # with open(os.path.join(predictions_folder, f'{test_name}.csv'), 'w') as predictions_file:
                # json.dump({'predictions': predictions, 'labels': labels}, predictions_file, indent=4)
        
        # with open(os.path.join(self.model_folder, 'test_results.json'), 'w') as results_file:
            # json.dump(results_dict, results_file, indent=4)