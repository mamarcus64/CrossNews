import os
import pickle
import re
import json
import argparse
import time
import numpy as np
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
# from sklearn.svm import SVC  # used in the original implementation but very slow on large datasets
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from typing import List, Callable, Tuple
from attribution_models.attribution_model import AttributionModel

def base_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = re.sub("[0-9]", "0", string)  # each digit will be represented as a 0
    string = re.sub(r'( \n| \t)+', '', string)
    # text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)
    return string


def char_diff_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    string = re.sub("[a-zA-Z]+", "*", string)
    # string = ''.join(['*' if char.isalpha() else char for char in string])
    return string


def word_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    # if model is a word n-gram model, remove all punctuation
    string = ''.join([char for char in string if char.isalnum() or char.isspace()])
    return string


def get_vectorizers(analyzer: str = 'char',
                    gram_range: List = (1, 2),
                    preprocessor: Callable = base_preprocessor,
                    max_features: int = 1000,
                    min_df: float = 0.1,
                    smooth_idf: bool = True,
                    sublinear_tf: bool = True) -> Tuple[CountVectorizer, TfidfTransformer]:
    """
    Get a vectorizer for this project
    """
    print(f'Building a {gram_range} TfidfVectorizer for {analyzer} with the {preprocessor} preprocessor.')
    print(f'Other params:\n\t\tmax_features: {max_features}\n\t\tmin_df: {min_df}\n\t\tsmooth_idf: '
                  f'{smooth_idf}\n\t\tsublinear_tf: {sublinear_tf}')
    count_vectorizer = CountVectorizer(decode_error='ignore', strip_accents='unicode', lowercase=False, stop_words=None,
                                       ngram_range=gram_range, analyzer=analyzer, min_df=min_df,
                                       max_features=max_features)
    tfidf_vectorizer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    return count_vectorizer, tfidf_vectorizer

class NGram_AA(AttributionModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        self.parameter_set = argparse.Namespace(**self.parameter_set)
        
    def get_model_name(self):
        return 'ngram_aa'
    
    def vectorizer_filename(self, analyzer, type):
        return f'{analyzer}_{type}_vectorizer.pkl'
    
    def get_analyzer_props(self, analyzer):
        params = self.parameter_set
        if analyzer == 'char':
            gram_range = params.char_range
            preprocessor = base_preprocessor
        elif analyzer == 'dist_char':
            gram_range = params.dist_char_range
            preprocessor = char_diff_preprocessor
        elif analyzer == 'word':
            gram_range = params.word_range
            preprocessor = word_preprocessor
        return gram_range, preprocessor
    
    def train_internal(self, params):
        
        self.train_texts = self.query_df['text'].tolist()
        self.train_labels = self.query_df['author'].tolist()
    
        self.vectorizers = {}
        self.train_term_matrix = {}
        self.train_data = {}
    
        for analyzer in params.analyzers:
            gram_range, preprocessor = self.get_analyzer_props(analyzer)
            print(f'{analyzer}: building the tf-idf vectorizer for the {analyzer} n-gram model')
            count_vectorizer, tfidf_transformer = get_vectorizers(analyzer=analyzer if 'dist' not in analyzer else 'char',
                                                                gram_range=gram_range,
                                                                preprocessor=preprocessor,
                                                                max_features=params.max_features,
                                                                min_df=params.min_df,
                                                                smooth_idf=True,
                                                                sublinear_tf=params.sublinear_tf)

            print(f'{analyzer}: fitting the count vectorizer')
            start = time.time()
            self.train_term_matrix[analyzer] = count_vectorizer.fit_transform(self.train_texts).toarray()
            print(f'took {(time.time() - start) / 60} minutes')
            print(f'{analyzer}: fitting the tfidf vectorizer')
            start = time.time()
            self.train_data[analyzer] = tfidf_transformer.fit_transform(self.train_term_matrix[analyzer]).toarray()
            print(f'took {(time.time() - start) / 60} minutes')
            
            self.vectorizers[analyzer] = {
                'count': count_vectorizer,
                'tfidf': tfidf_transformer
            }
            
    def save_model(self, folder):
        print(f'saving vectorizers to {folder}')
        for analyzer_name, vectorizer_dict in self.vectorizers.items():
            for vectorizer_name, vectorizer in vectorizer_dict.items():
                filename = self.vectorizer_filename(analyzer_name, vectorizer_name)
                with open(os.path.join(folder, filename), 'wb') as f:
                    pickle.dump(vectorizer, f)
                
    def load_model(self, folder):
        self.train_texts = self.query_df['text'].tolist()
        self.train_labels = self.query_df['author'].tolist()
        
        self.vectorizers = {}
        self.train_term_matrix = {}
        self.train_data = {}
        for vectorizer_type in ['count', 'tfidf']:
            for analyzer in ['char', 'dist_char', 'word']:
                filename = self.vectorizer_filename(analyzer, vectorizer_type)
                if os.path.exists(os.path.join(folder, filename)):
                    if analyzer not in self.vectorizers:
                        self.vectorizers[analyzer] = {}
                    with open(os.path.join(folder, filename), 'rb') as f:
                        self.vectorizers[analyzer][vectorizer_type] = pickle.load(f)
                    if vectorizer_type == 'count':
                        self.train_term_matrix[analyzer] = self.vectorizers[analyzer][vectorizer_type].transform(self.train_texts)
                    elif vectorizer_type == 'tfidf':
                        self.train_data[analyzer] = self.vectorizers[analyzer][vectorizer_type].transform(self.train_term_matrix[analyzer])
    
    def evaluate_internal(self, query_df, target_df, df_name=None):
        # ignoring query_df - in either load_model or train_internal, we have already created self.train_data
        test_texts = target_df['text'].tolist()
        
        self.test_prediction_probabilities = {}
        
        for analyzer in self.parameter_set.analyzers:

            print(f'{analyzer}: vectorizing the test texts')
            tfidf_vectorizer = self.vectorizers[analyzer]['tfidf']
            count_vectorizer = self.vectorizers[analyzer]['count']
            test_data = tfidf_vectorizer.transform(count_vectorizer.transform(test_texts).toarray()).toarray()

            print(f'{analyzer}: scaling the vectorized data')
            max_abs_scaler = preprocessing.MaxAbsScaler()
            scaled_train_data = max_abs_scaler.fit_transform(self.train_data[analyzer])
            scaled_test_data = max_abs_scaler.transform(test_data)

            print(f'{analyzer}: fitting the classifier')
            start = time.time()
            # This was the classifier used in the original implementation, but we need a more efficient one
            # char_std = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, kernel='linear',
            #                                                           gamma='auto', verbose=True)))
            if self.parameter_set.logistic_regression:
                # classifier = LogisticRegression(multi_class='multinomial', dual=dual)
                classifier = SGDClassifier(loss='log', n_jobs=-1, early_stopping=False, verbose=1)
            else:
                classifier = LogisticRegression(multi_class='multinomial', n_jobs=-1, dual=not self.parameter_set.primal)

            classifier.fit(scaled_train_data, self.train_labels)
            print(f'took {(time.time() - start) / 60} minutes')

            print(f'{analyzer}: inference on the test set')
            start - time.time()
            predicted_probs = classifier.predict_proba(scaled_test_data)
            print(f'took {(time.time() - start) / 60} minutes')

            self.test_prediction_probabilities[analyzer] = predicted_probs
            
        print('averaging the models')
        avg_probas = np.average(list(self.test_prediction_probabilities.values()), axis=0)
        return avg_probas.tolist()