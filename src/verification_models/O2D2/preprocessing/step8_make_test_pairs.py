import pandas as pd
import random
import pickle
import os
import numpy as np
from pathlib import Path

def run(model_folder, test_files):

    dir_results = os.path.join(model_folder, 'data_preprocessed')
    file_results = os.path.join(dir_results, 'results.txt')

    for test_file in test_files:
        test_name = Path(test_file).stem

        raw_data = pd.read_csv(test_file)
        
        same_genre = True
        for column in raw_data.columns:
            if 'genre' in column:
                same_genre = False
                break

        docs_L, docs_R, labels_a, labels_c = [], [], [], []

        for _, row in raw_data.iterrows():
            labels_a.append(row['label'])
            labels_c.append(1 if same_genre else 0)
            if random.random() < 0.5:
                docs_L.append(row['text0'])
                docs_R.append(row['text1'])
            else:
                docs_L.append(row['text1'])
                docs_R.append(row['text0'])
                
        
        dict_counts = {"SA_SF": 0,
                    "SA_DF": 0,
                    "DA_SF": 0,
                    "DA_DF": 0,
                    }
        for i in range(len(docs_L)):

            if labels_a[i] == 1 and labels_c[i] == 1:
                dict_counts["SA_SF"] += 1
            if labels_a[i] == 1 and labels_c[i] == 0:
                dict_counts["SA_DF"] += 1
            if labels_a[i] == 0 and labels_c[i] == 1:
                dict_counts["DA_SF"] += 1
            if labels_a[i] == 0 and labels_c[i] == 0:
                dict_counts["DA_DF"] += 1

        open(file_results, 'a').write(f'{test_name}: '
                                    + ', #pairs: ' + str(len(labels_a))
                                    + ', a=0: ' + str(np.sum(np.array(labels_a) == 0))
                                    + ', a=1: ' + str(np.sum(np.array(labels_a) == 1))
                                    + ', c=0: ' + str(np.sum(np.array(labels_c) == 0))
                                    + ', c=1: ' + str(np.sum(np.array(labels_c) == 1))
                                    + '\n')
        open(file_results, 'a').write('SA_SF: ' + str(dict_counts["SA_SF"])
                                    + ', SA_DF: ' + str(dict_counts["SA_DF"])
                                    + ', DA_SF: ' + str(dict_counts["DA_SF"])
                                    + ', DA_DF: ' + str(dict_counts["DA_DF"])
                                    + '\n')
        open(file_results, 'a').write('-----------------------------------------------\n')
        
        with open(os.path.join(dir_results, test_name), 'wb') as f:
            pickle.dump((docs_L, docs_R, labels_a, labels_c), f)