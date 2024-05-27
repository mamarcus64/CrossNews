# -*- coding: utf-8 -*-
import os
from verification_models.O2D2.helper_functions.resample_pairs import sample_pairs_single_epoch
import numpy as np
import pickle
import os


def run(model_folder):
    #####################
    # load validation set
    #####################
    dir_results = os.path.join(model_folder, 'data_preprocessed')
    file_results = os.path.join(dir_results, 'results.txt')

    open(file_results, 'a').write('-----------------------------------------------\n')

    with open(os.path.join(dir_results, 'dict_author_fandom_doc_val_tokenized'), 'rb') as f:
        dict_author_fandom_doc = pickle.load(f)

    #######################################
    # sample fixed pairs for validation set
    #######################################
    max_tries = 10
    tries = 0
    docs_L = []
    while len(docs_L) == 0:
        docs_L, docs_R, labels_a, labels_c = sample_pairs_single_epoch(dict_author_fandom_doc,
                                                                    delta_1=0.9,
                                                                    delta_2=0.9,
                                                                    delta_3=0.7,
                                                                    only_SADF_and_DASF=False,
                                                                    make_balanced=False,
                                                                    balance_factor=1.0,
                                                                    )
        if tries >= max_tries:
            raise Exception(f"Could not sample valid pairs within {max_tries} tries.")
        tries += 1

    ###################
    # check re-sampling
    ###################
    # counts
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

    open(file_results, 'a').write('val: '
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

    #######
    # store
    #######
    with open(os.path.join(dir_results, 'pairs_val'), 'wb') as f:
        pickle.dump((docs_L, docs_R, labels_a, labels_c), f)
