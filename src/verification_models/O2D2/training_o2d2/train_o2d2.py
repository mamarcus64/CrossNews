# -*- coding: utf-8 -*-
from verification_models.O2D2.training_o2d2.adhominem_o2d2 import AdHominem_O2D2
import pickle
import os

# import argparse
# parser = argparse.ArgumentParser(description='AdHominem for PAN 2020 and 2021')
# parser.add_argument('-thr_0', default=0.3, type=float)  # lower threshold for O2D2
# parser.add_argument('-thr_1', default=0.7, type=float)  # upper threshold for O2D2
# parser.add_argument('-epoch_trained', default=32, type=int)  # best epoch of trained AdHominem model
# parser.add_argument('-lr_start', default=0.001, type=float)  # initial learning rate
# parser.add_argument('-lr_end', default=0.0005, type=float)  # lower bound for learning rate
# parser.add_argument('-lr_epoch', default=100, type=float)  # epoch, when achieving the lower bound
# parser.add_argument('-epochs', default=60, type=int)  # total number of epochs
# parser.add_argument('-batch_size', default=30, type=int)  # batch size for training
# parser.add_argument('-batch_size_val', default=30, type=int)  # batch size for evaluation
# parser.add_argument('-retrain_chr_emb', default=False, type=bool)  # retrain certain layers
# parser.add_argument('-retrain_wrd_emb', default=False, type=bool)
# parser.add_argument('-retrain_cnn', default=False, type=bool)
# parser.add_argument('-retrain_bilstm', default=False, type=bool)
# parser.add_argument('-retrain_dml', default=False, type=bool)
# parser.add_argument('-retrain_loss_dml', default=False, type=bool)
# parser.add_argument('-retrain_bfs', default=False, type=bool)
# parser.add_argument('-retrain_ual', default=False, type=bool)
# parser.add_argument('-keep_prob_cnn', default=1.0, type=float)  # apply dropout when computing LEVs
# parser.add_argument('-keep_prob_lstm', default=1.0, type=float)
# parser.add_argument('-keep_prob_att', default=1.0, type=float)
# parser.add_argument('-keep_prob_metric', default=1.0, type=float)
# parser.add_argument('-keep_prob_bfs', default=1.0, type=float)
# parser.add_argument('-keep_prob_ual', default=1.0, type=float)
# parser.add_argument('-rate_o2d2', default=0.25, type=float) # O2D2 dropout rate

def run(model_folder, hyper_parameters_new):

    # create folder for results
    dir_results = os.path.join(model_folder, 'results_o2d2')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
        
    # find corresponding version of trained adhominem
    adhominem_epoch = hyper_parameters_new["epoch_trained"]
    if adhominem_epoch == -1: # find most recent epoch
        for file in os.listdir(os.path.join(model_folder, "results_adhominem", "weights_adhominem")):
            epoch = int(file.split('_')[1])
            adhominem_epoch = max(adhominem_epoch, epoch)

    # load trained model and hyper-parameters
    with open(os.path.join(model_folder,
                           "results_adhominem",
                           'weights_adhominem',
                           "weights_" + str(adhominem_epoch),
                           ), 'rb') as f:
        parameters = pickle.load(f)

    # overwrite old variables
    for hyper_parameter in hyper_parameters_new:
        parameters["hyper_parameters"][hyper_parameter] = hyper_parameters_new[hyper_parameter]

    # load validation set
    with open(os.path.join(model_folder, 'data_preprocessed', "pairs_cal"), 'rb') as f:
        docs_L, docs_R, labels, _ = pickle.load(f)
    val_set = (docs_L, docs_R, labels)
    parameters["hyper_parameters"]['N_val'] = len(labels)

    # file to store results epoch-wise
    file_results = os.path.join(dir_results, 'results.txt')
    # temporary file to store results batch-wise
    file_tmp = os.path.join(dir_results, 'tmp.txt')

    # delete already existing files
    if os.path.isfile(file_results):
        os.remove(file_results)
    if os.path.isfile(file_tmp):
        os.remove(file_tmp)

    # write hyper-parameters setup into file (results.txt)
    open(file_results, 'a').write('\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\nPARAMETER SETUP:\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\n'
                                  )
    for hp in sorted(parameters["hyper_parameters"].keys()):
        if hp in ['V_c', 'V_w']:
            open(file_results, 'a').write('num ' + hp + ': ' + str(len(parameters["hyper_parameters"][hp])) + '\n')
        else:
            open(file_results, 'a').write(hp + ': ' + str(parameters["hyper_parameters"][hp]) + '\n')

    # load neural network model
    adhominem_o2d2 = AdHominem_O2D2(hyper_parameters=parameters['hyper_parameters'],
                                    theta_init=parameters['theta'],
                                    theta_E_init=parameters['theta_E'],
                                    )
    # start training
    adhominem_o2d2.train_model(model_folder, val_set, file_results, file_tmp, dir_results)
    # close session
    adhominem_o2d2.sess.close()