# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
from verification_models.O2D2.inference.model_inference import AdHominem_O2D2
from verification_models.O2D2.helper_functions.evaluate import evaluate_all
from verification_models.O2D2.helper_functions.reliability_diagrams import compute_calibration
import json

def run(model_folder, test_file, o2d2_epoch=-1):

    def debug_print(text, other_text=None):
        if other_text is not None: # sloppy hack but who cares
            text = str(text) + str(other_text)
        print(text)
        log_file = open(os.path.join(model_folder, f'{test_file}.log'), 'a')
        log_file.write(f'{str(text)}\n')
        log_file.close()
        
    EPOCH = o2d2_epoch # epoch of best run (AdHominem-O2D2 model)
    if EPOCH == -1:
        for file in os.listdir(os.path.join(model_folder, "results_o2d2", "weights_o2d2")):
            epoch = int(file.split('_')[1])
            EPOCH = max(EPOCH, epoch)
        
    # define batch size
    BATCH_SIZE = 4

    # paths
    dir_data = os.path.join(model_folder, "data_preprocessed")
    dir_results = os.path.join(model_folder, "results_o2d2")

    # load dev set
    with open(os.path.join(dir_data, test_file), 'rb') as f:
        docs_L, docs_R, labels, _ = pickle.load(f)
    labels = np.array(labels)

    # docs_L, docs_R, labels = docs_L[100:201], docs_R[100:201], labels[100:201]

    dev_set = (docs_L, docs_R, labels)

    # load model
    debug_print("load trained model and hyper-parameters...")
    path = os.path.join(dir_results, "weights_o2d2", "weights_" + str(EPOCH))
    with open(path, 'rb') as f:
        parameters = pickle.load(f)

    # build Tensorflow graph with trained weights
    debug_print("build tensorflow graph...")
    adhominem = AdHominem_O2D2(hyper_parameters=parameters['hyper_parameters'],
                            theta_init=parameters['theta'],
                            theta_E_init=parameters['theta_E'],
                            )

    # inference
    debug_print("start inference...")
    pred_dml, pred_bfs, pred_ual, pred_o2d2, n_miss, conf_matrix, lev_L, lev_R, att_w_L, att_w_R, att_s_L, att_s_R \
        = adhominem.evaluate(docs_L, docs_R, batch_size=BATCH_SIZE)


    # compute confidence scores (p if p >= 0.5, otherwise 1-p)
    conf_dml, labels_dml = adhominem.compute_confidence(pred_dml)
    conf_bfs, labels_bfs = adhominem.compute_confidence(pred_bfs)
    conf_ual, labels_ual = adhominem.compute_confidence(pred_ual)
    conf_o2d2, labels_o2d2 = adhominem.compute_confidence(pred_o2d2)

    # store data
    debug_print("store results (predictions, levs)...")
    with open(os.path.join(dir_results, "results_att_lev_pred"), 'wb') as f:
        pickle.dump((pred_dml, pred_bfs, pred_ual, pred_o2d2,
                    n_miss, conf_matrix, lev_L, lev_R,
                    conf_dml, conf_bfs, conf_ual, conf_o2d2,
                    labels_dml, labels_bfs, labels_ual, labels_o2d2,
                    att_w_L, att_w_R, att_s_L, att_s_R,
                    ), f)

    # print results
    debug_print(f"evaluate {len(labels)} documents...")

    debug_print("PAN (dml): ", evaluate_all(pred_y=pred_dml, true_y=labels))
    debug_print("PAN (bfs)", evaluate_all(pred_y=pred_bfs, true_y=labels))
    debug_print("PAN (ual)", evaluate_all(pred_y=pred_ual, true_y=labels))
    debug_print("PAN (o2d2)", evaluate_all(pred_y=pred_o2d2, true_y=labels))

    predictions = pred_o2d2.tolist()
    labels = labels.tolist()

    with open(os.path.join(model_folder, f'{test_file}_predictions.json'), 'w') as out:
        json.dump({'predictions': predictions, 'labels': labels}, out, indent=4)

    # debug_print("# non-responses:", n_miss)
    # debug_print("Calibration (dml)", compute_calibration(true_labels=labels, pred_labels=labels_dml, confidences=conf_dml))
    # debug_print("Calibration (bfs)", compute_calibration(true_labels=labels, pred_labels=labels_bfs, confidences=conf_bfs))
    # debug_print("Calibration (ual)", compute_calibration(true_labels=labels, pred_labels=labels_ual, confidences=conf_ual))
    # debug_print("Calibration (o2d2)", compute_calibration(true_labels=labels, pred_labels=labels_o2d2, confidences=conf_o2d2))

    # debug_print("finished inference...")

    # close session
    adhominem.sess.close()
    
    return predictions, labels
    
    

