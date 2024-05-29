import pickle
from math import log
import os
import json
import time
import argparse
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from typing import Dict
import logging
import pandas as pd
from multiprocessing import Pool

from verification_models.verification_model import VerificationModel



class Model(object):
    # cnt - count of characters read
    # modelOrder - order of the model
    # orders - List of Order-Objects
    # alphSize - size of the alphabet
    def __init__(self, order, alphSize):
        self.cnt = 0
        self.alphSize = alphSize
        self.modelOrder = order
        self.orders = []
        for i in range(order + 1):
            self.orders.append(Order(i))

    # print the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printModel(self):
        s = "Total characters read: " + str(self.cnt) + "\n"
        for i in range(self.modelOrder + 1):
            self.printOrder(i)

    # print a specific order of the model
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printOrder(self, n):
        o = self.orders[n]
        s = "Order " + str(n) + ": (" + str(o.cnt) + ")\n"
        for cont in o.contexts:
            if n > 0:
                s += "  '" + cont + "': (" + str(o.contexts[cont].cnt) + ")\n"
            for char in o.contexts[cont].chars:
                s += "     '" + char + "': " + \
                     str(o.contexts[cont].chars[char]) + "\n"
        s += "\n"
        print(s)

    # updates the model with a character c in context cont
    def update(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than model order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            order.addContext(cont)
        context = order.contexts[cont]
        if not context.hasChar(c):
            context.addChar(c)
        context.incCharCount(c)
        order.cnt += 1
        if order.n > 0:
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if len(s) == 0:
            return
        for i in range(len(s)):
            cont = ""
            if i != 0 and i - self.modelOrder <= 0:
                cont = s[0:i]
            else:
                cont = s[i - self.modelOrder:i]
            self.update(s[i], cont)

    # return the models probability of character c in content cont
    def p(self, c, cont):
        if len(cont) > self.modelOrder:
            raise NameError("Context is longer than order!")

        order = self.orders[len(cont)]
        if not order.hasContext(cont):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if order.n == 0:
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])
        return float(context.getCharCount(c)) / context.cnt

    # merge this model with another model m, esentially the values for every
    # character in every context are added
    def merge(self, m):
        if self.modelOrder != m.modelOrder:
            raise NameError("Models must have the same order to be merged")
        if self.alphSize != m.alphSize:
            raise NameError("Models must have the same alphabet to be merged")
        self.cnt += m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].merge(m.orders[i])

    # make this model the negation of another model m, presuming that this
    # model was made my merging all models
    def negate(self, m):
        if self.modelOrder != m.modelOrder or self.alphSize != m.alphSize or self.cnt < m.cnt:
            raise NameError("Model does not contain the Model to be negated")
        self.cnt -= m.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].negate(m.orders[i])


class Order(object):
    # n - whicht order
    # cnt - character count of this order
    # contexts - Dictionary of contexts in this order
    def __init__(self, n):
        self.n = n
        self.cnt = 0
        self.contexts = {}

    def hasContext(self, context):
        return context in self.contexts

    def addContext(self, context):
        self.contexts[context] = Context()

    def merge(self, o):
        self.cnt += o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                self.contexts[c] = o.contexts[c]
            else:
                self.contexts[c].merge(o.contexts[c])

    def negate(self, o):
        if self.cnt < o.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= o.cnt
        for c in o.contexts:
            if not self.hasContext(c):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.contexts[c].negate(o.contexts[c])
        empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
        for c in empty:
            del self.contexts[c]


class Context(object):
    # chars - Dictionary containing character counts of the given context
    # cnt - character count of this context
    def __init__(self):
        self.chars = {}
        self.cnt = 0

    def hasChar(self, c):
        return c in self.chars

    def addChar(self, c):
        self.chars[c] = 0

    def incCharCount(self, c):
        self.cnt += 1
        self.chars[c] += 1

    def getCharCount(self, c):
        return self.chars[c]

    def merge(self, cont):
        self.cnt += cont.cnt
        for c in cont.chars:
            if not self.hasChar(c):
                self.chars[c] = cont.chars[c]
            else:
                self.chars[c] += cont.chars[c]

    def negate(self, cont):
        if self.cnt < cont.cnt:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.cnt -= cont.cnt
        for c in cont.chars:
            if (not self.hasChar(c)) or (self.chars[c] < cont.chars[c]):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.chars[c] -= cont.chars[c]
        empty = [c for c in self.chars if self.chars[c] == 0]
        for c in empty:
            del self.chars[c]


# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    n = len(s)
    h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]
        h -= log(m.p(s[i], context), 2)
    return h / n


# Calculates the cross-entropy of text2 using the model of text1 and vice-versa
# Returns the mean and the absolute difference of the two cross-entropies
def distance(text1, text2, ppm_order=5, label=None):
    mod1 = Model(ppm_order, 256)
    mod1.read(text1)
    d1 = h(mod1, text2)  # this is essentially perplexity of model trained on 1 and tested on 2
    mod2 = Model(ppm_order, 256)
    mod2.read(text2)
    d2 = h(mod2, text1)
    # peroplexity 1-on-2, perplexity 2-on-1, average perplexity, perplexity difference
    if label is not None:
        return [label, [d1, d2, (d1 + d2) / 2.0, abs(d1 - d2)]]
    return [d1, d2, (d1 + d2) / 2.0, abs(d1 - d2)]


class ppm_training_data_generator:
    def __init__(self, train_data, ppm_order=5, cache_dir='test', num_workers=0,):
        self.train_data = train_data
        self.ppm_order = ppm_order
        self.cache_dir = cache_dir
        self.num_workers = num_workers

        self.classifier_training_data = []

    def get_train_data(self):
        
        print('Getting train data...')
        result = []
        for _, row in tqdm(self.train_data.iterrows(), total=len(self.train_data)):
            same_diff, text0, text1 = row['label'], row['text0'], row['text1']
            
            result.append(distance(text0, text1, self.ppm_order, same_diff))
        
        return result

def train_classifier(X_train, y_train):
    logging.info('fitting the logistic regression model')
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg

def eval_sample(txt0, txt1, ppm_order, logreg, true_lbl=None, test_index=None):
    dists = distance(txt0, txt1, ppm_order)
    proba = logreg.predict_proba([dists])
    if true_lbl is not None:
        return proba[0][1], true_lbl, test_index
    return proba[0][1]

class PPM(VerificationModel):
    
    def __init__(self, args, parameter_set, num_workers=16):
        super().__init__(args, parameter_set)
        self.num_workers = num_workers
        
    def get_model_name(self):
        return 'ppm'
                
    def train_internal(self, params):
        
        self.ppm_order = params['order']
        
        self.temp_path = self.model_folder
        os.makedirs(self.temp_path, exist_ok=True)
        
        data_generator = ppm_training_data_generator(self.train_df, self.ppm_order, self.temp_path, num_workers=self.num_workers)
        classifier_training_data = data_generator.get_train_data()
        
        y_train = []
        X_train = []
        for lbl, dists in classifier_training_data:
            y_train.append(lbl)
            X_train.append(dists)
        print('Training the classifier...')
        # now train the logistic regression classifier
        self.logreg = train_classifier(X_train, y_train)
        
        
    def save_model(self, folder):
        save_file = os.path.join(folder, 'model.clf')
        with open(save_file, 'wb') as f:
            pickle.dump((self.ppm_order, self.logreg), f)
    
    def load_model(self, folder):
        load_file = os.path.join(folder, 'model.clf')
        with open(load_file, 'rb') as f:
            self.ppm_order, self.logreg = pickle.load(f)
    
    def evaluate_internal(self, df, df_name=None):
        test_data = []
        for _, row in df.iterrows():
            test_data.append([row['label'], row['text0'], row['text1']])
    
        probas_and_true_lbls = []

        with Pool(processes=self.num_workers) as pool:
            async_results, idx = {}, 0
            for test_index, (true_lbl, txt0, txt1) in enumerate(test_data):
                async_results[idx] = pool.apply_async(eval_sample, (txt0, txt1, self.ppm_order, self.logreg, true_lbl, test_index))
                idx += 1

            logging.info(f'PPM_AV: finished launching, now awaiting the processing')
            done = False
            start_time = time.time()
            last_check = time.time()
            loops = 0
            num_removed = 0
            while not done:
                remove_idxs = []
                loops += 1
                for idx, result in async_results.items():
                    if result.ready():
                        res = result.get()
                        # do the things with this data. . .
                        probas_and_true_lbls.append(res)
                        remove_idxs.append(idx)
                num_removed += len(remove_idxs)
                for idx in remove_idxs:
                    del async_results[idx]
                if len(async_results.keys()) < 1:
                    done = True
                if time.time() - last_check > 30:
                    elapsed = time.time() - start_time
                    res_left = len(list(async_results.keys()))
                    logging.info(
                        f'PPM_AV: {res_left} results remaining in queue, {elapsed:.2f}, {num_removed} removed, '
                        f'{loops} loops ran.')
                    logging.info(
                        f'PPM_AV: approximately {(res_left / (num_removed / 30)) / 60} minutes remaining')
                    last_check = time.time()
                    num_removed = 0

            logging.info(f'PPM_AV: all evaluation results gathered')
            probas, true_lbls, test_indices = [], [], []
            for proba, lbl, test_index in probas_and_true_lbls:
                probas.append(proba)
                true_lbls.append(lbl)
                test_indices.append(test_index)
            
        
        # sort by test_index to restore original order
        probas = [prob for _, prob in sorted(zip(test_indices, probas), key=lambda pair: pair[0])]
        true_lbls = [lbl for _, lbl in sorted(zip(test_indices, true_lbls), key=lambda pair: pair[0])]
        
        return probas, true_lbls