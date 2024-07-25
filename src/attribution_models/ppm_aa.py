"""
This is a modified version of the python file found here:
    https://github.com/pan-webis-de/teahan03/blob/master/teahan03.py
"""
from math import log
import pickle
import os
import pandas as pd
from tqdm import tqdm
import time
from attribution_models.attribution_model import AttributionModel
import scipy
import argparse
import pdb
from pathlib import Path

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
            if(n > 0):
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
        if (order.n > 0):
            self.update(c, cont[1:])
        else:
            self.cnt += 1

    # updates the model with a string
    def read(self, s):
        if (len(s) == 0):
            return
        for i in range(len(s)):
            cont = ""
            if (i != 0 and i - self.modelOrder <= 0):
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
            if (order.n == 0):
                return 1.0 / self.alphSize
            return self.p(c, cont[1:])

        context = order.contexts[cont]
        if not context.hasChar(c):
            if (order.n == 0):
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


# returns model object loaded from 'mpath' using pickle
def load_ppm_model(mpath):
    f = open(mpath, "rb")
    m = pickle.load(f)
    f.close()
    return m


# stores model object 'model' to 'mpath' using pickle
def store_ppm_model(model, mpath):
    f = open(mpath, "wb")
    pickle.dump(model, f)
    f.close()

# calculates the cross-entropy of the string 's' using model 'm'
def h(m, s):
    if type(s) != type('test'):
        s = str(s) # hack fix
    n = len(s)
    _h = 0
    for i in range(n):
        if i == 0:
            context = ""
        elif i <= m.modelOrder:
            context = s[0:i]
        else:
            context = s[i - m.modelOrder:i]
        _h -= log(m.p(s[i], context), 2)
    return _h / n

# creates models of candidates in 'candidates'
# updates each model with any files stored in the subdirectory of 'corpusdir' named with the candidates name
# stores each model named under the candidates name in 'modeldir'
def create_models(data, order, alphsize):
    models = {}
    for i in data['author'].unique():
        models[i] = Model(order, alphsize)
        print(f"creating model for author {i}")

        for doc in tqdm(data['text'][data['author'] == i]):
            models[i].read(str(doc))

    return models

# attributes the authorship, according to the cross-entropy ranking.
# attribution is saved in json-formatted structure 'answers'
def create_answers(test_data, models):
    print("attributing authors to unknown texts")
    candidates = list(sorted(models.keys(), key=lambda x: int(x)))
    predicted_authors, scores, probs = [], [], []
    
    true_authors = list(test_data['author'])
    texts = list(test_data['text'])
    for i in tqdm(range(len(true_authors)), desc='Evaluating on target documents'):
        true_author = true_authors[i]
        text = texts[i]
        hs = []
        for cand in candidates:
            hs.append(h(models[cand], text))
        m = min(hs)
        
        # convert to probabilities
        prob = scipy.special.softmax([-x for x in hs]).tolist()
        probs.append(prob)
        
        author = candidates[hs.index(m)]
        hs.sort()
        score = (hs[1] - m) / (hs[len(hs) - 1] - m)

        predicted_authors.append(author)
        scores.append(score)

    return probs

class PPM_AA(AttributionModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        self.parameter_set = argparse.Namespace(**self.parameter_set)
        os.makedirs(os.path.join(self.model_folder, 'ppm_models'), exist_ok=True)
        
    def get_model_name(self):
        return 'ppm_aa'
    
    def train_internal(self, params):
        self.models = create_models(self.query_df, params.order, params.alph_size)
        
    def save_model(self, folder):
        for author, model in self.models.items():
            store_ppm_model(model, os.path.join(folder, 'ppm_models', f'model_{author}.pkl'))
            
    def load_model(self, folder):
        self.models = {}
        for file in os.listdir(os.path.join(folder, 'ppm_models')):
            if file.endswith('.pkl'):
                author = '_'.split(Path(file).stem)[-1]
                self.models[author] = load_ppm_model(os.path.join(folder, 'ppm_models', file))
        
    def evaluate_internal(self, query_df, target_df, df_name=None):
        return create_answers(target_df, self.models)