# adapted from https://github.com/janithnw/pan2020_authorship_verification

import pickle
import numpy as np
from tqdm.auto import trange, tqdm
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
import os
import re
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from itertools import chain
from nltk.tag.perceptron import PerceptronTagger
import nltk
import nltk.data
import itertools
from typing_extensions import Self
import spacy
import argparse
import csv
from collections import defaultdict
import pdb

from verification_models.verification_model import VerificationModel

dirname = os.path.dirname(__file__)
regex_chunker = None
ml_chunker = None
tnlp_regex_chunker = None

treebank_tokenizer = nltk.tokenize.TreebankWordTokenizer()

nlp_stanza = None  # stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)
nlp_spacy = None  # spacy.load("en_core_web_sm", disable=['ner'])

tagger = nltk.data.load(os.path.join(dirname, "pos_tagger/treebank_brill_aubt.pickle"))
# if throwing error, might need to run this line in console:
# nltk.download('averaged_perceptron_tagger')
perceptron_tagger = PerceptronTagger()

ground_truth = {}

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}

"""

tweetNLP_grammar = r"""

    NP: {<X>?<D>?<\$>?<A>?(<R>?<A>)*<NOM>}
    NP: {(<O>|<\$>)+}         # Pronouns and propper nouns

    PP: {<P><NP>+}                 # Basic Prepositional Phrase
    PP: {<R|A>+<P><NP>+} 

    # Nominal is a noun, followed optionally by a series of post-modifiers
    # Post modifiers could be:
    # - Prepositional phrase
    # - non-finite postmodifiers (<V><NP>|<V><PP>|<V><NP><PP>)
    # - postnominal relative clause  (who | that) VP 
    NOM: {<L|\^|N>+(<PP>|<V><NP>|<V><PP>|<V><NP><PP>|<P|O><VP>)+}
    NOM: {<L|\^|N>+}
    NP: {<NP><\&><NP>}

    VP: {<R>*<V>+(<NP>|<PP>|<NP><PP>)+}
    VP: {<VP><\&><VP>}
"""


def merge_entries(entries):
    ret = {}
    for k in ['pos_tags', 'pos_tag_chunks', 'pos_tag_chunk_subtrees', 'tokens']:
        l = [e[k] for e in entries]
        ret[k] = list(itertools.chain.from_iterable(l))
    ret['preprocessed'] = '\n'.join([e['preprocessed'] for e in entries])
    return ret

def preprocess_text(text):
    # remove URLs
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' URL ', text)

    return text


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        histo = "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
        histo = history[-1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i + 1]
    return {"pos": pos,
            "word": word,
            "hist": histo,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos)
            }


class ConsecutiveNPChunkTagger(nltk.TaggerI):  # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)  # [_consec-use-fe]
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(  # [_consec-use-maxent]
            train_set, algorithm='IIS', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):  # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def get_nltk_pos_tag_based_regex_chunker():
    global regex_chunker
    if regex_chunker is not None:
        return regex_chunker
    regex_chunker = nltk.RegexpParser(grammar)
    return regex_chunker

def chunk_to_str(chunk):
    if type(chunk) is nltk.tree.Tree:
        return chunk.label()
    else:
        return chunk[1]


def extract_subtree_expansions(t, res):
    if type(t) is nltk.tree.Tree:
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)


def pos_tag_chunk(pos_tags, chunker):
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def tokenize(text, tokenizer):
    if tokenizer == 'treebank':
        return treebank_tokenizer.tokenize(text)
    if tokenizer == 'casual':
        return nltk.tokenize.casual_tokenize(text)
    if tokenizer == 'spacy':
        return map(lambda t: t.text, nlp_spacy(text))
    if tokenizer == 'stanza':
        return map(lambda t: t.text, nlp_stanza(text).iter_tokens())
    raise 'Unknown tokenizer type. Valid options: [treebank, casual, spacy, stanza]'


def prepare_entry(text, tokenizer='treebank'):
    tokens = []
    # Workaround because there re some docuemtns that are repitions of the same word which causes the regex chunker to hang
    prev_token = ''
    # for t in tokenizer.tokenize(text):
    if type(text) != str or len(text) <= 1:
        text = 'empty'
    
    for t in tokenize(text, tokenizer):
        if t != prev_token:
            tokens.append(t)
    tagger_output = tagger.tag(tokens)
    pos_tags = [t[1] for t in tagger_output]
    pos_chunks, subtree_expansions = pos_tag_chunk(tagger_output, get_nltk_pos_tag_based_regex_chunker())

    entry = {
        'preprocessed': text,
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': [preprocess_text(t) for t in tokens]
    }
    return entry

def word_count(entry):
    return len(entry['tokens'])


def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry['tokens']])
    return r


def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    if len(entry['tokens']) == 0:
        return counts
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c / len(entry['tokens']) for c in counts]
    #     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r


def character_count(entry):
    r = len(re.sub('\s+', '', entry['preprocessed']))
    return r


# def spell_err_freq(entry):


# https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    # return (len(hapax) / len(dis)) / len(entry['tokens'])
    return (len(hapax) / len(dis))

def handle_exceptions(func, *args):
    try:
        return func(*args)
    except:
        # print('Error occured', func, *args)
        return 0.0

def pass_fn(x):
    return x


class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=pass_fn, preprocessor=pass_fn,
                                              vocabulary=vocab, norm='l2', ngram_range=(1, n))
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l2',
                                              ngram_range=(1, n))

    def fit(self, x, y=None):
        # pdb.set_trace()
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        # self.vectorizer.fit(x, y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
        # return self.vectorizer.transform(x)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class CustomFreqTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm=None,
                                          ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx

    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames


class MaskedStopWordsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords, n):
        self.stopwords = set(stopwords)
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, min_df=0.1, ngram_range=(1, n))

    def _process(self, entry):
        return [
            entry['tokens'][i] if entry['tokens'][i] in self.stopwords else entry['pos_tags'][i]
            for i in range(len(entry['tokens']))
        ]

    def fit(self, X, y=None):
        X = [self._process(entry) for entry in X]
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X = [self._process(entry) for entry in X]
        return self.vectorizer.transform(X)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class POSTagStats(BaseEstimator, TransformerMixin):
    POS_TAGS = [
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
        'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
        'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
        'WP', 'WP$', 'WRB'
    ]

    def __init__(self):
        pass

    def _process(self, entry):
        tags_dict = defaultdict(set)
        tags_word_length = defaultdict(list)
        for i in range(len(entry['tokens'])):
            tags_dict[entry['pos_tags'][i]].add(entry['tokens'][i])
            tags_word_length[entry['pos_tags'][i]].append(len(entry['tokens'][i]))
        res_tag_fractions = np.array([len(tags_dict[t]) for t in self.POS_TAGS])
        if res_tag_fractions.sum() > 0:
            res_tag_fractions = res_tag_fractions / res_tag_fractions.sum()

        res_tag_word_lengths = np.array(
            [np.mean(tags_word_length[t]) if len(tags_word_length[t]) > 0 else 0 for t in self.POS_TAGS])
        return np.concatenate([res_tag_fractions, res_tag_word_lengths])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._process(entry) for entry in X]

    def get_feature_names(self):
        return ['tag_fraction_' + t for t in self.POS_TAGS] + ['tag_word_length_' + t for t in self.POS_TAGS]


class DependencyFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n=3):
        self.vectorizer = TfidfVectorizer(tokenizer=self.pass_fn, preprocessor=self.pass_fn)
        self.nlp = spacy.load("en_core_web_sm")

    def pass_fn(self, x):
        return x

    def extract_dep_n_grams(self, node, current_list, output, n, curr_depth=0):
        current_list.append(node.dep_)
        output.append(current_list)
        if curr_depth > 200:
            return
        for c in node.children:
            l = current_list.copy()
            if len(l) > n - 1:
                l = l[1:]
            self.extract_dep_n_grams(c, l, output, n, curr_depth + 1)

    def dep_ngrams_to_str(self, ngam_list):
        return ['_'.join(ngrams).lower() for ngrams in ngam_list]

    def process(self, text):
        dep_ngrams = []
        doc = self.nlp(text)
        for s in doc.sents:
            o = []
            self.extract_dep_n_grams(s.root, [], o, 4)
            dep_ngrams.extend(o)
        return self.dep_ngrams_to_str(dep_ngrams)

    def fit(self, x, y=None):
        xx = [self.process(entry['preprocessed']) for entry in x]
        self.vectorizer.fit(xx, y)
        return self

    def transform(self, x):
        xx = [self.process(entry['preprocessed']) for entry in x]
        return self.vectorizer.transform(xx)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


def get_transformer(selected_featuresets=None, char_n=3):
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=char_n)
    word_distr = CustomTfIdfTransformer('preprocessed', 'word', n=3)
    pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{¦}~'
    special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)

    featuresets = [
        ('char_distr', char_distr),
        ('word_distr', word_distr),
        ('pos_tag_distr', pos_tag_distr),
        ('pos_tag_chunks_distr', pos_tag_chunks_distr),
        ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),
        ('special_char_distr', special_char_distr),
        ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),
        ('character_count', CustomFuncTransformer(character_count)),
        ('distr_chars_per_word', CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        ('avg_chars_per_word', CustomFuncTransformer(avg_chars_per_word)),
        ('word_count', CustomFuncTransformer(word_count)),
        ('pos_tag_stats', POSTagStats())
    ]
    if selected_featuresets is None:
        transformer = FeatureUnion(featuresets)
    else:
        transformer = FeatureUnion([f for f in featuresets if f[0] in selected_featuresets])

    # pipeline = Pipeline([('features', transformer), ('selection', VarianceThreshold())])
    return transformer


def fit_transformers(data, selected_featuresets=None, char_n=3):  #, data_fraction=0.01):
    # docs_1 = []
    # docs_2 = []

    transformer = get_transformer(selected_featuresets=selected_featuresets, char_n=char_n)
    scaler = StandardScaler()
    secondary_scaler = StandardScaler()

    X = transformer.fit_transform(data).todense()  # docs_1 + docs_2).todense()
    X = scaler.fit_transform(X)
    X1 = X[:int(len(X)/2)]
    X2 = X[int(len(X)/2):]
    secondary_scaler.fit(np.abs(X1 - X2))

    return transformer, scaler, secondary_scaler


def vectorize(XX, Y, ordered_idxs, transformer, scaler, secondary_scaler, data, vector_Sz):

    batch_size = 5001
    i = 0
    docs1 = []
    docs2 = []
    idxs = []
    labels = []
    # for each sample
    for label, text0, text1 in tqdm(data, total=vector_Sz):

        labels.append(label)
        docs1.append(text0)
        docs2.append(text1)
        idxs.append(ordered_idxs[i])
        i += 1
        if len(labels) >= batch_size:
            x1 = scaler.transform(transformer.transform(docs1).todense())
            x2 = scaler.transform(transformer.transform(docs2).todense())
            XX[idxs, :] = secondary_scaler.transform(np.abs(x1 - x2))
            Y[idxs] = labels

            docs1 = []
            docs2 = []
            idxs = []
            labels = []

    if len(labels) > 0:
        x1 = scaler.transform(transformer.transform(docs1).todense())
        x2 = scaler.transform(transformer.transform(docs2).todense())
        XX[idxs, :] = secondary_scaler.transform(np.abs(x1 - x2))
        Y[idxs] = labels
    # XX.flush()
    # Y.flush()


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class NGram(VerificationModel):
    
    def __init__(self, args, parameter_set):
        super().__init__(args, parameter_set)
        
        if hasattr(self, 'train_df'):
            print('Processing train data...')
            self.train_data = []
            
            for i in tqdm(range(len(self.train_df))):
                row = self.train_df.iloc[i]
                self.train_data.append((row['label'], prepare_entry(row['text0']), prepare_entry(row['text1'])))
    
    
    def get_model_name(self):
        return 'ngram'
                
    def train_internal(self, params):
        
        self.char_n = params['char_n']
        self.num_search_iters = params['num_search_iters']
        
        self.temp_path = self.model_folder
        os.makedirs(self.temp_path, exist_ok=True)
        
        train_sz = len(self.train_data)

        train_raw_texts = []
        for _, text0, text1 in self.train_data:
            train_raw_texts.append(text0)
            train_raw_texts.append(text1)
        
        print('Fitting transformer...', flush=True)
        start_time = time.time()
        self.transformer, self.scaler, self.secondary_scaler = fit_transformers(train_raw_texts, char_n=self.char_n, selected_featuresets=params['selected_features'])
        print(f'took {(time.time() - start_time)/60} seconds')
        feature_sz = len(self.transformer.get_feature_names())
        
        del train_raw_texts
        
        print('Vectorizing train set...', flush=True)
        print(f'trying to allocate array of shape: {(train_sz, feature_sz)}')
        
        # XX_train = np.memmap(os.path.join(self.temp_path, 'vectorized_XX_train.npy'), dtype='float32', mode='w', shape=(train_sz, feature_sz))
        # Y_train = np.memmap(os.path.join(self.temp_path, 'Y_train.npy'), dtype='int32', mode='w', shape=(train_sz,))
        XX_train = np.ndarray(shape=(train_sz, feature_sz), dtype='float32')
        Y_train = np.ndarray(shape=(train_sz,), dtype='int32')
        train_idxs = np.array(range(train_sz))
        np.random.shuffle(train_idxs)

        vectorize(
            XX_train,
            Y_train,
            train_idxs,
            self.transformer,
            self.scaler,
            self.secondary_scaler,
            self.train_data,
            train_sz
        )
        
        print('Tuning parameters...', flush=True)

        param_dist = {'alpha': loguniform(1e-4, 1e0)}
        batch_size = 100
        clf = SGDClassifier(loss='log', alpha=0.01, n_jobs=32)
        n_iter_search = self.num_search_iters
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, verbose=2)
        for idxs in chunker(range(train_sz), batch_size):
                random_search.fit(XX_train[idxs, :], Y_train[idxs])
                break

        print('Best params:', random_search.best_params_)

        print('Training classifier...', flush=True)
        self.clf = SGDClassifier(loss='log', alpha=random_search.best_params_['alpha'], n_jobs=32)
        # batch_size = 50000
        batch_size = 64
        num_epochs = 50
        # num_epochs = 10
        for i in trange(num_epochs):
            for idxs in chunker(range(train_sz), batch_size):
                self.clf.partial_fit(XX_train[idxs, :], Y_train[idxs], classes=[0, 1])

    def save_model(self, folder):
        with open(os.path.join(folder, 'model.p'), 'wb') as f:
            pickle.dump((
                self.clf,
                self.transformer,
                self.scaler,
                self.secondary_scaler,
            ), f)
    
    def load_model(self, folder):
        self.temp_path = folder
        with open(os.path.join(folder, 'model.p'), 'rb') as f:
            self.clf, self.transformer, self.scaler, self.secondary_scaler = pickle.load(f)
    
    def evaluate_internal(self, df, df_name=None):
        
        print('Processing test data...')
        self.test_data = []
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            self.test_data.append((row['label'], prepare_entry(row['text0']), prepare_entry(row['text1'])))
        
        test_sz = len(self.test_data)
        feature_sz = len(self.transformer.get_feature_names())
    
        print('Vectorizing test set...', flush=True)
        # XX_test = np.memmap(os.path.join(self.temp_path, 'vectorized_XX_test.npy'), dtype='float32', mode='w', shape=(test_sz, feature_sz))
        # Y_test = np.memmap(os.path.join(self.temp_path, 'Y_test.npy'), dtype='int32', mode='w', shape=(test_sz,))
        XX_test = np.ndarray(shape=(test_sz, feature_sz), dtype='float32')
        Y_test = np.ndarray(shape=(test_sz,), dtype='int32')
        test_idxs = np.array(range(test_sz))
        # np.random.shuffle(test_idxs)

        vectorize(
            XX_test,
            Y_test,
            test_idxs,
            self.transformer,
            self.scaler,
            self.secondary_scaler,
            self.test_data,
            test_sz
        )
        
        probs = self.clf.predict_proba(XX_test)[:, 1]
        return [float(x) for x in list(probs)], [int(x) for x in list(Y_test)]