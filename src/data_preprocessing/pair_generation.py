import json
import random
import numpy as np
import copy
import pandas as pd
import os
import csv

def stack_documents(authors, min_char_threshold, upper_char_limit=3000, is_train=False):
    """
    Adds documents of a single genre until a certain minimum character threshold
    is reached, then returns the stacked documents. For training, splits large
    documents into multiple disjoint documents.
    """
    authors = copy.deepcopy(authors)
    new_authors = {author: [] for author in authors.keys()}
    for author, old_docs in authors.items():
        articles = [doc for doc in old_docs if doc['genre'] == 'Article']
        tweets = [doc for doc in old_docs if doc['genre'] == 'Tweet']
        
        for docs in [articles, tweets]:
            docs = sorted(docs, key=lambda x: len(x['text']), reverse=True)
            new_doc = None
            for doc in docs:
                if new_doc is None:
                    new_doc = doc
                else:
                    new_doc['text'] += f' <new> {doc["text"]}'
                if len(new_doc['text']) >= min_char_threshold:
                    text = new_doc['text']
                    # if training data, can split long data into multiple upper_char_limit-sized documents
                    for i in range(0, max(len(text) // upper_char_limit, 1) if is_train else 1):
                        new_doc['text'] = text[upper_char_limit*i:upper_char_limit*(i+1)]
                        if len(new_doc['text']) >= min_char_threshold:
                            new_authors[author].append(copy.deepcopy(new_doc))
                    new_doc = None
    return new_authors

def generate_pair_ids(first_docs, second_docs, max_docs_per_author=100, max_pairs=None):
    """
    Produces balanced verification pairs for the listed documents. Each document is present
    in exactly one positive (same-author) pair and one negative (different-author) pair,
    yielding an overal label distribution of 50-50 for same author/different author pairs.

    Args:
        first_docs (dict): dict of author name -> document ID's for the first set of docs
        second_docs (dict): dict of author name -> document ID's for the second set of docs
        max_docs_per_author (int, optional): max number of docs per author. Defaults to 100.

    Returns:
        _type_: _description_
    """
    pairs = []
    used_articles = []
    used_tweets = []
    id_to_auth = {}
    
    def to_pair_id(s1, s2):
        same = 1 if id_to_auth[s1] == id_to_auth[s2] else 0
        return f'{same}_{s1}_{s2}'
    
    # variables named "articles" and "tweets" for CrossNews genres, but works with any dataset

    # sample positive (same-author) pairs
    for i in range(max_docs_per_author):
        for auth in first_docs.keys():
            articles = first_docs[auth]
            tweets = second_docs[auth]
            if min(len(articles), len(tweets)) > i:
                used_articles.append(articles[i])
                used_tweets.append(tweets[i])
                id_to_auth[articles[i]] = auth
                id_to_auth[tweets[i]] = auth
                pairs.append(to_pair_id(articles[i], tweets[i]))
            if max_pairs and len(pairs) >= max_pairs // 2:
                break
        if max_pairs and len(pairs) >= max_pairs // 2:
                break
    
    # sample negative (different-author) pairs    
    for _ in range(len(used_articles)):
        article = used_articles.pop()
        tweet = random.choice(used_tweets)
        while id_to_auth[tweet] == id_to_auth[article]:
            tweet = random.choice(used_tweets)
        used_tweets.remove(tweet)
        pairs.append(to_pair_id(article, tweet))
    
    return pairs

def get_pair_entries(pair_ids, data, shuffle=True):
    """
    Takes in a list of pair ID's generated by generate_pair_ids() and returns
    the actual verification pair text and authors.
    """
    pairs = []
    id_to_doc = {}
    for author_docs in data.values():
        id_to_doc.update({str(doc['id']): doc for doc in author_docs})
    
    if shuffle:
        random.shuffle(pair_ids)
    
    for pair in pair_ids:
        same, first_id, second_id = tuple(pair.split('_')[:3])
        same = int(same)
        a = id_to_doc[first_id]
        b = id_to_doc[second_id]
        pairs.append((same, a['text'], b['text'], a['author'], b['author'], first_id, second_id))
    
    return pairs

def print_pair_stats(pairs, save=None, genres=None):
    print(f'Total pairs: {len(pairs)}; same-pair percent: {sum([pair[0] for pair in pairs]) / len(pairs)}')
    
    authors = set()
    first_genre_lengths, second_genre_lengths = [], []
    for pair in pairs:
        first_length, second_length = len(pair[1]), len(pair[2])
        first_author, second_author = pair[3], pair[4]
        authors.add(first_author)
        authors.add(second_author)
        first_genre_lengths.append(first_length)
        second_genre_lengths.append(second_length)
    
    print(f'Num authors: {len(authors)}')
    print(f'Avg. chars per first genre: {sum(first_genre_lengths) / len(first_genre_lengths)}')
    print(f'Avg. chars per second genre: {sum(second_genre_lengths) / len(second_genre_lengths)}')
    
    if save:
        print(f'Saving to {save}.')
        columns = ['label', 'text0', 'text1', 'author0', 'author1', 'id0', 'id1']
        if genres is None:
            df = pd.DataFrame([(pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6]) for pair in pairs], columns=columns)
        else:
            columns.extend(['genre0', 'genre1'])
            df = pd.DataFrame([(pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], genres[0], genres[1]) for pair in pairs], columns=columns)
        df.to_csv(save, index=False, quoting=csv.QUOTE_ALL)