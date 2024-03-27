import json
import random
import numpy as np
import copy
import pandas as pd
import os

random.seed(8888)

gold = json.load(open('data/crossnews_gold.json', 'r', encoding='utf-8'))
silver = json.load(open('data/crossnews_silver.json', 'r', encoding='utf-8'))

gold_authors = {}
silver_authors = {}

for docs, authors in [
    (gold, gold_authors),
    (silver, silver_authors)
]:
    for doc in docs:
        author = doc['author']
        authors[author] = authors.get(author, []) + [doc]
        
def doc_thresholds(authors, threshold, genre=None, filter=None):
    result = set()
    if filter is None:
        filter = lambda x: True
    for author_id, author_docs in authors.items():
        doc_num = sum([1 if filter(doc) and (genre is None or doc['genre'] == genre) else 0 for doc in author_docs])
        if doc_num >= threshold:
            result.add(author_id)
    return result

def length_filter(doc, length=100):
    return len(doc['text']) >= length

gold_filtered = doc_thresholds(gold_authors, 100, genre='Tweet', filter=length_filter)
silver_filtered = doc_thresholds(silver_authors, 1, genre='Article', filter=length_filter)

def stack_documents(authors, threshold, method='random'):
    authors = copy.deepcopy(authors)
    new_authors = {author: [] for author in authors.keys()}
    for author, old_docs in authors.items():
        articles = [doc for doc in old_docs if doc['genre'] == 'Article']
        tweets = [doc for doc in old_docs if doc['genre'] == 'Tweet']
        
        for docs in [articles, tweets]:
            if method == 'random':
                random.shuffle(docs)
            elif method == 'greedy':
                docs = sorted(docs, key=lambda x: len(x['text']), reverse=True)
            new_doc = None
            for doc in docs:
                if new_doc is None:
                    new_doc = doc
                else:
                    new_doc['text'] += f'<new> {doc["text"]}'
                if len(new_doc['text']) >= threshold:
                    new_authors[author].append(new_doc)
                    new_doc = None
    return new_authors

stacked_golds = stack_documents(gold_authors, 500, method='greedy')
stacked_silvers = stack_documents(silver_authors, 500, method='greedy')

def create_verification_pairs(data, first_genre, second_genre, add_imbalanced=False):
    data = copy.deepcopy(data)
    first_docs = {
        author_name: [doc for doc in author_docs if doc['genre'] == first_genre]
            for author_name, author_docs in data.items()
    }
    if first_genre == second_genre:
        second_docs = first_docs
    else:
        second_docs = {
            author_name: [doc for doc in author_docs if doc['genre'] == second_genre]
                for author_name, author_docs in data.items()
        }
        
    overflow_first, overflow_second = [], []
    
    for author in data.keys():
        random.shuffle(first_docs[author])
        random.shuffle(second_docs[author])
        
        if len(first_docs[author]) == 0 or len(second_docs[author]) == 0:
            overflow_first.extend(first_docs[author])
            overflow_second.extend(second_docs[author])
            
            del first_docs[author]
            del second_docs[author]
    
    # each pair is a 3-tuple of (label, first_text, second_text), where label == 0 if different authors and 1 if same author
    pairs = []
   
    next_pick = 'same'
    while len(first_docs) > 1:
        # pick authors for next pair
        
        
        first_author = sorted(list(first_docs.keys()), key=lambda author: len(first_docs[author]), reverse=True)[0]
        second_author = first_author
        if next_pick == 'diff':
            # second_author = sorted(list(first_docs.keys()), key=lambda author: len(first_docs[author]), reverse=True)[1]
            while second_author == first_author:
                second_author = random.choice(list(first_docs.keys()))
        
        if len(first_docs[first_author]) == 0 or len(second_docs[second_author]) == 0:
            print(first_author, second_author, len(first_docs[first_author]), len(second_docs[second_author]))
        pairs.append((1 if next_pick == 'same' else 0, first_docs[first_author].pop()['text'], second_docs[second_author].pop()['text']))
        
        delete_threshold = 0 if first_genre != second_genre else 1
        
        # now, if either list is empty (or list has one element and first_genre == second_genre), delete from both docs dicts
        if len(first_docs[first_author]) <= delete_threshold:
            del first_docs[first_author]
            if first_genre != second_genre:
                overflow_second.extend(second_docs[first_author])
                del second_docs[first_author]
        if second_author in first_docs and len(first_docs[second_author]) <= delete_threshold:
            del second_docs[second_author]
            if first_genre != second_genre:
                overflow_first.extend(first_docs[second_author])
                del first_docs[second_author]
        if first_author in second_docs and len(second_docs[first_author]) <= delete_threshold:
            del first_docs[first_author]
            if first_genre != second_genre:
                overflow_second.extend(second_docs[first_author])
                del second_docs[first_author]
        if second_author in second_docs and len(second_docs[second_author]) <= delete_threshold:
            del second_docs[second_author]
            if first_genre != second_genre:
                overflow_first.extend(first_docs[second_author])
                del first_docs[second_author]
                
        # alternate pair type
        next_pick = 'diff' if next_pick == 'same' else 'same'
        
    if add_imbalanced and first_genre != second_genre:
        # guaranteed to be diff pairs b/c overflow is added when length of one genre == 0
        random.shuffle(overflow_first)
        random.shuffle(overflow_second)
        for i in range(min(len(overflow_first), len(overflow_second))):
            pairs.append((0, overflow_first[i]['text'], overflow_second[i]['text']))
    
    return pairs

def print_pair_stats(pairs, save=None):
    print(f'Total pairs: {len(pairs)}; same-pair percent: {sum([pair[0] for pair in pairs]) / len(pairs)}')
    if save:
        print(f'Saving to {save}.')
        columns = ['label', 'text0', 'text1']
        df = pd.DataFrame(pairs, columns=columns)
        df.to_csv(save, index=False)
        
        
os.makedirs('pairs')
pairs = create_verification_pairs(stacked_golds, 'Tweet', 'Tweet')
print_pair_stats(pairs, save='pairs/test_X_X.csv')

pairs = create_verification_pairs(stacked_golds, 'Article', 'Tweet')
print_pair_stats(pairs, save='pairs/test_Article_X.csv')

pairs = create_verification_pairs(stacked_golds, 'Article', 'Article')
print_pair_stats(pairs, save='pairs/test_Article_Article.csv')

pairs = create_verification_pairs(stacked_silvers, 'Article', 'Tweet')
print_pair_stats(pairs, save='pairs/train_Article_X.csv')

pairs = create_verification_pairs(stacked_silvers, 'Article', 'Tweet', add_imbalanced=True)
print_pair_stats(pairs, save='pairs/train_Article_X_imbalanced.csv')