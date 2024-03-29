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
        
def list_stats(values):
    return f'total: {round(sum(values), 0)} count: {len(values)} mean: {round(np.mean(values), 3)} quartiles: {round(np.percentile(values, 25), 3)}/{round(np.percentile(values, 50), 3)}/{round(np.percentile(values, 75), 3)} std: {round(np.std(values), 3)}'

def stack_documents(authors, threshold, upper_char_limit=2500, method='random', is_test=False):
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
                    text = new_doc['text']
                    # only want to create multiple pairs from same datum in train setup
                    for i in range(0, max(len(text) // upper_char_limit, 1) if not is_test else 1):
                        new_doc['text'] = text[upper_char_limit*i:upper_char_limit*(i+1)]
                        if len(new_doc['text']) >= threshold:
                            new_authors[author].append(copy.deepcopy(new_doc))
                    new_doc = None
    return new_authors

stacked_golds = stack_documents(gold_authors, 500, method='greedy', is_test=True)
stacked_silvers = stack_documents(silver_authors, 500, method='greedy')

def create_verification_pairs(data, first_genre, second_genre):
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
        
    delete_threshold = 0 if first_genre != second_genre else 1
        
    for author in data.keys():
        random.shuffle(first_docs[author])
        random.shuffle(second_docs[author])
        
        if len(first_docs[author]) <= delete_threshold:
            del first_docs[author]
        if len(second_docs[author]) <= delete_threshold:
            del second_docs[author]
    
    # each pair is a 3-tuple of (label, first_text, second_text), where label == 0 if different authors and 1 if same author
    pairs = []
   
    next_pick = 'same'
    # need at least two authors for each genre to pick pairs
    while len(first_docs) > 1 and len(second_docs) > 1:
        # pick authors for next pair
        first_author_pool = list(first_docs.keys())
        second_author_pool = list(second_docs.keys())
        if next_pick == 'diff':
            first_author = random.choice(first_author_pool)
            second_author = random.choice(second_author_pool)
            while first_author == second_author:
                second_author = random.choice(second_author_pool)
        elif next_pick == 'same':
            first_author = random.choice(first_author_pool)
            # try picking 10 random authors, if this doesn't work, then iterate through all authors
            if first_author not in second_author_pool:
                for _ in range(10):
                    first_author = random.choice(first_author_pool)
                    if first_author in second_author_pool:
                        break
            if first_author not in second_author_pool:
                random.shuffle(first_author_pool)
                found_same_author = False
                for author in first_author_pool:
                    if author in second_author_pool:
                        first_author = author
                        found_same_author = True
                        break
                if not found_same_author:
                    break # bailing, no same authors left
            second_author = first_author
        
        if len(first_docs[first_author]) == 0 or len(second_docs[second_author]) == 0:
            print(first_author, second_author, len(first_docs[first_author]), len(second_docs[second_author]))
        pairs.append((1 if next_pick == 'same' else 0, first_docs[first_author].pop()['text'], second_docs[second_author].pop()['text'], first_author, second_author))
        
        if len(first_docs[first_author]) <= delete_threshold:
            del first_docs[first_author]
        if (first_genre != second_genre or first_author != second_author) and len(second_docs[second_author]) <= delete_threshold:
            del second_docs[second_author]
                
        # alternate pair type
        next_pick = 'diff' if next_pick == 'same' else 'same'

    return pairs

def print_pair_stats(pairs, save=None):
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
        columns = ['label', 'text0', 'text1']
        df = pd.DataFrame([(pair[0], pair[1], pair[2]) for pair in pairs], columns=columns)
        df.to_csv(save, index=False)
        
test_pairs_Article_X = create_verification_pairs(stacked_golds, 'Article', 'Tweet')
test_pairs_Article_Article = create_verification_pairs(stacked_golds, 'Article', 'Article')
test_pairs_X_X = create_verification_pairs(stacked_golds, 'Tweet', 'Tweet')
train_pairs = create_verification_pairs(stacked_silvers, 'Article', 'Tweet')
 
os.makedirs('pairs', exist_ok=True)
print_pair_stats(test_pairs_Article_X, save='pairs/test_Article_X.csv')
print_pair_stats(test_pairs_Article_Article, save='pairs/test_Article_Article.csv')
print_pair_stats(test_pairs_X_X, save='pairs/test_X_X.csv')
print_pair_stats(train_pairs, save='pairs/train_Article_X.csv')