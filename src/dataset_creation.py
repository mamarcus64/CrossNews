import random
import json
import os

from data_preprocessing.pair_generation import stack_documents, generate_pair_ids, get_pair_entries, print_pair_stats


def crossnews_helper(data, first_genre, second_genre, seed, max_docs_per_author, save_name):
    all_articles = {author: [str(doc['id']) for doc in docs if doc['genre'] == 'Article'] for author, docs in data.items()}
    all_tweets = {author: [str(doc['id']) for doc in docs if doc['genre'] == 'Tweet'] for author, docs in data.items()}

    random.seed(seed) # do all the shuffling here instead of during sample time
    for auth in all_articles.keys():
        random.shuffle(all_articles[auth])
        random.shuffle(all_tweets[auth])
    
    if first_genre != second_genre: # Article & Tweet
        first_docs = all_articles
        second_docs = all_tweets
    else: # either Article & Article or Tweet & Tweet
        first_docs, second_docs = {}, {} # split all of the single genre into two distinct dicts
        all_docs = all_articles if first_genre == 'Article' else all_tweets
        for auth in all_docs.keys():
            author_docs = all_docs[auth]
            if len(author_docs) >= 2:
                first_docs[auth] = author_docs[len(author_docs) // 2:]
                second_docs[auth] = author_docs[:len(author_docs) // 2]
                
    pair_ids = generate_pair_ids(first_docs, second_docs, max_docs_per_author=max_docs_per_author, max_pairs=100000)
    pairs = get_pair_entries(pair_ids, data)
    if first_genre != second_genre:
        print_pair_stats(pairs, save=save_name, genres=[first_genre, second_genre])
    else:
        print_pair_stats(pairs, save=save_name)

def create_crossnews():
    gold = json.load(open('raw_data/crossnews_gold.json', 'r', encoding='utf-8'))
    silver = json.load(open('raw_data/crossnews_silver.json', 'r', encoding='utf-8'))

    gold_authors = {}
    silver_authors = {}

    for docs, authors in [
        (gold, gold_authors),
        (silver, silver_authors)
    ]:
        for doc in docs:
            author = doc['author']
            authors[author] = authors.get(author, []) + [doc]
            
    stacked_golds = stack_documents(gold_authors, 500)
    stacked_silvers = stack_documents(silver_authors, 500, is_train=True)
    
    os.makedirs('verification_data', exist_ok=True)
    os.makedirs('verification_data/train', exist_ok=True)
    os.makedirs('verification_data/test', exist_ok=True)
    
    crossnews_helper(stacked_golds,
                            'Article',
                            'Tweet',
                            seed=111,
                            max_docs_per_author=5,
                            save_name='verification_data/test/CrossNews_Article_Tweet.csv')
    crossnews_helper(stacked_golds,
                            'Article',
                            'Article',
                            seed=222,
                            max_docs_per_author=5,
                            save_name='verification_data/test/CrossNews_Article_Article.csv')
    crossnews_helper(stacked_golds,
                            'Tweet',
                            'Tweet',
                            seed=333,
                            max_docs_per_author=5,
                            save_name='verification_data/test/CrossNews_Tweet_Tweet.csv')
    crossnews_helper(stacked_silvers,
                            'Article',
                            'Tweet',
                            seed=444,
                            max_docs_per_author=100,
                            save_name='verification_data/train/CrossNews_Article_Tweet.csv')
    crossnews_helper(stacked_silvers,
                            'Article',
                            'Article',
                            seed=555,
                            max_docs_per_author=100,
                            save_name='verification_data/train/CrossNews_Article_Article.csv')
    crossnews_helper(stacked_silvers,
                            'Tweet',
                            'Tweet',
                            seed=666,
                            max_docs_per_author=100,
                            save_name='verification_data/train/CrossNews_Tweet_Tweet.csv')
    
if __name__ == '__main__':
    create_crossnews()