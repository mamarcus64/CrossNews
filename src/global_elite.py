import random
import json
import csv
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pdb
import pandas as pd

random.seed(333)

global_ids = json.load(open('attribution_data/test/global_doc_ids.json', 'r'))
gold = json.load(open('raw_data/crossnews_gold.json', 'r', encoding='utf-8'))
# id_to_embedding = json.load(open('gold_embeddings/luar/CrossNews_Article.json', 'r'))
# id_to_embedding = json.load(open('gold_embeddings/part/CrossNews_Article.json', 'r'))
# id_to_embedding = json.load(open('gold_embeddings/stel/CrossNews_Article.json', 'r'))

# id_to_embedding = json.load(open('gold_embeddings/luar/CrossNews_Tweet.json', 'r'))
# id_to_embedding = json.load(open('gold_embeddings/part/CrossNews_Tweet.json', 'r'))
id_to_embedding = json.load(open('gold_embeddings/stel/CrossNews_Tweet.json', 'r'))

# t = 0.8 # LUAR
# t = 0.65 # PART
t = 0.6 # STEL


def generate_verification_pairs(pair_num=1000, seed=345):
    random.seed(seed)
    author_to_doc = {}
    for doc in gold:
        if doc['id'] in global_ids:
            author = doc['author']
            author_to_doc[author] = author_to_doc.get(author, []) + [doc]

    pairs = []
    same = True

    while len(pairs) < pair_num:
        authors = list(author_to_doc.keys())
        for author in authors:
            if len(author_to_doc[author]) < 2:
                del author_to_doc[author]
        authors = list(author_to_doc.keys())
        
        if same:
            author1 = author2 = random.choice(authors)
            doc1, doc2 = tuple(random.sample(author_to_doc[author1], 2))
        else:
            author1, author2 = tuple(random.sample(authors, 2))
            doc1 = random.choice(author_to_doc[author1])
            doc2 = random.choice(author_to_doc[author2])
        pairs.append((1 if same else 0, doc1['text'], doc2['text'], doc1['author'], doc2['author'], doc1['id'], doc2['id']))
        author_to_doc[author1].remove(doc1)
        author_to_doc[author2].remove(doc2)
        same = False if same else True

    print('Pairs generated.')
    columns = ['label', 'text0', 'text1', 'author0', 'author1', 'id0', 'id1']
    df = pd.DataFrame([(pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6]) for pair in pairs], columns=columns)
    df.to_csv('verification_data/test/global_elite.csv', index=False, quoting=csv.QUOTE_ALL)
    return pairs

def generate_attribution_pairs(seed=345, is_standard=False):
    random.seed(seed)
    author_to_doc = {}
    for doc in gold:
        if doc['id'] in global_ids or is_standard:
            author = doc['author']
            author_to_doc[author] = author_to_doc.get(author, []) + [doc]
            
    columns = ['author', 'genre', 'id', 'text']
    query_rows = []
    target_rows = []
    author_count = 0
    for author, docs in author_to_doc.items():
        if author_count >= 75:
            break
        if len(docs) < 3:
            continue
        author_count += 1
        if len(docs) >= 45:
            query_docs = docs[:30]
            target_docs = docs[30:45]
        else:
            thresh = int(len(docs) * 0.7)
            query_docs = docs[:thresh]
            target_docs = docs[thresh:]
            
        for doc in query_docs + target_docs:
            if doc['text'] == '' or len(doc['text']) <= 1:
                doc['text'] = 'text'
        
        for doc in query_docs:
            query_rows.append([doc['author'], doc['genre'], doc['id'], doc['text']])
            
        for doc in target_docs:
            target_rows.append([doc['author'], doc['genre'], doc['id'], doc['text']])
    
    query_df = pd.DataFrame(query_rows, columns=columns)
    target_df = pd.DataFrame(target_rows, columns=columns)
    print(author_count)
    
    if not is_standard:
        query_df.to_csv('attribution_data/query/global_elite.csv', index=False, quoting=csv.QUOTE_ALL)
        target_df.to_csv('attribution_data/test/global_elite.csv', index=False, quoting=csv.QUOTE_ALL)
    else:
        query_df.to_csv('attribution_data/query/global_standard.csv', index=False, quoting=csv.QUOTE_ALL)
        target_df.to_csv('attribution_data/test/global_standard.csv', index=False, quoting=csv.QUOTE_ALL)
    
def print_accuracy(pairs, threshold=t):
    predictions, labels = [], []
    for pair in tqdm(pairs):
        label, id1, id2 = pair[0], pair[-2], pair[-1]
        distance = cosine(id_to_embedding[str(id1)], id_to_embedding[str(id2)])
        predictions.append(1 if distance < threshold else 0)
        labels.append(label)
        
    print(sum(labels))
    print(sum([1 if label == prediction else 0 for label, prediction in zip(labels, predictions)]) / len(pairs))
    print(sum(predictions) / len(pairs))

# pairs = generate_verification_pairs()

df = pd.read_csv('verification_data/test/global_elite_tweet.csv')

pairs = [tuple(row) for row in df.itertuples(index=False)]

print_accuracy(pairs)

# pdb.set_trace()



""" For Tweet experiment

salloc -c 8
python src/run_verification.py --model ngram \
--load --load_folder models/ngram/CrossNews_Tweet_Tweet/05-26-20-30-52-pxefdm \
--parameter_sets default --test \
--save_folder tweet_topic \
--test_files verification_data/test/glclobal_elite_tweet.csv
exit

salloc -c 8
python src/run_verification.py --model ppm \
--load --load_folder models/ppm/CrossNews_Tweet_Tweet/05-26-20-31-27-pghdca \
--parameter_sets default --test \
--save_folder tweet_topic \
--test_files verification_data/test/global_elite_tweet.csv 
exit

salloc -c 8 -G a40
python src/run_verification.py --model o2d2 \
--load --load_folder models/O2D2/CrossNews_Tweet_Tweet/05-27-12-30-01-otvukz \
--parameter_sets default --test \
--save_folder tweet_topic \
--test_files verification_data/test/global_elite_tweet.csv
exit

Attribution


salloc -c 8

# dataset="global_elite_tweet.csv"
dataset="global_standard_tweet.csv"

# model="luar_aa"
# model="part_aa"
model="stel_aa"
conda activate luar

# model="ngram_aa"
# model="ppm_aa"
# conda activate AuthorID_copy

cd /nethome/mma81/storage/CrossNews

date
python src/run_attribution.py \
--model ${model} \
--train \
--query_file attribution_data/query/${dataset} \
--parameter_sets default \
--save_folder tweet_topic \
--test \
--target_file attribution_data/test/${dataset}
date

exit



"""