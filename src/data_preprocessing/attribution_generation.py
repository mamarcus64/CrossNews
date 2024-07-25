import pandas as pd
import pdb

def generate_attribution_data(docs, genres, query_num, target_num):
    author_to_doc = {}
    for doc in docs:
        author = doc['author']
        if author not in author_to_doc:
            author_to_doc[author] = {}
        genre = doc['genre']
        if genre not in author_to_doc[author]:
            author_to_doc[author][genre] = []
        author_to_doc[author][genre].append(doc)
        
    columns = ['author', 'genre', 'id', 'text']
    query_rows = []
    target_rows = []
    
    for author, docs in author_to_doc.items():
        for genre in genres:
            query_docs = docs[genre][:int(query_num / len(genres))]
            target_docs = docs[genre][int(query_num / len(genres)):int((query_num + target_num) / len(genres))]
            
            for doc in query_docs + target_docs:
                if doc['text'] == '' or len(doc['text']) <= 1:
                    doc['text'] = 'text'
            
            for doc in query_docs:
                query_rows.append([doc['author'], doc['genre'], doc['id'], doc['text']])
                
            for doc in target_docs:
                target_rows.append([doc['author'], doc['genre'], doc['id'], doc['text']])
    
    query_df = pd.DataFrame(query_rows, columns=columns)
    target_df = pd.DataFrame(target_rows, columns=columns)
    
    return query_df, target_df