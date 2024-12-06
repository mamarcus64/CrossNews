import os
from tqdm import tqdm
import json
import sys
import pandas as pd

sys_args = sys.argv

prompt_name = 'test_prompt_1' # change as needed
# prompt_name = 'test_prompt_av' # change as needed
# prompt_name = 'test_prompt_lip' # change as needed
# prompt_name = 'test_prompt_sem' # change as needed
# prompt_name = 'test_prompt_ignore' # change as needed

TEST_PROMPT = 'Instruct: Retrieve stylistically similar text.\nQuery:'
# TEST_PROMPT = 'Instruct: Retrieve stylistically similar text. Here are some relevant variables to this problem.\n1. punctuation style\n2. special characters\n3. acronyms and abbreviations\n4. writing style\n5. expressions and idioms\n6. tone and mood\n7. sentence structure\n8. any other relevant aspect\nQuery:'
# TEST_PROMPT = 'Instruct: Retrieve stylistically similar text. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reason based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings.\nQuery:'
# TEST_PROMPT = 'Instruct: Retrieve semantically similar text.\nQuery:'
# TEST_PROMPT = 'Instruct: Retrieve stylistically similar text, ignoring differences in content.\nQuery:'

print(prompt_name)

if 'combine' in sys_args:
    if 'train' in sys_args:
        partition_folder = f'gold_embeddings/mistral/train_partitions'
    else:
        partition_folder = f'gold_embeddings/mistral/{prompt_name}_partitions'
    results = {}
    for file in tqdm(os.listdir(partition_folder)):
        file = os.path.join(partition_folder, file)
        try:
            results.update(json.load(open(file, 'r')))
        except:
            print(f'ERROR: could not load {file}')
    if 'train' in sys_args:
        json_file = f'gold_embeddings/mistral/train.json'
        if os.path.exists(json_file):
            results.update(json.load(open(json_file, 'r')))
        json.dump(results, open(json_file, 'w'), indent=4)
    else:
        json_file = f'gold_embeddings/mistral/{prompt_name}.json'
        if os.path.exists(json_file):
            results.update(json.load(open(json_file, 'r')))
        json.dump(results, open(json_file, 'w'), indent=4)
else:
    from sentence_transformers import SentenceTransformer
    if 'stella' in sys_args:
        model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
        output_loc = 'gold_embeddings/stella'
    if 'mistral' in sys_args:
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct").cuda()
        output_loc = 'gold_embeddings/mistral'

    os.makedirs(output_loc, exist_ok=True)

    id_to_text = {} # id to text

    if 'train' in sys_args:
        df = pd.concat([pd.read_csv('attribution_data/query/CrossNews_Article.csv'),
                    pd.read_csv('attribution_data/query/CrossNews_Tweet.csv'),
                    pd.read_csv('attribution_data/query/CrossNews_Both.csv'),
                    pd.read_csv('attribution_data/query/global_elite.csv'),
                    pd.read_csv('attribution_data/query/global_standard.csv'),
                    pd.read_csv('attribution_data/query/global_elite_tweet.csv'),
                    pd.read_csv('attribution_data/query/global_standard_tweet.csv'),
                    ])
        
    if 'test' in sys_args:
        df = pd.concat([pd.read_csv('attribution_data/test/CrossNews.csv'),
                        pd.read_csv('attribution_data/test/global_elite.csv'),
                        pd.read_csv('attribution_data/test/global_standard.csv'),
                        pd.read_csv('attribution_data/test/global_elite_tweet.csv'),
                        pd.read_csv('attribution_data/test/global_standard_tweet.csv'),
                        ])
        
    df = df.drop_duplicates(subset='id', keep='first')
    df = df.reset_index(drop=True)
        
    print(f'dataset size: {len(df)}')
    
    all_ids = []

    for _, row in df.iterrows():
        
        id_to_text[str(row['id'])] = row['text'][:5000]
        all_ids.append(str(row['id']))

    ids, texts = [], []
    batch_size = 15 # small enough to fit on a single A40
    encodings = {} # id to encoding

    if 'mistral' in sys_args:
        for i in range(0, len(all_ids), batch_size):
            encodings = {}
            start_row = i
            end_row = i + batch_size
            if 'train' in sys_args:
                partition_name = 'train_partitions'
            else:
                partition_name = f'{prompt_name}_partitions'
            
            os.makedirs(os.path.join(output_loc, partition_name), exist_ok=True)
            save_name = os.path.join(output_loc, partition_name, f'partition_{start_row}_{end_row}.json')

            if not os.path.exists(save_name):
                print(f'Creating file {save_name}')
                open(save_name, 'w').close()
                
                ids = all_ids[start_row:end_row]
                texts = [id_to_text[id] for id in ids]
                
                if 'train' in sys_args:
                    encoding_list = model.encode(texts)
                else:
                    encoding_list = model.encode(texts, prompt=TEST_PROMPT)
                encoding_list = encoding_list.astype(float).tolist()
                assert len(encoding_list) == len(ids)
                for encoding, id in zip(encoding_list, ids):
                    encodings[id] = [round(feature, 4) for feature in encoding] 
                json.dump(encodings, open(save_name, 'w'), indent=4)

    if 'stella' in sys_args:
        for i in tqdm(range(len(all_ids))):
            id = all_ids[i]
            ids.append(id)
            texts.append(id_to_text[id])
            if len(texts) == batch_size or i == len(all_ids) - 1:
                if 'train' in sys_args:
                    batch_encodings = model.encode(texts)
                else:
                    batch_encodings = model.encode(texts, prompt=TEST_PROMPT)
                batch_encodings = batch_encodings.astype(float).tolist()
                assert len(batch_encodings) == len(ids)
                
                for batch_encoding, batch_id in zip(batch_encodings, ids):
                    encodings[batch_id] = [round(feature, 4) for feature in batch_encoding] 
                
                ids, texts = [], []
        if 'train' in sys_args:
            json_file = os.path.join(output_loc, 'train.json')
            if os.path.exists(json_file):
                encodings.update(json.load(open(json_file, 'r')))
            json.dump(encodings, open(json_file, 'w', encoding='utf-8'), indent=4)
        else:
            json_file = os.path.join(output_loc, f'{prompt_name}.json')
            if os.path.exists(json_file):
                encodings.update(json.load(open(json_file, 'r')))
            json.dump(encodings, open(json_file, 'w', encoding='utf-8'), indent=4)
    
    
"""

1. create query embeddings (and update prompt names above)

salloc -G a40 -c 6
conda activate stel
python src/collect_llm_embeddings.py mistral test
# python src/collect_llm_embeddings.py stella test
exit

1.5. combine mistral embeddings (for corresponding prompt name)

salloc -c 8
python src/collect_llm_embeddings.py mistral test combine
exit

2. update src/model_parameters/llm_embedding_aa.json
3. run 3 scripts (for each AA/TT/AT data) for each param set in quickstarts


1. remove last partition file
2. add other models
3. run train + test
4. combine
5. test on single part

"""