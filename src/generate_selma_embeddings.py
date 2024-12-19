import os
from tqdm import tqdm
import json
import sys
import pandas as pd

sys_args = sys.argv

"""
To generate SELMA embeddings, you will need to run this file several times with different options.
You will need to create both the known embeddings (train embeddings) which do not use instruction prompts,
then you will create the unknown embeddings (test embeddings) which do use instruction prompts.

To do this, you will run:

python src/generate_selma_embeddings.py train

This will generate many partition files into selma_embeddings/mistral/train_partitions. You can run the above python file
multiple times in parallel, as well as stop/start the file. Once all the partitions have been generated, then run:

python src/generate_selma_embeddings.py train combine

This will combine the partitions into a single file for use with the rest of the repository. Feel free to delete the partitions
folder after this. To create partitions for the unknown/test document embeddings, do similar, except you will need to specify the instruction:

python src/generate_selma_embeddings.py test test_prompt_lip
...
python src/generate_selma_embeddings.py test test_prompt_lip combine

If you modify any of the file locations, you will also need to modify the corresponding entry in src/model_parameters/selma.json.

"""

if 'no_prompt' in sys_args:
    TEST_PROMPT = ''
    prompt_name = 'no_prompt'
elif 'test_prompt_taskonly' in sys_args:
    TEST_PROMPT = 'Instruct: Retrieve stylistically similar text.\nQuery:'
    prompt_name = 'test_prompt_taskonly'
elif 'test_prompt_av' in sys_args:
    TEST_PROMPT = 'Instruct: Retrieve stylistically similar text. Here are some relevant variables to this problem.\n1. punctuation style\n2. special characters\n3. acronyms and abbreviations\n4. writing style\n5. expressions and idioms\n6. tone and mood\n7. sentence structure\n8. any other relevant aspect\nQuery:'
    prompt_name = 'test_prompt_av'
elif 'test_prompt_lip' in sys_args:
    TEST_PROMPT = 'Instruct: Retrieve stylistically similar text. Analyze the writing styles of the input texts, disregarding the differences in topic and content. Reason based on linguistic features such as phrasal verbs, modal verbs, punctuation, rare words, affixes, quantities, humor, sarcasm, typographical errors, and misspellings.\nQuery:'
    prompt_name = 'test_prompt_lip'

print(prompt_name)

if 'combine' in sys_args:
    if 'train' in sys_args:
        partition_folder = f'selma_embeddings/mistral/train_partitions'
    else:
        partition_folder = f'selma_embeddings/mistral/{prompt_name}_partitions'
    results = {}
    for file in tqdm(os.listdir(partition_folder)):
        file = os.path.join(partition_folder, file)
        try:
            results.update(json.load(open(file, 'r')))
        except:
            print(f'ERROR: could not load {file}')
    if 'train' in sys_args:
        json_file = f'selma_embeddings/mistral/train.json'
        if os.path.exists(json_file):
            results.update(json.load(open(json_file, 'r')))
        json.dump(results, open(json_file, 'w'), indent=4)
    else:
        json_file = f'selma_embeddings/mistral/{prompt_name}.json'
        if os.path.exists(json_file):
            results.update(json.load(open(json_file, 'r')))
        json.dump(results, open(json_file, 'w'), indent=4)
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct").cuda()
    output_loc = 'selma_embeddings/mistral'

    os.makedirs(output_loc, exist_ok=True)

    id_to_text = {} # id to text

    if 'train' in sys_args:
        df = pd.concat([pd.read_csv('attribution_data/query/CrossNews_Article.csv'),
                    pd.read_csv('attribution_data/query/CrossNews_Tweet.csv'),
                    pd.read_csv('attribution_data/query/CrossNews_Both.csv'),
                    ])
        
    if 'test' in sys_args:
        df = pd.concat([pd.read_csv('attribution_data/test/CrossNews.csv'),
                        ])
        
    df = df.drop_duplicates(subset='id', keep='first')
    df = df.reset_index(drop=True)
        
    print(f'dataset size: {len(df)}')
    
    all_ids = []

    for _, row in df.iterrows():
        
        id_to_text[str(row['id'])] = row['text'][:5000]
        all_ids.append(str(row['id']))

    ids, texts = [], []
    batch_size = 15 # small enough to fit on a single A40, feel free to adjust size
    encodings = {} # id to encoding

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
