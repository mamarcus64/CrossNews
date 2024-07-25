import sys
import json
from types import SimpleNamespace
from tqdm import tqdm
import os
from pathlib import Path

params = {
    "stel": {
        "CrossNews_Article_Article.csv": {
            "model_folder": "models/stel/CrossNews_Article_Article/06-04-09-08-54-lynhun",
            "parameter_set": "default"
        },
        "CrossNews_Article_Tweet.csv": {
            "model_folder": "models/stel/CrossNews_Article_Tweet/06-04-13-08-12-yxeulq",
            "parameter_set": "default"
        },
        "CrossNews_Tweet_Tweet.csv": {
            "model_folder": "models/stel/CrossNews_Tweet_Tweet/06-04-15-51-00-kdnzkk",
            "parameter_set": "default"
        },
    },
    "part": {
        "CrossNews_Article_Article.csv": {
            "model_folder": "models/part/CrossNews_Article_Article/05-28-04-24-31-bgmdzr",
            "parameter_set": "default"
        },
        "CrossNews_Article_Tweet.csv": {
            "model_folder": "models/part/CrossNews_Article_Tweet/05-28-04-24-42-wbjzxo",
            "parameter_set": "default"
        },
        "CrossNews_Tweet_Tweet.csv": {
            "model_folder": "models/part/CrossNews_Tweet_Tweet/05-28-04-24-54-hpxnqn",
            "parameter_set": "default"
        },
    },
    "luar": {
        "CrossNews_Article_Article.csv": {
            "model_folder": "models/luar/CrossNews_Article_Article/05-29-01-21-34-urinru",
            "parameter_set": "default"
        },
        "CrossNews_Article_Tweet.csv": {
            "model_folder": "models/luar/CrossNews_Article_Tweet/05-29-04-02-10-dcvtzh",
            "parameter_set": "default"
        },
        "CrossNews_Tweet_Tweet.csv": {
            "model_folder": "models/luar/CrossNews_Tweet_Tweet/05-29-04-18-39-xhxgej",
            "parameter_set": "default"
        },
    }
}

if sys.argv[1] == 'clear':
    # for model_type in ['luar', 'part', 'stel']:
    for model_type in ['part', 'stel']:
        for dataset in ['CrossNews_Article_Article', 'CrossNews_Article_Tweet', 'CrossNews_Tweet_Tweet']:
            print('            ', model_type, dataset)
            folder = os.path.join('gold_embeddings', model_type, dataset)
            to_delete = []
            for file in tqdm(os.listdir(folder)):
                embedding_file = os.path.join(folder, file)
                try:
                    embeddings = json.load(open(embedding_file, 'r'))
                    a = 1 / len(embeddings) # will throw error with size 0
                except:
                    to_delete.append(embedding_file)
                    print(f'Deleting {to_delete}...')
            for file in to_delete:
                os.remove(file)
    exit()


model_type = sys.argv[1] # PART, STEL, or LUAR
train_dataset = sys.argv[2]

load_folder = params[model_type][train_dataset]['model_folder']
param_name = params[model_type][train_dataset]['parameter_set']
    
args = {
    'model': model_type.lower(),
    'train': False,
    'load': True,
    'load_folder': load_folder,
    'parameter_sets': param_name,
    'evaluation_metric': 'F1'
}

        
parameter_set = json.load(open(f'src/model_parameters/{model_type.lower()}.json', 'r'))[param_name]

print('Loading model...')

if model_type == 'part':
    from embedding_models.part import PART
    model = PART(SimpleNamespace(**args), parameter_set)
elif model_type == 'stel':
    from embedding_models.stel import STEL
    model = STEL(SimpleNamespace(**args), parameter_set)
elif model_type == 'luar':
    from embedding_models.luar import LUAR
    model = LUAR(SimpleNamespace(**args), parameter_set)
    
print('Model loaded.')
step = 500
data = json.load(open('raw_data/crossnews_gold.json', 'r', encoding='utf-8'))
save_dir = os.path.join('gold_embeddings', model_type.lower(), Path(train_dataset).stem)



for i in range(0, len(data), step):
    start_row = i
    end_row = i + step
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f'partition_{start_row}_{end_row}.json')

    if not os.path.exists(save_name):
        print(f'Creating file {save_name}')
        open(save_name, 'w').close()
        
        texts = [x['text'] for x in data[start_row:end_row]]
        
        embeddings = model.get_embeddings(texts)
        json.dump(embeddings, open(save_name, 'w'), indent=4)

