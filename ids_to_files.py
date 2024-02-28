import json
import os
import sys
import pandas as pd
import argparse
from pathlib import Path

def generate_dataframes(id_location, save_location, data_location):
    os.makedirs(save_location, exist_ok=True)
    gold_data = json.load(open(data_location, 'r', encoding='utf-8'))
    id_to_doc = {}
    author_str_to_id = {}
    i = 0
    for datum in gold_data:
        id_to_doc[str(datum['id'])] = datum
        if datum['author'] not in author_str_to_id:
            author_str_to_id[datum['author']] = i
            i += 1
    id_file = json.load(open(id_location, 'r'))
    for id_key, csv_name in [
        ('train', 'train.csv'),
        ('val', 'val.csv'),
        ('test', 'test.csv')
    ]:
        rows = []
        ids = id_file[id_key]
        for id_list in ids:
            id_list = [str(doc_id) for doc_id in id_list]
            text = ''
            # clean up white space
            for doc_id in id_list: 
                text += id_to_doc[str(doc_id)]['text'].replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip() + ' '
            rows.append({
                "author": author_str_to_id[id_to_doc[id_list[0]]['author']],
                "genre": id_to_doc[id_list[0]]['genre'],
                "time": id_to_doc[id_list[0]]['date'],
                "id": ','.join([str(x) for x in id_list]),
                "text": text
            })
         
        pd.DataFrame(rows).to_csv(os.path.join(save_location, csv_name), sep=',', quotechar='"', index=False)
    json.dump(author_str_to_id, open(os.path.join(save_location, 'author_to_id.json'), 'w'), indent=4)


def main():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--id_folder', type=str, default='experiment_ids', help='Folder containing experiment ids.')
    parser.add_argument('--id_file', type=str, help='Name of experiment id file.')
    parser.add_argument('--output_folder', type=str, default='output', help='Name of output folder for train, val, and test dataframes.')
    parser.add_argument('--data_file', type=str, default='data/crossnews_gold.json', help='Location of data file.')
    
    params = parser.parse_args(sys.argv[1:])
    
    # remove file extension
    params.id_file = Path(params.id_file).stem
    
    id_location = os.path.join(params.id_folder, f'{params.id_file}.json')
    
    save_location = os.path.join(params.output_folder, params.id_file)
        
    
    generate_dataframes(id_location, save_location, params.data_file)
    
if __name__ == '__main__':
    main()