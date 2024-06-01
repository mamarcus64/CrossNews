import os
import subprocess
from tqdm import tqdm
import pandas as pd
import csv
import codecs
import string
import random
from pathlib import Path

marker = ''.join(random.choice(string.ascii_letters) for _ in range(10))

os.makedirs('tmp', exist_ok=True)
tmp_file = f'tmp/tmp_long_text_{marker}.txt'
tmp_out = f'tmp/tmp_text_split_{marker}.txt'

def shorten(text, max_length=500):
    with codecs.open(tmp_file, 'w', encoding='utf-8', errors='ignore') as out:
        out.write(text)
        
    subprocess.run(['ersatz', '--input', tmp_file, '--output', tmp_out])
    
    shortened_text = ''
    for sentence in codecs.open(tmp_out, 'r', encoding='utf-8', errors='ignore').readlines():
        sentence = sentence.strip()
        shortened_text += f' {sentence}'
        if len(shortened_text) > max_length:
            break
    return shortened_text

def shorten_df(input_file, output_file, start_row, end_row):
    
    output_file = output_file.replace('.csv', f'_{start_row}_{end_row}.csv').replace('/train/', '/train/partitions/')
    open(f'verification_data/claims/{Path(output_file).stem}.txt', 'w').close() # just to claim it
    
    df = pd.read_csv(input_file)
    
    new_rows = []
    df = df.iloc[int(start_row):int(end_row)]
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if 'genre0' in row:
            text0 = shorten(row['text0']) if row['genre0'] == 'Article' else row['text0']
            text1 = shorten(row['text1']) if row['genre1'] == 'Article' else row['text1']
        else:
            text0 = shorten(row['text0'])
            text1 = shorten(row['text1'])
        
        if 'genre0' in row:
            new_rows.append([row['label'], text0, text1, row['author0'], row['author1'], row['id0'], row['id1'], row['genre0'], row['genre1']])
        else:
            new_rows.append([row['label'], text0, text1, row['author0'], row['author1'], row['id0'], row['id1']])
    
    if len(new_rows[0]) == 9:
        pd.DataFrame(new_rows,
                    columns=['label', 'text0', 'text1', 'author0',
                            'author1', 'id0', 'id1', 'genre0', 'genre1']).to_csv(output_file,
                                                                                index=False, quoting=csv.QUOTE_ALL)
    else:
        pd.DataFrame(new_rows,
                    columns=['label', 'text0', 'text1', 'author0',
                            'author1', 'id0', 'id1']).to_csv(output_file,
                                                                                index=False, quoting=csv.QUOTE_ALL)
    
        