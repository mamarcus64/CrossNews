from data_preprocessing.paraphrasing.paraphrase_df import paraphrase_df
import pandas as pd
import pdb
from pathlib import Path
import os
import sys

dataset = f'verification_data/test/{sys.argv[1]}.csv'
model_id = sys.argv[2]

step = 20
for i in range(0, 5000, step):
    start_row = i
    end_row = i + step
    save_dir = os.path.join('prompt_testing', Path(dataset).stem, Path(model_id).stem)
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f'partition_{start_row}_{end_row}.csv')

    if not os.path.exists(save_name):
        print(f'Creating file {save_name}')
        open(save_name, 'w').close()
        df = pd.read_csv(dataset)
        new_df = paraphrase_df(df, model_id, start_row, end_row)
        new_df.to_csv(save_name, index=False)




"""

salloc -c 16 -G a40
conda activate stel
cd /nethome/mma81/storage/CrossNews
# python src/paraphrase_runner.py CrossNews_Article_Tweet meta-llama/Meta-Llama-3-8B-Instruct
python src/paraphrase_runner.py CrossNews_Article_Article meta-llama/Meta-Llama-3-8B-Instruct
# python src/paraphrase_runner.py CrossNews_Article_Tweet meta-llama/Meta-Llama-3-70B-Instruct
# python src/paraphrase_runner.py CrossNews_Article_Article meta-llama/Meta-Llama-3-70B-Instruct
exit

"""
