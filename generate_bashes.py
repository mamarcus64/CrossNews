import os
import sys


datasets = [
    "CrossNews_Article_Article.csv",
    "CrossNews_Article_Tweet.csv",
    "CrossNews_Tweet_Tweet.csv",
    # "CrossNews_Article_Article_short.csv",
    # "CrossNews_Article_Tweet_short.csv",
    ]

allocations = {
'ngram':
"""
salloc -c 8
model="ngram"
conda activate AuthorID_copy


""",

'ppm':
"""
salloc -c 16
model="ppm"
conda activate AuthorID_copy


""",

'o2d2':
"""
salloc -c 8 -G a40
model="o2d2"
conda activate AuthorID_copy


""",

'part_av':
"""
salloc -c 8 -G a40
model="part_av"
conda activate part


""",

'luar_av':
"""
salloc -c 8 -G a40
model="luar_av"
conda activate luar


""",

'stel_av':
"""
salloc -c 8 -G a40
model="stel_av"
conda activate stel


""",
}

model_allocation_each_time = {
    'ngram': False,
    'ppm': False,
    'o2d2': True,
    'part_av': True,
    'luar_av': True,
    'stel_av': True,
}

model_use_default_param = {
    'ngram': True,
    'ppm': True,
    'o2d2': True,
    'part_av': False,
    'luar_av': False,
    'stel_av': False,
}


for model in ['ngram', 'ppm', 'o2d2', 'part_av', 'luar_av', 'stel_av']:
    result = allocations[model]

    for experiment_num in range(1, 5):
        for dataset in datasets:
            save_folder = f'runs/experiment_{experiment_num}'
            if model_use_default_param[model]:
                param_str = 'default'
            else:
                param_str = '${dataset}'
            result += \
    f"""
dataset="{dataset}"

params="{param_str}"

save_folder="{save_folder}"


    """ + \
    """
date
cd /nethome/mma81/storage/CrossNews
current_date=$(date +"%m_%d-%H_%M")
output_file="logs/${current_date}_${model}_${dataset}.txt"
python src/run_verification.py \\
--model ${model} \\
--train \\
--save_folder ${save_folder} \\
--train_file verification_data/train/${dataset} \\
--parameter_sets ${params} \\
--test \\
--test_files verification_data/test/CrossNews_Tweet_Tweet.csv \\
verification_data/test/CrossNews_Article_Article.csv \\
verification_data/test/CrossNews_Article_Tweet.csv \\
| tee $output_file
date
exit

###########################


    """
            if model_allocation_each_time[model]:
                result += allocations[model]
                
    with open(os.path.join('bashes', f'{model}.sh'), 'w') as f:
        f.write(result)