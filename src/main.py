import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import csv
import shutil
import multiprocessing

from verification_models.random import Random
from verification_models.ngram import NGram
from verification_models.ppm import PPM
from verification_models.o2d2 import O2D2
from arguments import get_parser

def main(args):
    args.model = args.model.lower()
    args.evaluation_metric = args.evaluation_metric.lower()
    
    def debug_print(text):
        text = str(text)
        if not args.silent:
            print(f'{datetime.now().strftime("%A, %B %d, %Y %I:%M %p")}: {text}')
    
    debug_print(f'Creating model {args.model}...')
    
    parameter_file = json.load(open(f'src/verification_parameters/{args.model}.json', 'r'))
    
    if args.parameter_sets == ['all']:
        args.parameter_sets = list(parameter_file.keys())
    
    for parameter_set_name in args.parameter_sets:
        assert parameter_set_name in parameter_file
    
    models = []
    
    for parameter_set_name in args.parameter_sets:
        parameter_set = parameter_file[parameter_set_name]
    
        if args.model == 'random':
            model = Random(args, parameter_set)
        elif args.model == 'ngram':
            model = NGram(args, parameter_set)
        elif args.model == 'ppm':
            model = PPM(args, parameter_set, num_workers=max(multiprocessing.cpu_count() - 2, 1))
        elif args.model == 'o2d2':
            model = O2D2(args, parameter_set)
        
        debug_print(f"Created model (parameters {parameter_set_name}) at {model.model_folder}")
        
        if args.train:
            model.train()
            
        models.append(model)
        
        if args.load:
            break
        
    best_model = models[0]
        
    if args.train and not args.load: # find best model on the eval dataset
        if len(models) > 1:
            best_score = best_model.evaluate(best_model.eval_df)[0][args.evaluation_metric]
            debug_print(f"Model {best_model.model_folder} has {args.evaluation_metric} of {best_score}.")
            for model in models[1:]:
                model_score = model.evaluate(model.eval_df)[0][args.evaluation_metric]
                debug_print(f"Model {model.model_folder} has {args.evaluation_metric} of {model_score}.")
                if model_score > best_score:
                    best_score = model_score
                    best_model = model
            debug_print(f"Model {best_model.model_folder} performed best.")
            if not args.keep_all_models:
                for model in models:
                    if model != best_model:
                        debug_print(f"Deleting model {model.model_folder}.")
                        shutil.rmtree(model.model_folder) 

    if args.test:
        best_model.test_and_save()
        debug_print(f'Saved Model {best_model.model_folder}.')
    
if __name__ == '__main__':
    
    args = get_parser()
    main(args)
    
# export LD_LIBRARY_PATH=/nethome/mma81/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/

"""
O2D2 to do:
1. verify run in general
3. learn how to re-evaluate on different val set

salloc -G a40:2 -c 16
conda activate O2D2
cd /nethome/mma81/storage/O2D2
bash run_o2d2.sh


python train_adhominem.py -epochs 4

salloc -c 16 --partition nlprx-lab
conda activate O2D2
cd storage/O2D2
bash run_o2d2.sh


conda activate AuthorID
cd storage/CrossNews
python src/main.py --model PPM --experiment Train_Tweet_Tweet
exit


salloc -c 8
conda activate AuthorID_copy
python src/main.py \
--model ngram \
--train \
--train_file verification_data/train/CrossNews_Article_Article.csv \
--parameter_sets default \
--test \
--test_files verification_data/test/CrossNews_Tweet_Tweet.csv \
verification_data/test/CrossNews_Article_Article \
verification_data/test/CrossNews_Article_Tweet

"""


