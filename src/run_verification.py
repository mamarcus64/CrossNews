from datetime import datetime
import json
import shutil
import multiprocessing

from arguments import get_parser

def main(args):
    args.model = args.model.lower()
    args.evaluation_metric = args.evaluation_metric.lower()
    
    def debug_print(text):
        text = str(text)
        if not args.silent:
            print(f'{datetime.now().strftime("%A, %B %d, %Y %I:%M %p")}: {text}')
    
    debug_print(f'Creating model {args.model}...')
    
    parameter_file = json.load(open(f'src/model_parameters/{args.model}.json', 'r'))
    
    if args.parameter_sets == ['all']:
        args.parameter_sets = list(parameter_file.keys())
    
    for parameter_set_name in args.parameter_sets:
        assert parameter_set_name in parameter_file
    
    models = []
    
    for parameter_set_name in args.parameter_sets:
        parameter_set = parameter_file[parameter_set_name]
    
        if args.model == 'random':
            from verification_models.random import Random
            model = Random(args, parameter_set)
        elif args.model == 'ngram':
            from verification_models.ngram import NGram
            model = NGram(args, parameter_set)
        elif args.model == 'ppm':
            from verification_models.ppm import PPM
            model = PPM(args, parameter_set, num_workers=max(multiprocessing.cpu_count() - 2, 1))
        elif args.model == 'o2d2':
            from verification_models.o2d2 import O2D2
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