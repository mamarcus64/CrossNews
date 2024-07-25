from datetime import datetime
import json

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
        args.parameter_sets = ['default']
    
    parameter_set_name = args.parameter_sets[0]
    
    if args.model == 'part':
        from embedding_models.part import PART
        model = PART(args, parameter_file[parameter_set_name])
    if args.model == 'luar':
        from embedding_models.luar import LUAR
        model = LUAR(args, parameter_file[parameter_set_name])
    if args.model == 'stel':
        from embedding_models.stel import STEL
        model = STEL(args, parameter_file[parameter_set_name])
    
    debug_print(f"Created model (parameters {parameter_set_name}) at {model.model_folder}")
    
    if args.train:
        model.train()
    
if __name__ == '__main__':
    
    args = get_parser()
    main(args)