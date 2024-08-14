import json
import multiprocessing

from arguments import get_parser

def main(args):
    args.model = args.model.lower()
    
    print(f'Creating model {args.model}...')
    
    parameter_file = json.load(open(f'src/model_parameters/{args.model}.json', 'r'))
    
    parameter_set = parameter_file[args.parameter_set]

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
    elif args.model == 'part_av':
        from verification_models.part_av import PART_AV
        model = PART_AV(args, parameter_set)
    elif args.model == 'luar_av':
        from verification_models.luar_av import LUAR_AV
        model = LUAR_AV(args, parameter_set)
    elif args.model == 'stel_av':
        from verification_models.stel_av import STEL_AV
        model = STEL_AV(args, parameter_set)
    elif args.model == 'llm_prompting':
        from verification_models.llm_prompting import LLM_Prompting
        model = LLM_Prompting(args, parameter_set)
    
    print(f"Created model (parameters {args.parameter_set}) at {model.model_folder}")
    
    if args.train:
        model.train()
    elif args.load:
        model.load_model(args.load_folder)

    if args.test:
        model.test_and_save()
        print(f'Saved Model {model.model_folder}.')
    
if __name__ == '__main__':
    
    args = get_parser()
    main(args)