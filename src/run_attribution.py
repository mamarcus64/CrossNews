import json
import shutil

from arguments import get_parser

def main(args):
    args.model = args.model.lower()
    
    print(f'Creating model {args.model}...')
    
    parameter_file = json.load(open(f'src/model_parameters/{args.model}.json', 'r'))
    
    parameter_set = parameter_file[args.parameter_set]

    if args.model == 'luar_aa':
        from attribution_models.luar_aa import LUAR_AA
        model = LUAR_AA(args, parameter_set)
    elif args.model == 'part_aa':
        from attribution_models.part_aa import PART_AA
        model = PART_AA(args, parameter_set)
    elif args.model == 'stel_aa':
        from attribution_models.stel_aa import STEL_AA
        model = STEL_AA(args, parameter_set)
    elif args.model == 'ngram_aa':
        from attribution_models.ngram_aa import NGram_AA
        model = NGram_AA(args, parameter_set)
    elif args.model == 'ppm_aa':
        from attribution_models.ppm_aa import PPM_AA
        model = PPM_AA(args, parameter_set)
    elif args.model == 'llm_prompting_aa':
        from attribution_models.llm_prompting_aa import LLM_Prompting_AA
        model = LLM_Prompting_AA(args, parameter_set)
    
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