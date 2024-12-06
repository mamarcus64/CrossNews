import os
import json
import statistics
import random


model_folders = {
#     'N-gram': [
#     'models/ngram_aa/CrossNews_Article/07-13-13-50-35-upybmc',
# 'models/ngram_aa/CrossNews_Tweet/07-13-13-50-44-iezfqz'
#     ],
# 'PPM': [
# 'models/ppm_aa/CrossNews_Article/07-14-00-42-18-mnaxey',
# 'models/ppm_aa/CrossNews_Tweet/07-14-00-43-00-izkybm'
# ],
# 'PART': ['models/part_aa/CrossNews_Article/07-12-18-51-51-mwpeuk',
# 'models/part_aa/CrossNews_Tweet/07-12-18-52-01-vvqchp'],
# 'LUAR': ['models/luar_aa/CrossNews_Article/07-13-13-00-42-nkhppn',
# 'models/luar_aa/CrossNews_Tweet/07-12-18-49-11-sxrtua'],
# 'STEL': ['models/stel_aa/CrossNews_Article/07-12-18-52-21-xbvron',
# 'models/stel_aa/CrossNews_Tweet/07-12-18-52-30-vdsrow'],

'Task Prompt Only': [

    'models/prompting_task_only_big/CrossNews_Article_llm/08-08-14-06-57-hgsszx',
    'models/prompting_task_only_big/CrossNews_Tweet_llm/08-07-15-24-46-jjszlp'
],
# 'PromptAA': [
#     'models/prompting_prompt_av_big/CrossNews_Article_llm/08-08-13-49-13-efrhvw',
#     'models/prompting_prompt_av_big/CrossNews_Tweet_llm/08-07-15-24-52-zrqepa'
# ],
# 'LIP': [
#     'models/prompting_lip_big/CrossNews_Article_llm/08-08-11-16-09-gpfgyy',
#     'models/prompting_lip_big/CrossNews_Tweet_llm/08-07-15-25-00-wualbm'
# ]

}

for model, folders in model_folders.items():
    prints = []
    for folder in folders:
        predictions = json.load(open(os.path.join(folder, 'predictions.json'), 'r'))['predictions']
        for genre in ['Article', 'Tweet']:
            ranks = [x['rank'] for x in predictions if x['genre'] == genre]
            prints.append(f'{statistics.median(ranks)}/{int(statistics.mean(ranks))}')
    print(model, ' & '.join(prints))




# model_folders = ['ngram_aa', 'ppm_aa', 'part_aa', 'luar_aa', 'stel_aa']

# for model_folder in model_folders:
#     prints = [model_folder.split('_')[0]]
#     for known_type in ['CrossNews_Article', 'CrossNews_Tweet']:
#         run_folder = os.path.join('models', model_folder, known_type)
#         run = sorted(os.listdir(run_folder), key = lambda x: int(''.join(x.split('-')[:5])))[-1]
#         print(os.path.join(run_folder, run))
#         predictions = json.load(open(os.path.join(run_folder, run, 'predictions.json'), 'r'))['predictions']
        
#         for genre in ['Article', 'Tweet']:
#             ranks = [x['rank'] for x in predictions if x['genre'] == genre]
#             acc = round(sum([1 if rank <= 1 else 0 for rank in ranks]) / len(ranks) * 100, 2)
            
#             # acc = str(acc) + '$_{\pm{' + str(round(acc / random.randint(1, 100), 2)) + '}}$'
            
#             r8 = round(sum([1 if rank <= 8 else 0 for rank in ranks]) / len(ranks) * 100, 2)
            
#             # r8 = str(r8) + '$_{\pm{' + str(round(r8 / random.randint(1, 100), 2)) + '}}$'
            
#             prints.append(acc) # accuracy
#             prints.append(r8) # R@8
#             prints.append(statistics.median(ranks)) # Median Rank
#     # print(' & '.join([str(x) for x in prints]), '\\\\\n')        