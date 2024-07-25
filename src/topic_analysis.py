import os
import json
import pandas as pd
from collections import Counter
import pdb
import random
import numpy as np
from sklearn.metrics import confusion_matrix

model_folders = {
    'luar_aa':
        {
            'CrossNews_Article': 'models/luar_aa/CrossNews_Article/07-13-13-00-42-nkhppn',
            # 'CrossNews_Tweet': 'models/luar_aa/CrossNews_Tweet/07-12-18-49-11-sxrtua',
        },
    'part_aa':
        {
            'CrossNews_Article': 'models/part_aa/CrossNews_Article/07-12-18-51-51-mwpeuk',
            # 'CrossNews_Tweet': 'models/part_aa/CrossNews_Tweet/07-12-18-52-01-vvqchp',
        },
    'stel_aa':
        {
            'CrossNews_Article': 'models/stel_aa/CrossNews_Article/07-12-18-52-21-xbvron',
            # 'CrossNews_Tweet': 'models/stel_aa/CrossNews_Tweet/07-12-18-52-30-vdsrow',
        },
    'ngram_aa':
        {
            'CrossNews_Article': 'models/ngram_aa/CrossNews_Article/07-13-13-50-35-upybmc',
            # 'CrossNews_Tweet': 'models/ngram_aa/CrossNews_Tweet/07-13-13-50-44-iezfqz',
        },
    'ppm_aa':
        {
            'CrossNews_Article': 'models/ppm_aa/CrossNews_Article/07-14-00-42-18-mnaxey',
            # 'CrossNews_Tweet': 'models/ppm_aa/CrossNews_Tweet/07-14-00-43-00-izkybm',
        },
}

# def author_to_topic():
#     query_ids = pd.read_csv('attribution_data/query/CrossNews_Article.csv')['id'].tolist()
#     gold = json.load(open('raw_data/crossnews_gold_topics.json', 'r', encoding='utf-8'))
#     id_to_doc = {doc['id']: doc for doc in gold}
    
#     author_to_topic_list = {}
#     for query_id in query_ids:
#         doc = id_to_doc[str(query_id)]
#         author, topic = doc['author'], doc['topic']
#         author_to_topic_list[author] = author_to_topic_list.get(author, []) + [topic]
    
#     result = {}
#     for author, topic_list in author_to_topic_list.items():
#         result[author] = Counter(topic_list).most_common(1)[0][0]
        
#     json.dump(result, open('attribution_data/query/author_topics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
# author_to_topic()

# def average_index(float_list, label_list, target_label):
#     # Combine the float and label lists into a list of tuples (float, label, original_index)
#     combined_list = [(float_list[i], label_list[i], i) for i in range(len(float_list))]
    
#     # Sort the combined list by the float values
#     combined_list.sort(key=lambda x: x[0])
    
#     count = 0
#     for float_val, label, index in combined_list[:50]:
#         if label == target_label:
#             count += 1
#     return count
    
    # Extract the original indices of the target label from the sorted list
    # indices = [index for float_val, label, index in combined_list if label == target_label]
    # pdb.set_trace()
    
    # if not indices:
    #     return None  # Handle case where the target label is not found
    
    # # Calculate the average index
    # average_index = sum(indices) / len(indices)
    
    # return average_index



author_to_topic = json.load(open('attribution_data/query/author_topics.json', 'r', encoding='utf-8'))
global_ids = json.load(open('attribution_data/test/global_doc_ids.json', 'r'))

def print_topic_predictions():
    
    matrices = {}
    for model, model_folder in model_folders.items():
        for dataset, data_folder in model_folder.items():
            print('      ', model, dataset)
            prediction_file = json.load(open(os.path.join(data_folder, 'predictions.json'), 'r', encoding='utf-8'))
            predictions = prediction_file['predictions']
            author_list = prediction_file['author_list']
            topic_list = [author_to_topic[x] for x in author_list]
            
            mode = 'matrix'
            topics = ['Politics', 'Culture', 'Economy', 'Sports', 'Other']
            
            if mode == 'table':
                to_print = []
                to_print_random = []
                for topic in ['Politics', 'Culture', 'Economy', 'Sports', 'Other', 'All']:
                    count = 0
                    random_count = 0
                    for prediction in predictions:
                        if topic == 'All' or author_to_topic[prediction['label']] == topic:
                            if random.choice(topic_list) == author_to_topic[prediction['label']]:
                                random_count += 1
                            
                            if author_to_topic[prediction['prediction']] == author_to_topic[prediction['label']]:
                                count += 1
                            
                        # avg_idx.append(average_index(prediction['distances'], topic_list, author_to_topic[prediction['prediction']]))
                        # avg_random_idx.append(average_index(random.sample(list(range(len(topic_list))), len(topic_list)), topic_list, author_to_topic[prediction['prediction']]))
                            
                    # print(topic, count / len(predictions), random_count / len(predictions), count / random_count * 100 )
                    to_print.append(count / len(predictions))
                    to_print_random.append(random_count / len(predictions))
                print(to_print)
                # print(to_print_random)
                # print(np.mean(avg_idx), np.mean(avg_random_idx))
            if mode == 'matrix':
                y_true, y_pred = [], []
                for prediction in predictions:
                    y_true.append(author_to_topic[prediction['label']])
                    y_pred.append(author_to_topic[prediction['prediction']])
                c_mat = confusion_matrix(y_true, y_pred, labels=topics, normalize='true')
                matrices[model] = c_mat.tolist()
                print(c_mat)
                
    json.dump(matrices, open('matrices.json', 'w'), indent=4)
            
            
def global_stats():
    for model, model_folder in model_folders.items():
        for dataset, data_folder in model_folder.items():
            print('      ', model, dataset)
            prediction_file = json.load(open(os.path.join(data_folder, 'predictions.json'), 'r', encoding='utf-8'))
            predictions = prediction_file['predictions']
            author_list = prediction_file['author_list']
            
            all_count = 0
            global_count = 0
            for prediction in predictions:
                if prediction['label'] == prediction['prediction']:
                    all_count += 1
                    if prediction['id'] in global_ids:
                        global_count += 1
            
            print(all_count / len(predictions), global_count / 538)
            
            # topic_list = [author_to_topic[x] for x in author_list]
            
            # to_print = []
            # to_print_random = []
            # for topic in ['Politics', 'Culture', 'Economy', 'Sports', 'Other', 'All']:
            #     count = 0
            #     random_count = 0
            #     for prediction in predictions:
            #         if topic == 'All' or author_to_topic[prediction['label']] == topic:
            #             if random.choice(topic_list) == author_to_topic[prediction['label']]:
            #                 random_count += 1
                        
            #             if author_to_topic[prediction['prediction']] == author_to_topic[prediction['label']]:
            #                 count += 1
                        
            #         # avg_idx.append(average_index(prediction['distances'], topic_list, author_to_topic[prediction['prediction']]))
            #         # avg_random_idx.append(average_index(random.sample(list(range(len(topic_list))), len(topic_list)), topic_list, author_to_topic[prediction['prediction']]))
                        
            #     # print(topic, count / len(predictions), random_count / len(predictions), count / random_count * 100 )
            #     to_print.append(count / len(predictions))
            #     to_print_random.append(random_count / len(predictions))
            # print(to_print)
            # # print(to_print_random)
            # # print(np.mean(avg_idx), np.mean(avg_random_idx))
        
# global_stats()
print_topic_predictions()
            


    