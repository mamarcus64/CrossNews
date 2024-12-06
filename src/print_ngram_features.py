import pickle
import numpy as np

# model_path = 'models/ngram/CrossNews_Article_Article/05-27-12-27-49-hokftf/model.p'
# print_file = 'ngram_features.txt'

model_path = 'models/ngram/CrossNews_Tweet_Tweet/05-27-12-28-40-gjelip/model.p'
print_file = 'ngram_features_tweet.txt'
features_to_print = 20

clf, transformer, _, _ = pickle.load(open(model_path, 'rb'))

feature_names = transformer.get_feature_names()
coefficients = clf.coef_.flatten()
coefficients = np.abs(coefficients)

coefficient_indices = np.argsort(np.abs(coefficients))[::-1]

feature_rows = []
i = -1

valid_pos = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP\$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP\$', 'WRB']

while len(feature_rows) < features_to_print:
    i += 1
    index = coefficient_indices[i]
    feature_row = []
    
    feature = feature_names[index]
    feature = feature.replace('-None-', '')
    feature = feature.replace('&', '\&')
    feature = feature.replace('$', '\$')
    
    f_type, f_details = tuple(feature.split('__'))
    
    if f_type == 'word_distr':
        gram = f_details.split(' ')
        feature_row.append(f'Frequency of Word {len(gram)}-gram')
        feature_row.append(' '.join(gram))
    elif f_type == 'pos_tag_distr':
        gram = f_details.split(' ')
        # if any([1 if x not in valid_pos else 0 for x in gram]):
        if '' in gram:
            continue
        feature_row.append(f'Frequency of Part of Speech {len(gram)}-gram')
        feature_row.append(' '.join(gram))
    elif f_type == 'char_distr' or f_type == 'special_char_distr':
        gram = list(f_details)
        gram = ['[space]' if x == ' ' else x for x in gram]
        feature_row.append(f'Frequency of Char {len(gram)}-gram')
        feature_row.append( ''.join(gram))
    elif f_type =='distr_chars_per_word':
        feature_row.append(f'Average Number of Specific Char per Word')
        feature_row.append(f_details)
    elif f_type == 'pos_tag_stats' and f_details.startswith('tag_word_length_'):
        f_details = f_details.replace('tag_word_length_', '')
        feature_row.append(f'Average Word Length of Specific Part of Speech')
        feature_row.append(f_details)
    elif f_type == 'pos_tag_stats' and f_details.startswith('tag_fraction_'):
        f_details = f_details.replace('tag_fraction_', '')
        gram = f_details.split(' ')
        # if any([1 if x not in valid_pos else 0 for x in gram]):
        if '' in gram:
            continue
        feature_row.append(f'Frequency of Part of Speech {len(gram)}-gram')
        feature_row.append(' '.join(gram))
    else:
        continue
    coef_score = round(coefficients[index], 3)
    feature_row.append(str(coef_score))
    feature_rows.append(' & '.join(feature_row) + '\\\\')
    
open(print_file, 'w', encoding='utf-8').write('\n'.join(feature_rows))
