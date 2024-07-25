from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
import json
import statistics

def evaluate_scores(predictions, labels, threshold=0.5):
    rounded = [1 if val > threshold else 0 for val in predictions]
    labels = [int(label) for label in labels]
    
    try: # can throw errors in edge cases (all 1's, all 0's, etc.)
        roc = roc_auc_score(labels, predictions)
    except:
        roc = 0
    
    return {
        'accuracy': accuracy_score(labels, rounded),
        'precision': precision_score(labels, rounded),
        'recall': recall_score(labels, rounded),
        'f1': f1_score(labels, rounded),
        'auc': roc,
    }
    
def evaluate_attribution_scores(results, filter=None):
    if filter is None:
        ranks = [x['rank'] for x in results]
    else:
        ranks = [x['rank'] for x in results if filter(x)]
    
    scores = {}
    
    for k in [1, 8, 16, 32, 64]:
        scores[f'R@{k}' if k > 1 else 'Accuracy'] = sum([1 if rank <= k else 0 for rank in ranks]) / max(len(ranks), 1)
    
    scores['Mean_Reciprical_Rank'] = sum([1 / rank for rank in ranks]) / max(len(ranks), 1)
    if len(ranks) > 0:
        scores['Mean_Rank'] = statistics.mean(ranks)
        scores['Median_Rank'] = statistics.median(ranks)
    
    return scores