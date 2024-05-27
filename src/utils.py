from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
import json

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