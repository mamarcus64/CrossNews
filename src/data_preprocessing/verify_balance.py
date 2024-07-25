def verify_balance(df):
    
    doc_to_pos = {}
    doc_to_neg = {}
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        id0, id1 = row['id0'], row['id1']
        
        if row['label'] == 1:
            doc_to_pos[id0] = doc_to_pos.get(id1, []) + [id1]
            doc_to_pos[id1] = doc_to_pos.get(id0, []) + [id0]
        if row['label'] == 0:
            doc_to_neg[id0] = doc_to_neg.get(id1, []) + [id1]
            doc_to_neg[id1] = doc_to_neg.get(id0, []) + [id0]
    
    balance_diff = [len(doc_to_pos[k]) - len(doc_to_neg[k]) for k in list(doc_to_pos.keys())]
    if abs(sum(balance_diff)) > len(df) * 0.02: # account for edge cases, truncation, etc.
        print('error')