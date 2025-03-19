import json
import os

p = 'dataset/biasbench'

for n in os.listdir(p):
    with open(p + '/' + n, encoding='utf-8') as f:
        a = json.load(f)
    for x in a:
        if x['manual_score'] == 0:
            x['manual_score'] = -1
    
    with open(p + '/' + n, 'w',  encoding='utf-8') as f:
        json.dump(a, f, indent=1)