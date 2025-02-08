import json
import os
path = r'D:\workspace\llm_eval\dataset\biasbench'


for name in os.listdir(path):
    with open(os.path.join(path, name)) as f:
        xx = json.load(f)

    for i in range(len(xx)):
        x = xx[i]
        if i % 2 == 0:
            x['manual_score'] = 1 - x['manual_score']
            x['output'][0], x['output'][1] = x['output'][1], x['output'][0]

    
    with open(os.path.join(path, name), 'w') as f:
        json.dump(xx, f, indent=1)