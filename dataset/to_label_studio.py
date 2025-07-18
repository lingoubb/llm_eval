import json
import random

d = {
    'pandalm\\pandalm.json': 'pandalm.json',
    'mt_bench_train\\mt_bench_train.json': 'mt_bench_train.json',
    'arena_6000\\arena.json': 'arena.json',
}

half_1 = []
half_2 = []

for k, v in d.items():
    with open(k) as f:
        a = json.load(f)
    

    a = [
        {
            "data": {
                'source': v,
                'order_id': i,
                'instruction': x['question'],
                'output_a': x['output'][0],
                'output_b': x['output'][1],
                'original_label': x['manual_score']
            }
        }
        for i, x in enumerate(a)
    ]
    a = random.sample(a, 100)
    for i, x in enumerate(a):
        x['data']['id'] = i
    half_1 += a[:50]
    half_2 += a[50:]


    # with open(v, 'w') as f:
    #     json.dump(a, f, indent=1)



with open('half_1.json', 'w') as f:
    json.dump(half_1, f, indent=1)
with open('half_2.json', 'w') as f:
    json.dump(half_2, f, indent=1)