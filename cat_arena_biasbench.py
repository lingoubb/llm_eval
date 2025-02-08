import sys
import json
import os
import random

model_name = sys.argv[1]

bias_path = 'output/0116/biasbench' + model_name
arena_path = 'output/0116/arena_6000_wo_ep' + model_name + '/arena.json'

all = []

with open(arena_path) as f:
    all += [{**x, 'anno': None} for x in json.load(f)]

for name in os.listdir(bias_path):
    with open(os.path.join(bias_path, name)) as f:
        all += [{**x, 'anno': name} for x in json.load(f)]


random.shuffle(all)

with open(f'output/0116/biasbench_arena{model_name}/all.json', 'w') as f:
    json.dump(all, f, indent=1)
