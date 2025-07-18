import json
from eval import load_result
import os

model_name = '_qwen7b'

datasets = []
# datasets += load_result('dataset/topical-chat', f'output/0116/topical-chat' + model_name)
datasets += load_result('dataset/mt_bench_train', f'output/0116/mt_bench_train' + model_name)
# datasets += load_result('dataset/biasbench', f'output/0116/biasbench' + model_name)
datasets += load_result('dataset/pandalm', f'output/0116/pandalm' + model_name)
# datasets += load_result(None, f'output/0116/biasbench_arena' + model_name)
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000_wo_ep' + '' if model is model_deepseek else '_gpt3')
datasets += load_result('dataset/arena_6000', f'output/0116/arena_6000_wo_ep' + model_name)
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/llmeval2', f'output/0116/llmeval2' + model_name)
# datasets += load_result('dataset/JudgeBench', f'output/0116/JudgeBench' + model_name)


all_content = []
for dataset in datasets:
    all_content += dataset.content

path = f'output/0116/cat_all{model_name}'
if not os.path.exists(path):
    os.makedirs(path)

    with open(f'{path}/cat_all.json', 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=1, ensure_ascii=False)


datasets = []
datasets += load_result('dataset/llmeval2', f'output/0116/llmeval2' + model_name)
datasets += load_result('dataset/JudgeBench', f'output/0116/JudgeBench' + model_name)
for dataset in datasets:
    all_content += dataset.content


path = f'output/0116/cat_all_2{model_name}'
if not os.path.exists(path):
    os.makedirs(path)

    with open(f'{path}/cat_all_2.json', 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=1, ensure_ascii=False)