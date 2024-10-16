from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_judgelm, judge_probs
import summary.compare


for k, v in {
    'gpt_3': gpt_3,
    'deepseek': deepseek,
}.items():
    print(k)
    datasets = load_result('dataset/topical-chat', f'output/topical-chat/{k}')
    # datasets = load_result('dataset/arena_1000', f'output/arena_100/{k}')
    for dataset in datasets:
        gen_score(judge_probs.Judge(v.Model()), dataset)
        summary.compare.Summary().print_summary(dataset)
    print()

