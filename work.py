from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_probs
import summary.compare_metrics

import json
with open('metrics.json') as f:
    metrics = json.loads(f.read())

header_v1 = '''\
Please act as an impartial judge and evaluate as requested the responses provided by two AI assistants to the user question displayed below. 
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
{m} Output your final verdict without any explaination by strictly following this format: "A" if assistant A is better, "B" if assistant B is better, and "C" for a tie.'''

prompt_user = '''\
[User Question]
{question}
[The Start of Assistant A’s Answer]
{output_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{output_b}
[The End of Assistant B’s Answer]
'''

for metric, mp in metrics.items():
    for k, v in {
        # 'gpt_3': gpt_3,
        'deepseek': deepseek,
    }.items():
        datasets = load_result('dataset/arena_1000', f'output/arena_1000/{k}')
        # datasets = load_result('dataset/pandalm', f'output/pandalm/{k}')
        for dataset in datasets:
            gen_score(compare_probs.Judge(v.Model(), metric, header_v1.format(m=mp), prompt_user), dataset)
        print()

for dataset in datasets:
    summary.compare_metrics.Summary().print_summary(dataset)