from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_probs, judge_probs, conversation
import summary.metrics_regression

import json
with open('metrics.json', encoding='utf-8') as f:
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

header_scorer = '''\
Please act as an impartial judge and evaluate as requested the response provided by an AI assistant to the user question displayed below. Do not allow the length of the response to influence your evaluation. Be as objective as possible.
{m} After providing your explanation, output your final verdict by strictly following this format:
 "[[1]]" if the response is very bad (A completely invalid response. It would be difficult to recover the conversation after this.),
 "[[2]]" if the response is bad (Valid response, but otherwise poor in quality),
 "[[3]]" if the response is neutral (means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.),
 "[[4]]" if the response is good (means this is a good response, but falls short of being perfect because of a key flaw.),
 "[[5]]" if the response is very good (means this response is good and does not have any strong flaws).  
'''


header_scorer_v2 = '''\
Please act as an impartial judge and evaluate as requested the response provided by an AI assistant to the user question displayed below. Do not allow the length of the response to influence your evaluation. Be as objective as possible.
Requestion: {m}
After providing your explanation, output your final verdict by strictly following this format:
"[[2]]" if the response fully meets the requestion, "[[1]]" if the response basically meets the requestion, "[[0]]" if the response does not meets the requestion at all.
'''

prompt_user_scorer = '''\
[User Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
'''


header = header_scorer

# af_metrics = {}
# for metric, mp in metrics.items():
#     af_metrics["header_scorer_v2_" + metric] = mp
# metrics = af_metrics
# header = header_scorer_v2

for metric, mp in metrics.items():
    for k, v in {
        # 'gpt_3': gpt_3,
        'deepseek': deepseek,
    }.items():
        # datasets = load_result('dataset/arena_1000', f'output/arena_1000/{k}')
        datasets = load_result('dataset/topical-chat', f'output/topical-chat/{k}')
        for dataset in datasets:
            gen_score(conversation.Judge(v.Model(), metric, header.format(m=mp), prompt_user_scorer), dataset)
        print()

for dataset in datasets:
    summary.metrics_regression.Summary().print_summary(dataset, metric_names=[x for x in metrics.keys()])