from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_probs, judge_probs, conversation, network
import summary.metrics_regression

import json
with open('metrics.json', encoding='utf-8') as f:
    metrics = json.loads(f.read())


layer1_header = '''\
Please act as an impartial judge and evaluate as requested the response provided by an AI assistant to the user question displayed below. Do not allow the length of the response to influence your evaluation. Be as objective as possible.
{m} After providing your explanation, output your final verdict by strictly following this format:
 "[[1]]" if the response is very bad (A completely invalid response. It would be difficult to recover the conversation after this.),
 "[[2]]" if the response is bad (Valid response, but otherwise poor in quality),
 "[[3]]" if the response is neutral (means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.),
 "[[4]]" if the response is good (means this is a good response, but falls short of being perfect because of a key flaw.),
 "[[5]]" if the response is very good (means this response is good and does not have any strong flaws).  
'''

layer2_header = layer1_header

layer1_prompt = '''\
[User Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
'''

layer3_header = '''\
As a professional evaluator, you will receive answers from an artificial intelligence assistant to the user questions displayed below. Please choose an evaluation perspective to analyze the assistant's answers. Note that the angle you choose should be as unique as possible.
'''

def layer2_prompt(case, all_score):
    p = layer1_header + "\nYou and your colleagues in the expert group have conducted several rounds of evaluations."
    for i, score in enumerate(all_score[-1]):
        p += f"\n[The Start of Your Colleagues’ Evaluations {i}]\n{score}\n[The End of Your Colleagues’ Evaluations {i}]"
    p += '\nPlease provide your own final verdict.'
    return p

metrics = {k: v for k, v in metrics.items() if not k.startswith('gen')}

def pick_score(c):
    beg = c.index('[[')
    if beg >= 0:
        end = c.index(']]')
        ret = int(c[beg + 2:end])
    else:
        raise Exception(f'回答格式错误: {repr(c)}')
    return ret

judge = network.Judge([
    [network.Cell(deepseek.Model(kargs={'temperature': 1.0}), name=f"network2_1_{i}", system_prompt=layer3_header, fill_prompt=lambda case, all_score: layer1_prompt.format(**case)) for i in range(3)],
    # [network.Cell(deepseek.Model(), name="network1_1_" + metric, system_prompt=layer1_header.format(m=mp), fill_prompt=lambda case, all_score: layer1_prompt.format(**case)) for metric, mp in metrics.items()],
    [network.Cell(deepseek.Model(), name="network2_2_" + metric, system_prompt=layer2_header.format(m=mp), fill_prompt=layer2_prompt, deal_with_score=pick_score) for metric, mp in metrics.items()],
], overwrite=True)


datasets = load_result('dataset/topical-chat', f'output/topical-chat/deepseek')
for dataset in datasets:
    gen_score(judge, dataset) 
print()

for dataset in datasets:
    summary.metrics_regression.Summary().print_summary(dataset, metric_names=["network2_2_" + x for x in metrics.keys()])


'''
涉及的事实
存在的矛盾
问题的需求
'''