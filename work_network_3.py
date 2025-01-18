

from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_probs, judge_probs, conversation, network
import summary.metrics_regression

import json
with open('metrics.json', encoding='utf-8') as f:
    metrics = json.loads(f.read())

metrics = {k: v for k, v in metrics.items() if not k.startswith('gen')}

def pick_score(c):
    beg = c.index('[[')
    if beg >= 0:
        end = c.index(']]')
        ret = int(c[beg + 2:end])
    else:
        raise Exception(f'回答格式错误: {repr(c)}')
    return ret

# layer2
system_prompt = '''\
[System]
We would like to request your feedback on the performance of an AI assistant in response
to the user question displayed above.
{m}
Each assistant receives an overall score on a scale of 1 to 5, where a higher score indicates better overall performance. After providing your explanation, output your final verdict of overall score by strictly following this format: "[[1]]" or "[[2]]" or "[[3]]" or "[[4]]" or "[[5]]".
There are a few other referees assigned the same task, it’s your responsibility to discuss with them and think critically before you make your final judgment.
'''

user_prompt = '''\
Here is your discussion history:
{chat_history}
Now it’s your time to talk, please make your talk short and clear!
[Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
'''

name = ['Alice', 'Bob', 'White', 'Steve', 'Vex']

layers = []
i = 0
for metric, mp in metrics.items():
    # layer2.append(network.Cell(deepseek.Model(), name="network3_2_" + metric, system_prompt=system_prompt.format(m=mp), fill_prompt=lambda case, all_score: user_prompt.format(**case), deal_with_score=pick_score))

    layers.append([network.Cell(deepseek.Model(), 
                                name="network4_" + metric, 
                                system_prompt=system_prompt.format(m=mp), 
                                fill_prompt=lambda case, all_score: user_prompt.format(**case, chat_history='\n'.join([f'One of your colleagues： {x}' for j, x in enumerate(all_score)])),
                                deal_with_score=pick_score)])


judge = network.Judge(layers, overwrite=True)    

datasets = load_result('dataset/topical-chat', f'output/topical-chat/deepseek')
for dataset in datasets:
    gen_score(judge, dataset)
print()

for dataset in datasets:
    summary.metrics_regression.Summary().print_summary(dataset, metric_names=["network4_" + x for x in metrics.keys()])

