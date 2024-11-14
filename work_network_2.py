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

# layer1
user_prompt = '''\
[User Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
'''
user_prompt_wo = '''\
[User Question]
{question}
'''

layer1 = []

system_prompt = "You will receive a question and an assistant's answer to that question. What do you think are the drawbacks of that answer? If you're not sure if there are any drawbacks in the answer, please reply {NO}."
layer1.append(network.Cell(deepseek.Model(kargs={'temperature': 0}), name=f"network3_1_0", system_prompt=system_prompt, fill_prompt=lambda case, all_score: user_prompt.format(**case)))
              
system_prompt = "You will receive a question and an assistant's answer to that question. What do you think are the merits of that answer? If you're not sure if there are any merits in the answer, please reply {NO}."
layer1.append(network.Cell(deepseek.Model(kargs={'temperature': 0}), name=f"network3_1_1", system_prompt=system_prompt, fill_prompt=lambda case, all_score: user_prompt.format(**case)))
              
system_prompt = "You will receive a question and an assistant's answer to that question. What do you think are the aspects where the answer contradicts the facts? If you're not sure whether there is a contradiction with the facts, please reply {NO}."
layer1.append(network.Cell(deepseek.Model(kargs={'temperature': 0}), name=f"network3_1_2", system_prompt=system_prompt, fill_prompt=lambda case, all_score: user_prompt.format(**case)))
              
system_prompt = "You will receive a question. You don't actually need to answer this question, but you are required to point out the train of thought for answering it. If you think there isn't a definite train of thought, please reply {NO}."
layer1.append(network.Cell(deepseek.Model(kargs={'temperature': 0}), name=f"network3_1_3", system_prompt=system_prompt, fill_prompt=lambda case, all_score: user_prompt_wo.format(**case)))
              
system_prompt = "You will receive a question. You don't actually need to answer this question, but you are required to point out some factual bases that need to be known in order to answer it. If you think there are no facts that must be known, please reply {NO}."
layer1.append(network.Cell(deepseek.Model(kargs={'temperature': 0}), name=f"network3_1_4", system_prompt=system_prompt, fill_prompt=lambda case, all_score: user_prompt_wo.format(**case)))
              
# layer2
system_prompt = '''\
Please act as an impartial judge and evaluate as requested the response provided by an AI assistant to the user question displayed below. Do not allow the length of the response to influence your evaluation. Besides, you will also receive some suggestions given by others. You can refer to these suggestions to conduct your evaluation. Be as objective as possible.
{m} After providing your explanation, output your final verdict by strictly following this format:
 "[[1]]" if the response is very bad (A completely invalid response. It would be difficult to recover the conversation after this.),
 "[[2]]" if the response is bad (Valid response, but otherwise poor in quality),
 "[[3]]" if the response is neutral (means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.),
 "[[4]]" if the response is good (means this is a good response, but falls short of being perfect because of a key flaw.),
 "[[5]]" if the response is very good (means this response is good and does not have any strong flaws).  
'''
user_prompt_s = '''\
[User Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
[The Start of Others’s suggestions]
{suggestions}
[The End of Others’s suggestions]
'''

layer2 = []
for metric, mp in metrics.items():
    layer2.append(network.Cell(deepseek.Model(), name="network3_2_" + metric, system_prompt=system_prompt.format(m=mp), fill_prompt=lambda case, all_score: user_prompt_s.format(**case, suggestions='\n'.join([x for x in all_score[-1] if '{NO}' not in x])), deal_with_score=pick_score))

judge = network.Judge([
    layer1,
    layer2
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