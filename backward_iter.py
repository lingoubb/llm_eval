
from model import deepseek
m = deepseek.Model()

# 根据负样本生成新维度
dp = '''\
你是一个负责设计基于 LLM 的问答系统评估器的专家。现在你设计的评估器在以下用例上出现了偏差：
[User Question]
{question}
[The Start of Assistant A’s Answer]
{output_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{output_b}
[The End of Assistant B’s Answer]

你的评估器认为 Assistant A 的回答比 Assistant B 的回答更好，然而人类专家评估者给出的评估与此相反。

这是你的评估器在判题时所采用的所有评估角度：
{all}

该评估器产生误判可能是因为上面提到的评估角度还不够全面。请你分析人类专家认为 Assistant B 的回答更好的可能原因，并参考上面的评估角度的格式，在你输出的**最后一行**给出一个全新的评估角度（不需要加上 New Evaluation Angle 之类的额外说明），格式参考上面已有的评估角度（使用英文，尽可能和所有已有的评估角度都无关）。
'''

import json
with open(r'D:\workspace\llm_eval\output\0116\pandalm\pandalm.json', encoding='utf-8') as f:
    a = json.load(f)

exist = [
    "\"The assistant's answer can effectively solve the problem\"",
    "\"When evaluating, first briefly indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant's answer strictly follows these instructions\"",
    "\"The assistant's answer is related to the question, and isn't irrelevant to the question\"",
    "\"The assistant's response provides sufficient necessary information\"",
    "\"The assistant's response does not contain unnecessary redundant information\"",
    "\"The assistant's response does not contain tedious or repetitive content\"",
    "\"The information mentioned in the assistant's response does not contain any information that is inconsistent with facts or fabricated information\"",
]

rs = []
for c in a:
    try:    
        if abs(c['metrics']['baseline_0208'] - c['manual_score']) == 2:
            i = [
                {"role": "system", "content": dp.format(question=c['question'], output_a=c['output'][0], output_b=c['output'][1], all='\n'.join(exist))},
            ]
            r = m.get_outputs([i], text=True, temperature=1)[0].splitlines()[-1]
            exist.append(r)
            rs.append(r)
            print(r)
    except Exception:
        pass
import time
with open(f'backward_iter_r_{int(time.time()/1000)}.json', 'w', encoding='utf-8') as f:
    json.dump(rs, f, indent=1)