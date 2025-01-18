import json
from model.deepseek import Model
from tqdm import tqdm

path = r'D:\workspace\llm_eval\output\1226\pandalm\pandalm.json'

with open(path, encoding='utf-8') as f:
    all = json.load(f)

p = '''\
以下是一系列问题，请根据这些问题罗列出若干个问题类型，如计算、知识问答、日常对话、角色扮演等\n\n'''

deepseek = Model().get_outputs
qs = set()
for x in all:
    qs.add(repr(x['question']))

i = 1
for x in qs:
    p += f'问题{i}: {x}\n'
    i += 1

print(p)
# p = [
#     {"role": "system", "content": "你是一个助手。"},
#     {"role": "user", "content": p},
# ]
# print(deepseek([p], text=True)[0])
