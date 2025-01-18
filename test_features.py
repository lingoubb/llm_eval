import json
from model.deepseek import Model
from tqdm import tqdm

path = r'D:\workspace\llm_eval\output\1226\pandalm\pandalm.json'
# path = r'D:\workspace\llm_eval\output\1226\arena_1000\arena.json'

with open(path, encoding='utf-8') as f:
    all = json.load(f)

m = {}
for k in all[0]['metrics']:
    # if k != 'baseline':
    if k.startswith('with_'):
        m[k] = {}

todo = {}

for k, v in m.items():
    print(k)
    better, worse = [], []
    for x in all:
        t = x['features']['类型']
        v.setdefault(t, [0, 0, 0])
        v[t][0] += 1 
        if abs(x['metrics'][k] - x['manual_score']) > abs(x['metrics']['baseline'] - x['manual_score']):
            v[t][1] += 1
            better.append(x['question'])
        elif abs(x['metrics'][k] - x['manual_score']) < abs(x['metrics']['baseline'] - x['manual_score']):
            v[t][2] += 1
            worse.append(x['question'])
    todo[k] = [better, worse]
    print(v)


deepseek = Model().get_outputs
for k, v in tqdm(todo.items()):
    p = f'''\
现在有一个 ai 助手对一系列问题进行了回答，并有专家评估人员对它的回答进行了评估，分别指出了它回答得很好的问题和回答得很差的问题。现在你需要根据这些问题列表，帮助我们分别总结归纳该 ai 助手可能更擅长回答怎样的问题，以及可能更不擅长回答怎样的问题。

[ai 助手回答得很差的问题列表 开始]
{'\n'.join([(f"{i}: " + repr(v[1][i])) for i in range(len(v[1]))])}
[ai 助手回答得很差的问题列表 结束]

[ai 助手回答得很好的问题列表 开始]
{'\n'.join([(f"{i}: " + repr(v[0][i])) for i in range(len(v[0]))])}
[ai 助手回答得很好的问题列表 结束]
'''
    i = [
        {"role": "system", "content": "你是一个 ai 助手评估专家。"},
        {"role": "user", "content": p},
    ]
    
    r = deepseek([i], temperature=0, text=True)[0]
    v.append(r)

    print(k)
    print(r)
    print()


with open('tmp_features_log.json', 'w', encoding='utf-8') as f:
    json.dump(todo, f, ensure_ascii=False, indent=1)