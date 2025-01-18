import json
from model.deepseek import Model
from tqdm import tqdm

path = r'D:\workspace\llm_eval\output\1226\pandalm\pandalm.json'

with open(path, encoding='utf-8') as f:
    all = json.load(f)

p = '''\
以下是一个问题和两个 ai 助手给出的回答，人类评估者认为助手 A 比助手 B 的回答更好，请指出你认为人类评估者可能是基于什么原因给出这样的判断。你只能给出一个原因，请挑选最关键的原因。
[问题开始]
{question}
[问题结束]
[ai 助手 A 的回答开始]
{better}
[ai 助手 A 的回答结束]
[ai 助手 B 的回答开始]
{worse}
[ai 助手 B 的回答结束]
'''

deepseek = Model().get_outputs

for x in tqdm(all):
    if abs(x['metrics']['baseline'] - x['manual_score']) == 2:
        if x['manual_score'] == 1:
            prompt = p.format(question=x['question'], better=x['output'][0], worse=x['output'][1])
        else:
            prompt = p.format(question=x['question'], better=x['output'][1], worse=x['output'][0])
        
        i = [
            {"role": "system", "content": "你是一个 ai 助手评估专家。"},
            {"role": "user", "content": prompt},
        ]
        
        x.setdefault('info', {})
        r = deepseek([i], temperature=0.8, text=True)[0]
        print(r)
        x['info']['find_features_2'] = r




with open(path, 'w', encoding='utf-8') as f:
    json.dump(all, f, indent=1, ensure_ascii=False)
