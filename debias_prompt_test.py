import json

with open('debias_prompt.json') as f:
    ps = json.load(f)

file_name = {
    'Length Bias': 'length_bias',
    'Concreteness Bias': 'concreteness',
    'Empty Reference Bias': 'empty_reference',
    'Content Continuation Bias': 'content_continuation',
    'Nested Instruction Bias': 'nested_instruction',
    'Familiar Knowledge Bias': 'familiar_knowledge_preference_bias',
}

from model.deepseek import Model
model = Model()

with open(r"resource\biasbench\biasbench.json", encoding='utf-8') as f:
    pl = json.load(f)

for k, v in file_name.items():
    print(k)
    cot = 0
    for x in pl[v]:
        resp = model.get_outputs([[{'role': 'system', 'content': ps[k].format(question=x['instruction'], answer1=x['response1'], answer2=x['response2'])}]], text=True)[0]
        print('\t' + resp)
        if 'YES' in resp:
            cot += 1
    print(f'{cot/len(pl[v]):.3f}')
    print()

