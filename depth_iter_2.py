
# p = '''\
# 你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的提示词: "{prompt}"
# 请使用英文对其进行扩写，要求包含实现该评估目的的详细步骤。
# 请在你输出的最后一行，将你改写的提示词按原格式输出。
# '''

p = '''\
你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的提示词：
[提示词开始]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
**Output your final verdict by strictly following this format**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[提示词结束]

现在你需要改写这个提示词（使用英文），使其能够判断：{prompt}
你改写的提示词应该包含完成上述判断任务的详细步骤。请直接输出你改写后的提示词，不需要额外的说明。
'''

from model import deepseek, gpt_3

model = deepseek.Model()

ps = [
    'The assistant\'s answer can effectively solve the problem',
    'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    'The assistant\'s response provides sufficient necessary information',
    'The assistant\'s response does not contain unnecessary redundant information',
    'The assistant\'s response does not contain tedious or repetitive content',
    'The information mentioned in the assistant\'s response does not contain any information that is inconsistent with facts or fabricated information'
]

rs = []

for i in range(1):

    for x in ps:
        try:
            output = model.get_outputs([[
                {
                    'role': 'system',
                    'content': p.format(prompt=x)
                }
            ]], text=True, temperature=1)[0]
            rs.append(output)
            print(output)
        except Exception as e:
            print(e)

import json
with open('depth_iter_2.json', 'w') as f:
    json.dump(rs, f, indent=1)
