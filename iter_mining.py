
p = '''\
你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的提示词: "{prompt}"
请对其进行扩写，要求包含实现该评估目的的详细步骤。
将你改写的提示词按原格式输出。
'''

from model import deepseek, gpt_3

model = gpt_3.Model()

ps = [
    'The assistant\'s answer can effectively solve the problem',
    'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    'The assistant\'s response provides sufficient necessary information',
    'The assistant\'s response does not contain unnecessary redundant information',
    'The assistant\'s response does not contain tedious or repetitive content',
    'The information mentioned in the assistant\'s response does not contain any information that is inconsistent with facts or fabricated information'
]


output = model.get_outputs([[
    {
        'role': 'system',
        'content': p
    }
]], text=True, temperature=1)[0]

print(output)