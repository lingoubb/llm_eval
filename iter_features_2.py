# 同义转换



p = '''\
你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的一些提示词:
[
    'The assistant\'s answer can effectively solve the problem',
    'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    'Do you think it is necessary to provide clear and easy-to-understand explanations or steps to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to consider the user\'s possible level of background knowledge to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to offer multiple solutions or perspectives to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide additional resources or references to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide timely information on time-sensitive issues to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to appropriately simplify or break down complex issues to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide specific examples or cases to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to offer clear advice or action steps to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide warnings or considerations about potential risks to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide data support or statistical information to answer this question? If so, check if the assistant\'s answer did so.',
    'Do you think it is necessary to provide cross-cultural or cross-linguistic considerations to answer this question? If so, check if the assistant\'s answer did so.',
    'The assistant\'s response provides sufficient necessary information',
    'The assistant\'s response does not contain unnecessary redundant information',
    'The assistant\'s response does not contain tedious or repetitive content',
    'The information mentioned in the assistant\'s response does not contain any information that is inconsistent with facts or fabricated information'
]

请将这些提示词逐个改写，使它们在基本保持核心语义的情况下，变得和原句尽可能不同。
将你改写的提示词按原格式输出。
'''

from model import deepseek

model = deepseek.Model()

output = model.get_outputs([[
    {
        'role': 'system',
        'content': p
    }
]], text=True)[0]

print(output)