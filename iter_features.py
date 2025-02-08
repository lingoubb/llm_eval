p = '''
句子变换示例：
Does the assistant's response provide specific examples or cases when needed? -> Do you think it is necessary to provide specific examples or cases to answer this question? If so, check if the assistant's answer did so.

请按照示例的做法变换下列句子（你不需要输出原句子）：
Does the assistant's response provide clear and easy-to-understand explanations or steps? -> ?
Does the assistant's response consider the user's possible level of background knowledge? -> ?
Does the assistant's response offer multiple solutions or perspectives? -> ?
Does the assistant's response provide additional resources or references when necessary? -> ?
Does the assistant's response provide timely information on time-sensitive issues? -> ?
Does the assistant's response appropriately simplify or break down complex issues? -> ?
Does the assistant's response provide specific examples or cases when needed? -> ?
Does the assistant's response offer clear advice or action steps when needed? -> ?
Does the assistant's response provide warnings or considerations about potential risks when needed? -> ?
Does the assistant's response provide data support or statistical information when necessary? -> ?
Does the assistant's response provide cross-cultural or cross linguistic considerations when necessary? -> ?
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