prompt = '''
你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的提示词：

[提示词开始]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[提示词结束]

其中，{metric} 是需要填充的具体评估指令。我们需要丰富而全面的指令来对问答进行评估，以下是已有的指令：
- 助手的回答能够很好地解决问题
- 助手的回答与问题相关，没有答非所问
- 助手的回答提供了足够的必要信息
- 助手的回答不存在不必要的冗余信息
- 助手的回答不存在繁琐重复的内容
- 助手的回答中提到的信息不存在与事实不符的信息或编造的信息
- 评估时，先指出问题中是否存在关于回答格式、回答要求、角色扮演等的指令，然后判断助手的回答是否严格遵循了这些指令
- 助手的回答是否提供了清晰、易于理解的解释或步骤
- 助手的回答是否考虑了用户可能的背景知识水平
- 助手的回答是否提供了多个解决方案或视角
- 助手的回答是否在必要时提供了额外的资源或参考
- 助手的回答是否表现了适当的情感
- 助手的回答是否在时间敏感问题上提供了及时的信息
- 助手的回答是否在复杂问题上进行了适当的简化或分解
- 助手的回答是否在需要时提供了具体的例子或案例
- 助手的回答是否在需要时提供了明确的建议或行动步骤
- 助手的回答是否在需要时提供了对潜在风险的警告或注意事项
- 助手的回答是否在必要时提供了数据支持或统计信息
- 助手的回答是否在必要时提供了跨文化或跨语言的考虑

请你给出 10 个其他评估指令。要求它们关注的角度尽可能不同，与已有的指令关注的角度也尽可能不同，同时要足够简洁，并且对评估问答系统有帮助。
'''

t='''
'''

from model import deepseek

model = deepseek.Model()

output = model.get_outputs([[
    {
        'role': 'system',
        'content': prompt
    },
    {
        'role': 'assistant',
        'content': t
    },
    {
        'role': 'system',
        'content': "翻译为英文，按原格式输出"
    },
]], text=True)[0]

print(output)