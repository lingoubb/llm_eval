# p = '''
# 你是大语言模型评估领域的专家。研究发现，使用大语言模型作为评估者对问答进行评估存在 {bias_name}: {bias_ex}

# 下面给出了一些容易引导评估者模型产生 {bias_name} 的问答对：
# {p_list}

# 请你补全下面的 prompt 中的 {{text_to_be_completed}} 部分, 使其能用于判断待评估的问答对是否容易引导评估者产生 {bias_name}:
# "这是一个人类与 ai 助手的问答对：\\n<Q>{{question}}</Q>\\n<A>{{answer}}</A>\\n现在你需要判断 {{text_to_be_completed}}。\\n你的回答只能是 \\"YES\\" 或 \\"NO\\"，不需要其它任何解释。"
# '''

p = '''
You are an expert in the field of large language model evaluation. Research has found that when using a large language model as an evaluator to assess questions and answers, there exists {bias_name}: {bias_ep}.

The following are some question-answer pairs that are likely to lead the evaluator model to generate {bias_name}. In these examples, Answer1 is better than Answer2. However, due to {bias_name}, the evaluator model mistakenly thinks that Answer2 is better. :
{p_list}

Please complete the part of {{text_to_be_completed}} in the following prompt so that it can be used to determine whether the question-answer pair to be evaluated is likely to lead the evaluator to generate {bias_name}, The content you complete should clearly and comprehensively introduce this bias and analyze the reasons for the occurrence of this bias:
"This is a question-answer pair between a human and two AI assistant: \\n<Question>{{question}}</Question>\\n<Answer1>{{answer1}}</Answer1>\\n<Answer2>{{answer2}}</Answer2>\\nNow you need to determine {{text_to_be_completed}}. \\nYour answer can only be either \\"YES\\" or \\"NO\\", and no other explanations are needed."
'''

bias = {
    'Length Bias': 'A well - known and significant bias is length bias. This bias refers to the tendency of judge models to prefer longer responses, regardless of their quality or how well they adhere to the instruction.',
    'Concreteness Bias': 'Concreteness bias refers to the tendency to assign greater credibility to responses with specific details, including citation of authoritative sources, numerical values and complex terminologies. ',
    'Empty Reference Bias': 'In case of an incomplete instruction, such as a request for summary without target text, a good response would be to ask back to clarify the instruction or to honestly state the response’s uncertainty. Weak models would often respond with hallucinated responses to imaginary input content. Empty reference bias refers to the tendency of judge models to prefer such hallucinated content that seem to be associated with the instruction. ',
    'Content Continuation Bias': 'When instructions are accompanied by input text, weak models can give story completion responses that continue the input text. Content continuation bias means having a tendency to prefer responses that complete the input text instead of those that correctly follow the given instruction. This might be because the model gives a higher likelihood to the completion of the most recent text. ',
    'Nested Instruction Bias': 'Nested instruction bias is the tendency of judge models to favor responses to questions or requests embedded within the input text of a given instruction. It is similar to content continuation bias but more challenging. This is because the wrong response may seemingly follow the instruction, and the model needs to discern whether the response deals with the main instruction instead of the nested one. ',
    'Familiar Knowledge Bias': 'Familiar knowledge bias refers to the preference for responses that describe knowledge commonly encountered in real - world data. When an instruction is related to real - world knowledge such as idioms or commonly known facts, the judge models favor the more familiar text over responses that precisely meet the instruction.',
}

file_name = {
    'Length Bias': 'length_bias',
    'Concreteness Bias': 'concreteness',
    'Empty Reference Bias': 'empty_reference',
    'Content Continuation Bias': 'content_continuation',
    'Nested Instruction Bias': 'nested_instruction',
    'Familiar Knowledge Bias': 'familiar_knowledge_preference_bias',
}

import json
with open(r"resource\biasbench\biasbench.json", encoding='utf-8') as f:
    pl = json.load(f)

from model.deepseek import Model
model = Model()

r = {}
try:
    for k, fn in file_name.items():
        p_list = ["<Question>{question}</Question>\\n<Answer1>{answer1}</Answer1>\\n<Answer2>{answer2}</Answer2>".format(question=x['instruction'], answer1=x['response1'], answer2=x['response2']) for x in pl[fn]]
        np = p.format(
            bias_name = k,
            bias_ep = bias[k],
            p_list = '\n'.join(p_list)
        )
        print(np)
        resp = model.get_outputs([[{'role': 'system', 'content': np}]], text=True)[0]
        print('-'*10)
        print(resp)
        print('-'*10)
        r[k] = resp
finally:
    with open('debias_prompt.json', 'w') as f:
        json.dump(r, f, indent=1)