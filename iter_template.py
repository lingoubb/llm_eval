prompt_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
'''

append = '**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'

a = [
    # 'The assistant\'s answer can effectively solve the problem',
    # 'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    # 'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    # 'The assistant\'s response provides sufficient necessary information',
    # 'The assistant\'s response does not contain unnecessary redundant information',
    # 'The assistant\'s response does not contain tedious or repetitive content',
    'The information mentioned in the assistant\'s response does not contain any information that is inconsistent with facts or fabricated information'
]

p = '''\
你是一个问答系统评估专家。现在你正在利用一个大语言模型进行问答评估，以下是你使用的提示词：
[提示词开始]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[提示词结束]

现在你需要根据具体的判题要求改写这个提示词（使用英文），你改写的提示词需要**更详尽地、更清晰地**指示大语言模型判断问答是否符合下面的要求： {metric}

'''
from model import deepseek

model = deepseek.Model()

outputs = [
    '[Revised Prompt]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistant\'s answer effectively solves the user\'s problem.\n\n**Evaluation Criteria:**\n1. **Relevance**: Does the response directly address the user\'s question?\n2. **Accuracy**: Is the information provided correct and reliable?\n3. **Completeness**: Does the response cover all necessary aspects of the question?\n4. **Clarity**: Is the response easy to understand and free from ambiguity?\n5. **Practicality**: Does the response provide actionable and useful advice or information?\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\nThis revised prompt provides a more detailed and clear set of instructions for the language model to evaluate the responses based on specific criteria, ensuring a more objective and comprehensive assessment.',
    '[Revised Prompt]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should follow these steps:\n\n1. **Instruction Analysis**:\n   - First, carefully examine the user\'s question to identify if it contains any specific instructions regarding:\n     - Answer format requirements (e.g., bullet points, numbered lists, specific structure)\n     - Content requirements (e.g., must include certain elements, avoid specific topics)\n     - Role-playing instructions (e.g., act as a specific character or professional)\n     - Any other explicit directives\n\n2. **Compliance Evaluation**:\n   - For each assistant\'s response, determine whether it strictly adheres to all identified instructions from the user\'s question.\n   - Evaluate the response\'s completeness in addressing the question\'s requirements.\n   - Assess whether the response maintains the required format, content, and role-playing aspects (if specified).\n\n3. **Quality Assessment**:\n   - If no specific instructions are present in the question, evaluate the responses based on:\n     - Accuracy of information\n     - Clarity and coherence\n     - Depth of explanation\n     - Relevance to the question\n\n4. **Final Verdict**:\n   - Compare the two responses based on the above criteria.\n   - Output your final verdict by strictly following this format without providing any explanation:\n     - "[[A]]" if assistant A is better\n     - "[[B]]" if assistant B is better\n     - "[[C]]" for a tie\n\nRemember to:\n- Avoid any position biases\n- Ensure the order of responses does not influence your decision\n- Do not let response length affect your evaluation\n- Be as objective as possible in your assessment\n\n[End of Revised Prompt]',
    '[Revised Prompt]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on determining whether each assistant\'s answer is directly related to the user\'s question and does not contain irrelevant information.\n\n**Evaluation Criteria:**\n1. **Relevance**: The response must directly address the user\'s question without introducing unrelated topics or information.\n2. **Clarity**: The response should be clear and concise, avoiding unnecessary details that do not contribute to answering the question.\n3. **Accuracy**: The response should provide accurate information that is pertinent to the question asked.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on whether the response is relevant to the question and does not contain irrelevant information.\n\n**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n**Note**: If either response contains information that is not relevant to the user\'s question, it should be considered less favorable, regardless of other factors.',
    '[Revised Prompt Start]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on determining whether each assistant\'s response provides sufficient and necessary information to adequately address the user\'s query.\n\nKey Evaluation Criteria:\n1. Completeness: Does the response cover all essential aspects of the question?\n2. Relevance: Is the information provided directly pertinent to the user\'s query?\n3. Accuracy: Are the facts and details presented correct and reliable?\n4. Clarity: Is the information presented in a clear and understandable manner?\n5. Depth: Does the response provide enough detail to fully satisfy the user\'s information needs?\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A provides more sufficient and necessary information, "[[B]]" if assistant B provides more sufficient and necessary information, and "[[C]]" if both assistants provide equally sufficient and necessary information.\n[Revised Prompt End]',
    '[Revised Prompt]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on assessing whether the assistant\'s response contains unnecessary redundant information.\n\n**Evaluation Criteria:**\n1. **Conciseness**: The response should be concise and to the point, avoiding any unnecessary repetition or redundant information.\n2. **Relevance**: The response should directly address the user\'s question without including irrelevant details.\n3. **Clarity**: The response should be clear and easy to understand, without any convoluted or overly verbose explanations.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on the presence or absence of unnecessary redundant information in the responses.\n\n**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n**Note**: If both responses are equally concise and free of unnecessary redundant information, the verdict should be "[[C]]". If one response is clearly more concise and free of redundancy, the verdict should reflect that.',
    '[Revised Prompt]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the responses contain tedious or repetitive content.\n\n**Evaluation Criteria:**\n1. **Clarity and Conciseness**: The response should be clear and concise, avoiding unnecessary repetition or overly verbose explanations.\n2. **Relevance**: The response should directly address the user\'s question without straying into irrelevant details or redundant information.\n3. **Engagement**: The response should maintain the user\'s interest by avoiding monotonous or overly repetitive phrasing.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on the presence or absence of tedious or repetitive content in the responses.\n\n**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n[End of Revised Prompt]',
    
]

for x in a:
    print(f'改写前：{x}')
    output = model.get_outputs([[
        {
            'role': 'system',
            'content': p.format(metric=x)
        }
    ]], text=True)[0]
    print(output)
    outputs.append(output)
    print('------------------------------')
    
import json
with open('tmp_template_2.json', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=1)