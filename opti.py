from eval import load_result, gen_score, save_result
from model import deepseek, gpt_3, openai_api
from judge import compare_judgelm, judge_probs, judge_summ, judge_summ_new, judge_with_features
import summary.metrics_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr, kendalltau
import math
from sklearn import tree
from collections.abc import Iterable
from sklearn.feature_selection import RFE
import random
import sys

# 是否是打分数据集
is_score = False

ultra = False

# model_gpt3 = gpt_3.Model()
# model_deepseek = deepseek.Model()
# model_qwen7b = openai_api.Model('http://127.0.0.1:8050/v1', 'default', 'null')
# model_deepseek_2 = openai_api.Model('https://api.siliconflow.cn/v1', 'deepseek-ai/DeepSeek-V3', 'sk-yxzdnmrffeopxbqibtqkqtwtcigqqsjodjtahlmfshgcitnt')

# model_name = '_qwen7b'
# model_name = '_gpt3.5'
# model_name = '_deepseek_2'
# model_name, model = '', model_deepseek
# model_name, model = '_gpt3.5', gpt_3.Model()
# model_name, model = '_qwen72b', openai_api.Model('http://127.0.0.1:8051/v1', 'default', 'null')
model_name, model = '_qwen7b', openai_api.Model('http://127.0.0.1:8050/v1', 'default', 'null')

# model_name, model = '_deepseek_baidu', qianfan.Model()


datasets = []
# datasets += load_result('dataset/topical-chat', f'output/0116/topical-chat' + model_name)
# datasets += load_result('dataset/mt_bench_train', f'output/0116/mt_bench_train' + model_name)
# datasets += load_result('dataset/biasbench', f'output/0116/biasbench' + model_name)
# datasets += load_result('dataset/pandalm', f'output/0116/pandalm' + model_name)
datasets += load_result(None, f'output/0116/biasbench_arena' + model_name)
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000_wo_ep' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/arena_6000', f'output/0116/arena_6000_wo_ep' + model_name)
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/llmeval2', f'output/0116/llmeval2' + model_name)
# datasets += load_result('dataset/JudgeBench', f'output/0116/JudgeBench' + model_name)
def get_eval_res(features):
    for dataset in datasets:
        gen_score(judge_with_features.Judge(model, features=features), dataset, t=16)


# 初始维度
init_metrics = [
    '助手的回答能够很好地解决问题',
    '助手的回答与问题相关，没有答非所问',
    '助手的回答提供了足够的必要信息',
    '助手的回答不存在不必要的冗余信息',
    '助手的回答不存在繁琐重复的内容',
    '助手的回答中提到的信息不存在与事实不符的信息或编造的信息',
    '评估时，先指出问题中是否存在关于回答格式、回答要求、角色扮演等的指令，然后判断助手的回答是否严格遵循了这些指令',
    'The assistant\'s answer can effectively solve the problem',
    'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    'Does the assistant\'s response provide clear and easy-to-understand explanations or steps?',
    'Does the assistant\'s response consider the user\'s possible level of background knowledge?',
    'Does the assistant\'s response offer multiple solutions or perspectives?',
    'Does the assistant\'s response provide additional resources or references when necessary?',
    'Does the assistant\'s response exhibit appropriate emotional tone?',
    'Does the assistant\'s response provide timely information on time-sensitive issues?',
    'Does the assistant\'s response appropriately simplify or break down complex issues?',
    'Does the assistant\'s response provide specific examples or cases when needed?',
    'Does the assistant\'s response offer clear advice or action steps when needed?',
    'Does the assistant\'s response provide warnings or considerations about potential risks when needed?',
    'Does the assistant\'s response provide data support or statistical information when necessary?',
    'Does the assistant\'s response provide cross-cultural or cross linguistic considerations when necessary?',
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

pair_score_metircs = [    
    'The assistant\'s answer can effectively solve the problem',
    'When evaluating, first indicate whether there are instructions regarding answer format, answer requirements, role-playing, etc. in the question, and then determine whether the assistant\'s answer strictly follows these instructions',
    'The assistant\'s answer is related to the question, and isn\'t irrelevant to the question',
    'The assistant\'s response provides sufficient necessary information',
    'The assistant\'s response does not contain unnecessary redundant information',
    'The assistant\'s response does not contain tedious or repetitive content',
    'The information mentioned in the assistant\'s response does not contain any information that is inconsistent with facts or fabricated information'
]

# 改写1
init_metrics_2 = [
    'The assistant\'s response effectively addresses and resolves the issue',
    'During evaluation, first identify if the question includes any directives about response format, specific requirements, or role-playing scenarios, and then assess whether the assistant\'s response adheres strictly to these directives',
    'The assistant\'s response is pertinent to the question and does not deviate from the topic',
    'Is it essential to provide straightforward and comprehensible explanations or steps to answer this question? If yes, verify if the assistant\'s response includes these',
    'Is it crucial to consider the user\'s potential level of background knowledge when answering this question? If yes, check if the assistant\'s response takes this into account',
    'Is it important to present multiple solutions or viewpoints to answer this question? If yes, ensure the assistant\'s response offers these',
    'Is it necessary to include additional resources or references to answer this question? If yes, confirm if the assistant\'s response provides these',
    'Is it vital to provide up-to-date information for time-sensitive issues when answering this question? If yes, check if the assistant\'s response includes timely data',
    'Is it required to simplify or deconstruct complex issues appropriately to answer this question? If yes, verify if the assistant\'s response does so',
    'Is it necessary to include specific examples or cases to answer this question? If yes, ensure the assistant\'s response provides these',
    'Is it essential to offer clear advice or actionable steps to answer this question? If yes, check if the assistant\'s response includes these',
    'Is it important to provide warnings or considerations about potential risks when answering this question? If yes, verify if the assistant\'s response addresses these',
    'Is it necessary to include data support or statistical information to answer this question? If yes, confirm if the assistant\'s response provides these',
    'Is it crucial to consider cross-cultural or cross-linguistic factors when answering this question? If yes, check if the assistant\'s response takes these into account',
    'The assistant\'s response contains all the necessary information required',
    'The assistant\'s response is free from unnecessary or superfluous information',
    'The assistant\'s response avoids being overly verbose or repetitive',
    'The information provided in the assistant\'s response is accurate and does not contain any false or fabricated details'
]

# 改写2
init_metrics_3 = [
    "The answer given by the assistant can efficiently address the problem",
    "During the evaluation, first clarify whether there are any instructions about the answer format, answer requirements, role - playing, etc. in the query, and then judge whether the assistant's answer adheres strictly to these instructions",
    "The assistant's answer is relevant to the question and not off - topic",
    "Do you believe it is essential to present clear and comprehensible explanations or steps when answering this question? If so, verify if the assistant's answer has done that.",
    "Do you think it is necessary to take into account the user's possible background knowledge level when answering this question? If so, check if the assistant's answer has considered this aspect.",
    "Do you think it is required to provide multiple solutions or viewpoints when answering this question? If so, check if the assistant's answer has done so.",
    "Do you think it is necessary to supply additional resources or references when answering this question? If so, check if the assistant's answer has provided them.",
    "Do you think it is necessary to offer up - to - date information for time - sensitive issues when answering this question? If so, check if the assistant's answer has done that.",
    "Do you think it is necessary to properly simplify or break down complex issues when answering this question? If so, check if the assistant's answer has done so.",
    "Do you think it is necessary to provide specific examples or cases when answering this question? If so, check if the assistant's answer has provided them.",
    "Do you think it is necessary to offer clear suggestions or action steps when answering this question? If so, check if the assistant's answer has done so.",
    "Do you think it is necessary to give warnings or considerations about potential risks when answering this question? If so, check if the assistant's answer has done that.",
    "Do you think it is necessary to provide data support or statistical information when answering this question? If so, check if the assistant's answer has provided them.",
    "Do you think it is necessary to give cross - cultural or cross - linguistic considerations when answering this question? If so, check if the assistant's answer has done so.",
    "The information provided in the assistant's response is sufficient and necessary",
    "The assistant's response does not have any unnecessary surplus information",
    "The assistant's response does not include any tiresome or repetitive content",
    "The information stated in the assistant's response does not contain any information that contradicts facts or is fabricate",
]
# 特点
init_features = [
    '问题中是否存在关于回答格式、回答要求、角色扮演等的指令',
    '该问题是没有标准答案的开放式问题吗',
    '根据问题的类型和要求，你认为回答该问题应该尽可能简洁吗（反之应该尽可能提供更多信息）',
    '这是一道数学计算题吗',
    'Do you think it is necessary to provide clear and easy-to-understand explanations or steps to answer this question?',
    'Do you think it is necessary to consider the user\'s possible level of background knowledge to answer this question?',
    'Do you think it is necessary to offer multiple solutions or perspectives to answer this question?',
    'Do you think it is necessary to provide additional resources or references to answer this question?',
    'Do you think it is necessary to provide timely information on time-sensitive issues to answer this question?',
    'Do you think it is necessary to appropriately simplify or break down complex issues to answer this question?',
    'Do you think it is necessary to provide specific examples or cases to answer this question?',
    'Do you think it is necessary to offer clear advice or action steps to answer this question?',
    'Do you think it is necessary to provide warnings or considerations about potential risks to answer this question?',
    'Do you think it is necessary to provide data support or statistical information to answer this question?',
    'Do you think it is necessary to provide cross-cultural or cross-linguistic considerations to answer this question?',
]

init_template = [
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary criterion for evaluation should be whether the assistant's answer effectively solves the user's problem.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistants strictly follow any specific instructions regarding answer format, answer requirements, role-playing, etc., as indicated in the user's question.\n\nFirst, identify if there are any explicit instructions in the user's question regarding how the answer should be formatted, what specific requirements the answer must meet, or if the assistant should adopt a particular role. Then, assess whether each assistant's response adheres to these instructions.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistant's answer is related to the question and isn't irrelevant to the question.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate whether the responses provided by two AI assistants to the user question displayed below provide sufficient necessary information. Your evaluation should focus solely on the completeness and adequacy of the information provided in each response.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistant's response contains unnecessary redundant information.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistant's response contains tedious or repetitive content.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on determining whether the information mentioned in the assistant's response contains any information that is inconsistent with facts or fabricated information.\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the assistant's answer effectively solves the user's problem.\n\n**Evaluation Criteria:**\n1. **Relevance**: Does the response directly address the user's question?\n2. **Accuracy**: Is the information provided correct and reliable?\n3. **Completeness**: Does the response cover all necessary aspects of the question?\n4. **Clarity**: Is the response easy to understand and free from ambiguity?\n5. **Practicality**: Does the response provide actionable and useful advice or information?\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\nThis revised prompt provides a more detailed and clear set of instructions for the language model to evaluate the responses based on specific criteria, ensuring a more objective and comprehensive assessment.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should follow these steps:\n\n1. **Instruction Analysis**:\n   - First, carefully examine the user's question to identify if it contains any specific instructions regarding:\n     - Answer format requirements (e.g., bullet points, numbered lists, specific structure)\n     - Content requirements (e.g., must include certain elements, avoid specific topics)\n     - Role-playing instructions (e.g., act as a specific character or professional)\n     - Any other explicit directives\n\n2. **Compliance Evaluation**:\n   - For each assistant's response, determine whether it strictly adheres to all identified instructions from the user's question.\n   - Evaluate the response's completeness in addressing the question's requirements.\n   - Assess whether the response maintains the required format, content, and role-playing aspects (if specified).\n\n3. **Quality Assessment**:\n   - If no specific instructions are present in the question, evaluate the responses based on:\n     - Accuracy of information\n     - Clarity and coherence\n     - Depth of explanation\n     - Relevance to the question\n\n4. **Final Verdict**:\n   - Compare the two responses based on the above criteria.\n   - **Output your final verdict by strictly following this format without providing any explanation**:\n     - \"[[A]]\" if assistant A is better\n     - \"[[B]]\" if assistant B is better\n     - \"[[C]]\" for a tie\n\nRemember to:\n- Avoid any position biases\n- Ensure the order of responses does not influence your decision\n- Do not let response length affect your evaluation\n- Be as objective as possible in your assessment",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on determining whether each assistant's answer is directly related to the user's question and does not contain irrelevant information.\n\n**Evaluation Criteria:**\n1. **Relevance**: The response must directly address the user's question without introducing unrelated topics or information.\n2. **Clarity**: The response should be clear and concise, avoiding unnecessary details that do not contribute to answering the question.\n3. **Accuracy**: The response should provide accurate information that is pertinent to the question asked.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on whether the response is relevant to the question and does not contain irrelevant information.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n**Note**: If either response contains information that is not relevant to the user's question, it should be considered less favorable, regardless of other factors.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on determining whether each assistant's response provides sufficient and necessary information to adequately address the user's query.\n\nKey Evaluation Criteria:\n1. Completeness: Does the response cover all essential aspects of the question?\n2. Relevance: Is the information provided directly pertinent to the user's query?\n3. Accuracy: Are the facts and details presented correct and reliable?\n4. Clarity: Is the information presented in a clear and understandable manner?\n5. Depth: Does the response provide enough detail to fully satisfy the user's information needs?\n\nAvoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A provides more sufficient and necessary information, \"[[B]]\" if assistant B provides more sufficient and necessary information, and \"[[C]]\" if both assistants provide equally sufficient and necessary information.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on assessing whether the assistant's response contains unnecessary redundant information.\n\n**Evaluation Criteria:**\n1. **Conciseness**: The response should be concise and to the point, avoiding any unnecessary repetition or redundant information.\n2. **Relevance**: The response should directly address the user's question without including irrelevant details.\n3. **Clarity**: The response should be clear and easy to understand, without any convoluted or overly verbose explanations.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on the presence or absence of unnecessary redundant information in the responses.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n**Note**: If both responses are equally concise and free of unnecessary redundant information, the verdict should be \"[[C]]\". If one response is clearly more concise and free of redundancy, the verdict should reflect that.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should focus on whether the responses contain tedious or repetitive content.\n\n**Evaluation Criteria:**\n1. **Clarity and Conciseness**: The response should be clear and concise, avoiding unnecessary repetition or overly verbose explanations.\n2. **Relevance**: The response should directly address the user's question without straying into irrelevant details or redundant information.\n3. **Engagement**: The response should maintain the user's interest by avoiding monotonous or overly repetitive phrasing.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation. Be as objective as possible.\n- Focus solely on the presence or absence of tedious or repetitive content in the responses.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your primary focus should be on ensuring that the information provided by each assistant is factually accurate and free from any inconsistencies or fabricated details.\n\n**Evaluation Criteria:**\n1. **Factual Accuracy**: The response must not contain any information that is inconsistent with established facts or fabricated.\n2. **Consistency**: The information provided should be internally consistent and not contradict itself.\n3. **Relevance**: The response should directly address the user's question without introducing irrelevant or misleading information.\n\n**Instructions:**\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.\n- Do not allow the length of the responses to influence your evaluation.\n- Be as objective as possible, focusing solely on the factual accuracy and consistency of the information provided.\n\n**Output your final verdict by strictly following this format without providing any explanation**: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
]

init_for_pair = [

]

def get_score(score_map, output):
    r = None
    for k, s in score_map.items():
        if k in output:
            r = s 
            break
    if r is None:
        print(f'模型输出格式错误: {output}', file=sys.stderr)
    return r


pandalm_prompt = '''\
Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.

### Instruction: 
{instruction}

### Response 1: 
{response1}

### Response 2:
{response2}

**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''
def baseline_pandalm(self, c):
    i = [
        {"role": "user", "content": pandalm_prompt.format(instruction=c['question'], response1=c['output'][0], response2=c['output'][1])},
    ]
    output = self.model.get_outputs([i], max_tokens=256)[0].message.content
    # print(output)
    r = get_score({'[[A]]': 1, '[[B]]': -1, '[[C]]': 0}, output)
    return r

p_for_feature = '''\
这是一个人类向 ai 助手提出的问题：
{question}

现在你需要判断：{feature}
你的回答只能是 "YES" 或 "NO"，不需要其它任何解释。
'''

def get_feature(feature):
    def f(self, c):
        i = [
            {"role": "user", "content": p_for_feature.format(question=c['question'], feature=feature)},
        ]
        output = model.get_outputs([i], max_tokens=32)[0].message.content
        r = get_score({'YES': 1, 'NO': 0}, output)
        return r
    return f



# 根据负样本生成新维度
dp = '''\
你是一个负责设计基于 LLM 的问答系统评估器的专家。现在你设计的评估器在以下用例上出现了偏差：
[User Question]
{question}
[The Start of Assistant A’s Answer]
{output_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{output_b}
[The End of Assistant B’s Answer]

你的评估器认为 Assistant A 的回答比 Assistant B 的回答更好，然而人类专家评估者给出的评估与此相反。

这是你的评估器在判题时所采用的所有评估角度：
{metrics}

该评估器产生误判可能是因为上面提到的评估角度还不够全面。请你分析人类专家认为 Assistant B 的回答更好的可能原因，并参考上面的评估角度的格式，在你输出的**最后一行**给出一个全新的评估角度（尽可能和已有的评估角度不同）。
'''

baseline_prompt_0208 = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant tha follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. **Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''

baseline_prompt_wo_tie_0208 = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant tha follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. **Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better. You cannot give a verdict of a tie.
'''

# prompt_system_v1 = '''\
# Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.

# Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

# After providing your explanation, **output your final verdict by strictly following this format**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
# '''
baseline_prompt_pair_cot = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''

prompt_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''
prompt_score_pair_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.

You should choose an assistant that better meets this requirement: **{metric}**.

You need to give a score ranging from -5 to 5 according to the following rules:
- When the score is positive, it means that Assistant A's response is more in line with the requirements than Assistant B's. In this case, the closer the score is to 5, the greater the gap.
- When the score is negative, it indicates that Assistant B's response is more in line with the requirements than Assistant A's. Here, the closer the score is to -5, the greater the gap.
- The closer the score is to 0, the smaller the gap between their responses. 

**Output your final verdict by strictly following this format without providing any explanation**: "[[x]]", where x is the score you assign. For example, "[[3]]" represents 3 points, and "[[-3]]" represents -3 points.
'''
prompt_user = '''\
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
'''
# https://arxiv.org/abs/2005.00456
prompt_score_system_baseline = '''\
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Please score the assistant's response according to the following rules:
- A score of 1 (very bad). A completely invalid response. It would be difficult to recover the conversation after this.
- A score of 2 (bad). Valid response, but otherwise poor in quality.
- A score of 3 (neutral) means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.
- A score of 4 (good) means this is a good response, but falls short of being perfect because of a key flaw.
- A score of 5 (very good) means this response is good and does not have any strong flaws.

**Output your final verdict by strictly following this format without providing any explanation**: "[[1]]" if the score is 1, "[[2]]" if the score is 2, "[[3]]" if the score is 3, "[[4]]" if the score is 4, "[[5]]" if the score is 5.
'''
prompt_score_system = '''\
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistants to the user question displayed below.
You need to determine whether the assistant's response meets the following requirement: **{metric}**.
You need to give a score from 1 to 5. The higher the score, the more the assistant meets the requirements.
As two extreme examples, a score of 5 means the assistant's answer perfectly meets the requirements, and a score of 1 means the assistant's answer is completely contrary to the requirements.

**Output your final verdict by strictly following this format without providing any explanation**: "[[1]]" if the score is 1, "[[2]]" if the score is 2, "[[3]]" if the score is 3, "[[4]]" if the score is 4, "[[5]]" if the score is 5.
'''
prompt_score_user = '''\
[User Question]
{question}
[The Start of the Assistant’s Answer]
{answer}
[The End of the Assistant’s Answer]
'''

def get_m_origin(sys_prompt, default_score_map=None, max_tokens=300):
    def baseline(self, c):
        # print(c['id'])
        if is_score:
            i = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_score_user.format(question=c['question'], answer=c['output'])},
            ]
            output = self.model.get_outputs([i], max_tokens=max_tokens)[0].message.content
            score_map = default_score_map or {f'[[{i}]]': i for i in range(1, 6)}
            r = get_score(score_map, output)
        else:
            i = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
            ]
            # output = self.model.get_outputs([i], max_tokens=1024)[0].message.content
            output = self.model.get_outputs([i], max_tokens=max_tokens)[0].message.content
            score_map = default_score_map or {'[[A]]': 1, '[[B]]': -1, '[[C]]': 0}
            r = get_score(score_map, output)
        return r
    return baseline


def get_m(metric):
    if is_score:
        return get_m_origin(prompt_score_system.format(metric=metric))
    else:
        return get_m_origin(prompt_system.format(metric=metric))


    


# 训练集
# train_set = get_trainset()

# 测试集

# 现有维度
features = {
    'features':[],
    'metrics': [],
}


def ref_features():
    def fill_none(self, c):
        return None
    for i in range(len(init_metrics)):
        m = init_metrics[i]  
        features['metrics'].append((f'0115_{i}', get_m(m)))
        # features['metrics'].append((f'0115_{i}', fill_none))
    for i in range(len(init_features)):
        m = init_features[i]
        features['metrics'].append((f'0115_features_{i}', get_feature(m)))
        # features['metrics'].append((f'0115_features_{i}', fill_none))
    if not is_score:
        for i in range(len(init_template)):
            m = init_template[i]
            features['metrics'].append((f'0115_template_{i}', get_m_origin(m)))
        for i in range(len(pair_score_metircs)):
            m = pair_score_metircs[i]
            features['metrics'].append((f'0115_pair_score_{i}', get_m_origin(prompt_score_pair_system.format(metric=m), default_score_map={f'[[{i}]]': i for i in range(-5, 6)})))
        features['metrics'].append((f'baseline_pandalm', baseline_pandalm))
        features['metrics'].append((f'baseline_0208', get_m_origin(baseline_prompt_0208)))
        features['metrics'].append((f'baseline', get_m_origin(baseline_prompt_pair_cot, max_tokens=1000)))
        features['metrics'].append((f'baseline_wo_tie_0208', get_m_origin(baseline_prompt_wo_tie_0208, default_score_map={'[[A]]': 1, '[[B]]': -1})))
        if ultra:
            for i in range(len(init_metrics_2)):
                m = init_metrics_2[i]
                features['metrics'].append((f'0115_ultra_metrics_2_{i}', get_m(m)))
            
    if is_score:
        features['metrics'].append((f'baseline_score', get_m_origin(prompt_score_system_baseline)))

def get_x_y(content, features):
    X = [[x['metrics'].get(m[0]) for m in features['metrics']] for x in content]
    Y = [x['manual_score'] for x in content]
    return X, Y

def d_f(question, better, worse, origin_score):
    t_dp = dp.format(question=question, output_b=better, output_a=worse, metrics='\n'.join(['- ' + x for x in init_metrics]))
    resp = model.get_outputs([[{"role": "user", "content": t_dp}]], temperature=1, text=True)[0]
    new_m = resp.splitlines()[-1]
    print(new_m)
    i = [
        {"role": "system", "content": prompt_system.format(metric=new_m)},
        {"role": "user", "content": prompt_user.format(question=question, answer_a=better, answer_b=worse)},
    ]
    output = model.get_outputs([i], max_tokens=1024)[0].message.content
    score_map = {
        '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
    }
    r = None
    for k, s in score_map.items():
        if k in output:
            r = s 
            break
    if r is None:
        print(f'模型输出格式错误: {output}', file=sys.stderr)
    elif r != origin_score:
        print('有效')
        return new_m
    return None

# 迭代

# for dataset in datasets:
#     content = dataset.content
#     for x in content:
#         del x['metrics']['0115_features_0']
#     save_result(dataset)

    

epoch = 1
for i in range(epoch):
    # 判断维度相关性，筛选维度
    ref_features()
    get_eval_res(features)

    for dataset in datasets:
        # content = dataset.content
        # dataset.content = [x for x in dataset.content if 'metrics' in x and x['metrics'] is not None and x['metrics'].get('baseline')]
        # content = dataset.content
        # print(len(content))

        # X, Y = get_x_y(content, features)
        # pm = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
        
        # pm.fit(X, Y)
        # pred = pm.predict(X)
        
        # def get_cot_deta1(Y, pred):
        #     deta1 = 0
        #     cot = 0
        #     n = len(Y)
        #     for i in range(n):
        #         if abs(Y[i] - pred[i]) <= 1:
        #             deta1 += 1
        #             if Y[i] == pred[i]:
        #                 cot += 1
        #     return cot, deta1, n
        # baseline_pred = [x['metrics']['baseline'] for x in content]

        # cot, deta1, n = get_cot_deta1(Y, pred)
        print(dataset.name)
        # print(f'Pred Cot: {cot/n:.3f}, Deta<=1: {deta1/n:.3f}')
        # print(f'Pred spearmanr: {spearmanr(pred, Y)[0]:.3f}, kendalltau: {kendalltau(pred, Y)[0]:.3f}')
        # cot, deta1, n = get_cot_deta1(Y, baseline_pred)
        # print(f'Baseline Cot: {cot/n:.3f}, Deta<=1: {deta1/n:.3f}')
        # print(f'Baseline spearmanr: {spearmanr(baseline_pred, Y)[0]:.3f}, kendalltau: {kendalltau(baseline_pred, Y)[0]:.3f}')
        summary.metrics_regression.Summary().print_summary(dataset)
        print()
        

        # 获取负样本
        # tmp_m = []
        # bad_cases = []
        # for i in range(len(Y)):
        #     if Y[i] - pred[i] == 2:
        #         better = content[i]['output'][0]
        #         worse = content[i]['output'][1]
        #         bad_cases.append((content[i], better, worse))
        #     elif Y[i] - pred[i] == -2:
        #         better = content[i]['output'][1]
        #         worse = content[i]['output'][0]
        #         bad_cases.append((content[i], better, worse))
        #     else:
        #         continue

        #     new_m = d_f(content[i]['question'], better, worse, pred[i])
        #     if new_m is not None:
        #         tmp_m.append(new_m)
            

        # init_metrics += tmp_m
        # ref_features()

    
# for x in tmp_m:
#     print(x)

