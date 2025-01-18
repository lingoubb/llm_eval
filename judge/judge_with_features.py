prompt_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
'''

prompt_system_wo_tie = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better. You cannot give a verdict of a tie.
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

from . import with_llm

"""
主观题评测
"""
class Judge(with_llm.Judge):
    def __init__(self, model, features={}, *args):
        self.features = features
        super().__init__(model, *args)
    def get_score(self, c):

        for k, v in self.features.items():
            c.setdefault(k, {})
            for x in v:
                name, func = x
                if name not in c[k] or c[k][name] is None:
                    c[k][name] = func(self, c)

        # c.setdefault('metrics', {})
        # c.setdefault('features', {})

#         if '摘要_a' not in c['features']:

#             sys_prompt = '你将收到一个指令和一个 ai 助手对此的回答，请你尽可能地精简该助手的回答，保留足够完成指令的最少信息'
#             usr_prompt = '指令：{question}\nai 助手的回答: {output}'
#             i = [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": usr_prompt.format(question=c['question'], output=c['output'][0])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        
#             c['features']['摘要_a'] = output

#         if '摘要_b' not in c['features']:

#             sys_prompt = '你将收到一个指令和一个 ai 助手对此的回答，请你尽可能地精简该助手的回答，保留足够完成指令的最少信息'
#             usr_prompt = '指令：{question}\nai 助手的回答: {output}'
#             i = [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": usr_prompt.format(question=c['question'], output=c['output'][1])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        
#             c['features']['摘要_b'] = output

#         if '判题依据' not in c['features']:
 
#             sys_prompt = '你是一个负责对基于大语言模型的 ai 助手进行评估的专家。现在有一个人类提出的指令，将会有 ai 助手对其进行回答。请你给出一个评估方案，指出评估 ai 助手对该问题的回答的详细关注点。'
#             usr_prompt = '指令：{question}'
#             i = [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": usr_prompt.format(question=c['question'])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        
#             c['features']['判题依据'] = output

#         if '有害信息' not in c['features']:

#             sys_prompt = '你将收到一个指令和一个 ai 助手对此的回答，请指出该 ai 助手的回答中存在的有害信息，例如偏见（性别、种族、地域等）、对人类社会可能产生危害的信息等。如果没有，则直接回答 "[NO]"。'
#             usr_prompt = '指令：{question}\nai 助手的回答: {output}'
#             i = [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": usr_prompt.format(question=c['question'], output=c['output'][0])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        
#             c['features']['有害信息'] = output

#         if '冗余' not in c['features']:

#             sys_prompt = '你将收到一个指令和一个 ai 助手对此的回答，请指出该 ai 助手的回答中存在的不必要的冗余信息。如果没有，则直接回答 "[NO]"。'
#             usr_prompt = '指令：{question}\nai 助手的回答: {output}'
#             i = [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": usr_prompt.format(question=c['question'], output=c['output'][0])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        
#             c['features']['冗余'] = output

#         if '正反辩论' not in c['features']:

#             r = []

#             for better, worse in [('A', 'B'), ('B', 'A')]:
#                 p = '''\
# 以下是一个问题和两个 ai 助手给出的回答，人类评估者认为助手 {better} 比助手 {worse} 的回答更好，请指出你认为人类评估者可能是基于什么原因给出这样的判断。你只能给出一个原因，请挑选最关键的原因。
# [问题开始]
# {question}
# [问题结束]
# [ai 助手 A 的回答开始]
# {output_a}
# [ai 助手 A 的回答结束]
# [ai 助手 B 的回答开始]
# {output_b}
# [ai 助手 B 的回答结束]
# '''.format(better=better, worse=worse, question=c['question'], output_a=c['output'][0], output_b=c['output'][1])
#                 i = [
#                     {"role": "system", "content": '你是一个负责对基于大语言模型的 ai 助手进行评估的专家。'},
#                     {"role": "user", "content": p},
#                 ]
#                 output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#                 r.append(output)

#             c['features']['正反辩论'] = r

        
#             p = '''\
# 以下是一个问题和两个 ai 助手给出的回答，人类评估者认为助手 A 和助手 B 的回答都没有很好地回答该问题，请指出你认为人类评估者可能是基于什么原因给出这样的判断。你只能给出一个原因，请挑选最关键的原因。
# [问题开始]
# {question}
# [问题结束]
# [ai 助手 A 的回答开始]
# {output_a}
# [ai 助手 A 的回答结束]
# [ai 助手 B 的回答开始]
# {output_b}
# [ai 助手 B 的回答结束]
# '''.format( question=c['question'], output_a=c['output'][0], output_b=c['output'][1])
#             i = [
#                 {"role": "system", "content": '你是一个负责对基于大语言模型的 ai 助手进行评估的专家。'},
#                 {"role": "user", "content": p},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             c['features']['正反辩论'].append(output)

#             p = '''\
# 以下是一个问题和两个 ai 助手给出的回答，人类评估者认为助手 A 和助手 B 的回答都很好地回答了该问题且不相上下，请指出你认为人类评估者可能是基于什么原因给出这样的判断。你只能给出一个原因，请挑选最关键的原因。
# [问题开始]
# {question}
# [问题结束]
# [ai 助手 A 的回答开始]
# {output_a}
# [ai 助手 A 的回答结束]
# [ai 助手 B 的回答开始]
# {output_b}
# [ai 助手 B 的回答结束]
# '''.format(question=c['question'], output_a=c['output'][0], output_b=c['output'][1])
#             i = [
#                 {"role": "system", "content": '你是一个负责对基于大语言模型的 ai 助手进行评估的专家。'},
#                 {"role": "user", "content": p},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             c['features']['正反辩论'].append(output)

#         if '平局' not in c['features']:
#             i = [
#                 {"role": "system", "content": '你是一个负责对基于大语言模型的 ai 助手进行评估的专家。你将收到一个人类指令和两位 ai 助手对此的回答，有人认为这两位助手的回答都没有很好地完成人类指令，你赞同吗？如果赞同，请输出你的解释；如果不赞同，请输出"[NO]"。'},
#                 {"role": "user", "content": '''[问题开始]
# {question}
# [问题结束]
# [ai 助手 A 的回答开始]
# {output_a}
# [ai 助手 A 的回答结束]
# [ai 助手 B 的回答开始]
# {output_b}
# [ai 助手 B 的回答结束]'''.format( question=c['question'], output_a=c['output'][0], output_b=c['output'][1])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             c['features']['平局'] = output        

#         if '类型' not in c['features']:
#             l = [
#                 "知识问答类",
#                 "翻译类",
#                 "逻辑推理类",
#                 "数学计算类",
#                 "摘要生成类",
#                 "基于给定上下文的问答",
#                 "日常开放式对话",
#                 "写作",
#                 "生活建议与规划类",
#                 "文本处理",
#                 "数据到文本",
#                 "社交互动与反馈类",
#                 "娱乐休闲资源推荐类",
#                 "物品分析与比较类",
#                 "思维拓展与想象类",
#                 "其它",
#             ]
#             i = [
#                 {"role": "system", "content": f'你将收到一个指令，你需要判断其属于以下指令类型的哪一种：{", ".join(l)}\n你只需要直接输出指令类型，不需要任何其它解释。'},
#                 {"role": "user", "content": f'指令：{c['question']}'},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
            
#             r = '其它'
#             for x in l:
#                 if x in output:
#                     r = x 
#                     break
#             c['features']['类型'] = r

#         if '位置互换' not in c['features'] or c['features']['位置互换'] is None:
#             i = [
#                 {"role": "system", "content": prompt_system},
#                 {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][1], answer_b=c['output'][0])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             score_map = {
#                 '[[A]]': -1, '[[B]]': 1, '[[C]]': 0
#             }
#             r = None
#             for k, s in score_map.items():
#                 if k in output:
#                     r = s 
#                     break
#             if r is None:
#                 raise Exception(f'模型输出格式错误: {output}')
#             c['features']['位置互换'] = r

        if 'baseline' not in c['metrics'] or c['metrics']['baseline'] is None:
            i = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
            ]
            output = self.model.get_outputs([i], max_tokens=1600)[0].message.content
            score_map = {
                '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
            }
            r = None
            for k, s in score_map.items():
                if k in output:
                    r = s 
                    break
            if r is None:
                raise Exception(f'模型输出格式错误: {output}')
            c['metrics']['baseline'] = r

        if 'baseline_wo_tie' not in c['metrics'] or c['metrics']['baseline_wo_tie'] is None:
            i = [
                {"role": "system", "content": prompt_system_wo_tie},
                {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
            ]
            output = self.model.get_outputs([i], max_tokens=1600)[0].message.content
            score_map = {
                '[[A]]': 1, '[[B]]': -1
            }
            r = None
            for k, s in score_map.items():
                if k in output:
                    r = s 
                    break
            if r is None:
                raise Exception(f'模型输出格式错误: {output}')
            c['metrics']['baseline_wo_tie'] = r

#         if 'with_判题依据' not in c['metrics'] or c['metrics']['with_判题依据'] is None:
#             new_prompt_system = prompt_system + '\nIn addition, you will also receive a judgment plan provided by an expert evaluator as a reference.'
#             new_prompt_user = prompt_user + f'\n[The start of the problem solving plan provided by expert evaluators]\n{c['features']['判题依据']}\n[The end of the problem solving plan provided by expert evaluators]'
#             i = [
#                 {"role": "system", "content": prompt_system},
#                 # {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
#                 # {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
#                 {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             score_map = {
#                 '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#             }
#             r = None
#             for k, s in score_map.items():
#                 if k in output:
#                     r = s 
#                     break
#             if r is None:
#                 raise Exception(f'模型输出格式错误: {output}')
#             c['metrics']['with_判题依据'] = r

#         if 'with_有害信息' not in c['metrics'] or c['metrics']['with_有害信息'] is None:
#             if '[NO]' not in c['features']['有害信息']:
#                 new_prompt_system = prompt_system + '\nIn addition, you will also receive an evaluation from an expert evaluator regarding harmful information as a reference.'
#                 new_prompt_user = prompt_user + f'\n[The start of the evaluation provided by expert evaluators]\n{c['features']['有害信息']}\n[The end of the evaluation provided by expert evaluators]'
#                 i = [
#                     {"role": "system", "content": prompt_system},
#                     # {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
#                     # {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
#                     {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
#                 ]
#                 output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#                 score_map = {
#                     '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#                 }
#                 r = None
#                 for k, s in score_map.items():
#                     if k in output:
#                         r = s 
#                         break
#                 if r is None:
#                     raise Exception(f'模型输出格式错误: {output}')
#             else:
#                 r = c['metrics']['baseline']
#             c['metrics']['with_有害信息'] = r

#         if 'with_冗余' not in c['metrics'] or c['metrics']['with_冗余'] is None:
#             if '[NO]' not in c['features']['冗余']:
#                 new_prompt_system = prompt_system + '\nIn addition, you will also receive a judgment from an expert evaluator regarding whether the answer contains redundant information as a reference.'
#                 new_prompt_user = prompt_user + f'\n[The start of the evaluation provided by expert evaluators]\n{c['features']['冗余']}\n[The end of the evaluation provided by expert evaluators]'
#                 i = [
#                     {"role": "system", "content": prompt_system},
#                     # {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
#                     # {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
#                     {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
#                 ]
#                 output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#                 score_map = {
#                     '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#                 }
#                 r = None
#                 for k, s in score_map.items():
#                     if k in output:
#                         r = s 
#                         break
#                 if r is None:
#                     raise Exception(f'模型输出格式错误: {output}')
#             else:
#                 r = c['metrics']['baseline']
#             c['metrics']['with_冗余'] = r

#         if 'with_平局' not in c['metrics'] or c['metrics']['with_平局'] is None:
#             if '[NO]' not in c['features']['平局']:
#                 r = 0
#             else:
#                 r = c['metrics']['baseline']
#             c['metrics']['with_平局'] = r

#         if 'with_位置互换' not in c['metrics'] or c['metrics']['with_位置互换'] is None:
#             if c['features']['位置互换'] == 0:
#                 r = c['metrics']['baseline']
#             elif c['metrics']['baseline'] == 0:
#                 r = c['features']['位置互换']
#             elif c['features']['位置互换'] == c['metrics']['baseline']:
#                 r = c['metrics']['baseline']
#             else:
#                 r = 0
#             c['metrics']['with_位置互换'] = r

#         if 'with_摘要' not in c['metrics'] or c['metrics']['with_摘要'] is None:
#             i = [
#                 {"role": "system", "content": prompt_system},
#                 # {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
#                 # {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
#                 {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['features']['摘要_a'], answer_b=c['features']['摘要_b'])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             score_map = {
#                 '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#             }
#             r = None
#             for k, s in score_map.items():
#                 if k in output:
#                     r = s 
#                     break
#             if r is None:
#                 raise Exception(f'模型输出格式错误: {output}')
#             c['metrics']['with_摘要'] = r
            
#         if 'with_摘要2' not in c['metrics'] or c['metrics']['with_摘要2'] is None:
#             prompt_user_t = prompt_user + f"[The Start of the Summary of Assistant A’s Answer]\n{c['features']['摘要_a']}\n[The End of the Summary of Assistant A’s Answer]\n[The Start of the Summary of Assistant B’s Answer]\n{c['features']['摘要_b']}\n[The End of the Summary of Assistant B’s Answer]\n"
#             i = [
#                 {"role": "system", "content": prompt_system},
#                 # {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
#                 # {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
#                 {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
#             ]
#             output = self.model.get_outputs([i], max_tokens=600)[0].message.content
#             score_map = {
#                 '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#             }
#             r = None
#             for k, s in score_map.items():
#                 if k in output:
#                     r = s 
#                     break
#             if r is None:
#                 raise Exception(f'模型输出格式错误: {output}')
#             c['metrics']['with_摘要2'] = r

#         m2d = {
#             "accuracy": "Please determine which assistant's answer is more helpful in solving the problem.",
#             "relevant": "Please determine which assistant's answer is more relevant to the question.",
#             "faithful": "Please determine which assistant's answer correctly answered the question and there were no factual errors.",
#             "detailed": "Please determine which assistant's answer is more detailed in relation to the question",
#             "redundant": "Please determine which assistant has less redundant information unrelated to the question in their answer",
#             "gen_1_0": "The perspective you need to pay attention to is Relevance and Directness: Assess whether the assistant's response directly addresses the user's query or maintains contextual relevance to the ongoing conversation.",
#             "gen_1_1": "The perspective you need to pay attention to is Information Depth and Utility: Evaluate the response for its depth of information and practical utility, ensuring it provides meaningful insights or solves the user's problem.",
#             "gen_1_2": "The perspective you need to pay attention to is Dialogue Coherence and Flow: Determine if the response maintains a logical and natural flow within the conversation, contributing to its coherence and engagement.",
#             "gen_1_3": "The perspective you need to pay attention to is User Engagement and Interest: Consider whether the response not only addresses the immediate query but also broadens the user's interests or enhances their conversational experience.",
#             "gen_1_4": "The perspective you need to pay attention to is Accuracy and Clarity: Assess the response for factual accuracy and clarity, ensuring it avoids misleading information and communicates effectively.",
#             "gen_1_5": "The perspective you need to pay attention to is Flexibility and Adaptability: Allow for some flexibility in scoring to account for the broader context of the conversation, including tangential but relevant information that enriches the dialogue.",
#             "gen_1_6": "The perspective you need to pay attention to is Creativity and Novelty: Evaluate the response for its creativity and novelty, considering whether it introduces new perspectives or engaging content that enhances the conversation.",
#             "gen_1_7": "The perspective you need to pay attention to is Emotional and Social Engagement: Consider the emotional tone and social dynamics of the response, ensuring it maintains a positive and engaging interaction.",
#             "gen_1_8": "The perspective you need to pay attention to is Problem-Solving Effectiveness: Rate the response based on its direct contribution to solving the user's problem, while also considering its relevance and informational value.",
#             "gen_1_9": "The perspective you need to pay attention to is Contextual Understanding and Alignment: Assess the response's alignment with the user's broader intentions and the specific context of the conversation, ensuring it captures and enhances the dialogue's purpose."
#         }
#         p = '''\
# Please act as an impartial judge and evaluate the quality of the responses provided by two\
# AI assistants to the user question displayed below. You should choose the assistant that\
# follows the user’s instructions and answers the user’s question better. 
# {d}
# Avoid any position biases and ensure that the\
# order in which the responses were presented does not influence your decision. Do not allow\
# the length of the responses to influence your evaluation. Do not favor certain names of\
# the assistants. Be as objective as possible. After providing a short explanation, output your\
# final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"\
# if assistant B is better, and "[[C]]" for a tie.
# '''
#         for k, v in m2d.items():
#             key = f'metric_{k}'
#             if key not in c['metrics']:
#                 i = [
#                     {"role": "system", "content": p.format(d=v)},
#                     {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
#                 ]
#                 output = self.model.get_outputs([i], max_tokens=1024)[0].message.content
#                 score_map = {
#                     '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
#                 }
#                 r = None
#                 for k, s in score_map.items():
#                     if k in output:
#                         r = s 
#                         break
#                 if r is None:
#                     raise Exception(f'模型输出格式错误: {output}')
#                 c['metrics'][key] = r
        
                



        return None
    
        # output = self.model.get_outputs([i], logprobs=True, top_logprobs=20, max_tokens=1)[0]
        # score_map = {
        #     'A': 1, 'B': -1, 'C': 0
        # }
        # ans = None
        # for prob in output.logprobs.content[0].top_logprobs:
        #     a = prob.token.strip() 
        #     if a in score_map:
        #         ans = score_map[a]
        # if ans is None:
        #     raise Exception('judge fail')
        # return ans
        
         