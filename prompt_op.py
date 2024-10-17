from model.openai_api import Model

m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')

null = None
case  =  {
  "id": 22,
  "resp_id": 2,
  "question": "# Fact: \nif earth ’s entire history was viewed as a 24 hour period , humans would only represent 1 minute and 17 seconds .\n\n\n# Conversation: \nhello did you know that 80 percent of the earths natural forests have already been destroyed ? \n i heard it but i 'm skeptical . that sounds far too high and what 's the definition of \" natural . \n not sure in this case . when the earth was formed the days were only 5.5 hours long ! \n that was 5.5 hours of sunlit hell in magma and lava . bummer , they are saying we will be out of helium by the end of the century . \n\n",
  "output": "yeah , i think that is so cool . did you know that if earths history was viewed as a 24 hour period , humans would only represent 1 minute and 17 seconds .\n",
  "manual_score": 4,
  "metrics": {
   "accuracy": 2,
   "relevant": 4,
   "faithful": 2,
   "detailed": 3,
   "redundant": 2,
   "header_scorer_v2_accuracy": 0,
   "header_scorer_v2_relevant": 2,
   "header_scorer_v2_faithful": 0,
   "header_scorer_v2_detailed": 0,
   "header_scorer_v2_redundant": 2
  },
  "score": null,
  "err": null
}

p = '''\
以下是一位 AI 评分员对一个问答的评分，以及它评分的主要依据

问题：{question}

回答：{output}

评分：{ai_score}

评分依据: if the assistant's response is helpful in solving the problem.

对于这个问答，人类专家给出的评分是：{manual_score}

请分析 AI 评分员与人类专家评分存在差异的可能原因
'''.format(**case, ai_score=case['metrics']['accuracy'])

a = m.get_outputs([[
            {"role": "system", "content": "你是一个专家"},
            {"role": "user", "content": p},
        ]])[0].message.content

print(a)
print()

p = f'''\
以下是一个问答评分的评判标准: Please determine if the assistant's response is helpful in solving the problem.

但它存在以下问题：{a}

为了解决这个问题，请改写这个标准并直接输出（使用英文），不需要任何解释：
'''

a = m.get_outputs([[
            {"role": "system", "content": "你是一个专家"},
            {"role": "user", "content": p},
        ]])[0].message.content

print(a)
print()


header_scorer = f'''\
Please act as an impartial judge and evaluate as requested the response provided by an AI assistant to the user question displayed below. Do not allow the length of the response to influence your evaluation. Be as objective as possible.
{a} After providing your explanation, output your final verdict by strictly following this format:
 "[[1]]" if the response is very bad (A completely invalid response. It would be difficult to recover the conversation after this.),
 "[[2]]" if the response is bad (Valid response, but otherwise poor in quality),
 "[[3]]" if the response is neutral (means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.),
 "[[4]]" if the response is good (means this is a good response, but falls short of being perfect because of a key flaw.),
 "[[5]]" if the response is very good (means this response is good and does not have any strong flaws).  
'''

p = '''\
[User Question]
{question}
[The Start of Assistant’s Answer]
{output}
[The End of Assistant’s Answer]
'''.format(**case)


a = m.get_outputs([[
            {"role": "system", "content": header_scorer},
            {"role": "user", "content": p},
        ]])[0].message.content

print(a)
print()