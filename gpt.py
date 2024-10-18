from model.openai_api import Model

m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')


s = '''\
 {
  "id": 59,
  "resp_id": 0,
  "manual_score": 4,
  "metrics": {
   "accuracy": 4,
   "relevant": 5,
   "faithful": 5,
   "detailed": 2,
   "redundant": 5,
   "header_scorer_v2_accuracy": 2,
   "header_scorer_v2_relevant": 2,
   "header_scorer_v2_faithful": 0,
   "header_scorer_v2_detailed": 0,
   "header_scorer_v2_redundant": 2
  },
  "score": null,
  "err": null,
  "metrics_raw": {
   "accuracy": "The assistant's response directly addresses the user's question about the president of Indonesia being a pop star, which is relevant to the fact provided by the user. The response is concise and accurate, confirming the information given. However, it lacks depth or additional context that could enhance the user's understanding or engagement.\n\nFinal Verdict: [[4]]",
   "relevant": "The assistant's response is relevant to the user's question about the president of Indonesia releasing pop albums while in office. The assistant correctly acknowledges the fact and asks if the user is aware of this information. The response is concise and directly addresses the specific fact mentioned in the user's question.\n\nFinal verdict: [[5]]",
   "faithful": "The assistant's response directly addresses the user's question about the president of Indonesia being a pop star, which aligns with the provided fact that the president has released 3 pop albums while in office. The response is concise and accurate, correctly conveying the information without any factual errors.\n\nFinal verdict: [[5]]",
   "detailed": "The assistant's response to the user's question is very brief and does not provide any additional information or context about the fact that the president of Indonesia has released three pop albums while in office. The response simply restates the fact in a question format, which does not contribute to the conversation or enhance the user's understanding.\n\nThe response lacks detail and fails to explore any aspects of the president's musical career, such as the titles of the albums, the reception of the albums, or the impact of the president's musical endeavors on his political career. This omission makes the response inadequate in relation to the user's question.\n\nFinal verdict: [[2]]",
   "redundant": "The assistant's response, \"yeah . did you know the president of indonesia is a pop star ?\", directly addresses the user's question about the president of Indonesia releasing pop albums while in office. The response is concise and relevant, providing the necessary information without any extraneous details.\n\nThere is no redundant information unrelated to the question in the assistant's response. The response is focused and directly answers the user's query about the president of Indonesia's musical endeavors.\n\nFinal verdict: [[5]]"
  }
 }
'''
import json
print(json.loads(s))
exit()

p = '''\
以下是一位 AI 评分员对一个问答的评分，以及它评分的主要依据

问题：{question}

回答：{output}

评分：{ai_score}

评分依据: if the assistant's response is helpful in solving the problem.

对于这个问答，人类专家给出的评分是：{manual_score}

请分析 AI 评分员与人类专家评分存在差异的可能原因
'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个专家"},
            {"role": "user", "content": p},
        ]])[0].message.content)