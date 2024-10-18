from model.openai_api import Model
from model import deepseek, gpt_3
import json
from tqdm import tqdm

m = deepseek.Model()
null = None

def gen():

    with open('output/topical-chat/deepseek/topical-chat.json', encoding='utf-8') as f:
        cases = json.load(f)

    all_r = []
    avc, avt, total = 0, 0, 0
    for case in tqdm(cases):

        deta = abs(case['manual_score'] - case['metrics']['accuracy'])
        if deta < 2:
            continue
        # case  =  {
        #   "id": 22,
        #   "resp_id": 5,
        #   "question": "# Fact: \nif earth ’s entire history was viewed as a 24 hour period , humans would only represent 1 minute and 17 seconds .\n\n\n# Conversation: \nhello did you know that 80 percent of the earths natural forests have already been destroyed ? \n i heard it but i 'm skeptical . that sounds far too high and what 's the definition of \" natural . \n not sure in this case . when the earth was formed the days were only 5.5 hours long ! \n that was 5.5 hours of sunlit hell in magma and lava . bummer , they are saying we will be out of helium by the end of the century . \n\n",
        #   "output": "wow , i ca n't believe we 're destroying the world . if we viewed the earth 's history as a 24 hour period , humans would only have existed for 1 minute and 17 seconds . and in that time , we 've managed to completely destroy the planet .",
        #   "manual_score": 5,
        #   "metrics": {
        #    "accuracy": 2,
        #    "relevant": 4,
        #    "faithful": 2,
        #    "detailed": 3,
        #    "redundant": 4,
        #    "header_scorer_v2_accuracy": 0,
        #    "header_scorer_v2_relevant": 1,
        #    "header_scorer_v2_faithful": 0,
        #    "header_scorer_v2_detailed": 1,
        #    "header_scorer_v2_redundant": 0
        #   },
        #   "score": null,
        #   "err": null,
        #   "metrics_raw": {
        #    "accuracy": "The assistant's response is emotionally charged and does not directly address the user's skepticism about the 80 percent figure or the definition of \"natural forests.\" Instead, it reiterates the fact about the Earth's history and humans' brief existence in that context, which, while relevant, does not contribute to resolving the user's skepticism or providing additional information. The response also exaggerates the situation by stating that humans have \"completely destroyed the planet,\" which is an overstatement and not supported by the facts provided.\n\nFinal verdict: [[2]]",
        #    "relevant": "The assistant's response is relevant to the user's mention of the Earth's history being viewed as a 24-hour period, and it incorporates the fact provided about humans' brief existence in that timeline. However, the response diverges from the specific topic of natural forests being destroyed and instead generalizes about humanity's impact on the planet. This makes the response somewhat tangential to the immediate conversation about forests.\n\nThe response is emotionally charged and aligns with the user's expressed concern about environmental destruction, but it lacks specific details or clarifications that would directly address the user's skepticism about the 80% figure and the definition of \"natural.\"\n\nFinal Verdict: [[4]]",
        #    "faithful": "The assistant's response does not directly address the user's skepticism about the 80 percent destruction of natural forests. Instead, it shifts the focus to a general lament about humanity's impact on the planet, using the fact about the Earth's history as a 24-hour period. This does not provide the user with the clarification or information they were seeking regarding the specific statistic about forest destruction.\n\nAdditionally, the response contains a factual error: it incorrectly states that humans have \"completely destroyed the planet,\" which is an exaggeration and not supported by the fact provided about human existence in the context of Earth's history.\n\nTherefore, the response fails to correctly answer the user's question and contains a factual error.\n\nFinal verdict: [[2]]",
        #    "detailed": "The assistant's response addresses the user's mention of the Earth's history being viewed as a 24-hour period and the brief duration of human existence within that timeframe. However, it does not directly address the user's skepticism about the 80% destruction of natural forests or the definition of \"natural.\" The response also does not engage with the other facts mentioned in the conversation, such as the length of days when the Earth was formed or the impending shortage of helium.\n\nThe response is somewhat relevant but lacks depth and fails to fully engage with the user's concerns and the broader context of the conversation. It simplifies the issue and does not provide any additional information or perspective that could help clarify or expand on the topics discussed.\n\nFinal verdict: [[3]]",
        #    "redundant": "The assistant's response addresses the user's mention of the Earth's history being viewed as a 24-hour period and the brief existence of humans within that timeframe. However, the response includes an emotional and somewhat exaggerated statement about humans \"completely destroying the planet,\" which is not directly related to the factual information provided by the user. This part of the response introduces redundancy and a subjective viewpoint that does not align with the objective nature of the user's factual statement.\n\nFinal Verdict: [[4]]"
        #   }
        #  }

        p = '''\
以下是一位 AI 评分员对一个问答的评分和解释，以及它评分的主要依据

问题：{question}

回答：{output}

评分：{ai_score}

评分解释：{ai_explain}

评分依据: if the assistant's response is helpful in solving the problem.

对于这个问答，人类专家给出的评分是：{manual_score}

请根据 AI 评分员的评分依据和评分解释，分析 AI 评分员与人类专家评分存在差异的可能原因
'''.format(**case, ai_score=case['metrics']['accuracy'], ai_explain=case['metrics_raw']['accuracy'])

        a = m.get_outputs([[
                    {"role": "system", "content": "你是一个专家"},
                    {"role": "user", "content": p},
                ]])[0].message.content

        # print(a)
        # print()

        p = f'''\
以下是一个 AI 评分员用于对问答进行评分的评判标准: Please determine if the assistant's response is helpful in solving the problem.

但它存在以下问题：{a}

为了解决这个问题，请改写这个标准并直接输出（使用英文），不需要任何解释：
'''

        a = m.get_outputs([[
                    {"role": "system", "content": "你是一个专家"},
                    {"role": "user", "content": p},
                ]])[0].message.content

        all_r.append(a)

        # print(a)
        # print()


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
        
        beg = a.index('[[')
        if beg >= 0:
            end = a.index(']]')
            try:
                ret = a[beg + 2:end]
                ret = int(ret)
            except:
                pass
        else:
            ret = 999

        new_deta = abs(ret - case['manual_score'])
        total += 1
        if new_deta < deta:
            avc += 1
        elif new_deta == deta:
            avt += 1

        print(case['manual_score'], case['metrics']['accuracy'], ret, deta-new_deta)


    print(f'{avc/total:.3f}')
    print(f'{avt/total:.3f}')


    # with open('test_prompt_op.json', 'w') as f:
    #     json.dump(all_r, f)

def un():
    with open('test_prompt_op.json') as f: 
        ll = json.load(f)

    p = "以下是若干个评分规则列表，请根据这些规则，整合出十个规则（使用英文），这十个规则要求尽可能覆盖全部规则的关键点，同时互相之间尽可能独立：\n\n"
    
    i = 0
    for x in ll:
        i += 1
        p += f"规则{i}: {repr(x)}\n"

    a = m.get_outputs([[
            {"role": "system", "content": "你是一个专家"},
            {"role": "user", "content": p},
        ]])[0].message.content

    print(a)

# un()
gen()