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
    def get_score(self, c):
        i = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user.format(question='10+1=?', answer_a='2', answer_b='11')},
            {"role": "assistant", "content": "Assistant A provided an incorrect answer, Assistant B provided an accurate answer.\nfinal verdict: [[B]]"},
            {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
        ]
        output = self.model.get_outputs([i], max_tokens=600)[0].message.content
        score_map = {
            '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
        }
        for k, s in score_map.items():
            if k in output:
                return s
        return 'judge fail'
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
        
        