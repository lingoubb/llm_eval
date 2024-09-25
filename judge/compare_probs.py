from . import with_llm

"""
probs 主观题比较
"""
class Judge(with_llm.Judge):
    def __init__(self, model, judge_name, judge_head, judge_prompt, choice=['A', 'B', 'C']):
        self.judge_name = judge_name
        self.judge_head = judge_head
        self.judge_prompt = judge_prompt
        self.choice = choice
        super().__init__(model)


    def get_score(self, c):
        c.setdefault('metrics', {})
        if self.judge_name in c['metrics']:
            return 0

        prompt = [
            {"role": "system", "content": self.judge_head},
            {"role": "user", "content": self.judge_prompt.format(**c)},
        ]
        
        r = self.model.get_outputs([prompt], logprobs=True, top_logprobs=20)[0]

        probs = [0] * len(self.choice)
        ret = 0
        for t in r.logprobs.content[0].top_logprobs:
            token = t.token.strip()
            for i in range(len(self.choice)):
                c = self.choice[i]
                if c == token:
                    probs[i] = t.logprob
                if ret == 0:
                    ret = i + 1
        
        c['metrics'][self.judge_name] = probs
        return ret

        