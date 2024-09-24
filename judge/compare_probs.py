from . import with_llm

"""
probs 主观题比较
"""
class Judge(with_llm.Judge):
    def __init__(self, model, judge_name, judge_prompt, choice=['A', 'B']):
        self.judge_name = judge_name
        self.judge_prompt = judge_prompt
        self.choice = choice
        super().__init__(model)


    def get_score(self, c):
        prompt = self.judge_promt.format(**c)
        r = self.model.get_outputs([prompt], logprobs=True, top_logprobs=20)[0]

        probs = [0, 0]
        ret = 0
        for t in r.logprobs.content[0].top_logprobs:
            token = t.token.strip()
            for i in range(len(self.choice)):
                c = self.choice[i]
                if c == token:
                    probs[i] = t.probs
                if ret == 0:
                    ret = i + 1
        
        c.setdefault('metrics', {})
        c['metrics'][self.judge_name] = probs
        return ret

        