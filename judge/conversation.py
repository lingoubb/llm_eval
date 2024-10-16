from . import with_llm

"""
对话+fact
"""
class Judge(with_llm.Judge):
    def __init__(self, model, judge_name, judge_head, judge_prompt, choice=['A', 'B', 'C'], deal_with_score=int):
        self.judge_name = judge_name
        self.judge_head = judge_head
        self.judge_prompt = judge_prompt
        self.choice = choice
        self.deal_with_score = deal_with_score
        super().__init__(model)


    def get_score(self, case):
        case.setdefault('metrics', {})
        if self.judge_name in case['metrics']:
            return None

        prompt = [
            {"role": "system", "content": self.judge_head},
            {"role": "user", "content": self.judge_prompt.format(**case)},
        ]
        
        r = self.model.get_outputs([prompt])[0]
        c = r.message.content
        beg = c.index('[[')
        if beg >= 0:
            end = c.index(']]')
            ret = c[beg + 2:end]
        else:
            raise Exception(f'回答格式错误: {repr(c)}')
        
        case['metrics'][self.judge_name] = self.deal_with_score(ret)
        return None

        