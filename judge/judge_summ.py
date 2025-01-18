'Is this claim consistent with the document?'
from . import with_llm


class Judge(with_llm.Judge):
    def __init__(self, model):
        self.judge_name = 'score'
        super().__init__(model)


    def get_score(self, c):

        c.setdefault('metrics', {})
        if self.judge_name in c['metrics']:
            return None

        # prompt_t = f"question: Is this claim consistent with the document?</s>claim: {c['output']}</s>document: {c['question']}"
        prompt_t = f"claim: {c['output']}</s>document: {c['question']}"

        prompt = [
            # {"role": "system", "content": 'No matter what questions you receive, you can only answer "yes" or "no" without any explanation'},
            {"role": "system", "content": 'You will receive a claim and a document, and you need to determine if this claim is consistent with the document and give a score of 1-5, with higher scores indicating higher consistency. You must directly output the score without any other explanation.'},
            {"role": "user", "content": prompt_t},
        ]
        
        r = self.model.get_outputs([prompt])[0].message.content

        s = None
        for i in range(1, 6):
            if str(i) in r:
                s = i
        if s is None:
            raise Exception(f'模型回答格式错误: {r}')

        
        # if 'yes' in r:
        #     s = 1
        # elif 'no' in r:
        #     s = 0
        # else:
        #     raise Exception(f'模型回答格式错误: {r}')

        c['metrics'][self.judge_name] = s

        return None
        