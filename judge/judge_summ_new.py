'Is this claim consistent with the document?'
from . import with_llm


class Judge(with_llm.Judge):
    def __init__(self, model):
        self.judge_name = 'default'
        super().__init__(model)


    def get_score(self, c):

        c.setdefault('metrics', {})
        if self.judge_name in c['metrics']:
            return None

        info = []
        for line in c['output'].split('.'):
            line += '.'

            prompt_t = f"claim: {line}</s>document: {c['question']}"

            prompt = [
                {"role": "system", "content": 'You will receive a claim and a document. You need to determine if this claim is consistent with the document. If they are consistent, please answer "yes" directly without any explanation; If there are inconsistencies, please indicate the areas of inconsistency.'},
                {"role": "user", "content": prompt_t},
            ]
            
            r = self.model.get_outputs([prompt])[0].message.content
            if r.startswith('yes'):
                s = 1
            else:
                info.append(r)

        c['raw_info'] = info

        if info:
            prompt = [
                {"role": "system", "content": 'You will receive a claim and a document, and you need to determine if this claim is consistent with the document.'},
                {"role": "user", "content": prompt_t},
                {"role": "assistant", 'content': f'There are {len(info)} inconsistencie(s) between this claim and the document:\n' + '\n'.join(info)},
                {"role": "user", "content": "Give a score of 1-5, with higher scores indicating higher consistency. You must directly output the score without any other explanation."},
            ]
        else:
            prompt = [
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

        c['metrics'][self.judge_name] = s

        return None
        