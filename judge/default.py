from judge import qa
from judge import normal

class Judge(normal):
    def __init__(self):
        self.judge_map = {
            '简答题': qa.Judge,
            '主观题': qa.Judge,
        }


    def get_score(self, c):
        judge = self.judge_map.get(c['type'])
        if judge is None:
            raise Exception('unknown question type')
        return judge.get_score(c)