from judge import normal
from tools import tools

"""
主观题评测
"""
class Judge(normal.Judge):
    def __init__(self, model, *args):
        # self.model = tools.load_config('Model', model_config, *args)
        self.model = model

    def get_score(self, c):
        pass