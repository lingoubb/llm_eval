from . import with_llm

"""
主观题比较
"""
class Judge(with_llm.Judge):
    """
    1: A wins
    0: tie
    -1: B wins
    """
    def get_score(self, c):
        pass