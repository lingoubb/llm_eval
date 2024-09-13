from prompt import normal


class Prompt(normal.Prompt):
    def get_output(self, model, c):
        i = [
            {"role": "system", "content": "你是一个回答助手。"},
            {"role": "user", "content": c['question']},
        ],
        output = model.get_outputs([i])[0]

