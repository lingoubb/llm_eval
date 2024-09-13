from prompt import normal


class Prompt(normal.Prompt):
    def get_output(self, model, c):
        output = model.get_outputs([c])[0]

