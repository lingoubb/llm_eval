from . import openai_api

class Model(openai_api.Model):
    def get_outputs(self, inputs, *args, **kargs):
        for x in inputs:
            for y in x:
                y['content'] += ' /no_think'
        return super().get_outputs(inputs, *args, **kargs)
    