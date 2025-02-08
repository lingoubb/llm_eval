from qianfan import Qianfan


import os

class Model:
    def __init__(self, model='deepseek-v3', kargs={}):
        self.model = model
        self.client = Qianfan(access_key='1d4ca1d22b704043b25dc99f756f5e32', secret_key='a2f1b1710aa54811b146eca8bf3161eb')
        self.kargs = kargs


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def get_outputs(self, inputs, temperature=0.001, max_tokens=1024, logprobs=False, text=False, kargs={}):
        outputs = []
        for a_input in inputs:
            data = {
                "model": self.model,
                "logprobs": logprobs,
                "messages": a_input,
                "temperature": temperature,
                "stream": False,
                "max_tokens": max_tokens,
                **self.kargs,
                **kargs,
            }
            try:
                response = self.client.chat.completions.create(**data)
            except Exception as e:
                print(f'unexpected data: {data}')
                raise e
            try:
                if text:
                    outputs.append(response.choices[0].message.content)
                else:
                    outputs.append(response.choices[0])
            except Exception as e:
                print(f'unexpected resp: {response}')
                raise e
            print(f'[output] {repr(outputs[-1])}')
        # if logprobs:
        #     return outputs, [x.top_logprobs for x in response.choices[0].logprobs.content]
        # else:
        #     return outputs
        # print(outputs[0].message.content)
        return outputs
    