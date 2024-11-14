import os

class Model:
    def __init__(self, url, model, key=os.environ.get('OPENAI_API_KEY'), kargs={}):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=key, base_url=url) 
        self.kargs = kargs 


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def get_outputs(self, inputs, temperature=0, max_tokens=1024, logprobs=False, **karg):
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
                **karg,
            }
            response = self.client.chat.completions.create(**data)
            outputs.append(response.choices[0])
            # outputs.append(response.choices[0].message.content)
        # if logprobs:
        #     return outputs, [x.top_logprobs for x in response.choices[0].logprobs.content]
        # else:
        #     return outputs
        # print(outputs[0].message.content)
        return outputs
    