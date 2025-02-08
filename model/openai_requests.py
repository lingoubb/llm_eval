import requests
import json

import os

class Model:
    def __init__(self, url, model, key=os.environ.get('OPENAI_API_KEY'), kargs={}):
        self.url = url
        self.model = model
        self.kargs = kargs 
        self.key = key


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def get_outputs(self, inputs, temperature=0, max_tokens=1024, logprobs=False, text=False, kargs={}):
        outputs = []
        for a_input in inputs:
            
            data = {
                "model": self.model,
                "messages": a_input,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": logprobs,
                **self.kargs,
                **kargs,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": 'Bearer {self.key}'
            }
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            try:
                if response.status_code == 200:
                    result = response.json()
                    if text:
                        outputs.append(result['choices'][0]['message']['content'])
                    else:
                        outputs.append(result['choices'][0])
                else:
                    raise Exception(f"请求失败，状态码: {response.status_code}")
            except Exception as e:
                print(f'resp: {response.content}')
                raise e

        # if logprobs:
        #     return outputs, [x.top_logprobs for x in response.choices[0].logprobs.content]
        # else:
        #     return outputs
        # print(outputs[0].message.content)
        return outputs
    