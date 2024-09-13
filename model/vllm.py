from . import openai_api

import os

class Model(openai_api.Model):
    def __init__(self, port, model_path, model_name='default', key=os.environ.get('OPENAI_API_KEY')): 
        self.port = port
        self.model_path = model_path
        self.model_name = model_name
        super().__init__(f'http://127.0.0.1:{port}/v1', model_name, key)


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_value, traceback):
        pass