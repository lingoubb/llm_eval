from os import environ
from . import openai_api


class Model(openai_api.Model):
    def __init__(self, kargs={}):
        # super().__init__('https://api.deepseek.com', 'deepseek-chat', 'sk-ac86655dea034663aeffbf6371419bff', kargs=kargs)
        super().__init__('https://openrouter.ai/api/v1', 'deepseek/deepseek-chat', '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', kargs=kargs)