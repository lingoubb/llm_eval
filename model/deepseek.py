from os import environ
from . import openai_api


class Model(openai_api.Model):
    def __init__(self, kargs={}):
        # super().__init__('https://api.deepseek.com', 'deepseek-chat', 'sk-ac86655dea034663aeffbf6371419bff', kargs=kargs)
        super().__init__('https://openrouter.ai/api/v1', 'deepseek/deepseek-chat', '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', kargs=kargs)