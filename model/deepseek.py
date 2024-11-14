from os import environ
from . import openai_api


class Model(openai_api.Model):
    def __init__(self, kargs={}):
        super().__init__('https://api.deepseek.com', 'deepseek-chat', 'sk-ac86655dea034663aeffbf6371419bff', kargs=kargs)