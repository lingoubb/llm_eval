from os import environ
from . import openai_api


class Model(openai_api.Model):
    def __init__(self):
        super().__init__('https://api.chatanywhere.tech/v1', 'gpt-3.5-turbo', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')