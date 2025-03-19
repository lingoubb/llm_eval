from model import deepseek
# from model.deepseek import Model

# m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')
m = deepseek.Model()

p = '''\
'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": p},
        ]])[0].message.content)