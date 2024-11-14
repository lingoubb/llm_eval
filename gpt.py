from model.openai_api import Model
# from model.deepseek import Model

m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')

p = '''\

'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个专家"},
            {"role": "user", "content": p},
        ]])[0].message.content)