from model.openai_api import Model

m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')

p = '''\
详细介绍python sklearn mutual_info_regression参数和返回值的含义
'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": p},
        ]])[0].message.content)