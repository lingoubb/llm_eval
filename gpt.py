from model.openai_api import Model

m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')


p = '''\
为什么embedding模型要采用双向注意力机制
'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个大语言模型领域专家助手"},
            {"role": "user", "content": p},
        ]])[0].message.content)