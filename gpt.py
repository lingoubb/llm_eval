from model import deepseek
# from model.deepseek import Model

# m = Model('https://api.chatanywhere.com.cn/v1', 'gpt-4', 'sk-H4tGHk14FwSHTxKOjVwYh5W0uKlilh28WjCYciGsAV4lNjhb')
m = deepseek.Model()

p = '''\
以下是一些用于评判ai助手回答的标准，请翻译为英文并按原格式输出：
    '助手的回答提供了足够的必要信息',
    '助手的回答不存在不必要的冗余信息',
    '助手的回答不存在繁琐重复的内容',
    '助手的回答中提到的信息不存在与事实不符的信息或编造的信息',
'''

print(m.get_outputs([[
            {"role": "system", "content": "你是一个专家助手"},
            {"role": "user", "content": p},
        ]])[0].message.content)