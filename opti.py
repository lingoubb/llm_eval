from eval import load_result, gen_score, save_result
from model import deepseek, gpt_3
from judge import compare_judgelm, judge_probs, judge_summ, judge_summ_new, judge_with_features
import summary.metrics_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr, kendalltau
import math
from sklearn import tree
from collections.abc import Iterable
import random

model_gpt3 = gpt_3.Model()
model_deepseek = deepseek.Model()
model = model_deepseek

# 初始维度
init_metrics = [
    '助手的回答能够很好地解决问题',
    '助手的回答与问题相关，没有答非所问',
    '助手的回答提供了足够的必要信息',
    '助手的回答不存在不必要的冗余信息',
    '助手的回答不存在繁琐重复的内容',
    '助手的回答中提到的信息不存在与事实不符的信息或编造的信息',
    '评估时，先指出问题中是否存在关于回答格式、回答要求、角色扮演等的指令，然后判断助手的回答是否严格遵循了这些指令',
]

'助手的回答严格遵循了问题中关于回答格式、回答要求、角色扮演等的指令'
# 特点
init_features = [
    '问题中是否存在关于回答格式、回答要求、角色扮演等的指令',
    '该问题是没有标准答案的开放式问题吗',
    '根据问题的类型和要求，你认为回答该问题应该尽可能简洁吗（反之应该尽可能提供更多信息）',
    '这是一道数学计算题吗',
]

p_for_feature = '''\
这是一个人类向 ai 助手提出的问题：
{question}

现在你需要判断：{feature}
你的回答只能是 "YES" 或 "NO"，不需要其它任何解释。
'''

def get_feature(feature):
    def f(self, c):
        i = [
            {"role": "user", "content": p_for_feature.format(question=c['question'], feature=feature)},
        ]
        output = model.get_outputs([i], max_tokens=512)[0].message.content
        score_map = {
            'YES': 1, 'NO': 0
        }
        r = None
        for k, s in score_map.items():
            if k in output:
                r = s 
                break
        if r is None:
            raise Exception(f'模型输出格式错误: {output}')
        return r
    return f



# 根据负样本生成新维度
dp = '''\
你是一个负责设计基于 LLM 的问答系统评估器的专家。现在你设计的评估器在以下用例上出现了偏差：
[User Question]
{question}
[The Start of Assistant A’s Answer]
{output_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{output_b}
[The End of Assistant B’s Answer]

你的评估器认为 Assistant A 的回答比 Assistant B 的回答更好，然而人类专家评估者给出的评估与此相反。

这是你的评估器在判题时所采用的所有评估角度：
{metrics}

该评估器产生误判可能是因为上面提到的评估角度还不够全面。请你分析人类专家认为 Assistant B 的回答更好的可能原因，并参考上面的评估角度的格式，在你输出的**最后一行**给出一个全新的评估角度（尽可能和已有的评估角度不同）。
'''


prompt_system_v1 = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

After providing your explanation, **output your final verdict by strictly following this format**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''
prompt_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose an assistant that better meets this requirement: **{metric}**.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

**Output your final verdict by strictly following this format without providing any explanation**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
'''
prompt_user = '''\
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
'''

def get_m(metric):
    def baseline(self, c):
        # print(c['id'])
        i = [
            {"role": "system", "content": prompt_system.format(metric=metric)},
            {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
        ]
        output = self.model.get_outputs([i], max_tokens=1024)[0].message.content
        score_map = {
            '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
        }
        # print(output)
        r = None
        for k, s in score_map.items():
            if k in output:
                r = s 
                break
        if r is None:
            raise Exception(f'模型输出格式错误: {output}')
        return r
    return baseline


datasets = []
# datasets += load_result('dataset/mt_bench_train', f'output/0116/mt_bench_train' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/arena_1000', f'output/0116/arena_1000_wo_ep' + '' if model is model_deepseek else '_gpt3')
datasets += load_result('dataset/arena_6000', f'output/0116/arena_6000_wo_ep' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/pandalm', f'output/0116/pandalm' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/llmeval2', f'output/0116/llmeval2' + '' if model is model_deepseek else '_gpt3')
# datasets += load_result('dataset/JudgeBench', f'output/0116/JudgeBench' + '' if model is model_deepseek else '_gpt3')
def get_eval_res(features):
    for dataset in datasets:
        gen_score(judge_with_features.Judge(model, features=features), dataset)
    


# 训练集
# train_set = get_trainset()

# 测试集

# 现有维度
features = {
    'features':[],
    'metrics': [],
}
def ref_features():
    for i in range(len(init_metrics)):
        m = init_metrics[i]  
        features['metrics'].append((f'0115_{i}', get_m(m)))
    for i in range(len(init_features)):
        m = init_features[i]
        features['metrics'].append((f'0115_features_{i}', get_feature(m)))

def get_x_y(content, features):
    X = [[x['metrics'].get(m[0]) for m in features['metrics']] for x in content]
    Y = [x['manual_score'] for x in content]
    return X, Y

def d_f(question, better, worse, origin_score):
    t_dp = dp.format(question=question, output_b=better, output_a=worse, metrics='\n'.join(['- ' + x for x in init_metrics]))
    resp = model.get_outputs([[{"role": "user", "content": t_dp}]], temperature=1, text=True)[0]
    new_m = resp.splitlines()[-1]
    print(new_m)
    i = [
        {"role": "system", "content": prompt_system.format(metric=new_m)},
        {"role": "user", "content": prompt_user.format(question=question, answer_a=better, answer_b=worse)},
    ]
    output = model.get_outputs([i], max_tokens=1024)[0].message.content
    score_map = {
        '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
    }
    r = None
    for k, s in score_map.items():
        if k in output:
            r = s 
            break
    if r is None:
        print(f'模型输出格式错误: {output}')
    elif r != origin_score:
        print('有效')
        return new_m
    return None

# 迭代

# for dataset in datasets:
#     content = dataset.content
#     for x in content:
#         del x['metrics']['0115_features_0']
#     save_result(dataset)

    

epoch = 1
for i in range(epoch):
    # 判断维度相关性，筛选维度
    ref_features()
    get_eval_res(features)

    for dataset in datasets:
        content = dataset.content

        X, Y = get_x_y(content, features)
        pm = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
        
        pm.fit(X, Y)
        pred = pm.predict(X)
        
        def get_cot_deta1(Y, pred):
            deta1 = 0
            cot = 0
            n = len(Y)
            for i in range(n):
                if abs(Y[i] - pred[i]) <= 1:
                    deta1 += 1
                    if Y[i] == pred[i]:
                        cot += 1
            return cot, deta1, n
        baseline_pred = [x['metrics']['baseline'] for x in content]
        # cot, deta1, n = get_cot_deta1(Y, pred)
        print(dataset.name)
        # print(f'Pred Cot: {cot/n:.3f}, Deta<=1: {deta1/n:.3f}')
        # print(f'Pred spearmanr: {spearmanr(pred, Y)[0]:.3f}, kendalltau: {kendalltau(pred, Y)[0]:.3f}')
        # cot, deta1, n = get_cot_deta1(Y, baseline_pred)
        # print(f'Baseline Cot: {cot/n:.3f}, Deta<=1: {deta1/n:.3f}')
        # print(f'Baseline spearmanr: {spearmanr(baseline_pred, Y)[0]:.3f}, kendalltau: {kendalltau(baseline_pred, Y)[0]:.3f}')
        summary.metrics_regression.Summary().print_summary(dataset)
        print()
        

        # 获取负样本
        # tmp_m = []
        bad_cases = []
        for i in range(len(Y)):
            if Y[i] - pred[i] == 2:
                better = content[i]['output'][0]
                worse = content[i]['output'][1]
                bad_cases.append((content[i], better, worse))
            elif Y[i] - pred[i] == -2:
                better = content[i]['output'][1]
                worse = content[i]['output'][0]
                bad_cases.append((content[i], better, worse))
            else:
                continue

        #     new_m = d_f(content[i]['question'], better, worse, pred[i])
        #     if new_m is not None:
        #         tmp_m.append(new_m)
            

        # init_metrics += tmp_m
        # ref_features()

    
# for x in tmp_m:
#     print(x)

