from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_judgelm, judge_probs, judge_summ, judge_summ_new, judge_with_features
import summary.metrics_regression

prompt_system = '''\
Please act as an impartial judge and evaluate the quality of the responses provided by two\
AI assistants to the user question displayed below. You should choose the assistant that\
follows the user’s instructions and answers the user’s question better. Avoid any position biases and ensure that the\
order in which the responses were presented does not influence your decision. Do not allow\
the length of the responses to influence your evaluation. Be as objective as possible.

After providing your explanation, **output your final verdict by strictly following this format**: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
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

def baseline(self, c):
    i = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user.format(question=c['question'], answer_a=c['output'][0], answer_b=c['output'][1])},
    ]
    output = self.model.get_outputs([i], max_tokens=1024)[0].message.content
    score_map = {
        '[[A]]': 1, '[[B]]': -1, '[[C]]': 0
    }
    r = None
    for k, s in score_map.items():
        if k in output:
            r = s 
            break
    if r is None:
        raise Exception(f'模型输出格式错误: {output}')
    return r

features = {
    'features':[],
    'metrics': [
        ('baseline', baseline),
    ],
}

def f(datasets):
    for dataset in datasets:
        gen_score(judge_with_features.Judge(gpt_3.Model(), features=features), dataset)
        # gen_score(judge_summ_new.Judge(deepseek.Model()), dataset)
        summary.metrics_regression.Summary().print_summary(dataset)


f(load_result('dataset/biasbench', f'output/1226/biasbench_gpt_3'))
# f(load_result('dataset/pandalm', f'output/1226/pandalm'))
# f(load_result('dataset/arena_1000', f'output/1226/arena_1000')) 
# f(load_result('dataset/mt_bench', f'output/1226/mt_bench'))