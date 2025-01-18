from eval import load_result, gen_score
from model import deepseek, gpt_3, openai_api
from judge import compare_judgelm, judge_probs
import summary.compare
import math
from sklearn import tree
from collections.abc import Iterable
import random



import json


r = []

with open(r'D:\workspace\llm_eval\output\mt_bench_train\mt_bench_train.json', encoding='utf-8') as f:
    all = json.load(f)

def get_x_y(content):
    X = [[x['metrics'].get(m) for m in x['metrics'] if m != 'baseline'] for x in content]
    Y = [x['manual_score'] for x in content]
    return X, Y
X, Y = get_x_y(all)
pm = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
pm.fit(X, Y)
pred = pm.predict(X)
for i in range(len(all)):
    x = all[i]
    if abs(x['metrics']['baseline'] - x['manual_score']) > 1 or abs(pred[i] - x['manual_score']) > 1:
        r.append([x['question'], x['output'][0], x['output'][1], x['manual_score'], x['metrics']['baseline'], pred[i]])

import csv


def list_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

list_to_csv(r, 'deta.csv')