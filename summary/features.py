
import math
from collections.abc import Iterable
import random

def get_ans(f):
    return max(enumerate(f), key=lambda x: x[1])[0]

def inc(features, labels):

    size = len(features)
    cot = 0
    in_1 = 0
    total_deta = 0
    predict_labels = []
    for i in range(size):
        feature = features[i]

        # # probs
        # j = get_ans(feature)
        # score = {0: 1, 1: -1, 2: 0}[j]
        score = feature[0]

        predict_labels.append([score])
        deta = abs(score - labels[i])
        total_deta += deta
        if deta == 0:
            cot += 1
        if deta <= 1:
            in_1 += 1
    print(features == predict_labels)
    print(f'\tcorret: {cot/size:.3f}')
    print(f'\tdeta<=1: {in_1/size:.3f}')



class Summary:
    def print_summary(self, dataset, metric_names=None):
        dataset = dataset.content
        if metric_names is None:
            metric_names = dataset[0]['metrics'].keys()
        metrics = {}

        labels = []
        for x in dataset:
            labels.append(x['manual_score'])

        for metric in metric_names:
            print(metric)
            probs = []
            for x in dataset:
                v = x['metrics'][metric]
                if isinstance(v, Iterable):
                    probs.append(v)
                else:
                    probs.append([v])
            inc(probs, labels)
            metrics[metric] = probs
            print('-' * 20)