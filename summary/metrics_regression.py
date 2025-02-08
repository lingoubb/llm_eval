from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr, kendalltau
from sklearn.impute import SimpleImputer
import math
from sklearn import tree
from collections.abc import Iterable
import random
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def get_ans(f):
    return max(enumerate(f), key=lambda x: x[1])[0]

def inc(features, labels):
    
    if not isinstance(features[0][0], int):
        return

    total_size = len(features)
    size = total_size
    cot = 0
    in_1 = 0
    total_deta = 0
    predict_labels = []
    
    for i in range(total_size):
        feature = features[i]
    
        # # probs
        # j = get_ans(feature)
        # score = {0: 1, 1: -1, 2: 0}[j]
        score = feature[0]

        # 空值
        if score is None:
            size -= 1
            continue

        predict_labels.append([score])
        deta = abs(score - labels[i])
        total_deta += deta
        if deta == 0:
            cot += 1
        if deta <= 1:
            in_1 += 1
    # print(features == predict_labels)
    print(f'\tcorret: {cot/size:.3f}')
    print(f'\tdeta<=1: {in_1/size:.3f}')
    print(f'\tMAE: {total_deta/size:.3f}')
    # print(f'\tfeatures mutual info: {mutual_info_regression(features, labels, discrete_features=False)}')
    # print(f'\tmutual info: {mutual_info_regression(predict_labels, labels, discrete_features=False)}')
    print(f'\tspearmanr: {spearmanr([x[0] for x in predict_labels], labels)[0]}')
    print(f'\tkendalltau: {kendalltau([x[0] for x in predict_labels], labels)[0]}')


def pred(metrics, labels, batch_percent=0.2, predict_model=tree.DecisionTreeClassifier(criterion='gini', max_depth=3)):
    size = len(labels)
    batch_size = math.ceil(batch_percent * size)

    features = []
    for i in range(size):
        features.append([])
        for probs in metrics.values():
            # features[i].append(get_ans(probs[i]))
            features[i] += probs[i]

    # features = metrics['accuracy']
    # print(features)
    # print(labels[:size])
    cot = 0
    total_deta = 0
    in_1 = 0

    spearmanr_list = []
    kendalltau_list = []
    feature_importances_list = []

    
    for i in range(0, size, batch_size):
        end_i = min(i + batch_size, size)

        test_x = features[i:end_i]
        test_y = labels[i:end_i]
        train_x = features[:i] + features[end_i:size]
        train_y = labels[:i] + labels[end_i:size]
        
        # predict_model=tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
        predict_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        predict_model = RFE(predict_model, n_features_to_select=8, step=1)

        predict_model.fit(train_x, train_y)

        # print(predict_model.ranking_)

        # feature_importances = predict_model.feature_importances_
        # feature_importances_list.append(feature_importances)
        
        pred_y = predict_model.predict(test_x)
        
        cot_t = 0

        for p, t in zip(pred_y, test_y):
            # print(p,t)
            deta = abs(p - t)
            total_deta += deta
            if deta == 0:
                cot_t += 1
            if deta <= 1:
                in_1 += 1
        
        # print(pred_y)
        # print(test_y)
        spearmanr_list.append(spearmanr(pred_y, test_y)[0])
        kendalltau_list.append(kendalltau(pred_y, test_y)[0])
        # print(f'\tPred cot_t: {cot_t/(end_i-i):.3f}')
        cot += cot_t


    # print(feature_importances_list)
    # print(zip(*feature_importances_list))
    # feature_importances = [(sum(x)/len(x)) for x in zip(*feature_importances_list)]

    print(f'\tPred cot: {cot/size:.3f}')
    print(f'\tPred deta<=1: {in_1/size:.3f}')
    print(f'\tPred MAE: {total_deta/size:.3f}')
    print(f'\tPred spearmanr: {sum(spearmanr_list)/len(spearmanr_list):.3f}')
    print(f'\tPred kendalltau: {sum(kendalltau_list)/len(kendalltau_list):.3f}')
    # print(f'\tPred feature importances: {feature_importances}')


    # i, _ = min(enumerate(feature_importances), key=lambda x: x[1])
    # print(f'remove {i}')
    # features = [x[:i] + x[i+1:] for x in features]


def deal_none(probs):
    c = Counter([x[0] for x in probs])
    del c[None]
    z = c.most_common(1)[0][0]
    return [[z] if x[0] is None else x for x in probs]

class Summary:
    def print_summary(self, dataset, metric_names=None, feature_names=[]):
        dataset = dataset.content
        if metric_names is None:
            metric_names = dataset[0]['metrics'].keys()

        metrics = {}

        labels = []
        for x in dataset:
            labels.append(x['manual_score'])

        for metric in metric_names:
            probs = []
            for x in dataset:
                v = x['metrics'].get(metric)
                if not isinstance(v, str) and isinstance(v, Iterable):
                    probs.append(v)
                else:
                    probs.append([v])

            # 处理空值
            probs = deal_none(probs)

            if metric.startswith('baseline'):
                print(metric)
                inc(probs, labels)
                print('-' * 20)

            metrics[metric] = probs

        for metric in feature_names:
            print(metric)
            probs = []
            for x in dataset:
                v = x['features'].get(metric)
                if isinstance(v, Iterable):
                    probs.append(v)
                else:
                    probs.append([v])

            # 处理空值
            probs = deal_none(probs)

            inc(probs, labels)
            metrics[metric] = probs
            print('-' * 20)
            
        # pm = MLPClassifier(hidden_layer_sizes=(200,200,200,100), max_iter=2000)
        if len(metrics) > 1:
            pm = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
            pred(metrics, labels, predict_model=pm)