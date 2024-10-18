from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr, kendalltau
import math
from sklearn import tree
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
    print(f'\tMAE: {total_deta/size:.3f}')
    print(f'\tspearmanr: {spearmanr([x[0] for x in predict_labels], labels)[0]}')
    print(f'\tkendalltau: {kendalltau([x[0] for x in predict_labels], labels)[0]}')
    print(f'\tfeatures mutual info: {mutual_info_regression(features, labels, discrete_features=False)}')
    print(f'\tmutual info: {mutual_info_regression(predict_labels, labels, discrete_features=False)}')


def pred(metrics, labels, batch_percent=0.2, predict_model=tree.DecisionTreeClassifier(criterion='gini', max_depth=3)):
    size = len(labels)

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

    
    batch_size = math.ceil(batch_percent * size)
    for i in range(0, size, batch_size):
        end_i = min(i + batch_size, size)

        test_x = features[i:end_i]
        test_y = labels[i:end_i]
        train_x = features[:i] + features[end_i:size]
        train_y = labels[:i] + labels[end_i:size]

        predict_model.fit(train_x, train_y)
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
        
        spearmanr_list.append(spearmanr(pred_y, test_y)[0])
        kendalltau_list.append(kendalltau(pred_y, test_y)[0])
        # print(f'\tPred cot_t: {cot_t/(end_i-i):.3f}')
        cot += cot_t

    print(f'\tPred cot: {cot/size:.3f}')
    print(f'\tPred MAE: {total_deta/size:.3f}')
    print(f'\tPred deta<=1: {in_1/size:.3f}')
    print(f'\tPred spearmanr: {sum(spearmanr_list)/len(spearmanr_list):.3f}')
    print(f'\tPred kendalltau: {sum(kendalltau_list)/len(kendalltau_list):.3f}')

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
            
        # pm = MLPClassifier(hidden_layer_sizes=(200,200,200,100), max_iter=2000)
        pm = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
        pred(metrics, labels, predict_model=pm)