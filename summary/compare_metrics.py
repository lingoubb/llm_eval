from sklearn.feature_selection import mutual_info_regression
import math
from sklearn import tree

def inc(features, labels):
    size = len(features)
    cot = 0
    predict_labels = []
    for i in range(size):
        feature = features[i]
        j = max(enumerate(feature), key=lambda x: x[1])[0]
        score = {0: 1, 1: -1, 2: 0}[j]
        predict_labels.append([score])
        if score == labels[i]:
            cot += 1
    print(f'\tcorret: {cot/size:.3f}')
    print(f'\tmutual info: {mutual_info_regression(features, labels, discrete_features=False)}')
    print(f'\tmutual info: {mutual_info_regression(predict_labels, labels, discrete_features=False)}')


def pred(metrics, labels, batch_percent=0.2, predict_model=tree.DecisionTreeClassifier(criterion='gini', max_depth=3)):
    size = len(labels)

    features = [[]] * size
    for i in range(size):
        for probs in metrics.values():
            features[i] += probs[i]

    cot = 0
    
    batch_size = math.ceil(batch_percent * size)
    for i in range(0, size, batch_size):
        end_i = min(i + batch_size, size)

        test_x = features[i:end_i]
        test_y = labels[i:end_i]
        train_x = features[:i] + features[end_i:]
        train_y = labels[:i] + labels[end_i:]

        print(train_x)
        print(train_y)
        predict_model.fit(train_x, train_y)
        pred_y = predict_model.predict(test_x)
        
        for p, t in zip(pred_y, test_y):
            # print(p,t)
            if p == t:
                cot += 1

    print(f'\tPred cot: {cot/size:.3f}')

class Summary:
    def print_summary(self, dataset):
        dataset = dataset.content
        metric_names = dataset[0]['metrics'].keys()
        metrics = {}

        labels = []
        for x in dataset:
            labels.append(x['manual_score'])

        for metric in metric_names:
            print(metric)
            probs = []
            for x in dataset:
                probs.append(x['metrics'][metric])
            inc(probs, labels)
            metrics[metric] = probs
            print('-' * 20)
            
        pred(metrics, labels)