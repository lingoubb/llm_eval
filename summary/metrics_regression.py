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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.cluster import DBSCAN, KMeans

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

"""
feature_names 特征名称列表
train_set 训练集特征矩阵
labels 训练集标签列表
train_weight 训练集样本权重列表：
test_set 测试集特征矩阵

返回值 pred_labels 测试集预测标签列表
"""
# def pred(feature_names, train_set, labels, train_weight, test_set):
#     return pred_labels
def tmp(feature_names, train_set, labels, train_weight, test_set):
    
    train_set = np.array(train_set)
    test_set = np.array(test_set)

    pass


def pred(metrics, labels, batch_percent=0.2, get_predict_model=lambda: tree.DecisionTreeClassifier(criterion='gini', max_depth=3)):
    
    mode = 5
    
    size = len(labels)
    batch_size = math.ceil(batch_percent * size)

    def is_pre_feature(name):
        # return 'debias_features' in name
        return 'feature' in name 

    pre_features = []
    features = []
    for i in range(size):
        features.append([])
        pre_features.append([])
        for name, probs in metrics.items():
            # features[i].append(get_ans(probs[i]))
            t = probs[i] 
            if 'pair_score' in name:
                # normalize
                t = [t[0] / 5]

            if is_pre_feature(name):
                pre_features[i] += t
                if mode == 0 or mode == 4 or mode == 6:
                    features[i] += t
            else:
                features[i] += t

    pre_features_names = [name for name in metrics if is_pre_feature(name)]
    pre_features_name = '0115_features_13'
    pre_feature_index = pre_features_names.index(pre_features_name)
    # print(pre_features_names)

    # 观察每个pre_features对难度序列的影响
    # if True:            
    #     confidence_for_features = np.array([1 - (abs(np.array(x)-y) / 2) for x, y in zip(features, labels)])
    #     tmp = np.array(pre_features)

    #     new_pre_features = []
    #     for i, x in enumerate(tmp.T):
    #         print(pre_features_names[i], Counter(x))
    #         num = 0
    #         for y in confidence_for_features.T:
    #             p = spearmanr(x, y)[1]
    #             if p < 0.01:
    #                 num += 1
    #                 print(f'{p:.4f}', end=' ')
    #         if num >= 15:
    #             new_pre_features.append(x)
    #         print()
    #         print()

    #     print('len(pre_features[0])', len(pre_features[0]))
    #     pre_features = list(np.array(new_pre_features).T)
    #     print('len(pre_features[0])', len(pre_features[0]))

            
        


    # print(features)

    # features = metrics['accuracy']
    # print(features)
    # print(labels[:size])
    # feature_importances_list = []

    if mode == 2:
        # 聚类
        classes_num = 4
        # cl = DBSCAN(eps=0.2, min_samples=5)
        cl = KMeans(n_clusters=classes_num)
        cl.fit(pre_features)
        cl_labels = cl.labels_
        print(Counter(cl_labels))
        centroids = cl.cluster_centers_
        confidences = []
        for centroid in centroids:
            distances = []
            for i, point in enumerate(pre_features):
                distance = np.linalg.norm(np.array(point) - centroid)
                distances.append(distance)
            confidences.append(list(1 / (np.array(distances) + 0.5)))
        # print(confidences)


    all_pred_y = []
    all_pred_y_with_prob = []
    
    for i in range(0, size, batch_size):
        end_i = min(i + batch_size, size)

        test_x = features[i:end_i]
        # test_y = labels[i:end_i]
        train_x = features[:i] + features[end_i:size]
        train_y = labels[:i] + labels[end_i:size]
        
        if mode != 0:
            pre_features_train = pre_features[:i] + pre_features[end_i:size]

        
        bonus = False
        # 是否添加镜像样本
        if bonus:
            train_x += [list(-np.array(x)) for x in train_x]
            train_y += list(-np.array(train_y))
            if mode != 0:
                pre_features_train += pre_features_train


        def get_pred_y(test_x, predict_model):
            if len(test_x) == 0:
                return [], []
            pred_y_prob = predict_model.predict_proba(test_x)
            class_labels = predict_model.classes_
            pred_y = [class_labels[max(enumerate(x), key=lambda x: x[1])[0]] for x in pred_y_prob]
            pred_y_with_prob = [sum([label * prob for label, prob in zip(class_labels, x)]) for x in pred_y_prob]
            return pred_y, pred_y_with_prob
        
            
        
        if mode == 1 or mode == 0:
            predict_model = get_predict_model()
            predict_model.fit(train_x, train_y)
            if "best_params_" in dir(predict_model): 
                print("Best parameters found: ", predict_model.best_params_)
            pred_y, pred_y_with_prob = get_pred_y(test_x, predict_model)
            # print(predict_model.best_estimator_.feature_importances_)

        elif mode == 3:
            # predict_model = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
            predict_model = get_predict_model()
            predict_model.fit(train_x, train_y)
            pred_y, pred_y_with_prob = get_pred_y(train_x, predict_model)
            difficulty = [abs(p-l) for p, l in zip(pred_y_with_prob, train_y)]
            tmp = sorted(difficulty)
            classes_num = 6
            diff_classes = [tmp[len(tmp) * i // classes_num] for i in range(classes_num)] + [99]        
            print(diff_classes)
            
            predict_models = []
            for j in range(classes_num):
                predict_model = get_predict_model()
                def get_weight(d):
                    if diff_classes[j] <= d < diff_classes[j+1]:
                        return 1
                    return 1 - abs(d - ((diff_classes[j] + diff_classes[j+1]) / 2)) / 2
                predict_model.fit(train_x, train_y, sample_weight=[get_weight(d) for d in difficulty])
                if "best_params_" in dir(predict_model): 
                    print("Best parameters found: ", predict_model.best_params_)
                predict_models.append(predict_model)

            # difficulty_predictor = tree.DecisionTreeRegressor(max_depth=4)
            difficulty_predictor = RandomForestRegressor(max_depth=4, n_estimators=100, random_state=42)
            difficulty_predictor.fit(pre_features_train, difficulty)
            print(difficulty_predictor.feature_importances_ )

        elif mode == 4:
            predict_model = get_predict_model()
            predict_model.fit(train_x, train_y)
            pred_y, pred_y_with_prob = get_pred_y(train_x, predict_model)
            feature_importances = np.linalg.norm(predict_model.best_estimator_.feature_importances_)

            confidence_for_features = [feature_importances * (1 - (abs(np.array(x)-y) / 2)) for x, y in zip(train_x, train_y)]

            # difficulty = [abs(p-l) for p, l in zip(pred_y_with_prob, train_y)]
            # pre_features_num = len(pre_features[0])
            # dealt_pre_features = [[*pf, (d - 1) * (pre_features_num ** 0.5)] for pf, d in zip(pre_features[:i] + pre_features[end_i:size], difficulty)]

            # 聚类
            classes_num = 4
            # cl = DBSCAN(eps=0.2, min_samples=5)
            cl = KMeans(n_clusters=classes_num)
            cl.fit(confidence_for_features)
            cl_labels = cl.labels_
            print(Counter(cl_labels))
            centroids = cl.cluster_centers_

            confidences_x = []
            for centroid in centroids:
                distances = []
                for point in confidence_for_features:
                    distance = np.linalg.norm(np.array(point) - centroid)
                    distances.append(distance)
                confidences_x.append(list(1 / (np.array(distances) + 0.5)))

            # print('confidences', confidences_x)

            predict_models = []
            for j in range(classes_num):
                predict_model = get_predict_model()
                predict_model.fit(train_x, train_y, sample_weight=confidences_x[j])
                # predict_model.fit(train_x, train_y)
                if "best_params_" in dir(predict_model): 
                    print("Best parameters found: ", predict_model.best_params_)
                predict_models.append(predict_model)


            classes_predictor = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
            classes_predictor.fit(pre_features_train, cl_labels)

        elif mode == 5:
            classes_num = 2
            
            predict_models = []
            for j in range(classes_num):
                predict_model = get_predict_model()
                predict_model.fit(train_x, train_y, sample_weight=[int(pre_features_train[i][pre_feature_index] == j) for i in range(len(train_x))])
                # predict_model.fit(train_x, train_y)
                if "best_params_" in dir(predict_model): 
                    print("Best parameters found: ", predict_model.best_params_)
                predict_models.append(predict_model)

        elif mode == 6:
            pre_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                max_features=6
            )
            pre_model.fit(train_x, train_y)
            _, pred_y_with_prob = get_pred_y(train_x, pre_model)
            


            deep_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                max_features=50
            )
            confidence_threshold = 0.5
            deep_model.fit(train_x, train_y, [10 if abs(x) < confidence_threshold else 1 for x in pred_y_with_prob])
        


        elif mode == 2:
            predict_models = []
            for j in range(classes_num):
                confidences_x = confidences[j][:i] + confidences[j][end_i:size]
                
                # predict_model=tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
                predict_model = get_predict_model()
                # predict_model = RFE(predict_model, n_features_to_select=8, step=1)

                # if True:
                #     selector = SelectKBest(score_func=mutual_info_classif, k=70)
                #     X_new = selector.fit_transform(train_x, train_y)
                #     selected_indices = selector.get_support(indices=True)
                #     def mask(x):
                #         return [x[i] for i in selected_indices]
                #     train_x = [mask(x) for x in train_x]
                #     test_x = [mask(x) for x in test_x]


                predict_model.fit(train_x, train_y, sample_weight=confidences_x)
                if "best_params_" in dir(predict_model): 
                    print("Best parameters found: ", predict_model.best_params_)
                predict_models.append(predict_model)



        # print(predict_model.ranking_)

        # feature_importances = predict_model.best_estimator_.feature_importances_ED
        # feature_importances_list.append(feature_importances)
        
        # pred_y = predict_model.predict(test_x)

        if mode == 3 or mode == 4 or mode == 5 or mode == 6:
            if mode == 3:
                pre_features_test = pre_features[i:end_i]
                difficulty_test = difficulty_predictor.predict(pre_features_test)
                def get_diff_class(d):
                    for i in range(classes_num):
                        if d < diff_classes[i+1]:
                            return i
                classes_test = np.array(list(map(get_diff_class, difficulty_test)))
            elif mode == 4:
                pre_features_test = pre_features[i:end_i]
                classes_test = classes_predictor.predict(pre_features_test)
            elif mode == 5:
                pre_features_test = pre_features[i:end_i]
                classes_test = np.array([x[pre_feature_index] for x in pre_features_test])
            elif mode == 6:
                _, pred_y_with_prob = get_pred_y(test_x, pre_model)
                classes_test = np.array([1 if x < confidence_threshold else 0 for x in pred_y_with_prob])
                predict_models = [pre_model, deep_model]


            print(classes_test)

            test_x = np.array(test_x)
            pred_y, pred_y_with_prob = [0] * len(test_x), [0] * len(test_x)
            # for j in range(classes_num):
            for j in range(len(predict_models)):
                predict_model = predict_models[j]
                mask = classes_test == j
                if mask.any():
                    t_pred_y, t_pred_y_with_prob = get_pred_y(test_x[mask], predict_model)
                    t_pred_y, t_pred_y_with_prob = iter(t_pred_y), iter(t_pred_y_with_prob)
                    for k in range(len(mask)):
                        if mask[k]:
                            pred_y[k] = next(t_pred_y)
                            pred_y_with_prob[k] = next(t_pred_y_with_prob)

        elif mode == 2:
            test_x = np.array(test_x)
            pred_y, pred_y_with_prob = [0] * len(test_x), [0] * len(test_x)
            for j in range(classes_num):
                predict_model = predict_models[j]
                mask = cl_labels[i:end_i] == j
                t_pred_y, t_pred_y_with_prob = get_pred_y(test_x[mask], predict_model)
                t_pred_y, t_pred_y_with_prob = iter(t_pred_y), iter(t_pred_y_with_prob)
                for k in range(len(mask)):
                    if mask[k]:
                        pred_y[k] = next(t_pred_y)
                        pred_y_with_prob[k] = next(t_pred_y_with_prob)
            



        # print(pred_y)
        # print(pred_y_with_prob)

        all_pred_y += list(pred_y)
        all_pred_y_with_prob += list(pred_y_with_prob)



    def show(pred_y, test_y, index_list=None):
        if index_list is not None:
            pred_y = [pred_y[i] for i in index_list]
            test_y = [test_y[i] for i in index_list]

        size = len(test_y)
        if size == 0:
            return

        cot = 0
        total_deta = 0
        in_1 = 0

        # spearmanr_list = []
        # kendalltau_list = []
        # cot_t = 0

        for p, t in zip(pred_y, test_y):
            # print(p,t)
            deta = abs(p - t)
            total_deta += deta
            if deta == 0:
                cot += 1
            if deta <= 1:
                in_1 += 1
        
        # print(pred_y)
        # print(test_y)
        # spearmanr_list.append(spearmanr(pred_y, test_y)[0])
        # kendalltau_list.append(kendalltau(pred_y, test_y)[0])


        # print(feature_importances_list)
        # print(zip(*feature_importances_list))
        # feature_importances = [(sum(x)/len(x)) for x in zip(*feature_importances_list)]

        print(f'\tPred cot: {cot/size:.3f}')
        print(f'\tPred deta<=1: {in_1/size:.3f}')
        print(f'\tPred MAE: {total_deta/size:.3f}')
        print(f'\tPred spearmanr: {spearmanr(pred_y, test_y)[0]:.3f}')
        print(f'\tPred kendalltau: {kendalltau(pred_y, test_y)[0]:.3f}')
        print()
        # print(f'\tPred spearmanr: {sum(spearmanr_list)/len(spearmanr_list):.3f}')
        # print(f'\tPred kendalltau: {sum(kendalltau_list)/len(kendalltau_list):.3f}')
        # print(f'\tPred feature importances: {feature_importances}')

    show(all_pred_y, labels)
    for m in [0.8, 0.7, 0.6, 0.5]:
        indexs = [i for i, x in enumerate(all_pred_y_with_prob) if abs(x) >= m]
        print("\t>=", m, f'size: {len(indexs)/size:.3f}')
        show(all_pred_y, labels, indexs)
    # i, _ = min(enumerate(feature_importances), key=lambda x: x[1])
    # print(f'remove {i}')
    # features = [x[:i] + x[i+1:] for x in features]
    return all_pred_y


def deal_none(probs):
    c = Counter([x[0] for x in probs])
    del c[None]
    z = c.most_common(1)[0][0]
    return [[z] if x[0] is None else x for x in probs]

class Summary:
    def print_summary(self, dataset, metric_names=None, feature_names=[], metric_selector=None):
        dataset = dataset.content
        if metric_names is None:
            metric_names = dataset[0]['metrics'].keys()
        if metric_selector is not None:
            metric_names = [x for x in metric_names if metric_selector(x)]

        metrics = {}

        labels = []
        annos = []
        baselines = []

        for x in dataset:
            labels.append(x['manual_score'])
            annos.append(x.get('anno'))
            baselines.append([])

        for metric in metric_names:
            probs = []

            has_metric = False  # 是否至少有一个case存在值
            for x in dataset:
                v = x['metrics'].get(metric)
                if v is not None:
                    has_metric = True
                if not isinstance(v, str) and isinstance(v, Iterable):
                    probs.append(v)
                else:
                    probs.append([v])

            if not has_metric:
                continue

            # 处理空值
            probs = deal_none(probs)

            if metric.startswith('baseline'):
                for i in range(len(baselines)):
                    baselines[i] += probs[i]
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
        def get_pred_model():
            

            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            param_grid = {
                # 'min_samples_split': [2, 5, 8],
                # 'max_depth': [4, 5, 10, 15, 20],
                # 'min_samples_leaf': [1, 3, 5],
                # 'max_features': [20, 30, 50, 'sqrt', 'log2'],

                'max_depth': [5, 10],
                # 'n_estimators': [50, 100, 150],
                'max_features': [10, 20, 50, 500],
            }
            
            grid_search = GridSearchCV(rf, param_grid=param_grid, cv=4, n_jobs=-1)
            return grid_search
        
        if len(metrics) > 1:
            all_pred_y = pred(metrics, labels, get_predict_model=get_pred_model, batch_percent=0.25)
        
            annos_map = {}
            annos_map_count = {}
            baselines_num = len(baselines[0])
            for i in range(len(labels)):
                if annos[i] is not None:
                    annos_map.setdefault(annos[i], [0] * (1 + baselines_num))
                    annos_map_count.setdefault(annos[i], 0)
                    annos_map_count[annos[i]] += 1
                    if labels[i] == all_pred_y[i]:
                        annos_map[annos[i]][0] += 1
                    for j in range(baselines_num):
                        if labels[i] == baselines[i][j]:
                            annos_map[annos[i]][j+1] += 1

            # print('bad case:', [i for i in range(len(labels)) if abs(labels[i] - all_pred_y[i]) > 1])

            if annos_map:
                print('anno_map')
                for k, v in annos_map.items():
                    print(f'\t{k}: ' + ', '.join([f'{x/annos_map_count[k]:.3f}' for x in v]))
