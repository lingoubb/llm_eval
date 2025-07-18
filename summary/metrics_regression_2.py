import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau
import os
from joblib import dump
from collections import Counter

splits = 5
random_seed = 42

def extract_features(sample, baseline_features, all_features):
    metrics = sample.get("metrics", {})
    row = {"manual_score": sample["manual_score"]}

    for key, value in metrics.items():
        if key.startswith("baseline"):
            row[key] = value
            baseline_features.add(key)
        row[key] = value
        all_features.add(key)  # 使用集合自动去重

    return row

def calculate_metrics(predictions, true_values):
    accuracy = accuracy_score(true_values, predictions)
    spearman_corr, _ = spearmanr(true_values, predictions)
    kendall_corr, _ = kendalltau(true_values, predictions)
    precision = precision_score(true_values, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_values, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_values, predictions, average="weighted", zero_division=0)
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    return accuracy, spearman_corr, kendall_corr, precision, recall, f1, mae, mse

def split_features(names, values, cond):
    out_indexs, left_indexs = [], []
    out_names, left_names = [], []
    for i, x in enumerate(names):
        if cond(x):
            out_indexs.append(i)
            out_names.append(x)
        else:
            left_indexs.append(i)
            left_names.append(x)
    out_values = values[:, out_indexs]
    left_values = values[:, left_indexs]
    return out_names, out_values, out_indexs, left_names, left_values, left_indexs


def get_pred_y(test_x, predict_model):
    if len(test_x) == 0:
        return [], []
    pred_y_prob = predict_model.predict_proba(test_x)
    class_labels = predict_model.classes_
    pred_y = [class_labels[max(enumerate(x), key=lambda x: x[1])[0]] for x in pred_y_prob]
    pred_y_with_prob = [sum([label * prob for label, prob in zip(class_labels, x)]) for x in pred_y_prob]
    return pred_y, pred_y_with_prob

class DefaultPred:
    def __init__(self, **args):
        self.args = args
    def get_predictor(self, features_name, train_set, labels, train_weight):
        """
        feature_names 特征名称列表
        train_set 训练集特征矩阵
        labels 训练集标签列表
        train_weight 训练集样本权重列表
        test_set 测试集特征矩阵
        """
        args = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            **self.args
        }
        model = RandomForestClassifier(
            **args
        )

        model.fit(train_set, labels, sample_weight=train_weight)
        return model
    
    
    def pred(self, features_name, train_set, labels, train_weight, test_set, test_labels):

        model = self.get_predictor(features_name, train_set, labels, train_weight)
        pred_labels = model.predict(test_set)

        # accuracy = accuracy_score(test_labels, pred_labels)
        # precision = precision_score(test_labels, pred_labels, average='weighted', zero_division=0)
        # recall = recall_score(test_labels, pred_labels, average='weighted', zero_division=0)
        # f1 = f1_score(test_labels, pred_labels, average='weighted', zero_division=0)
        # spearman_corr, _ = spearmanr(test_labels, pred_labels)
        # kendall_corr, _ = kendalltau(test_labels, pred_labels)

        # print(f"Evaluation on the combined dataset:")
        # print(f"Random Forest (trained on all datasets):")
        # print(f"Accuracy: {accuracy:.4f} | Spearman: {spearman_corr:.4f} | Kendalltau: {kendall_corr:.4f}")
        # print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        return pred_labels

class PredWithoutFeatures:
    def pred(self, features_name, train_set, labels, train_weight, test_set, test_labels):
        def is_not_feature(name):
            return 'feature' not in name
        new_features_name, train_set, _, _, _, _ = split_features(features_name, train_set, is_not_feature)
        _, test_set, _, _, _, _ = split_features(features_name, test_set, is_not_feature)
        return DefaultPred().pred(new_features_name, train_set, labels, train_weight, test_set, test_labels)
    
class ExpertPred_V1:
    def pred(self, features_name, train_set, labels, train_weight, test_set, test_labels):
        residual = True

        def is_feature(name):
            return 'feature' in name
        if residual:
            pre_features_name, pre_train_set, _, _, left_train_set, _ = split_features(features_name, train_set, is_feature)
            _, pre_test_set, _, _, _, _ = split_features(features_name, test_set, is_feature)
        else:
            pre_features_name, pre_train_set, _, _, train_set, _ = split_features(features_name, train_set, is_feature)
            _, pre_test_set, _, features_name, test_set, _ = split_features(features_name, test_set, is_feature)
            left_train_set = train_set

        def classify(x):
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=50) 
            r = dbscan.fit_predict(x)
            return r, set(r)
        difficulty_train_set = [abs(x - labels[i]) for i, x in enumerate(left_train_set)]
        train_set_classes, class_set = classify(difficulty_train_set)
        classify_model = DefaultPred().get_predictor(pre_features_name, pre_train_set, train_set_classes, None)

        
        # pre_judge_model = DefaultPred().get_predictor(features_name, train_set, labels, None)
        # pre_judge_labels = pre_judge_model.predict(test_set)
        # difficulty_test_set = [abs(x - pre_judge_labels[i]) for i, x in enumerate(test_set)]
        test_set_classes = classify_model.predict(pre_test_set)
        print(Counter(train_set_classes))
        print(Counter(test_set_classes))

        pred_models = []
        pred_labels = np.zeros(len(test_set))
        for i in class_set:
            mask = (train_set_classes==i)
            # pred_model = DefaultPred().get_predictor(features_name, train_set[mask], labels[mask], None)
            pred_model = DefaultPred().get_predictor(features_name, train_set, labels, mask * 3 + 1)

            mask = (test_set_classes==i)
            if mask.any():
                pred_labels[mask] = pred_model.predict(test_set[mask])

        return pred_labels
    

class ExpertPred_V2:
    def __init__(self, pre_feature_num=6, threshold=None):
        self.threshold = threshold
        self.pre_feature_num = pre_feature_num
    def pred(self, features_name, train_set, labels, train_weight, test_set, test_labels):
        def is_feature(name):
            return 'feature' in name
        pre_features_name, pre_train_set, pre_indexs, left_features_name, left_train_set, left_indexs = split_features(features_name, train_set, is_feature)
        _, pre_test_set, _, _, left_test_set, _ = split_features(features_name, test_set, is_feature)

        pre_model = DefaultPred().get_predictor(features_name, train_set, labels, None)
        feature_importances = pre_model.feature_importances_
        # print(list(feature_importances))
        # feature_importances[pre_indexs]
        
        # 重要性前6名
        pre_feature_num = self.pre_feature_num
        if pre_feature_num > 0:
            indices = np.argsort(feature_importances[left_indexs])[-pre_feature_num:]
            
            # 第一轮预测用到的特征
            all_pre_indexs = sorted(pre_indexs + list(np.array(left_indexs)[indices]))
        else:
            all_pre_indexs = pre_indexs

        feature_importances = feature_importances[all_pre_indexs]
        
        pre_model = DefaultPred().get_predictor(np.array(features_name)[all_pre_indexs], train_set[:, all_pre_indexs], labels, None)
        pre_pred, probs = get_pred_y(test_set[:, all_pre_indexs], pre_model)
        probs = np.array(probs)
        # print(np.sum(probs>0.9), np.sum(probs>0.8), np.sum(probs>0.5), np.sum(probs>0))
        
        # 聚类
        os.environ['OMP_NUM_THREADS'] = '4' 
        from sklearn.cluster import KMeans
        classes_num = 5
        pre_x = train_set[:,all_pre_indexs] * feature_importances
        pre_x_test = test_set[:,all_pre_indexs] * feature_importances

        cl = KMeans(n_clusters=classes_num)
        cl.fit(np.vstack((pre_x, pre_x_test)))
        cl_labels = cl.labels_
        print(Counter(cl_labels))
        centroids = cl.cluster_centers_

        # confidences_x = []
        # for centroid in centroids:
        #     distances = []
        #     for point in pre_x:
        #         distance = np.linalg.norm(np.array(point) - centroid)
        #         distances.append(distance)
        #     confidences_x.append(list(1 / (np.array(distances) + 0.5)))

        final_pre_models = []
        pred_labels = np.zeros(len(test_set))
        for i in range(classes_num):
            model = DefaultPred().get_predictor(features_name, train_set, labels, 1 / (np.linalg.norm(pre_x - centroids[i], axis=1) + 0.0001))
            final_pre_models.append(model)
            
            mask = (cl_labels[len(train_set):] == i)
            if mask.any():
                pred_labels[mask] = model.predict(test_set[mask])
        
        if self.threshold is not None:
            easy = (probs > self.threshold)
            print(f'easy num: {np.sum(easy)}/{len(easy)}')
            pred_labels[easy] = np.array(pre_pred)[easy]

        return pred_labels



def cross_validate(data, pred_cls):

    print(f'{pred_cls.__class__.__name__}')
    # 初始化总的数据集
    all_data = []
    all_baseline_features = set()
    all_features = set()

    for sample in data:
        all_data.append(extract_features(sample, all_baseline_features, all_features))

    # 生成固定的特征顺序
    feature_cols = sorted(list(all_features))
    all_baseline_features = sorted(list(all_baseline_features))
    # print(feature_cols)

    # 创建总的 DataFrame
    df = pd.DataFrame(all_data, columns=["manual_score"] + feature_cols)



    # 最大值归一化
    tmp = df[feature_cols].abs().max()
    tmp[tmp==0] = 1
    df[feature_cols] = df[feature_cols] / tmp
    # 处理缺失值
    df[feature_cols] = df[feature_cols].interpolate(method='linear', axis=0)
    df = df.fillna(df.mean()) # 再次填充剩余的 NaN 值
    # df.fillna(0) # 全NaN

    # 准备数据
    # unique_values = sorted(df["manual_score"].unique())
    # y = pd.Categorical(df["manual_score"], categories=unique_values).codes
    y = df["manual_score"]
    X = df[feature_cols].values

    # print(X.tolist())
    # print(y.tolist())

    # 交叉验证
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_seed)
    y_pred_all = []
    y_true_all = []


    fold = 1
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_weight = None

        pred = pred_cls.pred
        # ✅ 使用封装好的函数进行训练和预测
        y_pred = pred(
            features_name=feature_cols,
            train_set=X_train,
            labels=y_train,
            train_weight=train_weight,
            test_set=X_test,
            test_labels=y_test
        )

        # 收集预测结果
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)
        fold += 1

    accuracy, spearman, kendall, precision, recall, f1, mae, mse = calculate_metrics(
        y_pred_all, y_true_all
    )

    print(f"\tOurs:")
    print(f"\t\tAccuracy: {accuracy:.4f} | Spearman: {spearman:.4f} | Kendalltau: {kendall:.4f}")
    print(f"\t\tPrecision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"\t\tMAE: {mae:.4f} | MSE: {mse:.4f}")

    # Baseline evaluation on the combined dataset
    y_true = df["manual_score"].values
    for baseline_feature in all_baseline_features:
        if baseline_feature in df.columns:
            baseline_X = df[baseline_feature].fillna(0).values
            baseline_pred = np.round(baseline_X).astype(int)

            # Align lengths
            min_len = min(len(baseline_pred), len(y_true))
            baseline_pred = baseline_pred[:min_len]
            y_true_tmp = y_true[:min_len]

            # Calculate metrics
            accuracy, spearman, kendall, precision, recall, f1, mae, mse = calculate_metrics(
                baseline_pred, y_true_tmp
            )

            print(f"\tBaseline '{baseline_feature}':")
            print(f"\t\tAccuracy: {accuracy:.4f} | Spearman: {spearman:.4f} | Kendall: {kendall:.4f}")
            print(f"\t\tPrecision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"\t\tMAE: {mae:.4f} | MSE: {mse:.4f}")

    print()



class Summary:
    def print_summary(self, dataset, metric_names=None, feature_names=[], metric_selector=None):
        for pred_cls in [ExpertPred_V2(threshold=0.75), ExpertPred_V2(pre_feature_num=0, threshold=0.75)]:
            cross_validate(dataset.content, pred_cls=pred_cls)