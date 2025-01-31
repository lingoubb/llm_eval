def get_pred_score(features, labels, model_func, score_func, batch_percent=0.2):

    size = len(labels)
    batch_size = math.ceil(batch_percent * size)

    score_list = []

    for i in range(0, size, batch_size):

        end_i = min(i + batch_size, size)

        test_x = features[i:end_i]
        test_y = labels[i:end_i]
        train_x = features[:i] + features[end_i:size]
        train_y = labels[:i] + labels[end_i:size]

        predict_model = model_func()
        predict_model.fit(train_x, train_y)
        pred_y = predict_model.predict(test_x)

        score = score_func(pred_y, test_y)
        score.append(score_list)

    return sum(score_list)/len(score_list)


def select_new_features(features, new_features, labels, model_func, score_func):
    n = len(features)
    best_i, best_score = None, None
    for i, new_feature in enumerate(new_features):
        new_feature_t = [features[i] + [new_feature[i]] for i in range(n)]
        score = get_pred_score(new_feature_t, labelsm, model_func, score_func)
        if best_score is None or score > best_score:
            best_score = score
            best_i = i
    return best_i, best_score


def forward_select(features, labels, model_funcs, score_func):
    n = len(features)
    now_features = [[] for _ in range(n)]
    features_pool = features
    while True:
        best_i, best_score = select_new_features(now_features, )
    pass