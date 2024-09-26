def inc(features, labels):
    size = len(features)
    cot = 0
    for i in range(size):
        feature = features[i]
        j = max(enumerate(feature), key=lambda x: x[1])[0]
        score = {0: 1, 1: -1, 2: 0}[j]
        if score == labels[i]:
            cot += 1
    print(f'corret: {cot/size:.3f}')

class Summary:
    def print_summary(self, dataset):
        dataset = dataset.content
        metric_names = dataset[0]['metrics'].keys()
        for metric in metric_names:
            print(metric)
            probs = []
            labels = []
            for x in dataset:
                probs.append(x['metrics'][metric])
                labels.append(x['manual_score'])
            inc(probs, labels)
            print('-' * 20)
            
        
