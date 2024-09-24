def inc(features, labels):
    pass

class Summary:
    def print_summary(self, dataset):
        metric_names = dataset[0]['metrics'].keys()
        for metric in metric_names:
            probs = []
            labels = []
            for x in dataset:
                probs.append(x['metrics'][metric])
                labels.append(x['manual_score'])
            
        
