from sklearn.metrics import precision_recall_fscore_support

class Summary:
    def print_summary(self, dataset):
        total = len(dataset)  # case 总数
        available = 0  # 合法格式的 case 数
        cot = 0  # 回答正确的 case 数
        for x in dataset:
            if x['score'] in [-1, 0, 1]:
                available += 1
                if x['score'] == x['manual_score']:
                    cot += 1
        micro_f1 = precision_recall_fscore_support([x['manual_score'] for x in dataset], [x['score'] for x in dataset], average='micro')
        macro_f1 = precision_recall_fscore_support([x['manual_score'] for x in dataset], [x['score'] for x in dataset], average='macro')
        ret = [
            ('available', f'{available}/{len(dataset)}'),
            ('accuracy', f'{cot/len(dataset)*100:.1f}%'),
            ('micro-precision', f'{micro_f1[0]:.3f}'),
            ('micro-recall', f'{micro_f1[1]:.3f}'),
            ('micro-f1', f'{micro_f1[2]:.3f}'),
            ('macro-precision', f'{macro_f1[0]:.3f}'),
            ('macro-recall', f'{macro_f1[1]:.3f}'),
            ('macro-f1', f'{macro_f1[2]:.3f}'),
        ]
        print(f'{dataset.name}')
        for k, v in ret:
            print(f'\t{k}: {v}')
