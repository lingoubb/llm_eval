class Summary:
    def print_summary(self, dataset):
        available = 0
        cot = 0
        for x in dataset:
            if x['score'] in [-1, 0, 1]:
                available += 1
                if x['score'] == x['manual_score']:
                    cot += 1
        print(f'{dataset.name}\n\tavailable: {available}/{len(dataset)} corret: {cot/len(dataset)*100:.1f}%')