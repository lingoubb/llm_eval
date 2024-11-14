from . import with_llm, normal
from concurrent.futures import ThreadPoolExecutor


class Judge(normal.Judge):
    def __init__(self, network, overwrite=False):
        self.network = network
        self.overwrite = overwrite


    def get_score(self, case):
        
        case.setdefault('metrics', {})
        case.setdefault('metrics_raw', {})
        
        all_score = []
        layer_score = []
        for layer in self.network:
            layer_score = []
            fs = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for cell in layer:
                    if not self.overwrite and cell.name in case['metrics_raw']:
                        f = pool.submit(lambda: case['metrics_raw'][cell.name])
                    else:
                        f = pool.submit(cell.forward, case, all_score)
                    fs.append((cell, f))
                for cell, f in fs:
                    r = f.result()
                    case['metrics_raw'][cell.name] = r
                    r = cell.deal_with_score(r)
                    case['metrics'][cell.name] = r
                    layer_score.append(r)
            all_score.append(layer_score)

        return None

    
class Cell:
    '''
    self.fill_prompt(case, all_score)
    '''
    def __init__(self, model, name, system_prompt, fill_prompt, deal_with_score=lambda x: x):
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.fill_prompt = fill_prompt
        self.deal_with_score = deal_with_score


    def forward(self, case, all_score):

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.fill_prompt(case, all_score)},
        ]
        r = self.model.get_outputs([prompt])[0]
        c = r.message.content

        return c

    
    def backward(self):
        return