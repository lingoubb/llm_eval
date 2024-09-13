import json

class Case(dict):
    def __init__(self, **kargs):
        for k, v in kargs.items():
            self[k] = v


    def __init_subclass__(cls):
        return super().__init_subclass__()
    

    def __getitem__(self, key):
        return super().get(key)
        

    def selections_format(self):
        r = ''
        t = 'A'
        for selection in self.selections:
            r += f'{t}. {selection}\n'
            t = chr(ord(t) + 1)
        return r


    def to_json(self):
        return json.dumps(vars(self), indent=1, ensure_ascii=False)


    @classmethod
    def list_to_json(cls, case_list):
        return json.dumps(case_list, indent=1, ensure_ascii=False, default=vars)


    @classmethod
    def list_from_json(cls, j):
        raw = json.loads(j)
        return [Case(**x) for x in raw]


    """
    设置模型对该 case 的回答
    """
    def set_output(self, output):
        self['output'] = output


    """
    标记该 case 在评测过程中出现了异常
    """
    def set_err(self, err):
        self['err'] = err


    def is_available(self):
        return self['err'] is None
    

class Dataset:
    def __init__(self, name, content, path):
        self.content = content
        self.name = name
        self.path = path

    
    def __iter__(self):
        return iter(self.content)
    

    def __len__(self):
        return len(self.content)