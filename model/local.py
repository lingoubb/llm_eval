import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    def __init__(self, model_path, kargs={}):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.kargs = kargs 


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def get_outputs(self, inputs, temperature=0, max_tokens=1024, logprobs=False, text=False, kargs={}):

        outputs = []

        for input_text in inputs:

            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

            # 生成文本
            output = self.model.generate(input_ids, max_length=max_tokens, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

            # 解码生成的词元为文本
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            outputs.append(generated_text)

        return outputs
    