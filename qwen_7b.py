import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

from transformers import AutoTokenizer, AutoModelForCausalLM

# 本地模型路径
model_path = r"D:\workspace\models\qwen-7b-instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)

# 示例输入
input_text = "你是谁"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成文本
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# 解码生成的词元为文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}")