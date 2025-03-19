from model import openai_api
from summary import metrics_regression
from eval import load_result

datasets = load_result(None, '数据集目录')
for dataset in datasets:
    metrics_regression.Summary().print_summary(dataset)
