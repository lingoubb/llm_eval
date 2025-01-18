from eval import load_result, gen_score
from model import deepseek, gpt_3
from judge import compare_judgelm, judge_probs, judge_summ, judge_summ_new, judge_with_features
import summary.metrics_regression


# datasets = load_result('dataset/summeval', f'output/summeval/new')

a = {   "metric_accuracy": 1,
   "metric_relevant": 1,
   "metric_faithful": 1,
   "metric_detailed": 1,
   "metric_redundant": 1}.keys()

# a = {      "metric_gen_1_0": -1,
#    "metric_gen_1_1": -1,
#    "metric_gen_1_2": -1,
#    "metric_gen_1_3": 1,
#    "metric_gen_1_4": -1,
#    "metric_gen_1_5": -1,
#    "metric_gen_1_6": 1,
#    "metric_gen_1_7": -1,
#    "metric_gen_1_8": -1,
#    "metric_gen_1_9": -1}.keys()

def f(datasets):
    for dataset in datasets:
        gen_score(judge_with_features.Judge(deepseek.Model()), dataset)
        # gen_score(judge_summ_new.Judge(deepseek.Model()), dataset)
        summary.metrics_regression.Summary().print_summary(dataset)


# f(load_result('dataset/pandalm', f'output/1226/pandalm'))
# f(load_result('dataset/arena_1000', f'output/1226/arena_1000')) 
f(load_result('dataset/mt_bench', f'output/1226/mt_bench'))

# CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server --model /NVME1/elecLLM/models/qwen/Qwen2-7B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.9  --served-model-name default  --block-size 16 --port 9901
