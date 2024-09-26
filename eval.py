import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import json
from tqdm import tqdm

from tools.log import Logger
from tools.dataset import Case, Dataset
from tools import tools

logger = Logger('DEBUG')

def save_result(dataset):
    with open(dataset.path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset.content, indent=1, ensure_ascii=False, default=vars))


def gen_output(prompt, dataset, model, t=16):
    try:
        with ThreadPoolExecutor(max_workers=t) as pool:
            fs = []
            for c in dataset:
                if c['output'] is None:
                    fs.append((pool.submit(prompt.get_output, model, c), c))
            for f, c in tqdm(fs, total=len(fs)):
                try:
                    c.set_output(f.result(timeout=15))
                    c.set_err(None)
                except Exception as e:
                    c.set_err(str(e))
    finally:
        save_result(dataset)


def gen_score(judge, dataset, t=16):
    try:
        with ThreadPoolExecutor(max_workers=t) as pool:
            fs = []
            for c in dataset:
                if c['score'] is None:
                    fs.append((pool.submit(judge.get_score, c), c))
                # fs.append((pool.submit(judge.get_score, c), c))
            for f, c in tqdm(fs, total=len(fs)):
                try:
                    c['score'] = f.result()
                    c.set_err(None)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    c.set_err(str(e))
    finally:
        save_result(dataset)


def load_result(dataset_path, target_path):
    # 创建结果保存目录
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    # 拷贝数据集
    f_list = os.listdir(target_path)
    for fname in os.listdir(dataset_path):
        if fname not in f_list:
            print(f'Create: {fname}')
            shutil.copy(os.path.join(dataset_path, fname), os.path.join(target_path, fname))

    # 解析数据集
    datasets = {}
    for fname in os.listdir(target_path): 
        try:
            fp = os.path.join(target_path, fname)
            with open(fp, encoding='utf-8') as f:
                raw = json.loads(f.read())
                dataset = [Case(**x) for x in raw]
                datasets[fname] = Dataset(fname, dataset, fp)
        except Exception as e:
            print(f'Load failed: {fname}, {e}')

    return [datasets[k] for k in sorted(datasets.keys())]


def run_eval(models, judges, prompts, datasets, summary):

    for model in models:
        with model:
            for judge in judges:
                with judge:
                    for prompt in prompts:
                        for dataset in datasets:
                            gen_output(prompt, dataset, model)
                            gen_score(judge, dataset)
    
    summary.print_summary(dataset)
                    

# def run_eval(model, judge, prompt, datasets, summary):
#     with model:
#         with judge:
#             for dataset in datasets:
#                 try:
#                     gen_output(prompt, dataset, model)
#                 finally:
#                     save_result(dataset)
#                 try:
#                     gen_score(judge, dataset)
#                 finally:
#                     save_result(dataset)
    
#     summary.print_summary(dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', nargs='*', action='append', help='数据集配置')
    parser.add_argument('-m', '--model', nargs='*', action='append', help='模型配置')
    parser.add_argument('-p', '--prompt', nargs='*', action='append', help='模型输出方案配置')
    parser.add_argument('-j', '--judge', nargs='*', action='append', help='打分方案配置')
    parser.add_argument('-t', '--target', help='结果保存目录')
    args = parser.parse_args()
    return args


def main():
    # args parse
    args = parse_args()

    # load model
    models = []
    for x in args.model:
        models.append(tools.load_config('Model', *x))

    # load judge
    judges = []
    for x in args.judge:
        judges.append(tools.load_config('Judge', *x))

    # load prompt
    prompts = []
    for x in args.prompt:
        prompts.append(tools.load_config('Prompt', *x))

    # load dataset
    # datasets = []
    # for x in args.dataset:
    #     datasets.append(load_config('Dataset', *x))
    datasets = []
    for x in args.dataset:
        datasets += load_result(x, args.target)

    # run eval
    run_eval(models, judges, prompts, datasets, None)

    return

if __name__ == '__main__':
    main() 