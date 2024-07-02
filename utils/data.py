from datasets import load_dataset
from dataclasses import dataclass

import os
import random

@dataclass
class DatasetInfo:
    path: str
    name: str
    exemple: str
    eval: str

def get_dataset_info(dataset):
    """
    根据数据集名返回数据集信息"""
    return {
        "ac": DatasetInfo(
            path="ai2_arc",
            name="ARC-Challenge",
            exemple="train",
            eval="test",
        )
    }[dataset]

def load_data(dataset):
    """
    加载数据集"""
    os.makedirs("data/", exist_ok=True)
    dataset_info = get_dataset_info(dataset)
    print(f"Loading data for {dataset_info.name}")
    data = load_dataset(dataset_info.path, dataset_info.name, 
                        cache_dir="data/")
    return data[dataset_info.exemple], data[dataset_info.eval]

def prep_data(dataset, k_shots):
    """
    获取数据集中所有问题"""
    exp_data, eval_questions = load_data(dataset)
    exp_questions = []
    print(f"Questions Count: {len(eval_questions)}")

    if k_shots == 0:
        exp_questions = [{} for _ in range(len(eval_questions))]
    else:
        random.seed(42)
        for _ in range(len(eval_questions)): 
            idxs = random.sample(range(len(exp_data)), k_shots)
            exp_questions.append([exp_data[idx] for idx in idxs])

    return eval_questions, exp_questions