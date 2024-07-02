from argparse import ArgumentParser
from eval import get_model
from utils.data import prep_data
from tqdm import tqdm
from utils.prompt import choice2idx
from utils.acc import calc_mcp_acc, calc_cp_acc

import json
import os

def get_args():
    """
    获取待评估模型名、数据集名以及评估方法"""
    parser = ArgumentParser(description='llm evaluation')
    parser.add_argument('--model', type=str, default='qwen2-0.5B-ins', help='model name')
    parser.add_argument('--dataset', type=str, default='ac', help='dataset name')
    parser.add_argument('--k_shots', type=int, default=0, help='number of shots')
    parser.add_argument('--mcp', action='store_true', help='use mcp strategy')
    parser.add_argument('--cp', action='store_true', help='use cp stategy')
    parser.add_argument('--que_idx', type=int, default=0, help='index of question to debug')

    return parser.parse_args()


def run_debug(args):
    """
    进行评估"""

    # 加载数据和模型
    questions, exmples = prep_data(args.dataset, args.k_shots)
    que = questions[args.que_idx], exps = exmples[args.que_idx]
    model= get_model(args.model)

    # 评估并保存结果
    if args.mcp:
        pred_ans = model.mcp_eval(que, exps)
    elif args.cp:
        pred_ans = model.cp_eval(que, exps)
    else:
        raise ValueError("Please choose a strategy to evaluate the model.")
    

if __name__ == "__main__":
    args = get_args()
    run_debug(args)