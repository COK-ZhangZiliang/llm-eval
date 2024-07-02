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

    return parser.parse_args()


def run_exp(args):
    """
    进行评估"""
    # 创建结果文件
    strategy = "mcp" if args.mcp else "cp"
    os.makedirs(f"results/{args.model}", exist_ok=True)
    results_path = f"results/{args.model}/{args.dataset}-{strategy}-{args.k_shots}.json"
    if os.path.exists(results_path): # 如果已经评估过，直接返回
        return

    # 加载数据和模型
    questions, exmples = prep_data(args.dataset, args.k_shots)
    model= get_model(args.model)

    # 评估并保存结果
    with open(results_path, "w", encoding='utf-8') as f:
        for que, exps in zip(tqdm(questions), exmples):
            if args.mcp:
                pred_ans = model.mcp_eval(que, exps)
            elif args.cp:
                pred_ans = model.cp_eval(que, exps)
            else:
                raise ValueError("Please choose a strategy to evaluate the model.")
            json.dump({
                'Ans': choice2idx(que['answerKey']), 
                'Pred': pred_ans
                }, f, ensure_ascii=False)
            f.write("\n")   


if __name__ == "__main__":
    args = get_args()
    run_exp(args)
    if args.mcp:
        acc = calc_mcp_acc(f"results/{args.model}/{args.dataset}-mcp-{args.k_shots}.json")
    elif args.cp:
        acc = calc_cp_acc(f"results/{args.model}/{args.dataset}-cp-{args.k_shots}.json")

    print(f"Accuracy: {acc}")