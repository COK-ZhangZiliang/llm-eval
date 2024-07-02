from argparse import ArgumentParser
from eval import get_model
from utils.data import prep_data

def get_args():
    """
    获取待评估模型名、数据集名以及评估方法"""
    parser = ArgumentParser(description='llm evaluation')
    parser.add_argument('--model', type=str, default='qwen2-0.5B-Ins', help='model name')
    parser.add_argument('--dataset', type=str, default='ac', help='dataset name')
    parser.add_argument('--k_shots', type=int, default=0, help='number of shots')
    parser.add_argument('--mcp', action='store_true', default=True, help='use mcp strategy')
    parser.add_argument('--cp', action='store_true', default=False, help='use cp stategy')

    return parser.parse_args()


def run_exp(args):
    """
    进行评估"""
    questions, exmples = prep_data(args.dataset, args.k_shots)
    model= get_model(args.model)
    for que, exps in zip(questions, exmples):
        print(que, exps)
        if args.mcp:
            print(model.mcp_eval(que, exps))
        elif args.cp:
            print(model.cp_eval(que, exps))
        else:
            raise ValueError("Please choose a strategy to evaluate the model.")
    


if __name__ == "__main__":
    args = get_args()
    run_exp(args)
