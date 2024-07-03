from argparse import ArgumentParser
from eval import get_model
from utils.data import prep_data

def get_args():
    parser = ArgumentParser(description='llm evaluation')
    parser.add_argument('--model', type=str, default='qwen2-0.5B-ins', help='model name')
    parser.add_argument('--dataset', type=str, default='ac', help='dataset name')
    parser.add_argument('--k_shots', type=int, default=0, help='number of shots')
    parser.add_argument('--que_idx', type=int, default=0, help='index of question to debug')

    return parser.parse_args()


def run_debug(args):
    questions, exmples = prep_data(args.dataset, args.k_shots)
    que, exps= questions[args.que_idx], exmples[args.que_idx]
    model = get_model(args.model)

    print(model.cp_eval(que, exps, isdebug=True))
    

if __name__ == "__main__":
    args = get_args()
    run_debug(args)