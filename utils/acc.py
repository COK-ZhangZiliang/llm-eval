import json
from .prompt import idx2choice

def calc_mcp_acc(file):
    """
    计算mcp策略的准确率"""
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        correct = 0
        for line in lines:
            data = json.loads(line)
            if idx2choice(data['Ans']) == data['Pred']:
                correct += 1
        return correct / len(lines)


def calc_cp_acc(file):
    """
    计算cp策略的准确率"""
    # raw
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        correct = 0
        for line in lines:
            data = json.loads(line)
            raw_ans = data['Pred']['cond_log_probs'].index(max(data['Pred']['cond_log_probs']))
            if  raw_ans == data['Ans']:
                correct += 1
        raw_acc = correct / len(lines)
    
    # unconditional normalization
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        correct = 0
        for line in lines:
            data = json.loads(line)
            log_probs = [cond_log_probs-uncond_log_probs 
                         for cond_log_probs, uncond_log_probs 
                         in zip(data['Pred']['cond_log_probs'], data['Pred']['uncond_log_probs'])]
            uncond_ans = log_probs.index(max(log_probs))
            if uncond_ans == data['Ans']:
                correct += 1
        unconditional_acc = correct / len(lines)

    # length normalization
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        correct = 0
        for line in lines:
            data = json.loads(line)
            log_probs = [cond_log_probs/lens if lens != 0 else 0
                         for cond_log_probs, lens 
                         in zip(data['Pred']['cond_log_probs'], data['Pred']['answer_token_lens'])]
            len_ans = log_probs.index(max(log_probs))
            if len_ans == data['Ans']:
                correct += 1
        length_acc = correct / len(lines)

    return raw_acc, unconditional_acc, length_acc