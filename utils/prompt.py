from dataclasses import dataclass
from utils.data import load_data

def idx2choice(idx):
    return chr(idx + ord("A"))

def choice2idx(choice):
    return ord(choice) - ord("A")

@dataclass
class Prompt:
    question: str
    choices: list
    ans_idx: int

    def get_cp_prompt(self, idx):
        prompt = f"Question: {self.question}\n"
        prompt += f"Answer: {self.choices[idx]}"
        return prompt

    def get_mcp_prompt(self, is_exp=False):
        prompt = f"Question: {self.question}\n"
        for i, choice in enumerate(self.choices):
            prompt += f"{idx2choice(i)}. {choice}\n"
        prompt += "Answer:"
        
        if is_exp:
            prompt += " " + idx2choice(self.ans_idx)
        return prompt
    
def cp_prompt(data, idx):
    question = data['question']
    choices = data['choices']['text']
    ans_idx = choice2idx(data['answerKey'])
    return Prompt(question, choices, ans_idx).get_cp_prompt(idx)

def mcp_prompt(data, is_exp=False):
    print(data)
    question = data['question']
    choices = data['choices']['text']
    ans_idx = choice2idx(data['answerKey'])
    return Prompt(question, choices, ans_idx).get_mcp_prompt(is_exp)