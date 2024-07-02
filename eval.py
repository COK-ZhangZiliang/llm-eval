from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt import mcp_prompt, cp_prompt

import os

model_path = {
    'qwen2-0.5B-Ins': 'Qwen/Qwen2-0.5B-Instruct',
}

class qwen:
    def __init__(self, model):
        """
        加载模型"""
        model_name = model_path[model]
        os.makedirs("model/", exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="model/")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="model/")
        self.idx2val = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def cp_eval(self, question, exmaples):
        pass

    def mcp_eval(self, question, exmaples):
        prompts = ""
        for exp in exmaples:
            prompts += mcp_prompt(exp, is_exp=True) + "\n\n"
        prompts += mcp_prompt(question)
        print(prompts)
        inputs = self.tokenizer(prompts, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        probs = logits.softmax(dim=-1)
        pred_idx = probs.argmax().item()
        return self.idx2val[pred_idx].replace("Ġ", "")

def get_model(model):
    return {
        "qwen2-0.5B-Ins": qwen(model)
    }[model]
        
            