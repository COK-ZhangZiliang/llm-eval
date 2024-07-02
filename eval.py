from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt import mcp_prompt, cp_prompt

import os
import torch

model_path = {
    'qwen2-0.5B-ins': 'Qwen/Qwen2-0.5B-Instruct',
    'qwen2-7B-ins': 'Qwen/Qwen2-7B-Instruct'
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

    def _get_log_probs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0].log_softmax(dim=-1)
    
    def cp_eval(self, question, exmaples, isdebug=False):
        exp_prompts = ""
        for exp in exmaples:
            exp_prompts += cp_prompt(exp, -1) + "\n\n"

        answer_token_lens = []
        uncond_log_probs = []
        cond_log_probs = []
        for i, choice in enumerate(question["choices"]["text"]):
            # 计算非条件概率
            uncond_prompts = f"Answer: {choice}"
            log_probs = self._get_log_probs(uncond_prompts)[2: -1]
            ans_token_len = log_probs.size(0)
            answer_token_lens.append(ans_token_len)
            uncond_log_probs.append(log_probs.sum().item())
            
            # 计算条件概率
            cond_prompts = exp_prompts + cp_prompt(question, i)
            log_probs = self._get_log_probs(cond_prompts)[-(ans_token_len+1):-1]
            cond_log_probs.append(log_probs.sum().item())
        
        return {'uncond_log_probs': uncond_log_probs, 
                'cond_log_probs': cond_log_probs, 
                'answer_token_lens': answer_token_lens}

    def mcp_eval(self, question, exmaples):
        prompts = ""
        for exp in exmaples:
            prompts += mcp_prompt(exp, is_exp=True) + "\n\n"
        prompts += mcp_prompt(question)
        inputs = self.tokenizer(prompts, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        log_probs = logits.log_softmax(dim=-1)
        pred_idx = log_probs.argmax().item()
        return self.idx2val[pred_idx].replace("Ġ", "")


def get_model(model):
    return {
        "qwen2-0.5B-ins": qwen(model),
        "qwen2-7B-ins": qwen(model)
    }[model]
           