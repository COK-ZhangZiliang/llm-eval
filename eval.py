from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt import mcp_prompt, cp_prompt

import os
import torch

model_path = {
    'qwen2-0.5B-ins': 'Qwen/Qwen2-0.5B-Instruct',
    'qwen2-1.5B-ins': 'Qwen/Qwen2-1.5B-Instruct'
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _get_log_probs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0].log_softmax(dim=-1), inputs["input_ids"][0]
    
    def cp_eval(self, question, exmaples):
        exp_prompts = ""
        for exp in exmaples:
            exp_prompts += cp_prompt(exp, -1) + "\n\n"

        answer_token_lens = []
        uncond_log_probs = []
        cond_log_probs = []
        for i, choice in enumerate(question["choices"]["text"]):
            # 计算非条件概率
            uncond_prompts = f"Answer: {choice} "
            log_probs, inputids = self._get_log_probs(uncond_prompts)
            log_probs, inputids= log_probs[2:-1], inputids[2:-1]
            token_log_probs = [log_probs[i, j].item() for i, j in enumerate(inputids)]
            ans_token_len = len(token_log_probs)
            answer_token_lens.append(ans_token_len)
            uncond_log_probs.append(sum(token_log_probs))
            
            # 计算条件概率
            cond_prompts = exp_prompts + cp_prompt(question, i) + " "
            log_probs, inputids = self._get_log_probs(cond_prompts)
            log_probs, inputids = log_probs[-(ans_token_len+1):-1], inputids[-(ans_token_len+1):-1]
            token_log_probs = [log_probs[i, j].item() for i, j in enumerate(inputids)]
            cond_log_probs.append(sum(token_log_probs))
        
        return {'uncond_log_probs': uncond_log_probs, 
                'cond_log_probs': cond_log_probs, 
                'answer_token_lens': answer_token_lens}

    def mcp_eval(self, question, exmaples):
        prompts = ""
        for exp in exmaples:
            prompts += mcp_prompt(exp, is_exp=True) + "\n\n"
        prompts += mcp_prompt(question) + " "
        inputs = self.tokenizer(prompts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]
        log_probs = logits.log_softmax(dim=-1)
        pred_idx = log_probs.argmax().item()
        return self.idx2val[pred_idx].replace("Ġ", "")


def get_model(model):
    return {
        "qwen2-0.5B-ins": qwen(model),
        "qwen2-1.5B-ins": qwen(model)
    }[model]
           