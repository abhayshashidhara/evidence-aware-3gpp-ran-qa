import re
import torch

from utils import normalize_text


def build_prompt(question, evidence, source_name="retrieved evidence"):
    evidence_text = "\n".join([f"E{i+1}. {ev['sentence']}" for i, ev in enumerate(evidence)])
    return f"""You are answering a 3GPP RAN/RRC technical question.
Use only the provided evidence from {source_name}.
Do not use outside knowledge.
If the evidence is insufficient, answer exactly: Not enough information.
Avoid unsupported message names, timers, states, procedures, causes, or configuration fields.

Question:
{question}

Evidence:
{evidence_text}

Answer:"""


class LocalGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", allow_cpu=False, no_4bit=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError("CUDA is not available. Use --allow_cpu for CPU mode.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if self.device == "cuda" and not no_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            kwargs["device_map"] = "auto" if self.device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def generate(self, question, evidence, source_name="retrieved evidence", max_new_tokens=450, temperature=0.0):
        prompt = build_prompt(question, evidence, source_name)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return parse_answer(raw)


def parse_answer(raw: str) -> str:
    answer = normalize_text(raw)
    answer = re.sub(r"^(Answer:|Final Answer:)\s*", "", answer, flags=re.I)
    if not answer:
        return "Not enough information."
    if "not enough information" in answer.lower():
        return "Not enough information."
    return answer
