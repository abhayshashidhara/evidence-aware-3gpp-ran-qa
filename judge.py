import torch
from utils import normalize_text, is_bad_answer


class NLIJudge:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base", cpu=False):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    def score_pair(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        out = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for i, p in enumerate(probs):
            label = self.id2label.get(i, str(i))
            if "entail" in label:
                out["entailment"] = p
            elif "contrad" in label:
                out["contradiction"] = p
            elif "neutral" in label:
                out["neutral"] = p
        return out

    def judge(self, answer, evidence, entailment_threshold=0.50, partial_threshold=0.25, max_contradiction=0.75, min_support_score=0.34):
        answer = normalize_text(answer)
        if is_bad_answer(answer) or answer.lower() == "not enough information.":
            return {"verdict": "unsupported", "entailment": 0.0, "contradiction": 0.0, "support_score": 0.0}
        premise = " ".join(ev.get("sentence", "") for ev in evidence)
        scores = self.score_pair(premise, answer)
        entail = scores["entailment"]
        contradiction = scores["contradiction"]
        support = entail * (1.0 - contradiction)
        if contradiction > max_contradiction:
            verdict = "unsupported"
        elif entail >= entailment_threshold or support >= min_support_score:
            verdict = "supported"
        elif entail >= partial_threshold:
            verdict = "partial"
        else:
            verdict = "unsupported"
        return {"verdict": verdict, "entailment": entail, "contradiction": contradiction, "support_score": support}
