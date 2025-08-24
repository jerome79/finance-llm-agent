from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

_model = None
_pipe = None

def finbert():
    global _pipe
    if _pipe is None:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _pipe = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True)
    return _pipe

def score_text(text: str):
    p = finbert()
    return p(text[:4000])  # truncate long text
