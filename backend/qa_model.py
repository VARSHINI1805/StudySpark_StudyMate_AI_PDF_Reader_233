from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import Tuple, List, Dict

# Reader model: deepset/roberta-base-squad2
READER_MODEL_ID = "deepset/roberta-base-squad2"

# Lazy globals for cached model/tokenizer
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_reader():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_ID)
        _model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL_ID).to(_device)
    return _tokenizer, _model


def _qa_on_context(question: str, context: str) -> Tuple[str, float]:
    """Run QA on a single context block and return (answer, score)."""
    tokenizer, model = _load_reader()

    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_device)

    with torch.inference_mode():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    all_tokens = inputs["input_ids"][0]
    answer_tokens = all_tokens[start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    start_prob = torch.softmax(start_scores, dim=-1)[0, start_idx].item()
    end_prob = torch.softmax(end_scores, dim=-1)[0, end_idx].item()
    score = (start_prob + end_prob) / 2.0

    # No-answer handling via CLS
    cls_index = (inputs["input_ids"] == tokenizer.cls_token_id).nonzero(as_tuple=False)
    if cls_index.numel() > 0:
        cls_idx = cls_index[0, 1]
        no_ans_score = (torch.softmax(start_scores, dim=-1)[0, cls_idx].item() +
                        torch.softmax(end_scores, dim=-1)[0, cls_idx].item()) / 2.0
        if no_ans_score > score:
            return "", no_ans_score

    return answer, score


def answer_over_passages(question: str, passages: List[str]) -> Dict:
    """
    Run the reader against each passage, aggregate the answers, and return a combined one.
    This version attempts to combine answers from multiple passages.
    Returns dict: {"answer": str, "score": float, "passage_index": int}
    """
    all_answers = []
    best_score = -1.0
    best_answer_obj = {"answer": "", "score": -1.0, "passage_index": -1}

    for idx, ctx in enumerate(passages):
        ans, sc = _qa_on_context(question, ctx)
        if ans:
            all_answers.append(ans)
            if sc > best_score:
                best_score = sc
                # Store the original best answer object, but we will override the 'answer' text later
                best_answer_obj = {"answer": ans, "score": sc, "passage_index": idx}

    # Deduplicate and join answers. Sorting by length can help with ordering.
    unique_answers = sorted(list(set(all_answers)), key=len, reverse=True)
    combined_answer = ". ".join(unique_answers)

    # Return the combined answer, but keep the score and index from the single best find.
    best_answer_obj["answer"] = combined_answer if combined_answer else best_answer_obj["answer"]
    return best_answer_obj


def answer_question(question: str, context: str) -> Tuple[str, float]:
    """
    Backward-compatible wrapper: runs on a single combined context.
    """
    ans, sc = _qa_on_context(question, context)
    return ans, sc
