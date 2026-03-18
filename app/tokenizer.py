"""Token estimation: tiktoken for DeepSeek/OpenAI-compatible, fallback to char/4."""

from typing import Optional


def estimate_tokens(text: str, model_id: Optional[str] = None) -> int:
    """
    Estimate token count for *text*. When model_id is provided and indicates DeepSeek (or OpenAI),
    use tiktoken if available for accurate counts; otherwise fall back to len(text)//4.
    """
    if not text:
        return 0
    model_id = (model_id or "").strip().lower()
    if model_id and "deepseek" in model_id:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)
