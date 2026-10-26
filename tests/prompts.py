"""Deterministic synthetic prompts with exact token targeting.

Prompt text is generated from a seeded template; no external dataset is read.
`build_prompts` is the only place that talks to the tokenizer, and it caches by
text digest so warm/cold rounds do not re-tokenize identical documents.
"""

import hashlib

FILLER = (
    "The maintenance log records a routine inspection of the cooling loop, the "
    "supply manifold, and the return line. Readings stayed inside the published "
    "band for the whole shift, and the operator noted nothing unusual."
)
SHORT = "The log entry is unremarkable."
RETRIEVAL_INSTRUCTION = (
    "Using only the document above, list every ACCESS CODE exactly as written, "
    "one per line, with no commentary."
)
DECODE_INSTRUCTION = (
    "Write a detailed technical explanation of continuous batching, KV caches, "
    "prefix reuse, speculative decoding, and latency/throughput trade-offs in an "
    "LLM inference server. Use worked examples and do not stop early."
)
CODE_FRACTIONS = (0.20, 0.55, 0.85)
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _code(marker, index):
    raw = hashlib.sha256(f"{marker}:{index}".encode()).digest()
    digits = "".join(_ALPHABET[byte % len(_ALPHABET)] for byte in raw[:6])
    return f"{digits[:3]}-{digits[3:]}"


def render(marker, sentences, short_tail, codes):
    """Build the document body for a sentence count and a short-clause tail."""
    lines = [f"SESSION {marker}"]
    # A retrieval document always needs at least one sentence to carry its codes;
    # a decode prompt may consist of the preamble alone.
    total = max(1, sentences) if codes else sentences
    code_positions = {int(total * fraction): code for fraction, code in zip(CODE_FRACTIONS, codes)}
    for index in range(total):
        lines.append(FILLER)
        if index in code_positions:
            lines.append(f"ACCESS CODE {code_positions[index]}.")
    lines.extend(SHORT for _ in range(short_tail))
    lines.append(RETRIEVAL_INSTRUCTION if codes else DECODE_INSTRUCTION)
    return "\n".join(lines) + "\n"


def _largest_fit(measure, target, ceiling):
    """Largest n in [0, ceiling] whose measured token count does not exceed target."""
    if measure(0) > target:
        return 0
    low, high = 0, 1
    while high < ceiling and measure(high) <= target:
        low, high = high, min(ceiling, high * 2)
    if measure(high) <= target:
        return high
    while low + 1 < high:
        middle = (low + high) // 2
        if measure(middle) <= target:
            low = middle
        else:
            high = middle
    return low


def fit_text(count, target, marker, codes, sentence_ceiling=20000, short_ceiling=64):
    """Closest text at or just below `target` tokens, using two monotone searches.

    `count` must be a deterministic tokenizer call. The result never exceeds the
    target; callers record the achieved size instead of assuming it.
    """
    if target < 1:
        raise ValueError("Target token count must be positive")
    cache = {}

    def measured(make):
        text = make()
        if text not in cache:
            cache[text] = count(text)
        return cache[text]

    minimum = measured(lambda: render(marker, 0, 0, codes))
    if minimum > target:
        raise ValueError(
            f"Target {target} tokens is below the {minimum}-token prompt preamble")

    sentences = _largest_fit(
        lambda n: measured(lambda: render(marker, n, 0, codes)),
        target, sentence_ceiling,
    )
    short = _largest_fit(
        lambda n: measured(lambda: render(marker, sentences, n, codes)),
        target, short_ceiling,
    )
    text = render(marker, sentences, short, codes)
    return {"text": text, "tokens": cache[text], "sentences": sentences, "short_clauses": short}


def build_prompts(count, target_tokens, concurrency, prompt_set_id, seed, purpose="decode"):
    """One prompt per client slot; identical targets, distinct content.

    Cold runs pass a per-round `prompt_set_id`, so early session markers differ
    before any shared prefix can form. Warm runs reuse the identifier to force
    prefix-cache reuse.
    """
    prompts = []
    for index in range(concurrency):
        marker = f"{prompt_set_id}-{seed}-{index}"
        codes = tuple(_code(marker, position) for position in range(3)) if purpose == "retrieval" else ()
        fitted = fit_text(count, target_tokens, marker, codes)
        prompts.append({
            "marker": marker,
            "purpose": purpose,
            "target_tokens": target_tokens,
            "actual_tokens": fitted["tokens"],
            "sentences": fitted["sentences"],
            "short_clauses": fitted["short_clauses"],
            "expected_codes": list(codes),
            "sha256": hashlib.sha256(fitted["text"].encode()).hexdigest(),
            "text": fitted["text"],
        })
    return prompts
