# BatchEncoding token-count regression

Evidence confidence: `ARTIFACT_BACKED`

The first T2 stress generator called `len(encoded)` after `apply_chat_template(..., tokenize=True)`.
The returned `BatchEncoding` had two fields, so the code recorded two tokens and constructed an
oversized prompt. vLLM rejected all 64 requests with HTTP 400 before inference; GPU/KV activity did
not occur and both engines remained healthy.

The correction counts `encoded.input_ids`, asserts exact local token length, and cross-checks one
request with `usage.prompt_tokens`. Regression tests cover the two-field trap and exact 128/1024-token
targets. HTTP failure from an invalid generator must never be interpreted as GPU capacity evidence.
