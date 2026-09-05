# Tokenizer-aware workloads

Stress and benchmark prompts must be measured after the model chat template is applied. A
Transformers `BatchEncoding` is mapping-like: `len(encoded)` counts fields, not tokens. Count
`encoded.input_ids`, assert the exact target locally, and cross-check at least one request against
the server's `usage.prompt_tokens`. Fixing the container-length bug is necessary but insufficient:
the local tokenizer must use the same chat-template configuration as the serving endpoint. For the
Qwen3 V1 workload, `enable_thinking=false` is frozen provenance and must be passed explicitly during
both prompt search and final verification. Keep template kwargs, prompt/output token assertions, and
the server cross-check in final evidence; an HTTP success alone does not prove the intended workload ran.
