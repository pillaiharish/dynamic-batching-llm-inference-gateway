# Tokenizer-aware workloads

Stress and benchmark prompts must be measured after the model chat template is applied. A
Transformers `BatchEncoding` is mapping-like: `len(encoded)` counts fields, not tokens. Count
`encoded.input_ids`, assert the exact target locally, and cross-check at least one request against
the server's `usage.prompt_tokens`. Keep prompt and output token assertions in the final evidence;
an HTTP success alone does not prove the intended workload ran.
