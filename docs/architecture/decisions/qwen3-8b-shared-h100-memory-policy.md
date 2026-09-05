# Qwen3-8B shared-H100 memory policy

Status: V1 candidate accepted for the next T1/T2/T3 campaign  
Evidence confidence: `ARTIFACT_BACKED`

Use `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16 weights,
FP8 KV, TP=1, `max_model_len=4096`, `max_num_seqs=128`,
`max_num_batched_tokens=8192`, `gpu_memory_utilization=0.44`, and
`kv_cache_memory_bytes=18,700,000,000` per replica. Thinking is disabled as part of the serving and
workload contract; vLLM uses `--default-chat-template-kwargs '{"enable_thinking": false}'`, and local
token generation passes the equivalent `enable_thinking=False` explicitly.

Automatic KV sizing at 0.44 let two replicas fit but produced unequal capacities. Fixed KV alone
left vLLM's default 0.92 admission guard active and blocked the second start. The combined policy
gave each replica 17.42 GiB KV, 253,632 KV tokens, reported 61.92x maximum concurrency at 4096,
and 35,172 MiB idle EngineCore allocation in both startup orders. The corrected T2 qualification
then completed 64/64 exact 1024-input/1024-output requests on A while B remained resident and idle.

This decision establishes a reproducible capacity/configuration candidate for the recorded
environment. It does not establish latency, throughput, multi-GPU scaling, or portability to a
different vLLM/model/GPU revision. Recalibrate after any such change or any admission/OOM failure.
