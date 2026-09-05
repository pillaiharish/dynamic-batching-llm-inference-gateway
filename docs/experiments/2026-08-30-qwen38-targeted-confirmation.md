# 2026-08-30 Qwen3.8-27B targeted confirmation

Status: `BENCHMARK_INVALID` for clean performance claims  
Evidence confidence: `ARTIFACT_BACKED`

## Question and procedure

Would the earlier aggregation direction repeat at concurrency 4, 8, and 16? The matrix used A/B/C,
three rotated repetitions, 20 warmups and 300 measured requests per arm, with the same deterministic
generation settings. It ran on one H100 with Qwen3.8-27B, vLLM 0.26.0, BF16/FP8 KV, TP=1, max
length 32768, max sequences 16, and gateway SHA `4473e78c41f33f69c0a0860ca6275743201f0324`.

## Failure and validity

C=8 repeat 1 arm B attempted 300 requests: 275 succeeded and 25 returned HTTP 503
`no_healthy_backend`. Gateway logs show its single backend becoming unhealthy for about five seconds
and then recovering. Across the matrix, 8075/8100 requests succeeded. vLLM remained running; the
artifacts do not prove a crash, GPU fault, network root cause, or health-probe root cause.

Every result could pass the contemporary structural schema with `valid=true`; that field did not
satisfy the experiment's 300/300 success criterion. No descriptive performance value from the failed
arm supports a clean confirmation claim. The invalid matrix prompted a fresh isolated C=8 redesign,
not repair or deletion of this evidence. Archive SHA-256:
`9226f34e3cb60e8f9092a3db7d5a9199147fee88e5d4079ca01a96758567b9e7`.
