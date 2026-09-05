# vLLM replica lifecycle

Launch each replica as its own process group with a stable logical name and port. The helper accepts
the real command after `--`, for example `python deploy/vllm_replica.py start A 18001 -- vllm serve …`.
Record the complete non-secret command, model revision, PID/PGID, port, log, and memory policy.

Before declaring ready, require process alive, correct `/health` and `/v1/models`, metrics reachable,
expected listener owner, GPU process allocation, and a minimal exact-token request. Stop through
`python deploy/vllm_replica.py stop A`; it sends TERM to the process group and deliberately refuses to
force-kill a survivor. Afterward inspect descendants, process group, listeners, NVIDIA compute apps,
and HBM baseline. Supervisor `STOPPED` or a vanished parent PID alone is insufficient.
