# Vast supervisor reported stopped while vLLM descendants remained

Evidence confidence: `CHAT_AND_LOG_BACKED`

During replacement of the old Qwen3.8-27B service, supervisor reported `STOPPED` while API/EngineCore
descendants still existed, the API port remained bound, and roughly 71 GiB HBM remained allocated.
The preserved engineering context supports the incident and its operational lesson, but the curated
archive does not contain a complete independently replayable event timeline; root cause is not claimed.

Supervisor state is control-plane intent, not proof of CUDA teardown. Completion now requires checking
the process tree and group, relevant listeners, `nvidia-smi` compute applications, and return to the
recorded HBM baseline. Directly killing a child can also trigger supervisor restart, so stop through the
owner first and escalate only after inspection.
