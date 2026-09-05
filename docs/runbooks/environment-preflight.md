# Environment preflight

Before model or benchmark work, record UTC time, repo SHA/clean status, pinned model revision, GPU name
and UUID, driver/CUDA, total and baseline HBM, compute applications, OS/allocation CPU/RAM, Python,
PyTorch/CUDA, vLLM, Transformers, Go, mounts/free disk, persistent-volume status, process tree, and
relevant listeners. `python deploy/preflight.py > preflight.json` captures a concise starting point;
supplement allocation/storage facts from the provider control plane.

Do not capture environment variables, credentials, Authorization headers, SSH material, private host
coordinates, or full machine dumps. Refuse to proceed if the repo/model revision is ambiguous, the
expected ports are occupied, unexplained GPU processes exist, free storage is inadequate, or evidence
cannot survive the planned instance lifecycle.
