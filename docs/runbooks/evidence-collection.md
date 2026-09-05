# Evidence collection

Capture a run manifest before traffic: UTC, repo/model revisions, software/GPU fingerprint, topology,
full non-secret vLLM/gateway/benchmark configuration, dataset/payload hash, frozen request parameters,
and planned success criteria. During and after the run retain compact result summaries, request counts,
health transitions, relevant Prometheus deltas, GPU snapshots, exit codes, and problem scans.

Keep raw evidence external in a checksummed archive. Curate only the smallest files needed to support
claims into Git, record the archive filename/digest, and label interpretation separately. Verify tar
path safety, checksum, structured-file parsing, secret/private-path absence, and member inventory.
Never substitute a Markdown narrative for missing raw evidence or copy giant logs into prose.
