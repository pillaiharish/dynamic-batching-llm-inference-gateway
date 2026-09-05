# Vast H100 experiment runbook

1. Select a current offer satisfying exactly one H100 80GB and the required disk/network constraints.
2. Review `infra/vast`, pin the image, plan Terraform, and explicitly acknowledge billable creation.
3. Register SSH access through the Vast account; never place private keys or API keys in Terraform.
4. Run the environment preflight and preserve its sanitized fingerprint before changing services.
5. Acquire and verify the pinned model through Chakra Vault; record the resolved revision.
6. Establish one validated topology from `deploy/topologies/`; for T2 keep B out of `BACKENDS_JSON`.
7. Smoke health, model identity, metrics, exact token accounting, and evidence destinations.
8. Run one isolated campaign, fail closed on any request/health/config mismatch, then collect evidence.
9. Teardown using the dedicated checklist. Copy verified archives off ephemeral storage before destroy.

Terraform creates the laboratory; deploy code changes topology; benchmark code measures. Do not mix
these phases or treat same-GPU co-location as multi-GPU scaling.
