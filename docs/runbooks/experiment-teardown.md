# Experiment teardown

1. Stop traffic and evidence monitors; record exit codes and final health/metrics/GPU snapshots.
2. Stop gateways, then each vLLM through its process-group/supervisor owner.
3. Verify no descendants or process-group members remain.
4. Verify experiment ports are no longer listening.
5. Verify `nvidia-smi` lists no experiment compute applications and HBM returns to baseline.
6. Hash archives and sidecars; copy them to durable external storage and re-verify there.
7. Remove the billable instance with `terraform destroy` only after evidence is durable.

If any process, port, or HBM allocation survives, teardown is incomplete. Record the incident and
inspect ownership before escalation; do not indiscriminately kill PIDs or destroy the instance first.
