# T0/T1/T2/T3 experiment topologies

All comparisons use the same model revision and frozen request parameters. The machine-readable
forms live in `deploy/topologies/` and fail closed under `deploy/validate_topology.py`.

| Topology | Runtime state | Question |
|---|---|---|
| T0 | One vLLM with a near-full H100 memory budget | Capacity ceiling |
| T1 | One vLLM alone with the same per-instance policy used for co-location | Shared-policy baseline |
| T2 | A and B resident on one H100; direct traffic only to A; B absent from `BACKENDS_JSON` | Co-resident reservation/runtime effect without B compute |
| T3-A | Both serve; client performs deterministic request-level A/B round-robin | Direct multi-process baseline |
| T3-B | Gateway routes to both; compatible aggregation off | Gateway/router effect |
| T3-C | Gateway routes to both; compatible aggregation on | Aggregation effect under multi-backend routing |

T0 versus T1 measures the memory/KV-budget effect; T1 versus T2 measures the resident second
process effect. T3-B minus T3-A is gateway/router overhead. T3-C minus T3-B is gateway-compatible
aggregation. T3-A versus T3-C compares the full gateway stack with deterministic direct routing.

T0 is not a fair causal comparator for co-located instances because its memory budget differs.
Two processes sharing one physical GPU are not 2-GPU scaling. A future experiment with one vLLM
per independent H100 is separate; tensor parallelism `TP=2` is also separate.
