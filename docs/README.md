# Engineering and experiment record

This documentation separates experiment narrative from curated machine evidence. Historical notes
use one of three confidence labels:

- `ARTIFACT_BACKED`: directly supported by preserved files, logs, results, or checksums.
- `CHAT_AND_LOG_BACKED`: supported by logs/artifacts plus reconstructed engineering context.
- `RECONSTRUCTED`: interpreted after the fact without complete original evidence.

Start with the [chronological experiment index](experiments/experiment-index.md), the
[system overview](architecture/system-overview.md), and the [T0/T1/T2/T3 topology design](architecture/experiment-topologies.md).
The [claim boundaries](architecture/decisions/benchmark-claim-boundaries.md) define A/B/C language.

Operational guidance is under [runbooks](runbooks/vast-h100-experiment.md); failed paths and hazards
are preserved in [incidents](incidents/README.md). The [learning notes](learning/README.md) explain
the resulting design. Machine-readable and source evidence is indexed in
[`benchmark/evidence`](../benchmark/evidence/README.md).
