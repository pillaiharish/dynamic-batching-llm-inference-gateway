# Matrix resume identity

Evidence confidence: `ARTIFACT_BACKED` (Git commit `400d79fae3269a6d3722a44bc9d3e824f78617bc`)

The old expression `matrix_id = config.run_id or timestamp` allowed `--resume` without a stable ID to
create a fresh timestamp directory instead of discovering the intended matrix. PR10 established:

1. use CLI `--matrix-id` when supplied;
2. otherwise use config `run_id`;
3. fail if both exist and differ;
4. allow a timestamp only for non-resume with neither;
5. fail resume with neither;
6. resume old timestamp runs by passing that explicit ID;
7. validate the ID against path traversal and unsafe characters.

The lesson is that resumability requires identity, not merely skip logic. Identity resolution must
fail before execution so a typo cannot split one scientific matrix across directories.
