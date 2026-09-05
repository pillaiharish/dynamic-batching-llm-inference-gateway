# Lessons learned

- Preserve invalid experiments: they explain why protocol and topology changed.
- Separate schema validity from experiment validity and capacity evidence from performance evidence.
- Give resumed matrices a stable, explicit identity; never synthesize one during resume.
- Treat supervisor state as one signal, not proof that a CUDA process ended.
- Freeze model revision, memory policy, topology, request parameters, and evidence checksums.
- Validate locally generated token counts against server usage before a stress campaign.
- Use T1/T2 to isolate memory reservation and co-residency before interpreting T3 routing results.
