# C=8 `no_healthy_backend`

Evidence confidence: `ARTIFACT_BACKED`  
Impact: the 27-run targeted confirmation was invalid for clean performance claims.

At C=8 repeat 1, gateway aggregation off, 300 requests were attempted: 275 succeeded and 25 returned
HTTP 503 `no_healthy_backend`. The gateway marked its sole backend unhealthy for about five seconds,
then recovered. vLLM remained running. Logs prove the health transition and lack of a routable backend;
they do not prove vLLM crashed or establish the root cause.

The structural validator still emitted `valid=true`, exposing a boundary between schema validity and
experiment validity. The response was to preserve the failed matrix, require complete request counts,
and run a fresh C=8 confirmation rather than overwrite or resume the affected data.
