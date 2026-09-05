# Gateway aggregation versus vLLM continuous batching

vLLM continuously batches work inside its scheduler even when HTTP requests arrive separately.
Gateway aggregation is an additional compatibility and transport layer. Its effect depends on the
workload: the 2026-08-29 sweep favored C−B around concurrency 4–8 for its exact workload, but was
inconsistent higher; C=1 paid the wait. Always retain A, B, and C so router cost and aggregation
effect remain separately observable.
