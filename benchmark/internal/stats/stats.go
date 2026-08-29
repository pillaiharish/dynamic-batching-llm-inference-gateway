// Package stats contains deliberately small, deterministic summary algorithms.
package stats

import (
	"math"
	"sort"
)

// Percentile returns the nearest-rank percentile. For N sorted observations and
// p in (0, 1], the selected one-based rank is ceil(p*N).
func Percentile(values []float64, p float64) *float64 {
	if len(values) == 0 || p <= 0 || p > 1 {
		return nil
	}
	ordered := append([]float64(nil), values...)
	sort.Float64s(ordered)
	rank := int(math.Ceil(p * float64(len(ordered))))
	value := ordered[rank-1]
	return &value
}

// Latencies is the common percentile set, represented in milliseconds.
type Latencies struct {
	P50MS *float64 `json:"p50_ms"`
	P90MS *float64 `json:"p90_ms"`
	P95MS *float64 `json:"p95_ms"`
	P99MS *float64 `json:"p99_ms"`
}

// Summarize calculates the documented nearest-rank percentiles.
func Summarize(values []float64) Latencies {
	return Latencies{
		P50MS: Percentile(values, 0.50),
		P90MS: Percentile(values, 0.90),
		P95MS: Percentile(values, 0.95),
		P99MS: Percentile(values, 0.99),
	}
}
