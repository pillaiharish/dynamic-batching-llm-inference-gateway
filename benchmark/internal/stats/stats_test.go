package stats

import "testing"

func TestNearestRankPercentiles(t *testing.T) {
	values := make([]float64, 100)
	for index := range values {
		values[index] = float64(index + 1)
	}
	summary := Summarize(values)
	for name, test := range map[string]struct {
		actual *float64
		want   float64
	}{
		"p50": {summary.P50MS, 50},
		"p90": {summary.P90MS, 90},
		"p95": {summary.P95MS, 95},
		"p99": {summary.P99MS, 99},
	} {
		t.Run(name, func(t *testing.T) {
			if test.actual == nil || *test.actual != test.want {
				t.Fatalf("got %v, want %.0f", test.actual, test.want)
			}
		})
	}
}

func TestPercentileDoesNotMutateInput(t *testing.T) {
	values := []float64{3, 1, 2}
	_ = Percentile(values, 0.5)
	if values[0] != 3 || values[1] != 1 || values[2] != 2 {
		t.Fatal("Percentile mutated input")
	}
}
