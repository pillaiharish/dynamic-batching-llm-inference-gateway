package metrics

import "testing"

const beforeFixture = `
# TYPE gateway_observed_output_tokens_total counter
gateway_observed_output_tokens_total{mode="non_streaming"} 100
gateway_batches_total{flush_reason="size",outcome="success"} 3
gateway_batch_size_bucket{le="1"} 1
gateway_batch_size_bucket{le="2"} 3
gateway_batch_size_bucket{le="4"} 4
gateway_batch_size_bucket{le="+Inf"} 4
gateway_batch_size_sum 9
gateway_batch_size_count 4
`

const afterFixture = `
gateway_observed_output_tokens_total{mode="non_streaming"} 140
gateway_batches_total{flush_reason="size",outcome="success"} 5
gateway_batches_total{flush_reason="timeout",outcome="success"} 1
gateway_batch_size_bucket{le="1"} 1
gateway_batch_size_bucket{le="2"} 4
gateway_batch_size_bucket{le="4"} 7
gateway_batch_size_bucket{le="+Inf"} 7
gateway_batch_size_sum 19
gateway_batch_size_count 7
`

func TestCounterAndHistogramDeltas(t *testing.T) {
	before, err := Parse(beforeFixture)
	if err != nil {
		t.Fatal(err)
	}
	after, err := Parse(afterFixture)
	if err != nil {
		t.Fatal(err)
	}
	tokens := CounterDelta(before, after, "gateway_observed_output_tokens_total", nil)
	if !tokens.Found || tokens.Reset || tokens.Value != 40 {
		t.Fatalf("unexpected token delta: %#v", tokens)
	}
	timeout := CounterDelta(before, after, "gateway_batches_total", map[string]string{"flush_reason": "timeout"})
	if timeout.Value != 1 {
		t.Fatalf("new label series should start at zero: %#v", timeout)
	}
	mean, reset := HistogramMean(before, after, "gateway_batch_size")
	if reset || mean == nil || *mean != 10.0/3.0 {
		t.Fatalf("unexpected mean: %v reset=%v", mean, reset)
	}
	p95, reset := HistogramPercentile(before, after, "gateway_batch_size", 0.95)
	if reset || p95 == nil || *p95 != 4 {
		t.Fatalf("unexpected p95: %v reset=%v", p95, reset)
	}
}

func TestCounterResetAndMissingMetric(t *testing.T) {
	before, _ := Parse("counter_total{label=\"a\"} 10\n")
	after, _ := Parse("counter_total{label=\"a\"} 2\n")
	delta := CounterDelta(before, after, "counter_total", nil)
	if !delta.Reset {
		t.Fatalf("expected reset: %#v", delta)
	}
	missing := CounterDelta(before, after, "optional_missing", nil)
	if !missing.Missing || missing.Found {
		t.Fatalf("unexpected missing result: %#v", missing)
	}
}

func TestLabelsAndEscapes(t *testing.T) {
	snapshot, err := Parse("metric_total{a=\"hello\\nworld\",b=\"quote\\\"\"} 3\n")
	if err != nil {
		t.Fatal(err)
	}
	for _, series := range snapshot {
		if series.Labels["a"] != "hello\nworld" || series.Labels["b"] != "quote\"" {
			t.Fatalf("unexpected labels: %#v", series.Labels)
		}
	}
}
