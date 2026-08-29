package runner

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/client"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/workload"
)

func runnerClient(t *testing.T, server *httptest.Server, concurrency int) *client.Client {
	t.Helper()
	value, err := client.New(client.Config{
		BaseURL: server.URL, Endpoint: "/v1/chat/completions", Model: "m", Mode: client.NonStreaming,
		Generation: client.Generation{TopP: 1, MaxTokens: 4, Seed: 1}, Timeout: time.Second,
		Concurrency: concurrency,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(value.Close)
	return value
}

func corpus() workload.Corpus {
	return workload.Corpus{SHA256: "hash", Records: []workload.Record{{
		ID: "one", Messages: []workload.Message{{Role: "user", Content: "hello"}},
	}}}
}

func TestClosedLoopNeverExceedsAndReachesConcurrency(t *testing.T) {
	var active atomic.Int64
	var maximum atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		current := active.Add(1)
		defer active.Add(-1)
		for {
			old := maximum.Load()
			if current <= old || maximum.CompareAndSwap(old, current) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		fmt.Fprint(writer, `{"usage":{"completion_tokens":1}}`)
	}))
	defer server.Close()
	runner, err := New(Config{Client: runnerClient(t, server, 4), Corpus: corpus(), Concurrency: 4, Requests: 12})
	if err != nil {
		t.Fatal(err)
	}
	measured := runner.Measure(context.Background())
	if maximum.Load() != 4 {
		t.Fatalf("observed max concurrency %d, want 4", maximum.Load())
	}
	if measured.Summary.Successful != 12 {
		t.Fatalf("unexpected summary: %#v", measured.Summary)
	}
}

func TestWarmupLatencyIsExcluded(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) <= 3 {
			time.Sleep(60 * time.Millisecond)
		}
		fmt.Fprint(writer, `{"usage":{"completion_tokens":1}}`)
	}))
	defer server.Close()
	runner, err := New(Config{Client: runnerClient(t, server, 1), Corpus: corpus(), Concurrency: 1, Requests: 5, Warmup: 3})
	if err != nil {
		t.Fatal(err)
	}
	warmup := runner.Warmup(context.Background())
	measured := runner.Measure(context.Background())
	if warmup.Success != 3 || measured.Summary.E2E.P99MS == nil || *measured.Summary.E2E.P99MS >= 40 {
		t.Fatalf("warmup leaked into measured latency: warmup=%#v measured=%#v", warmup, measured.Summary)
	}
}

func TestErrorRateAndCategories(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		call := calls.Add(1)
		switch call {
		case 9:
			writer.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(writer, `{"error":{"code":"queue_full"}}`)
		case 10:
			writer.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(writer, `{"error":{"code":"unavailable"}}`)
		default:
			fmt.Fprint(writer, `{"usage":{"completion_tokens":1}}`)
		}
	}))
	defer server.Close()
	runner, err := New(Config{Client: runnerClient(t, server, 1), Corpus: corpus(), Concurrency: 1, Requests: 10})
	if err != nil {
		t.Fatal(err)
	}
	summary := runner.Measure(context.Background()).Summary
	if summary.Attempted != 10 || summary.Successful != 8 || summary.Failed != 2 || summary.ErrorRate != 0.2 {
		t.Fatalf("unexpected summary: %#v", summary)
	}
	if summary.ErrorsByCategory["http_429"] != 1 || summary.ErrorsByCategory["http_5xx"] != 1 {
		t.Fatalf("unexpected categories: %#v", summary.ErrorsByCategory)
	}
}
