// Package runner implements bounded closed-loop concurrency and warmup exclusion.
package runner

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/client"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/stats"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/workload"
)

// Config defines one closed-loop workload point.
type Config struct {
	Client      *client.Client
	Corpus      workload.Corpus
	Concurrency int
	Requests    int
	Warmup      int
}

// WarmupSummary records lifecycle counts without retaining warmup latencies.
type WarmupSummary struct {
	Attempted int `json:"attempted"`
	Success   int `json:"success"`
	Failed    int `json:"failed"`
}

// Summary contains only measured observations.
type Summary struct {
	Attempted                  int              `json:"attempted"`
	Successful                 int              `json:"successful"`
	Failed                     int              `json:"failed"`
	ErrorRate                  float64          `json:"error_rate"`
	ErrorsByCategory           map[string]int   `json:"errors_by_category"`
	E2E                        stats.Latencies  `json:"e2e"`
	TTFT                       *stats.Latencies `json:"ttft"`
	RequestThroughputRPS       float64          `json:"request_throughput_rps"`
	OutputThroughputTPS        *float64         `json:"output_throughput_tps"`
	OutputTokenSource          string           `json:"output_token_source,omitempty"`
	AuthoritativeTokenCoverage int              `json:"authoritative_token_coverage"`
}

// MeasuredRun is the timed phase and its monotonic wall-clock duration.
type MeasuredRun struct {
	StartedAt time.Time
	EndedAt   time.Time
	Duration  time.Duration
	Requests  []client.Result
	Summary   Summary
}

// Runner reuses one client for warmup and measurement.
type Runner struct {
	config Config
}

// New validates a runner configuration.
func New(config Config) (*Runner, error) {
	if config.Client == nil || len(config.Corpus.Records) == 0 {
		return nil, fmt.Errorf("client and non-empty corpus are required")
	}
	if config.Concurrency <= 0 || config.Requests <= 0 || config.Warmup < 0 {
		return nil, fmt.Errorf("concurrency and requests must be positive; warmup cannot be negative")
	}
	return &Runner{config: config}, nil
}

// Warmup runs a separate unmeasured phase using the same client and concurrency.
func (runner *Runner) Warmup(ctx context.Context) WarmupSummary {
	if runner.config.Warmup == 0 {
		return WarmupSummary{}
	}
	results, _, _ := runner.runPhase(ctx, runner.config.Warmup, true)
	summary := WarmupSummary{Attempted: len(results)}
	for _, result := range results {
		if result.Success {
			summary.Success++
		} else {
			summary.Failed++
		}
	}
	return summary
}

// Measure runs N requests with at most C outstanding. Each worker starts its
// next request only after its previous request completes, so this is closed-loop.
func (runner *Runner) Measure(ctx context.Context) MeasuredRun {
	results, started, ended := runner.runPhase(ctx, runner.config.Requests, false)
	duration := ended.Sub(started)
	return MeasuredRun{
		StartedAt: started,
		EndedAt:   ended,
		Duration:  duration,
		Requests:  results,
		Summary:   summarize(results, duration),
	}
}

func (runner *Runner) runPhase(ctx context.Context, count int, warmup bool) ([]client.Result, time.Time, time.Time) {
	jobs := make(chan int)
	results := make(chan client.Result, count)
	workers := min(runner.config.Concurrency, count)
	var start time.Time
	var wait sync.WaitGroup
	wait.Add(workers)
	for range workers {
		go func() {
			defer wait.Done()
			for sequence := range jobs {
				record := runner.config.Corpus.Records[sequence%len(runner.config.Corpus.Records)]
				persistedSequence := sequence
				if warmup {
					persistedSequence = -sequence - 1
				}
				results <- runner.config.Client.Do(ctx, client.Request{
					Sequence: persistedSequence,
					Record:   record,
					RunStart: start,
				})
			}
		}()
	}
	start = time.Now()
	go func() {
		for sequence := 0; sequence < count; sequence++ {
			jobs <- sequence
		}
		close(jobs)
		wait.Wait()
		close(results)
	}()
	ordered := make([]client.Result, count)
	for result := range results {
		index := result.Sequence
		if warmup {
			index = -index - 1
		}
		ordered[index] = result
	}
	return ordered, start, time.Now()
}

func summarize(results []client.Result, duration time.Duration) Summary {
	summary := Summary{
		Attempted:        len(results),
		ErrorsByCategory: make(map[string]int),
	}
	var e2e []float64
	var ttft []float64
	var outputTokens int64
	for _, result := range results {
		if !result.Success {
			summary.Failed++
			summary.ErrorsByCategory[result.ErrorCategory]++
			continue
		}
		summary.Successful++
		e2e = append(e2e, result.EndToEndMS)
		if result.TTFTMS != nil {
			ttft = append(ttft, *result.TTFTMS)
		}
		if result.CompletionTokens != nil {
			summary.AuthoritativeTokenCoverage++
			outputTokens += *result.CompletionTokens
		}
	}
	if summary.Attempted > 0 {
		summary.ErrorRate = float64(summary.Failed) / float64(summary.Attempted)
	}
	seconds := duration.Seconds()
	if seconds > 0 {
		summary.RequestThroughputRPS = float64(summary.Successful) / seconds
		if summary.Successful > 0 && summary.AuthoritativeTokenCoverage == summary.Successful {
			throughput := float64(outputTokens) / seconds
			summary.OutputThroughputTPS = &throughput
			summary.OutputTokenSource = "client_usage"
		}
	}
	summary.E2E = stats.Summarize(e2e)
	if len(ttft) > 0 {
		value := stats.Summarize(ttft)
		summary.TTFT = &value
	}
	return summary
}
