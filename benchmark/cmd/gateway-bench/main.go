// Command gateway-bench runs one reproducible closed-loop workload point.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/client"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/metrics"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/results"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/runner"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/sampler"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/workload"
)

const harnessVersion = "0.8.0"

var gatewayMetrics = []string{
	"gateway_requests_total",
	"gateway_request_duration_seconds",
	"gateway_admission_queue_wait_seconds",
	"gateway_client_ttft_seconds",
	"gateway_backend_ttft_seconds",
	"gateway_observed_output_tokens_total",
	"gateway_token_accounting_requests_total",
	"gateway_errors_total",
	"gateway_batch_eligibility_total",
	"gateway_batches_total",
	"gateway_batch_size",
	"gateway_batch_wait_seconds",
	"gateway_admission_inflight",
	"gateway_admission_queued",
	"gateway_backend_healthy",
	"gateway_backend_inflight",
	"gateway_batch_pending",
	"gateway_batch_inflight",
}

var vllmMetrics = []string{
	"vllm:generation_tokens",
	"vllm:prompt_tokens",
	"vllm:request_success",
	"vllm:num_preemptions",
	"vllm:num_requests_running",
	"vllm:num_requests_waiting",
	"vllm:kv_cache_usage_perc",
	"vllm:time_to_first_token_seconds",
	"vllm:e2e_request_latency_seconds",
	"vllm:request_queue_time_seconds",
	"vllm:request_time_per_output_token_seconds",
}

type options struct {
	baseURL            string
	endpoint           string
	model              string
	dataset            string
	mode               string
	concurrency        int
	requests           int
	warmup             int
	timeout            time.Duration
	label              string
	output             string
	temperature        float64
	topP               float64
	maxTokens          int
	seed               int
	gatewayMetricsURL  string
	vllmMetricsURL     string
	sampleInterval     time.Duration
	gpuSampleInterval  time.Duration
	outputTokenCounter string
	runID              string
	repeat             int
	executionOrder     int
	batchingEnabled    string
	batchMaxSize       int
	batchMaxWait       float64
	tenantMaxInflight  int
	globalMaxInflight  int
	prefixCaching      string
	gatewayVersion     string
	gatewayGitSHA      string
	vllmVersion        string
	vllmConfigPath     string
	gatewayConfigPath  string
	environmentPath    string
}

func main() {
	if err := run(context.Background(), os.Args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "gateway-bench:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, arguments []string) error {
	options, err := parseFlags(arguments)
	if err != nil {
		return err
	}
	corpus, err := workload.Load(options.dataset)
	if err != nil {
		return err
	}
	mode := client.Mode(options.mode)
	benchmarkClient, err := client.New(client.Config{
		BaseURL:     options.baseURL,
		Endpoint:    options.endpoint,
		Model:       options.model,
		Mode:        mode,
		Generation:  client.Generation{Temperature: options.temperature, TopP: options.topP, MaxTokens: options.maxTokens, Seed: options.seed},
		Timeout:     options.timeout,
		AuthToken:   os.Getenv("BENCH_AUTH_TOKEN"),
		Concurrency: options.concurrency,
	})
	if err != nil {
		return err
	}
	defer benchmarkClient.Close()
	benchmarkRunner, err := runner.New(runner.Config{
		Client: benchmarkClient, Corpus: corpus, Concurrency: options.concurrency,
		Requests: options.requests, Warmup: options.warmup,
	})
	if err != nil {
		return err
	}

	artifactDir := filepath.Dir(options.output)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		return err
	}
	warmup := benchmarkRunner.Warmup(ctx)
	before := make(map[string]metrics.Snapshot)
	after := make(map[string]metrics.Snapshot)
	artifacts := results.Artifacts{Manifest: "manifest.json", Summary: "summary.json", Samples: "samples.csv"}
	if options.gatewayMetricsURL != "" {
		artifacts.GatewayBefore = "gateway-before.prom"
		artifacts.GatewayAfter = "gateway-after.prom"
		if before["gateway"], err = captureMetrics(ctx, options.gatewayMetricsURL, filepath.Join(artifactDir, artifacts.GatewayBefore)); err != nil {
			return fmt.Errorf("capture gateway metrics before run: %w", err)
		}
	}
	if options.vllmMetricsURL != "" {
		artifacts.VLLMBefore = "vllm-before.prom"
		artifacts.VLLMAfter = "vllm-after.prom"
		if before["vllm"], err = captureMetrics(ctx, options.vllmMetricsURL, filepath.Join(artifactDir, artifacts.VLLMBefore)); err != nil {
			return fmt.Errorf("capture vLLM metrics before run: %w", err)
		}
	}

	sampleFile, err := os.Create(filepath.Join(artifactDir, artifacts.Samples))
	if err != nil {
		return err
	}
	samplerStart := time.Now()
	runtimeSampler := sampler.StartRuntime(ctx, sampleFile, options.gatewayMetricsURL, options.vllmMetricsURL, options.sampleInterval, samplerStart)
	gpuInfo := sampler.ProbeGPU(ctx)
	gpuInfo.Enabled = options.gpuSampleInterval > 0
	var gpuSampler *sampler.GPU
	var gpuFile *os.File
	if gpuInfo.Available && gpuInfo.Enabled {
		artifacts.GPU = "gpu.csv"
		gpuFile, err = os.Create(filepath.Join(artifactDir, artifacts.GPU))
		if err != nil {
			runtimeSampler.Stop()
			_ = sampleFile.Close()
			return err
		}
		gpuSampler = sampler.StartGPU(ctx, gpuFile, options.gpuSampleInterval, samplerStart)
	}
	measured := benchmarkRunner.Measure(ctx)
	runtimeSampler.Stop()
	if err := sampleFile.Close(); err != nil {
		return err
	}
	if gpuSampler != nil {
		gpuSampler.Stop()
		if err := gpuFile.Close(); err != nil {
			return err
		}
	}
	if options.gatewayMetricsURL != "" {
		if after["gateway"], err = captureMetrics(ctx, options.gatewayMetricsURL, filepath.Join(artifactDir, artifacts.GatewayAfter)); err != nil {
			return fmt.Errorf("capture gateway metrics after run: %w", err)
		}
	}
	if options.vllmMetricsURL != "" {
		if after["vllm"], err = captureMetrics(ctx, options.vllmMetricsURL, filepath.Join(artifactDir, artifacts.VLLMAfter)); err != nil {
			return fmt.Errorf("capture vLLM metrics after run: %w", err)
		}
	}

	vllmConfig, err := loadNonSecretJSON(options.vllmConfigPath)
	if err != nil {
		return fmt.Errorf("vLLM config: %w", err)
	}
	gatewayConfig, err := loadNonSecretJSON(options.gatewayConfigPath)
	if err != nil {
		return fmt.Errorf("gateway config: %w", err)
	}
	environmentExtra, err := loadNonSecretJSON(options.environmentPath)
	if err != nil {
		return fmt.Errorf("environment manifest: %w", err)
	}
	configuration, err := makeConfiguration(options, corpus, mode, gatewayConfig, vllmConfig)
	if err != nil {
		return err
	}
	validity := results.Validity{Valid: measured.Summary.Attempted > 0, Reasons: []string{}}
	if !validity.Valid {
		validity.Reasons = append(validity.Reasons, "zero_measured_requests")
	}
	if options.vllmConfigPath == "" {
		validity.Valid = false
		validity.Reasons = append(validity.Reasons, "vllm_launch_config_not_recorded")
	}
	metricSummary := analyzeMetrics(before, after, options.outputTokenCounter, measured.Duration, &measured.Summary, &validity)
	metadata := results.NewMetadata(options.runID, options.label, options.repeat, options.executionOrder)
	environment := buildEnvironment(options, gpuInfo, environmentExtra)
	result := results.Result{
		SchemaVersion: 1,
		Metadata:      metadata,
		Configuration: configuration,
		Timing: results.Timing{
			MeasuredStartUTC: measured.StartedAt.UTC().Format(time.RFC3339Nano),
			MeasuredEndUTC:   measured.EndedAt.UTC().Format(time.RFC3339Nano),
			DurationSeconds:  measured.Duration.Seconds(),
		},
		Warmup: warmup, Summary: measured.Summary, PerRequest: measured.Requests,
		Metrics: metricSummary, Artifacts: artifacts, Environment: environment, Validity: validity,
	}
	manifest := buildManifest(result, options, vllmConfig, gatewayConfig, gpuInfo)
	if err := results.WriteJSON(filepath.Join(artifactDir, artifacts.Manifest), manifest); err != nil {
		return err
	}
	if err := results.WriteJSON(filepath.Join(artifactDir, artifacts.Summary), map[string]any{
		"schema_version": 1, "metadata": metadata, "configuration": configuration,
		"timing": result.Timing, "warmup": warmup, "summary": measured.Summary,
		"metrics": metricSummary, "validity": validity,
	}); err != nil {
		return err
	}
	if err := results.WriteJSON(options.output, result); err != nil {
		return err
	}
	printSummary(result)
	return nil
}

func parseFlags(arguments []string) (options, error) {
	var value options
	flags := flag.NewFlagSet("gateway-bench", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	flags.StringVar(&value.baseURL, "base-url", "", "target HTTP(S) base URL")
	flags.StringVar(&value.endpoint, "endpoint", "/v1/chat/completions", "Chat Completions endpoint")
	flags.StringVar(&value.model, "model", "", "model identifier")
	flags.StringVar(&value.dataset, "dataset", "", "deterministic JSONL corpus")
	flags.StringVar(&value.mode, "mode", "non_streaming", "streaming or non_streaming")
	flags.IntVar(&value.concurrency, "concurrency", 1, "maximum outstanding requests")
	flags.IntVar(&value.requests, "requests", 200, "measured request count")
	flags.IntVar(&value.warmup, "warmup", 20, "unmeasured warmup count")
	flags.DurationVar(&value.timeout, "timeout", 120*time.Second, "per-request HTTP timeout")
	flags.StringVar(&value.label, "label", "", "target label only; does not change request behavior")
	flags.StringVar(&value.output, "output", "", "client result JSON path")
	flags.Float64Var(&value.temperature, "temperature", 0, "generation temperature")
	flags.Float64Var(&value.topP, "top-p", 1, "generation top_p")
	flags.IntVar(&value.maxTokens, "max-tokens", 128, "maximum generated tokens")
	flags.IntVar(&value.seed, "seed", 1, "fixed generation seed")
	flags.StringVar(&value.gatewayMetricsURL, "gateway-metrics-url", "", "optional gateway Prometheus URL")
	flags.StringVar(&value.vllmMetricsURL, "vllm-metrics-url", "", "optional vLLM Prometheus URL")
	flags.DurationVar(&value.sampleInterval, "sample-interval", 0, "runtime metric interval; zero disables samples")
	flags.DurationVar(&value.gpuSampleInterval, "gpu-sample-interval", 0, "nvidia-smi interval; zero disables sampling")
	flags.StringVar(&value.outputTokenCounter, "output-token-counter", "", "authoritative Prometheus counter name")
	flags.StringVar(&value.runID, "run-id", "", "stable run identifier")
	flags.IntVar(&value.repeat, "repeat", 1, "one-based repetition number")
	flags.IntVar(&value.executionOrder, "execution-order", 1, "one-based arm execution position")
	flags.StringVar(&value.batchingEnabled, "batching-enabled", "not_applicable", "true, false, or not_applicable metadata")
	flags.IntVar(&value.batchMaxSize, "batch-max-size", 0, "recorded gateway max batch size")
	flags.Float64Var(&value.batchMaxWait, "batch-max-wait-seconds", 0, "recorded gateway max batch wait")
	flags.IntVar(&value.tenantMaxInflight, "tenant-max-inflight", 0, "recorded tenant admission limit")
	flags.IntVar(&value.globalMaxInflight, "global-max-inflight", 0, "recorded global admission limit")
	flags.StringVar(&value.prefixCaching, "prefix-caching", "unknown", "enabled, disabled, or unknown")
	flags.StringVar(&value.gatewayVersion, "gateway-version", harnessVersion, "recorded gateway version")
	flags.StringVar(&value.gatewayGitSHA, "gateway-git-sha", "", "recorded gateway Git SHA")
	flags.StringVar(&value.vllmVersion, "vllm-version", "unknown", "recorded vLLM version")
	flags.StringVar(&value.vllmConfigPath, "vllm-config", "", "non-secret exact vLLM launch config JSON")
	flags.StringVar(&value.gatewayConfigPath, "gateway-config", "", "non-secret gateway config JSON")
	flags.StringVar(&value.environmentPath, "environment", "", "optional sample_system.py JSON")
	flags.Usage = func() {
		fmt.Fprintln(flags.Output(), "Usage: gateway-bench --base-url URL --model MODEL --dataset FILE --label LABEL --output FILE [options]")
		fmt.Fprintln(flags.Output(), "Authorization is read only from BENCH_AUTH_TOKEN and is never serialized.")
		flags.PrintDefaults()
	}
	if err := flags.Parse(arguments); err != nil {
		return value, err
	}
	if flags.NArg() != 0 {
		return value, fmt.Errorf("unexpected positional arguments")
	}
	if value.baseURL == "" || value.model == "" || value.dataset == "" || value.label == "" || value.output == "" {
		flags.Usage()
		return value, fmt.Errorf("base-url, model, dataset, label, and output are required")
	}
	if value.mode != string(client.Streaming) && value.mode != string(client.NonStreaming) {
		return value, fmt.Errorf("mode must be streaming or non_streaming")
	}
	if value.requests <= 0 || value.concurrency <= 0 || value.warmup < 0 || value.maxTokens <= 0 {
		return value, fmt.Errorf("requests, concurrency, and max-tokens must be positive; warmup cannot be negative")
	}
	if value.repeat <= 0 || value.executionOrder <= 0 {
		return value, fmt.Errorf("repeat and execution-order must be positive")
	}
	if value.batchingEnabled != "true" && value.batchingEnabled != "false" && value.batchingEnabled != "not_applicable" {
		return value, fmt.Errorf("batching-enabled must be true, false, or not_applicable")
	}
	if value.gatewayGitSHA != "" && !isFullGitSHA(value.gatewayGitSHA) {
		return value, fmt.Errorf("gateway-git-sha must be a full 40-character hexadecimal SHA")
	}
	for name, target := range map[string]string{
		"base-url":            value.baseURL,
		"gateway-metrics-url": value.gatewayMetricsURL,
		"vllm-metrics-url":    value.vllmMetricsURL,
	} {
		if target != "" {
			parsed, err := url.Parse(target)
			if err != nil || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
				return value, fmt.Errorf("%s must not contain credentials, query, or fragment", name)
			}
		}
	}
	return value, nil
}

func isFullGitSHA(value string) bool {
	if len(value) != 40 {
		return false
	}
	for _, character := range value {
		if !((character >= '0' && character <= '9') ||
			(character >= 'a' && character <= 'f') ||
			(character >= 'A' && character <= 'F')) {
			return false
		}
	}
	return true
}

func captureMetrics(ctx context.Context, target, output string) (metrics.Snapshot, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d", response.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, (32<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > 32<<20 {
		return nil, fmt.Errorf("metrics response exceeds 32 MiB")
	}
	if err := os.WriteFile(output, raw, 0o644); err != nil {
		return nil, err
	}
	return metrics.Parse(string(raw))
}

func loadNonSecretJSON(path string) (map[string]any, error) {
	if path == "" {
		return map[string]any{"recorded": false}, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	if err := results.ValidateNonSecretConfig(value); err != nil {
		return nil, err
	}
	value["recorded"] = true
	return value, nil
}

func makeConfiguration(options options, corpus workload.Corpus, mode client.Mode, gatewayConfig, vllmConfig map[string]any) (results.Configuration, error) {
	workloadValue := map[string]any{
		"dataset_sha256": corpus.SHA256, "model": options.model, "mode": mode,
		"temperature": options.temperature, "top_p": options.topP, "max_tokens": options.maxTokens,
		"seed": options.seed, "n": 1, "requests": options.requests, "concurrency": options.concurrency,
	}
	workloadFingerprint, err := results.Fingerprint(workloadValue)
	if err != nil {
		return results.Configuration{}, err
	}
	gatewayFingerprint, err := results.Fingerprint(map[string]any{
		"launch_configuration":   gatewayConfig,
		"gateway_version":        options.gatewayVersion,
		"gateway_git_sha":        options.gatewayGitSHA,
		"batching_enabled":       options.batchingEnabled,
		"batch_max_size":         options.batchMaxSize,
		"batch_max_wait_seconds": options.batchMaxWait,
		"tenant_max_inflight":    options.tenantMaxInflight,
		"global_max_inflight":    options.globalMaxInflight,
	})
	if err != nil {
		return results.Configuration{}, err
	}
	vllmFingerprint, err := results.Fingerprint(map[string]any{
		"launch_configuration": vllmConfig,
		"version":              options.vllmVersion,
		"prefix_caching":       options.prefixCaching,
	})
	if err != nil {
		return results.Configuration{}, err
	}
	return results.Configuration{
		BaseURL: options.baseURL, Endpoint: options.endpoint, Model: options.model, Mode: mode,
		Concurrency: options.concurrency, Requests: options.requests, Warmup: options.warmup,
		TimeoutSeconds: options.timeout.Seconds(), DatasetSHA256: corpus.SHA256,
		Temperature: options.temperature, TopP: options.topP, MaxTokens: options.maxTokens, Seed: options.seed,
		N: 1, Stream: mode == client.Streaming, StreamIncludeUsage: mode == client.Streaming,
		BatchingEnabled: options.batchingEnabled, BatchMaxSize: options.batchMaxSize,
		BatchMaxWaitSeconds: options.batchMaxWait, TenantMaxInflight: options.tenantMaxInflight,
		GlobalMaxInflight: options.globalMaxInflight, PrefixCaching: options.prefixCaching,
		WorkloadFingerprint: workloadFingerprint, GatewayFingerprint: gatewayFingerprint, VLLMFingerprint: vllmFingerprint,
	}, nil
}

func analyzeMetrics(before, after map[string]metrics.Snapshot, outputCounter string, duration time.Duration, summary *runner.Summary, validity *results.Validity) results.MetricsSummary {
	metricSummary := results.MetricsSummary{
		GatewayAvailable: []string{}, GatewayMissing: []string{},
		VLLMAvailable: []string{}, VLLMMissing: []string{}, CounterDeltas: make(map[string]any),
	}
	if snapshot := after["gateway"]; snapshot != nil {
		metricSummary.GatewayAvailable, metricSummary.GatewayMissing = results.Availability(
			gatewayMetrics, func(name string) bool { return metricFamilyAvailable(snapshot, name) },
		)
		batch := &results.BatchSummary{}
		var reset bool
		batch.MeanSize, reset = metrics.HistogramMean(before["gateway"], after["gateway"], "gateway_batch_size")
		markReset(reset, "gateway_batch_size", validity)
		batch.SizeP95, reset = metrics.HistogramPercentile(before["gateway"], after["gateway"], "gateway_batch_size", 0.95)
		markReset(reset, "gateway_batch_size", validity)
		batch.WaitP95MS, reset = metrics.HistogramPercentile(before["gateway"], after["gateway"], "gateway_batch_wait_seconds", 0.95)
		markReset(reset, "gateway_batch_wait_seconds", validity)
		secondsToMilliseconds(batch.WaitP95MS)
		batch.QueueWaitP95MS, reset = metrics.HistogramPercentile(before["gateway"], after["gateway"], "gateway_admission_queue_wait_seconds", 0.95)
		markReset(reset, "gateway_admission_queue_wait_seconds", validity)
		secondsToMilliseconds(batch.QueueWaitP95MS)
		size := metrics.CounterDelta(before["gateway"], after["gateway"], "gateway_batches_total", map[string]string{"flush_reason": "size"})
		timeout := metrics.CounterDelta(before["gateway"], after["gateway"], "gateway_batches_total", map[string]string{"flush_reason": "timeout"})
		if size.Found && !size.Reset {
			batch.SizeFlushes = &size.Value
		}
		if timeout.Found && !timeout.Reset {
			batch.TimeoutFlushes = &timeout.Value
		}
		markReset(size.Reset, "gateway_batches_total", validity)
		markReset(timeout.Reset, "gateway_batches_total", validity)
		metricSummary.Batch = batch
	}
	if snapshot := after["vllm"]; snapshot != nil {
		metricSummary.VLLMAvailable, metricSummary.VLLMMissing = results.Availability(
			vllmMetrics, func(name string) bool { return metricFamilyAvailable(snapshot, name) },
		)
	}
	if outputCounter != "" {
		source := ""
		for _, candidate := range []string{"gateway", "vllm"} {
			if after[candidate].Has(outputCounter) {
				source = candidate
				break
			}
		}
		if source == "" {
			validity.Valid = false
			validity.Reasons = append(validity.Reasons, "output_token_counter_missing")
			metricSummary.CounterDeltas[outputCounter] = map[string]any{"missing": true}
		} else {
			delta := metrics.CounterDelta(before[source], after[source], outputCounter, nil)
			metricSummary.CounterDeltas[outputCounter] = delta
			if delta.Reset {
				markReset(true, outputCounter, validity)
				summary.OutputThroughputTPS = nil
				summary.OutputTokenSource = ""
			} else if delta.Found && !delta.Missing && duration > 0 {
				throughput := delta.Value / duration.Seconds()
				summary.OutputThroughputTPS = &throughput
				summary.OutputTokenSource = "prometheus_counter:" + outputCounter
			}
		}
	}
	return metricSummary
}

func metricFamilyAvailable(snapshot metrics.Snapshot, name string) bool {
	return snapshot.Has(name) || snapshot.Has(name+"_count") || snapshot.Has(name+"_bucket")
}

func markReset(reset bool, metric string, validity *results.Validity) {
	if !reset {
		return
	}
	validity.Valid = false
	reason := "invalid_counter_reset:" + metric
	for _, existing := range validity.Reasons {
		if existing == reason {
			return
		}
	}
	validity.Reasons = append(validity.Reasons, reason)
}

func secondsToMilliseconds(value *float64) {
	if value != nil {
		*value *= 1000
	}
}

func buildEnvironment(options options, gpu sampler.GPUInfo, extra map[string]any) map[string]any {
	hostname, _ := os.Hostname()
	sha := options.gatewayGitSHA
	if sha == "" {
		sha = commandOutput("git", "rev-parse", "HEAD")
	}
	environment := map[string]any{
		"gateway_git_sha": sha, "gateway_version": options.gatewayVersion,
		"benchmark_harness_git_sha": commandOutput("git", "rev-parse", "HEAD"),
		"benchmark_harness_version": harnessVersion, "host": hostname,
		"kernel": commandOutput("uname", "-srvm"), "os": runtime.GOOS, "architecture": runtime.GOARCH,
		"go_version": runtime.Version(), "python_version": commandOutput("python3", "--version"),
		"cuda_version": commandOutput("nvcc", "--version"),
		"vllm_version": options.vllmVersion, "gpu_sampling": gpu,
		"operator_environment": extra,
	}
	return environment
}

func buildManifest(result results.Result, options options, vllmConfig, gatewayConfig map[string]any, gpu sampler.GPUInfo) map[string]any {
	return map[string]any{
		"schema_version": 1, "benchmark_run_id": result.Metadata.RunID,
		"utc_timestamp": result.Metadata.TimestampUTC, "target_label": result.Metadata.Label,
		"repeat": result.Metadata.Repeat, "execution_order": result.Metadata.ExecutionOrder,
		"gateway_git_sha": result.Environment["gateway_git_sha"], "gateway_version": options.gatewayVersion,
		"benchmark_harness_git_sha": result.Environment["benchmark_harness_git_sha"],
		"model":                     options.model, "endpoint": options.endpoint, "stream_mode": options.mode,
		"concurrency": options.concurrency, "request_count": options.requests, "warmup": result.Warmup,
		"dataset_sha256":        result.Configuration.DatasetSHA256,
		"generation_parameters": map[string]any{"temperature": options.temperature, "top_p": options.topP, "max_tokens": options.maxTokens, "seed": options.seed, "n": 1},
		"gateway": map[string]any{
			"batching_enabled": options.batchingEnabled, "batch_max_size": options.batchMaxSize,
			"batch_max_wait_seconds": options.batchMaxWait, "tenant_max_inflight": options.tenantMaxInflight,
			"global_max_inflight": options.globalMaxInflight, "launch_configuration": gatewayConfig,
			"config_fingerprint": result.Configuration.GatewayFingerprint,
		},
		"vllm": map[string]any{
			"version": options.vllmVersion, "prefix_caching": options.prefixCaching,
			"launch_configuration": vllmConfig, "config_fingerprint": result.Configuration.VLLMFingerprint,
		},
		"workload_config_fingerprint": result.Configuration.WorkloadFingerprint,
		"gpu":                         gpu, "host": result.Environment,
		"client_timeout_seconds": options.timeout.Seconds(), "validity": result.Validity,
	}
}

func commandOutput(name string, arguments ...string) string {
	raw, err := exec.Command(name, arguments...).CombinedOutput()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(raw))
}

func printSummary(result results.Result) {
	fmt.Printf("label: %s\nconcurrency: %d\nrequests: %d\nsuccess: %d\nerrors: %d\n\n", result.Metadata.Label, result.Configuration.Concurrency, result.Summary.Attempted, result.Summary.Successful, result.Summary.Failed)
	printLatencies("E2E", result.Summary.E2E.P50MS, result.Summary.E2E.P95MS, result.Summary.E2E.P99MS)
	if result.Summary.TTFT != nil {
		printLatencies("TTFT", result.Summary.TTFT.P50MS, result.Summary.TTFT.P95MS, result.Summary.TTFT.P99MS)
	}
	fmt.Printf("request throughput:\n%.3f req/s\n\n", result.Summary.RequestThroughputRPS)
	if result.Summary.OutputThroughputTPS != nil {
		fmt.Printf("observed output throughput:\n%.3f tok/s (%s)\n\n", *result.Summary.OutputThroughputTPS, result.Summary.OutputTokenSource)
	}
	if result.Metrics.Batch != nil {
		fmt.Println("batch:")
		fmt.Printf("mean size %s\np95 wait %s ms\nsize flushes %s\ntimeout flushes %s\n", formatOptional(result.Metrics.Batch.MeanSize), formatOptional(result.Metrics.Batch.WaitP95MS), formatOptional(result.Metrics.Batch.SizeFlushes), formatOptional(result.Metrics.Batch.TimeoutFlushes))
	}
	if !result.Validity.Valid {
		fmt.Printf("\nvalidity: invalid (%s)\n", strings.Join(result.Validity.Reasons, ", "))
	}
}

func printLatencies(name string, p50, p95, p99 *float64) {
	fmt.Printf("%s:\np50 %s ms\np95 %s ms\np99 %s ms\n\n", name, formatOptional(p50), formatOptional(p95), formatOptional(p99))
}

func formatOptional(value *float64) string {
	if value == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.3f", *value)
}
