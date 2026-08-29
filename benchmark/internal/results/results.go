// Package results defines and writes the versioned, secret-free result contract.
package results

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/client"
	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/runner"
)

// Metadata identifies an independently retained repetition.
type Metadata struct {
	RunID          string `json:"run_id"`
	TimestampUTC   string `json:"timestamp_utc"`
	Label          string `json:"label"`
	Repeat         int    `json:"repeat"`
	ExecutionOrder int    `json:"execution_order"`
}

// Configuration is the endpoint-neutral workload and target configuration.
type Configuration struct {
	BaseURL             string         `json:"base_url"`
	Endpoint            string         `json:"endpoint"`
	Model               string         `json:"model"`
	Mode                client.Mode    `json:"mode"`
	Concurrency         int            `json:"concurrency"`
	Requests            int            `json:"requests"`
	Warmup              int            `json:"warmup"`
	TimeoutSeconds      float64        `json:"timeout_seconds"`
	DatasetSHA256       string         `json:"dataset_sha256"`
	Temperature         float64        `json:"temperature"`
	TopP                float64        `json:"top_p"`
	MaxTokens           int            `json:"max_tokens"`
	Seed                int            `json:"seed"`
	N                   int            `json:"n"`
	Stream              bool           `json:"stream"`
	StreamIncludeUsage  bool           `json:"stream_include_usage"`
	BatchingEnabled     string         `json:"batching_enabled"`
	BatchMaxSize        int            `json:"batch_max_size,omitempty"`
	BatchMaxWaitSeconds float64        `json:"batch_max_wait_seconds,omitempty"`
	TenantMaxInflight   int            `json:"tenant_max_inflight,omitempty"`
	GlobalMaxInflight   int            `json:"global_max_inflight,omitempty"`
	PrefixCaching       string         `json:"prefix_caching"`
	WorkloadFingerprint string         `json:"workload_fingerprint"`
	GatewayFingerprint  string         `json:"gateway_config_fingerprint"`
	VLLMFingerprint     string         `json:"vllm_config_fingerprint"`
	Extra               map[string]any `json:"extra,omitempty"`
}

// Timing records UTC boundaries while the duration itself comes from Go's monotonic clock.
type Timing struct {
	MeasuredStartUTC string  `json:"measured_start_utc"`
	MeasuredEndUTC   string  `json:"measured_end_utc"`
	DurationSeconds  float64 `json:"duration_seconds"`
}

// BatchSummary is derived only from before/after gateway histogram/counter deltas.
type BatchSummary struct {
	MeanSize       *float64 `json:"mean_size"`
	SizeP95        *float64 `json:"size_p95"`
	WaitP95MS      *float64 `json:"wait_p95_ms"`
	QueueWaitP95MS *float64 `json:"queue_wait_p95_ms"`
	SizeFlushes    *float64 `json:"size_flushes"`
	TimeoutFlushes *float64 `json:"timeout_flushes"`
}

// MetricsSummary tracks metric presence and counter validity.
type MetricsSummary struct {
	GatewayAvailable []string       `json:"gateway_available"`
	GatewayMissing   []string       `json:"gateway_missing"`
	VLLMAvailable    []string       `json:"vllm_available"`
	VLLMMissing      []string       `json:"vllm_missing"`
	CounterDeltas    map[string]any `json:"counter_deltas"`
	Batch            *BatchSummary  `json:"batch,omitempty"`
}

// Artifacts points to sibling raw evidence files using non-secret basenames.
type Artifacts struct {
	Manifest      string `json:"manifest"`
	Summary       string `json:"summary"`
	GatewayBefore string `json:"gateway_before,omitempty"`
	GatewayAfter  string `json:"gateway_after,omitempty"`
	VLLMBefore    string `json:"vllm_before,omitempty"`
	VLLMAfter     string `json:"vllm_after,omitempty"`
	Samples       string `json:"samples"`
	GPU           string `json:"gpu,omitempty"`
}

// Validity makes invalid comparisons/counters explicit rather than silently useful-looking.
type Validity struct {
	Valid   bool     `json:"valid"`
	Reasons []string `json:"reasons"`
}

// Result is schema_version 1.
type Result struct {
	SchemaVersion int                  `json:"schema_version"`
	Metadata      Metadata             `json:"metadata"`
	Configuration Configuration        `json:"configuration"`
	Timing        Timing               `json:"timing"`
	Warmup        runner.WarmupSummary `json:"warmup"`
	Summary       runner.Summary       `json:"summary"`
	PerRequest    []client.Result      `json:"per_request"`
	Metrics       MetricsSummary       `json:"metrics"`
	Artifacts     Artifacts            `json:"metric_artifacts"`
	Environment   map[string]any       `json:"environment"`
	Validity      Validity             `json:"validity"`
}

// WriteJSON creates a deterministic, indented JSON artifact.
func WriteJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

// Fingerprint hashes normalized JSON. encoding/json sorts string map keys.
func Fingerprint(value any) (string, error) {
	normalized, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(normalized)
	return hex.EncodeToString(sum[:]), nil
}

// ValidateNonSecretConfig rejects likely secret-bearing keys from persisted config files.
func ValidateNonSecretConfig(value any) error {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			lower := strings.ToLower(key)
			for _, forbidden := range []string{"secret", "password", "api_key", "authorization", "auth_token", "credential"} {
				if strings.Contains(lower, forbidden) {
					return fmt.Errorf("configuration key %q may contain a secret and cannot be persisted", key)
				}
			}
			if err := ValidateNonSecretConfig(child); err != nil {
				return err
			}
		}
	case []any:
		for _, child := range typed {
			if err := ValidateNonSecretConfig(child); err != nil {
				return err
			}
		}
	}
	return nil
}

// Availability partitions an expected metric list deterministically.
func Availability(expected []string, has func(string) bool) (available []string, missing []string) {
	for _, name := range expected {
		if has(name) {
			available = append(available, name)
		} else {
			missing = append(missing, name)
		}
	}
	sort.Strings(available)
	sort.Strings(missing)
	return available, missing
}

// NewMetadata uses a UTC identifier and preserves repeat/order explicitly.
func NewMetadata(runID, label string, repeat, order int) Metadata {
	now := time.Now().UTC()
	if runID == "" {
		runID = now.Format("20060102T150405.000000000Z") + "-" + label
	}
	return Metadata{
		RunID:          runID,
		TimestampUTC:   now.Format(time.RFC3339Nano),
		Label:          label,
		Repeat:         repeat,
		ExecutionOrder: order,
	}
}
