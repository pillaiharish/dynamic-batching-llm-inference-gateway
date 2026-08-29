// Package sampler captures optional runtime Prometheus gauges and NVIDIA data.
package sampler

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/metrics"
)

var runtimeColumns = []struct {
	Source string
	Name   string
}{
	{"vllm", "vllm:num_requests_running"},
	{"vllm", "vllm:num_requests_waiting"},
	{"vllm", "vllm:kv_cache_usage_perc"},
	{"gateway", "gateway_admission_inflight"},
	{"gateway", "gateway_admission_queued"},
	{"gateway", "gateway_batch_pending"},
	{"gateway", "gateway_batch_inflight"},
	{"gateway", "gateway_backend_inflight"},
}

// Runtime writes a best-effort time series. Individual scrape errors produce
// blank fields plus a safe error indicator; they never become fabricated zeros.
type Runtime struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// StartRuntime starts sampling and always writes a CSV header.
func StartRuntime(parent context.Context, output io.Writer, gatewayURL, vllmURL string, interval time.Duration, start time.Time) *Runtime {
	ctx, cancel := context.WithCancel(parent)
	runtimeSampler := &Runtime{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(runtimeSampler.done)
		writer := csv.NewWriter(output)
		header := []string{"offset_seconds"}
		for _, column := range runtimeColumns {
			header = append(header, column.Name)
		}
		header = append(header, "scrape_error")
		_ = writer.Write(header)
		writer.Flush()
		if interval <= 0 || (gatewayURL == "" && vllmURL == "") {
			return
		}
		httpClient := &http.Client{Timeout: min(2*time.Second, interval)}
		sampleRuntime(ctx, writer, httpClient, gatewayURL, vllmURL, start)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				sampleRuntime(ctx, writer, httpClient, gatewayURL, vllmURL, start)
			}
		}
	}()
	return runtimeSampler
}

// Stop flushes and joins the sampler.
func (runtimeSampler *Runtime) Stop() {
	runtimeSampler.cancel()
	<-runtimeSampler.done
}

func sampleRuntime(ctx context.Context, writer *csv.Writer, httpClient *http.Client, gatewayURL, vllmURL string, start time.Time) {
	snapshots := make(map[string]metrics.Snapshot)
	var errors []string
	for source, target := range map[string]string{"gateway": gatewayURL, "vllm": vllmURL} {
		if target == "" {
			continue
		}
		snapshot, err := scrape(ctx, httpClient, target)
		if err != nil {
			errors = append(errors, source+"_scrape")
			continue
		}
		snapshots[source] = snapshot
	}
	row := []string{strconv.FormatFloat(time.Since(start).Seconds(), 'f', 6, 64)}
	for _, column := range runtimeColumns {
		value, found := snapshots[column.Source].Sum(column.Name)
		if !found {
			row = append(row, "")
		} else {
			row = append(row, strconv.FormatFloat(value, 'g', -1, 64))
		}
	}
	row = append(row, strings.Join(errors, ";"))
	_ = writer.Write(row)
	writer.Flush()
}

func scrape(ctx context.Context, httpClient *http.Client, target string) (metrics.Snapshot, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return nil, err
	}
	response, err := httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("metrics HTTP %d", response.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, 16<<20))
	if err != nil {
		return nil, err
	}
	return metrics.Parse(string(raw))
}

// GPUInfo is safe environment metadata from nvidia-smi.
type GPUInfo struct {
	Available     bool   `json:"available"`
	Enabled       bool   `json:"enabled"`
	Name          string `json:"name,omitempty"`
	MemoryTotalMB string `json:"memory_total_mib,omitempty"`
	DriverVersion string `json:"driver_version,omitempty"`
}

// ProbeGPU discovers nvidia-smi without making GPU support mandatory.
func ProbeGPU(ctx context.Context) GPUInfo {
	probeContext, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return GPUInfo{}
	}
	command := exec.CommandContext(probeContext, path,
		"--query-gpu=name,memory.total,driver_version",
		"--format=csv,noheader,nounits",
	)
	raw, err := command.Output()
	if err != nil {
		return GPUInfo{}
	}
	line := strings.Split(strings.TrimSpace(string(raw)), "\n")[0]
	fields := strings.Split(line, ",")
	if len(fields) < 3 {
		return GPUInfo{}
	}
	return GPUInfo{
		Available:     true,
		Name:          strings.TrimSpace(fields[0]),
		MemoryTotalMB: strings.TrimSpace(fields[1]),
		DriverVersion: strings.TrimSpace(fields[2]),
	}
}

// GPU samples utilization, memory, and power into CSV when available.
type GPU struct {
	cancel context.CancelFunc
	done   chan struct{}
	once   sync.Once
}

// StartGPU starts a best-effort sampler. The caller should invoke it only when
// ProbeGPU reports availability and sampling is enabled.
func StartGPU(parent context.Context, output io.Writer, interval time.Duration, start time.Time) *GPU {
	ctx, cancel := context.WithCancel(parent)
	gpu := &GPU{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(gpu.done)
		writer := csv.NewWriter(output)
		_ = writer.Write([]string{"offset_seconds", "gpu_index", "utilization_percent", "memory_used_mib", "memory_total_mib", "power_draw_watts"})
		writer.Flush()
		if interval <= 0 {
			return
		}
		sampleGPU(ctx, writer, start)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				sampleGPU(ctx, writer, start)
			}
		}
	}()
	return gpu
}

// Stop flushes and joins the GPU sampler.
func (gpu *GPU) Stop() {
	gpu.once.Do(func() { gpu.cancel() })
	<-gpu.done
}

func sampleGPU(ctx context.Context, writer *csv.Writer, start time.Time) {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return
	}
	command := exec.CommandContext(ctx, path,
		"--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
		"--format=csv,noheader,nounits",
	)
	raw, err := command.Output()
	if err != nil {
		return
	}
	offset := strconv.FormatFloat(time.Since(start).Seconds(), 'f', 6, 64)
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		fields := strings.Split(line, ",")
		if len(fields) != 5 {
			continue
		}
		row := []string{offset}
		for _, field := range fields {
			row = append(row, strings.TrimSpace(field))
		}
		_ = writer.Write(row)
	}
	writer.Flush()
}
