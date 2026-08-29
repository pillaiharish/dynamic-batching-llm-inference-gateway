// Package client implements the endpoint-neutral Chat Completions benchmark client.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/workload"
)

const (
	maxResponseBytes = 16 << 20
	maxErrorBytes    = 1 << 20
	defaultSSELimit  = 1 << 20
)

var errSSEEventTooLarge = errors.New("SSE event too large")

// Mode selects response parsing; it does not select a target implementation.
type Mode string

const (
	Streaming    Mode = "streaming"
	NonStreaming Mode = "non_streaming"
)

// Generation contains request parameters shared by every target arm.
type Generation struct {
	Temperature float64
	TopP        float64
	MaxTokens   int
	Seed        int
}

// Config configures one reusable client and connection pool.
type Config struct {
	BaseURL          string
	Endpoint         string
	Model            string
	Mode             Mode
	Generation       Generation
	Timeout          time.Duration
	AuthToken        string
	Concurrency      int
	MaxSSEEventBytes int
}

// Request identifies one measured attempt and its deterministic corpus record.
type Request struct {
	Sequence int
	Record   workload.Record
	RunStart time.Time
}

// Result is safe to persist: it contains no credentials or generated content.
type Result struct {
	Sequence         int      `json:"sequence"`
	RequestID        string   `json:"request_id"`
	StartOffsetMS    float64  `json:"start_offset_ms"`
	EndToEndMS       float64  `json:"e2e_ms"`
	TTFTMS           *float64 `json:"ttft_ms"`
	HTTPStatus       int      `json:"http_status"`
	Success          bool     `json:"success"`
	ErrorCategory    string   `json:"error_category,omitempty"`
	ErrorCode        string   `json:"error_code,omitempty"`
	ResponseBytes    int64    `json:"response_bytes"`
	CompletionTokens *int64   `json:"completion_tokens"`
}

// Client owns one reusable http.Client and Transport for a run.
type Client struct {
	config Config
	http   *http.Client
	url    string
}

// New validates config and creates a concurrency-sized keepalive pool.
func New(config Config) (*Client, error) {
	if config.Mode != Streaming && config.Mode != NonStreaming {
		return nil, fmt.Errorf("mode must be streaming or non_streaming")
	}
	if config.Timeout <= 0 || config.Concurrency <= 0 {
		return nil, fmt.Errorf("timeout and concurrency must be positive")
	}
	base, err := url.Parse(strings.TrimRight(config.BaseURL, "/"))
	if err != nil || (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" || base.User != nil {
		return nil, fmt.Errorf("invalid base URL")
	}
	endpoint := "/" + strings.TrimLeft(config.Endpoint, "/")
	if config.MaxSSEEventBytes <= 0 {
		config.MaxSSEEventBytes = defaultSSELimit
	}
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          max(100, config.Concurrency*2),
		MaxIdleConnsPerHost:   max(16, config.Concurrency),
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &Client{
		config: config,
		http: &http.Client{
			Transport: transport,
			Timeout:   config.Timeout,
		},
		url: strings.TrimRight(base.String(), "/") + endpoint,
	}, nil
}

// Close releases idle keepalive connections.
func (client *Client) Close() {
	client.http.CloseIdleConnections()
}

// Do executes one request using exactly the same construction for every label.
func (client *Client) Do(ctx context.Context, request Request) Result {
	started := time.Now()
	result := Result{
		Sequence:      request.Sequence,
		RequestID:     request.Record.ID,
		StartOffsetMS: float64(started.Sub(request.RunStart).Nanoseconds()) / 1e6,
	}
	payload := map[string]any{
		"model":       client.config.Model,
		"messages":    request.Record.Messages,
		"temperature": client.config.Generation.Temperature,
		"top_p":       client.config.Generation.TopP,
		"max_tokens":  client.config.Generation.MaxTokens,
		"seed":        client.config.Generation.Seed,
		"n":           1,
		"stream":      client.config.Mode == Streaming,
	}
	if client.config.Mode == Streaming {
		payload["stream_options"] = map[string]bool{"include_usage": true}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return finish(result, started, "other", "request_encode_error")
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, client.url, bytes.NewReader(body))
	if err != nil {
		return finish(result, started, "other", "request_build_error")
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "application/json")
	if client.config.Mode == Streaming {
		httpRequest.Header.Set("Accept", "text/event-stream")
	}
	if client.config.AuthToken != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+client.config.AuthToken)
	}
	response, err := client.http.Do(httpRequest)
	if err != nil {
		category := "transport"
		var networkError net.Error
		if errors.Is(err, context.DeadlineExceeded) ||
			errors.Is(ctx.Err(), context.DeadlineExceeded) ||
			(errors.As(err, &networkError) && networkError.Timeout()) {
			category = "timeout"
		}
		return finish(result, started, category, category)
	}
	defer response.Body.Close()
	result.HTTPStatus = response.StatusCode
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBytes))
		result.ResponseBytes = int64(len(raw))
		result.ErrorCode = safeErrorCode(raw)
		result.ErrorCategory = statusCategory(response.StatusCode)
		return finish(result, started, result.ErrorCategory, result.ErrorCode)
	}
	if client.config.Mode == Streaming {
		return client.readStream(response.Body, result, started)
	}
	return readJSON(response.Body, result, started)
}

func readJSON(body io.Reader, result Result, started time.Time) Result {
	raw, err := io.ReadAll(io.LimitReader(body, maxResponseBytes+1))
	result.ResponseBytes = int64(len(raw))
	if err != nil || len(raw) > maxResponseBytes {
		return finish(result, started, "other", "response_read_error")
	}
	var response struct {
		Usage *struct {
			CompletionTokens *int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return finish(result, started, "other", "response_json_error")
	}
	if response.Usage != nil && response.Usage.CompletionTokens != nil && *response.Usage.CompletionTokens >= 0 {
		result.CompletionTokens = response.Usage.CompletionTokens
	}
	result.Success = true
	return finish(result, started, "", "")
}

type countingReader struct {
	reader io.Reader
	count  int64
}

func (reader *countingReader) Read(buffer []byte) (int, error) {
	n, err := reader.reader.Read(buffer)
	reader.count += int64(n)
	return n, err
}

func (client *Client) readStream(body io.Reader, result Result, started time.Time) Result {
	counted := &countingReader{reader: body}
	reader := bufio.NewReaderSize(counted, 32*1024)
	var event bytes.Buffer
	done := false
	for {
		line, err := readBoundedLine(reader, client.config.MaxSSEEventBytes-event.Len())
		if errors.Is(err, errSSEEventTooLarge) {
			result.ResponseBytes = counted.count
			return finish(result, started, "sse_protocol", "sse_event_too_large")
		}
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				completed, parseErr := parseEvent(event.String(), started, &result)
				event.Reset()
				if parseErr != nil {
					result.ResponseBytes = counted.count
					return finish(result, started, "sse_protocol", "sse_parse_error")
				}
				if completed {
					done = true
					break
				}
			} else {
				event.WriteString(line)
			}
		}
		if err != nil {
			if err == io.EOF {
				if event.Len() > 0 {
					completed, parseErr := parseEvent(event.String(), started, &result)
					if parseErr != nil {
						result.ResponseBytes = counted.count
						return finish(result, started, "sse_protocol", "sse_parse_error")
					}
					done = completed
				}
				break
			}
			result.ResponseBytes = counted.count
			return finish(result, started, "transport", "stream_read_error")
		}
	}
	result.ResponseBytes = counted.count
	if !done {
		return finish(result, started, "sse_protocol", "sse_missing_done")
	}
	result.Success = true
	return finish(result, started, "", "")
}

func readBoundedLine(reader *bufio.Reader, remaining int) (string, error) {
	if remaining <= 0 {
		return "", errSSEEventTooLarge
	}
	var line bytes.Buffer
	for {
		fragment, err := reader.ReadSlice('\n')
		if line.Len()+len(fragment) > remaining {
			return "", errSSEEventTooLarge
		}
		line.Write(fragment)
		if err == nil {
			return line.String(), nil
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return line.String(), err
	}
}

func parseEvent(raw string, started time.Time, result *Result) (bool, error) {
	var data []string
	for _, line := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			value := strings.TrimPrefix(line, "data:")
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if len(data) == 0 {
		return false, nil
	}
	payload := strings.Join(data, "\n")
	if payload == "[DONE]" {
		return true, nil
	}
	var event struct {
		Choices []struct {
			Delta struct {
				Content any `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
		Usage *struct {
			CompletionTokens *int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		return false, err
	}
	if result.TTFTMS == nil {
		for _, choice := range event.Choices {
			content, ok := choice.Delta.Content.(string)
			if ok && content != "" {
				value := float64(time.Since(started).Nanoseconds()) / 1e6
				result.TTFTMS = &value
				break
			}
		}
	}
	if event.Usage != nil && event.Usage.CompletionTokens != nil && *event.Usage.CompletionTokens >= 0 {
		result.CompletionTokens = event.Usage.CompletionTokens
	}
	return false, nil
}

func safeErrorCode(raw []byte) string {
	var payload struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &payload) == nil && payload.Error.Code != "" {
		if len(payload.Error.Code) <= 64 {
			valid := true
			for _, character := range payload.Error.Code {
				if !((character >= 'a' && character <= 'z') ||
					(character >= 'A' && character <= 'Z') ||
					(character >= '0' && character <= '9') ||
					strings.ContainsRune("_.-", character)) {
					valid = false
					break
				}
			}
			if valid {
				return payload.Error.Code
			}
		}
	}
	return "http_error"
}

func statusCategory(status int) string {
	if status == http.StatusTooManyRequests {
		return "http_429"
	}
	if status >= 400 && status < 500 {
		return "http_4xx"
	}
	if status >= 500 {
		return "http_5xx"
	}
	return "other"
}

func finish(result Result, started time.Time, category string, code string) Result {
	result.EndToEndMS = float64(time.Since(started).Nanoseconds()) / 1e6
	if !result.Success && category != "" {
		result.ErrorCategory = category
		if result.ErrorCode == "" {
			result.ErrorCode = code
		}
	}
	return result
}
