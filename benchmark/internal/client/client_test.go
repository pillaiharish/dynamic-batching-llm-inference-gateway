package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pillaiharish/dynamic-batching-llm-inference-gateway/benchmark/internal/workload"
)

func newTestClient(t *testing.T, server *httptest.Server, mode Mode, timeout time.Duration) *Client {
	t.Helper()
	value, err := New(Config{
		BaseURL: server.URL, Endpoint: "/v1/chat/completions", Model: "test-model", Mode: mode,
		Generation: Generation{Temperature: 0, TopP: 1, MaxTokens: 8, Seed: 7},
		Timeout:    timeout, Concurrency: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(value.Close)
	return value
}

func testRequest() Request {
	return Request{
		Sequence: 3, RunStart: time.Now(),
		Record: workload.Record{ID: "req-1", Messages: []workload.Message{{Role: "user", Content: "hello"}}},
	}
}

func TestNonStreamingSuccessAndRequestContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "" {
			t.Error("unexpected authorization")
		}
		var payload map[string]any
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload["model"] != "test-model" || payload["stream"] != false || payload["seed"] != float64(7) {
			t.Fatalf("unexpected payload: %#v", payload)
		}
		writer.Header().Set("Content-Type", "application/json")
		fmt.Fprint(writer, `{"choices":[{"message":{"content":"not persisted"}}],"usage":{"completion_tokens":5}}`)
	}))
	defer server.Close()

	result := newTestClient(t, server, NonStreaming, time.Second).Do(context.Background(), testRequest())
	if !result.Success || result.HTTPStatus != 200 || result.CompletionTokens == nil || *result.CompletionTokens != 5 {
		t.Fatalf("unexpected result: %#v", result)
	}
	encoded, _ := json.Marshal(result)
	if strings.Contains(string(encoded), "not persisted") {
		t.Fatal("generated output leaked into result")
	}
}

func TestGatewayErrorAndStatusCategories(t *testing.T) {
	for _, test := range []struct {
		status   int
		category string
	}{
		{400, "http_4xx"},
		{429, "http_429"},
		{503, "http_5xx"},
	} {
		t.Run(fmt.Sprint(test.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.WriteHeader(test.status)
				fmt.Fprint(writer, `{"error":{"code":"safe_code","message":"unsafe detail"}}`)
			}))
			defer server.Close()
			result := newTestClient(t, server, NonStreaming, time.Second).Do(context.Background(), testRequest())
			if result.Success || result.ErrorCategory != test.category || result.ErrorCode != "safe_code" {
				t.Fatalf("unexpected result: %#v", result)
			}
		})
	}
}

func TestTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		time.Sleep(50 * time.Millisecond)
		fmt.Fprint(writer, `{}`)
	}))
	defer server.Close()
	result := newTestClient(t, server, NonStreaming, 10*time.Millisecond).Do(context.Background(), testRequest())
	if result.ErrorCategory != "timeout" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestStreamingTTFTWaitsForFirstContentAndHandlesFragmentation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher := writer.(http.Flusher)
		fmt.Fprint(writer, ": comment\n\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
		flusher.Flush()
		time.Sleep(20 * time.Millisecond)
		fmt.Fprint(writer, "data: {\"choices\":[{\"delta\":{\"cont")
		flusher.Flush()
		time.Sleep(10 * time.Millisecond)
		fmt.Fprint(writer, "ent\":\"hello\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":4}}\n\ndata: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	result := newTestClient(t, server, Streaming, time.Second).Do(context.Background(), testRequest())
	if !result.Success || result.TTFTMS == nil || *result.TTFTMS < 20 || result.CompletionTokens == nil || *result.CompletionTokens != 4 {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestStreamingProtocolFailures(t *testing.T) {
	for name, body := range map[string]string{
		"malformed":    "data: not-json\n\n",
		"missing done": "data: {\"choices\":[]}\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(writer, body)
			}))
			defer server.Close()
			result := newTestClient(t, server, Streaming, time.Second).Do(context.Background(), testRequest())
			if result.ErrorCategory != "sse_protocol" {
				t.Fatalf("unexpected result: %#v", result)
			}
		})
	}
}

func TestStreamingEventBufferIsBounded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(writer, "data: "+strings.Repeat("x", 128)+"\n\n")
	}))
	defer server.Close()
	value, err := New(Config{
		BaseURL: server.URL, Endpoint: "/v1/chat/completions", Model: "m", Mode: Streaming,
		Generation: Generation{TopP: 1, MaxTokens: 4}, Timeout: time.Second, Concurrency: 1,
		MaxSSEEventBytes: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	result := value.Do(context.Background(), testRequest())
	if result.ErrorCode != "sse_event_too_large" {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestUnsafeErrorCodeIsNotPersisted(t *testing.T) {
	raw := []byte(`{"error":{"code":"contains a secret-like sentence"}}`)
	if code := safeErrorCode(raw); code != "http_error" {
		t.Fatalf("unexpected safe code %q", code)
	}
}

func TestRejectsCredentialInBaseURL(t *testing.T) {
	_, err := New(Config{BaseURL: "http://secret@example.test", Endpoint: "/", Model: "m", Mode: NonStreaming, Timeout: time.Second, Concurrency: 1})
	if err == nil {
		t.Fatal("expected userinfo URL rejection")
	}
}
