package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunDoesNotPersistAuthorizationSecret(t *testing.T) {
	var tokens int
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/metrics" {
			fmt.Fprintf(writer, "vllm:generation_tokens %d\n", tokens)
			return
		}
		if request.Header.Get("Authorization") != "Bearer SENTINEL-DO-NOT-PERSIST" {
			t.Error("authorization was not sent from the environment")
		}
		tokens += 2
		fmt.Fprint(writer, `{"choices":[],"usage":{"completion_tokens":2}}`)
	}))
	defer server.Close()

	directory := t.TempDir()
	dataset := filepath.Join(directory, "dataset.jsonl")
	config := filepath.Join(directory, "vllm.json")
	output := filepath.Join(directory, "artifacts", "client-result.json")
	if err := os.WriteFile(dataset, []byte(`{"id":"one","messages":[{"role":"user","content":"hello"}]}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(config, []byte(`{"model":"fake","prefix_caching":"disabled"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("BENCH_AUTH_TOKEN", "SENTINEL-DO-NOT-PERSIST")
	err := run(context.Background(), []string{
		"--base-url", server.URL,
		"--model", "fake",
		"--dataset", dataset,
		"--label", "direct",
		"--output", output,
		"--requests", "2",
		"--warmup", "1",
		"--concurrency", "1",
		"--vllm-metrics-url", server.URL + "/metrics",
		"--output-token-counter", "vllm:generation_tokens",
		"--vllm-config", config,
	})
	if err != nil {
		t.Fatal(err)
	}
	err = filepath.Walk(filepath.Dir(output), func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil || info.IsDir() {
			return walkErr
		}
		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		if strings.Contains(string(raw), "SENTINEL-DO-NOT-PERSIST") {
			t.Errorf("secret leaked into %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestFlagsRejectURLsThatCouldPersistCredentials(t *testing.T) {
	for _, target := range []string{
		"http://user:secret@example.test",
		"http://example.test?token=secret",
		"http://example.test#secret",
	} {
		_, err := parseFlags([]string{
			"--base-url", target,
			"--model", "m",
			"--dataset", "data.jsonl",
			"--label", "direct",
			"--output", "result.json",
		})
		if err == nil {
			t.Fatalf("expected rejection for %s", target)
		}
	}
}
