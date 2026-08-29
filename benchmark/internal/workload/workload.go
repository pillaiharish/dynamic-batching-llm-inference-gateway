// Package workload loads the deterministic JSONL request corpus.
package workload

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
)

const maxRecordBytes = 1 << 20

// Message is the text-only OpenAI-compatible message subset used by the harness.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Record is one deterministic corpus entry.
type Record struct {
	ID       string    `json:"id"`
	Messages []Message `json:"messages"`
}

// Corpus retains ordered records and the SHA-256 of the exact input bytes.
type Corpus struct {
	Records []Record
	SHA256  string
}

// Load reads and validates a JSONL corpus without reordering it.
func Load(path string) (Corpus, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Corpus{}, fmt.Errorf("read dataset: %w", err)
	}
	sum := sha256.Sum256(raw)
	corpus := Corpus{SHA256: hex.EncodeToString(sum[:])}
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 64*1024), maxRecordBytes)
	line := 0
	for scanner.Scan() {
		line++
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			continue
		}
		var record Record
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return Corpus{}, fmt.Errorf("dataset line %d: %w", line, err)
		}
		if record.ID == "" || len(record.Messages) == 0 {
			return Corpus{}, fmt.Errorf("dataset line %d: id and messages are required", line)
		}
		if _, exists := seen[record.ID]; exists {
			return Corpus{}, fmt.Errorf("dataset line %d: duplicate id %q", line, record.ID)
		}
		seen[record.ID] = struct{}{}
		for _, message := range record.Messages {
			if (message.Role != "system" && message.Role != "user" && message.Role != "assistant") || message.Content == "" {
				return Corpus{}, fmt.Errorf("dataset line %d: invalid message", line)
			}
		}
		corpus.Records = append(corpus.Records, record)
	}
	if err := scanner.Err(); err != nil {
		return Corpus{}, fmt.Errorf("scan dataset: %w", err)
	}
	if len(corpus.Records) == 0 {
		return Corpus{}, fmt.Errorf("dataset contains no records")
	}
	return corpus, nil
}
