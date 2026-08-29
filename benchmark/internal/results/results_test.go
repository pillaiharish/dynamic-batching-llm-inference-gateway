package results

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSecretCannotEnterConfigOrArtifact(t *testing.T) {
	secret := "SENTINEL-DO-NOT-PERSIST"
	if ValidateNonSecretConfig(map[string]any{"api_key": secret}) == nil {
		t.Fatal("expected secret-bearing key rejection")
	}
	path := filepath.Join(t.TempDir(), "result.json")
	if err := WriteJSON(path, map[string]any{"label": "gateway_batch"}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), secret) {
		t.Fatal("sentinel secret leaked")
	}
}

func TestFingerprintIsIndependentOfMapInsertionOrder(t *testing.T) {
	first, _ := Fingerprint(map[string]any{"a": 1, "b": 2})
	second, _ := Fingerprint(map[string]any{"b": 2, "a": 1})
	if first != second {
		t.Fatalf("fingerprints differ: %s != %s", first, second)
	}
}
