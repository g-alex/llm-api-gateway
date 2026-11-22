package zai

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTokenFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokens.txt")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadPoolParsesAndDedups(t *testing.T) {
	path := writeTokenFile(t, "# comment\n\ntok-a\ntok-b\n  tok-a  \ntok-c\n")
	p, err := LoadPool(path)
	if err != nil {
		t.Fatalf("LoadPool: %v", err)
	}
	if got := p.Size(); got != 3 {
		t.Fatalf("Size = %d, want 3", got)
	}
	if got := p.Current(); got != "tok-a" {
		t.Fatalf("Current = %q, want tok-a", got)
	}
}

func TestLoadPoolEmptyFile(t *testing.T) {
	path := writeTokenFile(t, "# only comments\n\n")
	if _, err := LoadPool(path); err == nil {
		t.Fatal("expected error for file with no usable tokens")
	}
}

func TestLoadPoolMissingFile(t *testing.T) {
	if _, err := LoadPool(filepath.Join(t.TempDir(), "nope.txt")); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestPoolRotationWraps(t *testing.T) {
	p, err := LoadPool(writeTokenFile(t, "a\nb\nc\n"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"b", "c", "a", "b"}
	for _, w := range want {
		if got := p.Rotate(); got != w {
			t.Fatalf("Rotate = %q, want %q", got, w)
		}
	}
	if got := p.Current(); got != "b" {
		t.Fatalf("Current after rotates = %q, want b", got)
	}
}

func TestNilPoolSafe(t *testing.T) {
	var p *Pool
	if p.Size() != 0 || p.Current() != "" || p.Rotate() != "" {
		t.Fatal("nil pool must degrade to zero values")
	}
}

func TestClassifyAuthCheck(t *testing.T) {
	cases := []struct {
		status int
		body   string
		want   string
	}{
		{200, `{"data":{"id":"u1"}}`, "alive"},
		{204, "", "ambiguous"},                // 2xx without id: no proof
		{401, `{"detail":"expired"}`, "dead"}, //
		{403, "USER_BLOCKED", "dead"},         //
		{410, "", "dead"},                     //
		{500, "boom", "ambiguous"},            // server error says nothing about the account
		{-1, "request error: x", "ambiguous"}, // network failure
	}
	for _, c := range cases {
		if got := ClassifyAuthCheck(c.status, c.body); got != c.want {
			t.Errorf("ClassifyAuthCheck(%d, %q) = %q, want %q", c.status, c.body, got, c.want)
		}
	}
}
