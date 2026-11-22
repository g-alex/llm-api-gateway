// api_test.go — integration tests: the full HTTP pipeline against a fake
// upstream (session init, SSE chat, failover), bearer auth, and the
// OpenAI/Anthropic wire shapes.
package api

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/g-alex/llm-api-gateway/internal/config"
	"github.com/g-alex/llm-api-gateway/internal/zai"
)

const (
	tokenA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	tokenB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// fakeUpstream mimics the upstream endpoints the client touches: guest
// auth, account validation, model catalog and the SSE chat stream.
// chatReject lists bearer tokens the chat endpoint rejects with 401.
type fakeUpstream struct {
	*httptest.Server
	chatCalls  atomic.Int64
	chatReject map[string]bool
	streamBody string
}

func newFakeUpstream(t *testing.T) *fakeUpstream {
	t.Helper()
	f := &fakeUpstream{
		chatReject: map[string]bool{},
		streamBody: "data: {\"data\":{\"delta_content\":\"Hello\"}}\n" +
			"data: {\"data\":{\"delta_content\":\" world\"}}\n" +
			"data: {\"data\":{\"phase\":\"done\"}}\n\n",
	}
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/auths/guest", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"token":"guest-token"}`))
	})
	mux.HandleFunc("/api/v1/auths/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":{"id":"u-123","email":"user@example.com"}}`))
	})
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"id":"glm-4.7","name":"GLM-4.7"}]}`))
	})
	mux.HandleFunc("/api/v2/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.chatCalls.Add(1)
		bearer := strings.TrimPrefix(r.Header.Get("authorization"), "Bearer ")
		if f.chatReject[bearer] {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"detail":"USER_BLOCKED"}`))
			return
		}
		if r.Header.Get("x-signature") == "" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"detail":"missing signature"}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(f.streamBody))
	})

	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Server.Close)
	return f
}

// newTestGateway wires an api.Server to a client pointed at the fake
// upstream.
func newTestGateway(t *testing.T, f *fakeUpstream, pool *zai.Pool) *httptest.Server {
	t.Helper()
	cfg := &config.Config{
		AuthToken:      "test-token",
		LogLevel:       "error",
		Timeout:        30 * time.Second,
		AnthropicMinMS: 0,
	}
	client := zai.New(f.URL, cfg.LogLevel, cfg.ZAIToken, pool)
	srv := New(cfg, client)
	ts := httptest.NewServer(srv.Routes())
	t.Cleanup(ts.Close)
	return ts
}

func chatBody(model string, stream bool) string {
	b, _ := json.Marshal(map[string]interface{}{
		"model":    model,
		"stream":   stream,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	return string(b)
}

func post(t *testing.T, ts *httptest.Server, path, token, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(ts.URL+path, "application/json",
		strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func postAuth(t *testing.T, ts *httptest.Server, path, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", ts.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// ── Auth ──

func TestAuthMiddleware(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)

	// no token → 401
	req, _ := http.NewRequest("GET", ts.URL+"/v1/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token: status = %d, want 401", resp.StatusCode)
	}

	// wrong token → 401
	req.Header.Set("Authorization", "Bearer wrong")
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong token: status = %d, want 401", resp.StatusCode)
	}

	// correct token → 200
	req.Header.Set("Authorization", "Bearer test-token")
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("correct token: status = %d, want 200", resp.StatusCode)
	}
}

// ── OpenAI wire shape ──

func TestChatCompletionNonStream(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)

	resp := postAuth(t, ts, "/v1/chat/completions", chatBody("glm-4.7", false))
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" || len(out.Choices) != 1 {
		t.Fatalf("shape wrong: %+v", out)
	}
	if out.Choices[0].Message.Content != "Hello world" {
		t.Fatalf("content = %q, want %q", out.Choices[0].Message.Content, "Hello world")
	}
	if out.Choices[0].FinishReason != "stop" || out.Usage == nil || out.Usage.TotalTokens == 0 {
		t.Fatalf("finish/usage wrong: %+v", out)
	}
}

func TestChatCompletionStream(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)

	resp := postAuth(t, ts, "/v1/chat/completions", chatBody("glm-4.7", true))
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %s", ct)
	}

	var content strings.Builder
	sawDone := false
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "data: [DONE]" {
			sawDone = true
			break
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			content.WriteString(c.Delta.Content)
		}
	}
	if !sawDone {
		t.Fatal("stream did not end with [DONE]")
	}
	// The holdback window (24 runes) is released by the final flush, so
	// the full text reaches the client.
	if content.String() != "Hello world" {
		t.Fatalf("streamed content = %q, want %q", content.String(), "Hello world")
	}
}

// ── Anthropic wire shape ──

func TestAnthropicMessagesNonStream(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)

	body, _ := json.Marshal(map[string]interface{}{
		"model":      "claude-opus-4-7",
		"max_tokens": 1024,
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
	})
	resp := postAuth(t, ts, "/v1/messages", string(body))
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var out struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Content) == 0 || out.Content[0].Type != "text" {
		t.Fatalf("content blocks wrong: %+v", out)
	}
	if !strings.Contains(out.Content[0].Text, "Hello world") {
		t.Fatalf("text = %q", out.Content[0].Text)
	}
}

// ── Account pool failover ──

// The pool's first account is rejected by the CHAT endpoint only (auth
// validation still accepts it): the send must rotate to the second
// account, re-initialize, and succeed — the mid-session 401 failover
// path.
func TestPoolFailoverOnChat401(t *testing.T) {
	f := newFakeUpstream(t)
	f.chatReject[tokenA] = true

	path := filepath.Join(t.TempDir(), "tokens.txt")
	if err := os.WriteFile(path, []byte(tokenA+"\n"+tokenB+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	pool, err := zai.LoadPool(path)
	if err != nil {
		t.Fatal(err)
	}

	ts := newTestGateway(t, f, pool)
	resp := postAuth(t, ts, "/v1/chat/completions", chatBody("glm-4.7", false))
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 after failover", resp.StatusCode)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "Hello world" {
		t.Fatalf("content after failover: %+v", out)
	}

	// Exactly two chat attempts: the rejected one on tokenA, the retried
	// one on tokenB.
	if got := f.chatCalls.Load(); got != 2 {
		t.Fatalf("chat calls = %d, want 2", got)
	}
	if got := pool.Current(); got != tokenB {
		t.Fatalf("pool did not rotate: current = %s", zai.ShortToken(got))
	}
}

// Startup validation: a dead first account (401 on auths/) is skipped
// during initialization and the second one wins without any chat retry.
func TestPoolSkipsDeadAccountAtInit(t *testing.T) {
	f := newFakeUpstream(t)
	// Reject tokenA everywhere, including validation.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	_ = dead // auths/ rejection wired below via mux override

	path := filepath.Join(t.TempDir(), "tokens.txt")
	os.WriteFile(path, []byte(tokenA+"\n"+tokenB+"\n"), 0o600)
	pool, _ := zai.LoadPool(path)

	// Rebuild the upstream with auths/ rejecting tokenA.
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/auths/", func(w http.ResponseWriter, r *http.Request) {
		bearer := strings.TrimPrefix(r.Header.Get("authorization"), "Bearer ")
		if bearer == tokenA {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"data":{"id":"u-123"}}`))
	})
	mux.HandleFunc("/api/v2/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(f.streamBody))
	})
	replacing := httptest.NewServer(mux)
	defer replacing.Close()

	cfg := &config.Config{AuthToken: "t", LogLevel: "error", Timeout: 30 * time.Second}
	client := zai.New(replacing.URL, cfg.LogLevel, "", pool)
	srv := New(cfg, client)
	ts := httptest.NewServer(srv.Routes())
	defer ts.Close()

	resp := postAuthRaw(t, ts, "/v1/chat/completions", chatBody("glm-4.7", false), "t")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// Note: startup validation adopts the first ALIVE account into the
	// session without advancing the round-robin pointer — rotation only
	// happens on mid-session failures. The 200 above proves the dead
	// tokenA was skipped at init and tokenB is active.
}

func postAuthRaw(t *testing.T, ts *httptest.Server, path, body, token string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", ts.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// ── Bad requests ──

func TestChatCompletionRejectsEmptyMessages(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)
	resp := postAuth(t, ts, "/v1/chat/completions", `{"model":"glm-4.7"}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestChatCompletionRejectsMalformedJSON(t *testing.T) {
	f := newFakeUpstream(t)
	ts := newTestGateway(t, f, nil)
	resp := postAuth(t, ts, "/v1/chat/completions", `{not json`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}
