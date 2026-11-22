// Package zai implements the upstream client: session management, the
// browser-fingerprinted HTTP layer, an account pool with mid-session
// rotation, the model catalog, and the streaming chat completion request
// with retry/failover semantics.
package zai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	fhttp "github.com/bogdanfinn/fhttp"
	"github.com/g-alex/llm-api-gateway/internal/sse"
)

// DefaultBaseURL is the upstream the client talks to when not overridden
// (tests point it at an httptest server).
const DefaultBaseURL = "https://chat.z.ai"

// Request describes one logical chat completion.
type Request struct {
	Model           string
	WebSearch       *bool  // nil = per-model default
	Thinking        *bool  // nil = per-model default
	ReasoningEffort string // "" = none
	ChatID          string
	// Messages is the session-history form (map messages); used when
	// ClientMessages is empty.
	Messages []map[string]interface{}
	// ClientMessages is the structured client message array forwarded
	// as-is to the upstream (preferred over Messages).
	ClientMessages []json.RawMessage
	// Files carries already-uploaded file entries (see vision.go) to
	// attach as the top-level "files" array. nil for text-only requests.
	Files []map[string]interface{}

	features map[string]interface{} // resolved once in Send
}

// Client is the upstream client. Safe for concurrent use.
type Client struct {
	BaseURL     string
	LogLevel    string
	SingleToken string // legacy single-account mode (no pool)
	Pool        *Pool  // nil = single-token mode
	// PersistHistory gates appending request/response turns to the
	// conversation store (in-memory context for follow-up requests).
	PersistHistory bool

	session *Session
	convs   *ConversationStore

	modelsMu          sync.Mutex
	modelsCache       []ModelInfo
	modelsCacheTime   time.Time
	modelsLastFailure time.Time

	featureMu     sync.Mutex
	featureStates map[string]*ModelFeatureState

	sends atomic.Int64 // completed logical sends (stats)
}

// New creates a Client for the given upstream.
func New(baseURL, logLevel, singleToken string, pool *Pool) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		BaseURL:       baseURL,
		LogLevel:      logLevel,
		SingleToken:   singleToken,
		Pool:          pool,
		session:       NewSession(baseURL),
		convs:         NewConversationStore(),
		featureStates: make(map[string]*ModelFeatureState),
	}
}

// Conversations exposes the conversation store.
func (c *Client) Conversations() *ConversationStore { return c.convs }

// Session exposes the underlying session (status endpoints).
func (c *Client) Session() *Session { return c.session }

// Sends reports the number of completed logical sends.
func (c *Client) Sends() int64 { return c.sends.Load() }

// ── Utility ──

func uuidV4() string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = (b[6] & 0x0F) | 0x40
	b[8] = (b[8] & 0x3F) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// GenerateID returns a random 32-hex-char identifier for API responses.
func GenerateID() string {
	var b [16]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// EstimateTokens gives a cheap byte/4 approximation of token count for
// usage reporting.
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

// GetMessageContent extracts text from an OpenAI message content field,
// which can be a string or an array of content parts.
func GetMessageContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try string first
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	// Try array of parts
	var parts []map[string]interface{}
	if json.Unmarshal(raw, &parts) == nil {
		var texts []string
		for _, p := range parts {
			if t, ok := p["type"].(string); ok && t != "text" {
				continue
			}
			if text, ok := p["text"].(string); ok {
				texts = append(texts, text)
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

// MessagesToPrompt flattens a messages array → prompt string. Used ONLY
// for signature_prompt computation (HMAC signing). The actual messages
// array is forwarded as-is to the upstream.
func MessagesToPrompt(messages []json.RawMessage) string {
	var sb strings.Builder
	for _, raw := range messages {
		var msg struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(raw, &msg) != nil {
			continue
		}
		sb.WriteString(GetMessageContent(msg.Content))
		sb.WriteString("\n\n")
	}
	return strings.TrimSpace(sb.String())
}

// ── Chat completion ──

// Sentinels classifying request failures for the retry loop.
var (
	// ErrUnauthorized marks a dead account or expired session: the retry
	// loop rotates to the next account and re-initializes.
	ErrUnauthorized = errors.New("upstream unauthorized")
	// ErrStreamCut marks a stream that ended before the explicit
	// done-phase with nothing usable emitted; retryable.
	ErrStreamCut = errors.New("upstream stream cut before done")
)

// sendAttempts: base 2 covers the direct route plus one retry after a 401
// session re-init.
const sendAttempts = 2

// Send performs one logical chat completion, calling onContent for each
// content delta and onReasoning (may be nil) for each reasoning delta.
// Returns the AUTHORITATIVE final content/reasoning from the parser —
// buffered consumers must use these, NOT concatenated deltas: upstream
// edit_content rewrites already-emitted text in place, so delta
// concatenation produces garbled mixes of pre- and post-edit fragments.
func (c *Client) Send(ctx context.Context, prompt string, req Request, onContent, onReasoning func(string)) (string, string, error) {
	if !c.session.IsInitialized() {
		if err := c.session.Initialize(c.nextToken(), c.Pool); err != nil {
			return "", "", err
		}
	}

	// Resolve features dynamically from per-model state
	// (server capabilities + stored overrides)
	featuresMap := c.ResolveFeaturesForModel(req.Model)

	// Apply per-request overrides (highest precedence)
	if req.WebSearch != nil {
		if *req.WebSearch {
			featuresMap["auto_web_search"] = true
			featuresMap["web_search"] = true
		} else {
			delete(featuresMap, "auto_web_search")
			delete(featuresMap, "web_search")
		}
	}
	if req.Thinking != nil {
		featuresMap["enable_thinking"] = *req.Thinking
	}

	c.applyReasoningEffort(featuresMap, req.Model, req.ReasoningEffort)

	// Remove 'think' entirely; ALWAYS force image_generation to false
	delete(featuresMap, "think")
	featuresMap["image_generation"] = false

	req.features = featuresMap

	for attempt := 0; attempt < sendAttempts; attempt++ {
		content, reasoning, err := c.doRequest(ctx, prompt, req, onContent, onReasoning)
		switch {
		case errors.Is(err, ErrUnauthorized) && attempt+1 < sendAttempts:
			// Dead account or expired session: rotate to the next account
			// (pool) and re-initialize before the next attempt.
			next := c.rotateAccount()
			log.Printf("[ZAI] unauthorized/blocked — rotating account, next candidate %s...", ShortToken(next))
			c.session.MarkUninitialized()
			if initErr := c.session.Initialize(next, c.Pool); initErr != nil {
				return "", "", initErr
			}
		case errors.Is(err, ErrStreamCut) && attempt+1 < sendAttempts:
			log.Printf("[ZAI] stream cut before done — retrying")
		default:
			if err == nil {
				c.sends.Add(1)
			}
			return content, reasoning, err
		}
	}
	return "", "", fmt.Errorf("upstream request failed after retry")
}

// applyReasoningEffort applies the per-request reasoning_effort parameter
// to the feature map. Defensive semantics: stale values are always
// stripped first so unsupported models never receive a placeholder value;
// the parameter is re-added only when the model's capabilities declare
// "reasoning_effort": true AND the value is a valid level. When applied,
// enable_thinking is forced on.
func (c *Client) applyReasoningEffort(featuresMap map[string]interface{}, modelID, effort string) {
	delete(featuresMap, "reasoning_effort")
	if effort == "" {
		return
	}
	if !c.ModelSupportsReasoningEffort(modelID) {
		log.Printf("[reasoning_effort] model=%s does not support reasoning_effort; parameter ignored", modelID)
		return
	}
	if !isValidReasoningEffort(effort) {
		log.Printf("[reasoning_effort] invalid value '%s' for model=%s (accepted: low, high, max); ignored", effort, modelID)
		return
	}
	featuresMap["reasoning_effort"] = effort
	featuresMap["enable_thinking"] = true
	log.Printf("[reasoning_effort] model=%s effort=%s enabled (enable_thinking forced true)", modelID, effort)
}

// nextToken picks the account to (re)initialize with: the pool's current
// candidate, or the single-token config when no pool is loaded.
func (c *Client) nextToken() string {
	if t := c.Pool.Current(); t != "" {
		return t
	}
	return c.SingleToken
}

// rotateAccount advances past a dead account and returns who comes next
// (same single-token fallback rule).
func (c *Client) rotateAccount() string {
	if t := c.Pool.Rotate(); t != "" {
		return t
	}
	return c.SingleToken
}

// requestBody is the upstream completion payload.
type requestBody struct {
	Model           string                 `json:"model"`
	ChatID          string                 `json:"chat_id"`
	Messages        []json.RawMessage      `json:"messages"`
	SignaturePrompt string                 `json:"signature_prompt"`
	Stream          bool                   `json:"stream"`
	Features        map[string]interface{} `json:"features"`
	// Files carries already-uploaded file entries (see vision.go) to
	// attach as the top-level "files" array. nil for text-only requests
	// (the field is omitted entirely).
	Files []map[string]interface{} `json:"files,omitempty"`
}

// doRequest performs one HTTP attempt against the upstream completion
// endpoint and parses the SSE stream.
func (c *Client) doRequest(ctx context.Context, prompt string, req Request, onContent, onReasoning func(string)) (string, string, error) {
	zs := c.session
	token, userID, feVersion := zs.tokenValue(), zs.userIDValue(), zs.feVersionValue()

	signature := zs.generateSignature(prompt, token, userID)

	headers := map[string]string{
		"authorization": "Bearer " + token,
		"User-Agent":    userAgent,
		"content-type":  "application/json",
		"x-fe-Version":  feVersion,
		"x-region":      "overseas",
		"x-signature":   signature,
	}

	// Forward structured messages, NOT flattened prompt
	var forwardedMessages []json.RawMessage
	if len(req.ClientMessages) > 0 {
		forwardedMessages = req.ClientMessages
	} else {
		// Fallback: session messages + new user message
		for _, m := range req.Messages {
			raw, _ := json.Marshal(m)
			forwardedMessages = append(forwardedMessages, raw)
		}
		userMsg, _ := json.Marshal(map[string]string{"role": "user", "content": prompt})
		forwardedMessages = append(forwardedMessages, userMsg)
	}

	featuresPayload := make(map[string]interface{}, len(req.features)+2)
	for k, v := range req.features {
		featuresPayload[k] = v
	}
	delete(featuresPayload, "think")
	featuresPayload["flags"] = []interface{}{}
	featuresPayload["image_generation"] = false

	body := requestBody{
		Model:           req.Model,
		ChatID:          req.ChatID,
		Messages:        forwardedMessages,
		SignaturePrompt: prompt,
		Stream:          true,
		Features:        featuresPayload,
		Files:           req.Files,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return "", "", err
	}

	if c.LogLevel == "debug" {
		log.Printf("[DEBUG] request body: %s", string(bodyBytes))
	}

	// No per-attempt timeout here: thinking-heavy streams legitimately run
	// past 90s WITH data flowing (a hard 90s cap killed live streams at
	// exactly 90.00s). Bounds instead: the outer request context and the
	// 120s idle watchdog below (fires only when NO data arrives).
	client, err := zaiHTTP()
	if err != nil {
		return "", "", fmt.Errorf("http client: %w", err)
	}
	httpReq, _ := fhttp.NewRequestWithContext(ctx, "POST", c.BaseURL+"/api/v2/chat/completions", bytes.NewReader(bodyBytes))
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return "", "", fmt.Errorf("upstream connection error: %w", err)
	}
	defer resp.Body.Close()

	if c.LogLevel == "debug" {
		log.Printf("[DEBUG] response status: %d %s", resp.StatusCode, resp.Status)
	}

	if resp.StatusCode == 401 {
		return "", "", ErrUnauthorized
	}

	if resp.StatusCode != 200 {
		errText, _ := io.ReadAll(resp.Body)
		errStr := string(errText)
		// HTML edge pages → clean sentinel instead of dumping raw HTML
		if strings.Contains(errStr, "<html") || strings.Contains(errStr, "<!doctype") {
			return "", "", fmt.Errorf("upstream error %d: blocked by edge filter (rate limit/security)", resp.StatusCode)
		}
		if len(errStr) > 200 {
			errStr = errStr[:200] + "..."
		}
		if c.LogLevel == "debug" {
			log.Printf("[DEBUG] error body: %s", errStr)
		}
		// Account-level rejections behave like unauthorized so the retry
		// loop rotates to the next account instead of hammering a dead one
		if strings.Contains(errStr, "USER_BLOCKED") || strings.Contains(errStr, "\"detail\"") && resp.StatusCode == 403 {
			return "", "", fmt.Errorf("%w: account rejected by platform (%d)", ErrUnauthorized, resp.StatusCode)
		}
		return "", "", fmt.Errorf("upstream error %d: %s", resp.StatusCode, errStr)
	}

	// Parse the SSE stream through the stateful parser (edit_index,
	// details split).
	reader := bufio.NewReaderSize(resp.Body, 256*1024)
	parser := sse.NewParser(c.LogLevel)
	var buffer string

	// Track whether ANYTHING was forwarded to the client: a stream that
	// dies before the done-phase is retryable only while nothing was
	// emitted (retrying after partial deltas would duplicate output).
	emitted := false
	origContent, origReasoning := onContent, onReasoning
	clientContent, clientReasoning := origContent, origReasoning
	if origContent != nil {
		clientContent = func(chunk string) { emitted = true; origContent(chunk) }
	}
	if origReasoning != nil {
		clientReasoning = func(chunk string) { emitted = true; origReasoning(chunk) }
	}
	var sawDone bool

	// Idle watchdog: the upstream occasionally stalls mid-stream (no
	// bytes, no done-phase) and the SSE would hang until the outer
	// timeout. Close the body after 120s of silence — the blocked
	// ReadString unblocks and we return an error instead of keeping the
	// client on hold.
	var watchdogFired atomic.Bool
	streamIdle := time.AfterFunc(120*time.Second, func() {
		watchdogFired.Store(true)
		log.Printf("[ZAI] stream idle >120s — aborting")
		resp.Body.Close()
	})
	defer streamIdle.Stop()

	if c.LogLevel == "debug" {
		log.Printf("[DEBUG] response headers: %v", resp.Header)
	}

	for {
		line, err := reader.ReadString('\n')
		if line != "" {
			streamIdle.Reset(120 * time.Second)
			buffer += line
			lines := strings.Split(buffer, "\n")
			buffer = lines[len(lines)-1] // keep incomplete last line

			for _, l := range lines[:len(lines)-1] {
				trimmed := strings.TrimSpace(l)
				if trimmed != "" && c.LogLevel == "debug" {
					log.Printf("[DEBUG] SSE line: %s", trimmed)
				}
				if ferr := parser.FeedLine(l, clientContent, clientReasoning); ferr != nil {
					if ferr == io.EOF {
						// [DONE] / phase done — normal end; release any
						// holdback-tail content before returning finals
						sawDone = true
						parser.FlushFinal(clientContent, clientReasoning)
						return parser.Content(), parser.Reasoning(), nil
					}
					return "", "", ferr // actual error from the upstream
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				// Process any remaining buffer
				if strings.TrimSpace(buffer) != "" {
					if c.LogLevel == "debug" {
						log.Printf("[DEBUG] SSE line (final): %s", strings.TrimSpace(buffer))
					}
					parser.FeedLine(buffer, clientContent, clientReasoning)
				}
			} else if watchdogFired.Load() {
				return "", "", fmt.Errorf("upstream stream stalled >120s — aborted by idle watchdog")
			}
			break
		}
	}

	// Stream ended without an explicit done-phase: the upstream dropped
	// the connection mid-flight. Release held-back tail, then either retry
	// (nothing emitted yet) or surface the truncation to the client.
	parser.FlushFinal(clientContent, clientReasoning)
	if !sawDone {
		if !emitted {
			log.Printf("[ZAI] stream ended without done-phase (%d bytes content buffered) — treated as cut", len(parser.Content()))
			return "", "", ErrStreamCut
		}
		return "", "", fmt.Errorf("upstream stream cut before completion (partial output already delivered)")
	}
	return parser.Content(), parser.Reasoning(), nil
}
