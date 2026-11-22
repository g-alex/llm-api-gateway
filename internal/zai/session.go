// session.go — Z.AI session state, guest flow, account adoption, and the
// HMAC-SHA256 request signature.
package zai

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"regexp"
	"strings"
	"sync"
	"time"

	fhttp "github.com/bogdanfinn/fhttp"
)

const (
	saltKey          = "key-@@@@)))()((9))-xxxx&&&%%%%%"
	defaultFEVersion = "prod-fe-1.0.185"
	sessionTTL       = 30 * time.Minute
)

// Session is the shared Z.AI connection state: the active token, user
// identity and the frontend version used in request headers.
type Session struct {
	mu           sync.Mutex
	baseURL      string
	token        string
	userID       string
	userName     string
	chatID       string
	feVersion    string
	initialized  bool
	initializing bool
}

// NewSession creates an uninitialized guest session.
func NewSession(baseURL string) *Session {
	return &Session{
		baseURL:   baseURL,
		userName:  "Guest",
		chatID:    uuidV4(),
		feVersion: defaultFEVersion,
	}
}

// Conversation is per-client-session upstream chat state.
type Conversation struct {
	ChatID   string
	Messages []map[string]interface{}
	lastUsed time.Time
}

// ConversationStore keeps per-client-session conversations with a TTL
// cleanup loop.
type ConversationStore struct {
	mu       sync.Mutex
	sessions map[string]*Conversation
}

// NewConversationStore starts the store and its cleanup goroutine.
func NewConversationStore() *ConversationStore {
	cs := &ConversationStore{sessions: make(map[string]*Conversation)}
	go cs.cleanup()
	return cs
}

// GetOrCreate returns the conversation for sessionID, creating a fresh
// one when absent or when fresh is true.
func (cs *ConversationStore) GetOrCreate(sessionID string, fresh bool) *Conversation {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	if fresh || cs.sessions[sessionID] == nil {
		cs.sessions[sessionID] = &Conversation{
			ChatID:   uuidV4(),
			lastUsed: time.Now(),
		}
	}
	s := cs.sessions[sessionID]
	s.lastUsed = time.Now()
	return s
}

// Clear drops every stored conversation.
func (cs *ConversationStore) Clear() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.sessions = make(map[string]*Conversation)
}

// Count reports the number of stored conversations.
func (cs *ConversationStore) Count() int {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return len(cs.sessions)
}

// TotalMessages reports the total number of stored messages.
func (cs *ConversationStore) TotalMessages() int {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	total := 0
	for _, s := range cs.sessions {
		total += len(s.Messages)
	}
	return total
}

func (cs *ConversationStore) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		cs.mu.Lock()
		now := time.Now()
		for id, s := range cs.sessions {
			if now.Sub(s.lastUsed) > sessionTTL {
				delete(cs.sessions, id)
			}
		}
		cs.mu.Unlock()
	}
}

// ── Session lifecycle ──

// jwtPayload decodes the payload section of a JWT (base64url, no padding).
func jwtPayload(token string) (map[string]interface{}, error) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return nil, fmt.Errorf("not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	if err := json.Unmarshal(payload, &m); err != nil {
		return nil, err
	}
	return m, nil
}

var feVersionRegex = regexp.MustCompile(`prod-fe-\d+\.\d+\.\d+`)

// scrapeConfig fetches the upstream homepage and records the current
// frontend version (sent as x-fe-Version on chat requests).
func (s *Session) scrapeConfig() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client, err := zaiHTTP()
	if err != nil {
		log.Printf("[Config] http client: %v", err)
		return
	}
	req, _ := fhttp.NewRequestWithContext(ctx, "GET", s.baseURL, nil)
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[Config] scrape error: %v, using default feVersion", err)
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if m := feVersionRegex.FindSubmatch(body); m != nil {
		s.mu.Lock()
		s.feVersion = string(m[0])
		s.mu.Unlock()
		log.Printf("[Config] fe_version: %s", s.feVersion)
	}
}

// Initialize establishes the session: with a token, the account fast path
// (validated against auths/, pool-aware); otherwise the guest flow.
func (s *Session) Initialize(zaiToken string, pool *Pool) error {
	s.mu.Lock()
	if s.initializing {
		s.mu.Unlock()
		// Wait for another goroutine's init to finish.
		for i := 0; i < 300; i++ {
			time.Sleep(100 * time.Millisecond)
			s.mu.Lock()
			done := !s.initializing
			s.mu.Unlock()
			if done {
				return nil
			}
		}
		return fmt.Errorf("session init timeout")
	}
	s.initializing = true
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		s.initializing = false
		s.mu.Unlock()
	}()

	// Fast path: provided Z.AI account(s) — validate against auths/ and
	// pick a live one (see pool.go for the pool policy).
	if zaiToken != "" {
		if s.initFromAccounts(zaiToken, pool) {
			return nil
		}
		log.Println("[Session] provided account(s) unusable — falling back to guest session")
	}

	log.Println("[Session] initializing guest flow...")

	s.scrapeConfig()

	headers := map[string]string{
		"Origin":       s.baseURL,
		"Referer":      s.baseURL + "/",
		"Content-Type": "application/json",
		"User-Agent":   userAgent,
	}

	// POST /api/v1/auths/guest (warm up)
	guestCtx, guestCancel := context.WithTimeout(context.Background(), 15*time.Second)
	s.post(guestCtx, "/api/v1/auths/guest", "{}", headers)
	guestCancel()

	// GET /api/v1/auths/
	client, err := zaiHTTP()
	if err != nil {
		return fmt.Errorf("http client: %w", err)
	}
	ctx2, cancel2 := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel2()
	req, _ := fhttp.NewRequestWithContext(ctx2, "GET", s.baseURL+"/api/v1/auths/", nil)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("auth request failed: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("auth failed: %d", resp.StatusCode)
	}

	var authData struct {
		Token string `json:"token"`
	}
	json.Unmarshal(body, &authData)
	token := authData.Token

	if token == "" {
		// Retry guest endpoint
		retryCtx, retryCancel := context.WithTimeout(context.Background(), 15*time.Second)
		guestBody := s.post(retryCtx, "/api/v1/auths/guest", "{}", headers)
		retryCancel()
		var guestData struct {
			Token string `json:"token"`
		}
		json.Unmarshal([]byte(guestBody), &guestData)
		token = guestData.Token
	}

	if token == "" {
		return fmt.Errorf("no token received from upstream")
	}

	s.mu.Lock()
	s.token = token
	s.mu.Unlock()

	if payload, err := jwtPayload(token); err == nil {
		s.mu.Lock()
		if id, ok := payload["id"].(string); ok {
			s.userID = id
		}
		if email, ok := payload["email"].(string); ok && email != "" {
			s.userName = strings.Split(email, "@")[0]
		}
		userID, userName := s.userID, s.userName
		s.mu.Unlock()
		if len(userID) > 8 {
			userID = userID[:8]
		}
		log.Printf("[Session] connected. UserID: %s... (%s)", userID, userName)
	} else {
		log.Println("[Session] token decode failed, but continuing.")
	}

	s.mu.Lock()
	s.initialized = true
	s.mu.Unlock()
	return nil
}

// initFromAccounts tries the configured accounts (the caller's primary —
// the pool's current pick or the single token — followed by the rest of
// the pool) against the live auths/ endpoint and adopts the first usable
// one. Reports true when the session fields are set.
func (s *Session) initFromAccounts(primary string, pool *Pool) bool {
	var extraTokens []string
	if pool != nil {
		pool.mu.Lock()
		extraTokens = append(extraTokens, pool.tokens...)
		pool.mu.Unlock()
	}

	var candidates []string
	seen := map[string]bool{}
	add := func(t string) {
		if t != "" && !seen[t] {
			seen[t] = true
			candidates = append(candidates, t)
		}
	}
	add(primary)
	for _, t := range extraTokens {
		add(t)
	}
	if len(candidates) == 0 {
		return false
	}

	for i, tok := range candidates {
		status, body := s.probeAccount(tok)
		switch ClassifyAuthCheck(status, body) {
		case "alive":
			log.Printf("[Session] account %d/%d accepted (HTTP %d)", i+1, len(candidates), status)
			s.setAccount(tok, body)
			return true
		case "dead":
			log.Printf("[Session] account %d/%d unusable (HTTP %d) — trying next", i+1, len(candidates), status)
		default:
			// Network/parse ambiguity: accept locally so a flaky network
			// never bricks startup.
			log.Printf("[Session] account %d/%d check ambiguous (HTTP %d) — accepting without live validation", i+1, len(candidates), status)
			s.setAccount(tok, "")
			return true
		}
	}
	return false
}

// setAccount installs a validated-or-trusted account into session state,
// deriving userID/email from the auths/ response first, then from the
// JWT payload as fallback.
func (s *Session) setAccount(token, authBody string) {
	id, email := "", ""
	var respData struct {
		Data struct {
			ID    string `json:"id"`
			Email string `json:"email"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(authBody), &respData); err == nil {
		id, email = respData.Data.ID, respData.Data.Email
	}
	if id == "" || email == "" {
		if payload, perr := jwtPayload(token); perr == nil {
			if v, ok := payload["id"].(string); ok && id == "" {
				id = v
			}
			if v, ok := payload["email"].(string); ok && email == "" {
				email = v
			}
		}
	}

	name := ""
	if email != "" {
		name = strings.Split(email, "@")[0]
	}

	s.mu.Lock()
	s.token = token
	s.userID = id
	if name != "" {
		s.userName = name
	}
	s.initialized = true
	s.mu.Unlock()

	uid := id
	if len(uid) > 8 {
		uid = uid[:8]
	}
	log.Printf("[Session] token user: %s... (%s)", uid, name)
}

// probeAccount validates one account against GET /api/v1/auths/.
// Returns status + raw body for classification.
func (s *Session) probeAccount(token string) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := zaiHTTP()
	if err != nil {
		return -1, "client error: " + err.Error()
	}
	req, _ := fhttp.NewRequestWithContext(ctx, "GET", s.baseURL+"/api/v1/auths/", nil)
	req.Header.Set("Origin", s.baseURL)
	req.Header.Set("Referer", s.baseURL+"/")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("authorization", "Bearer "+token)

	resp, err := client.Do(req)
	if err != nil {
		return -1, "request error: " + err.Error()
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	return resp.StatusCode, string(body)
}

// post is a simple POST helper returning the response body as string.
func (s *Session) post(ctx context.Context, path, body string, headers map[string]string) string {
	client, err := zaiHTTP()
	if err != nil {
		return ""
	}
	req, _ := fhttp.NewRequestWithContext(ctx, "POST", s.baseURL+path, strings.NewReader(body))
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return string(data)
}

// ── Accessors ──

// tokenValue returns the active session token (client-internal use).
func (s *Session) tokenValue() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.token
}

// userIDValue returns the active user ID (client-internal use).
func (s *Session) userIDValue() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.userID
}

// feVersionValue returns the frontend version (client-internal use).
func (s *Session) feVersionValue() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.feVersion
}

// IsInitialized reports whether the session has a usable token.
func (s *Session) IsInitialized() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.initialized
}

// IsConnected reports whether a session token is present.
func (s *Session) IsConnected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.token != ""
}

// MarkUninitialized forces re-initialization on the next send (used after
// mid-session 401s).
func (s *Session) MarkUninitialized() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialized = false
}

// Snapshot returns the session state for status endpoints.
func (s *Session) Snapshot() (connected bool, userName, userID, feVersion string, init bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.token != "", s.userName, s.userID, s.feVersion, s.initialized
}

// generateSignature computes the HMAC-SHA256 request signature:
// wKey = HMAC(saltKey, 5-minute bucket), then HMAC(wKey, sorted payload |
// base64(prompt) | timestamp).
func (s *Session) generateSignature(prompt, token, userID string) string {
	timestamp := fmt.Sprintf("%d", time.Now().UnixMilli())
	requestID := uuidV4()
	bucket := time.Now().UnixMilli() / 300000

	// wKey = HMAC-SHA256(saltKey, bucket) hex
	mac := hmac.New(sha256.New, []byte(saltKey))
	fmt.Fprintf(mac, "%d", bucket)
	wKey := hex.EncodeToString(mac.Sum(nil))

	// sorted payload: requestId,timestamp,user_id
	type kv struct{ k, v string }
	sorted := []kv{
		{"requestId", requestID},
		{"timestamp", timestamp},
		{"user_id", userID},
	}
	var sp strings.Builder
	for i, kv := range sorted {
		if i > 0 {
			sp.WriteByte(',')
		}
		sp.WriteString(kv.k)
		sp.WriteByte(',')
		sp.WriteString(kv.v)
	}

	promptB64 := base64.StdEncoding.EncodeToString([]byte(strings.TrimSpace(prompt)))
	dataToSign := sp.String() + "|" + promptB64 + "|" + timestamp

	mac2 := hmac.New(sha256.New, []byte(wKey))
	mac2.Write([]byte(dataToSign))
	return hex.EncodeToString(mac2.Sum(nil))
}
