// Package api exposes the gateway over HTTP: OpenAI-, Anthropic- and
// Responses-compatible endpoints, bearer auth, CORS, status and admin
// routes.
package api

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/g-alex/llm-api-gateway/internal/config"
	"github.com/g-alex/llm-api-gateway/internal/zai"
)

// Server wires the protocol handlers to the upstream client.
type Server struct {
	cfg    *config.Config
	client *zai.Client
}

// New creates the API server.
func New(cfg *config.Config, client *zai.Client) *Server {
	return &Server{cfg: cfg, client: client}
}

// ── Middleware ──

func (s *Server) withAuth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		provided := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		provided = strings.TrimSpace(provided)
		if provided == "" || provided != s.cfg.AuthToken {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"type": "error",
				"error": map[string]interface{}{
					"type":    "authentication_error",
					"message": "Invalid or missing authentication token",
				},
			})
			return
		}
		h(w, r)
	}
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Session-Id, X-Fresh-Session")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ── Routes ──

// Routes builds the HTTP handler with every endpoint.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/status", s.handleStatus)

	mux.HandleFunc("/v1/models", s.withAuth(s.handleV1Models))
	mux.HandleFunc("/models", s.withAuth(s.handleModels))
	mux.HandleFunc("/v1/chat/completions", s.withAuth(s.handleChatCompletions))
	mux.HandleFunc("/v1/messages", s.withAuth(s.handleAnthropicMessages))
	mux.HandleFunc("/v1/responses", s.withAuth(s.handleResponses))

	mux.HandleFunc("/admin/stats", s.handleAdminStats)
	mux.HandleFunc("/admin/health", s.handleAdminHealth)
	mux.HandleFunc("/admin/session/clear", s.withAuth(s.handleSessionClear))

	return corsMiddleware(mux)
}

// ── Handlers ──

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	host := r.Host
	if host == "" {
		host = "localhost:" + s.cfg.Port
	}
	html := dashboardHTML
	html = strings.ReplaceAll(html, "__HOST__", host)
	html = strings.ReplaceAll(html, "__AUTH_TOKEN__", s.cfg.AuthToken)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(html))
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	_, userName, userID, feVersion, init := s.client.Session().Snapshot()
	if !init {
		userID = ""
	}
	features := s.client.ResolveFeaturesForModel("glm-4.7")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"connected":       init,
		"userName":        userName,
		"userId":          userID,
		"feVersion":       feVersion,
		"activeSessions":  s.client.Conversations().Count(),
		"features":        features,
		"mode":            "direct",
		"port":            s.cfg.Port,
		"accountPoolSize": s.client.Pool.Size(),
		"completedSends":  s.client.Sends(),
		"persistHistory":  s.cfg.PersistHistory,
		"claudeDesktop":   s.cfg.ClaudeDesktop,
	})
}

func (s *Server) handleAdminStats(w http.ResponseWriter, r *http.Request) {
	totalMsgs := s.client.Conversations().TotalMessages()
	_, _, _, _, init := s.client.Session().Snapshot()
	connected := 0
	if init {
		connected = 1
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"mode":           "direct",
		"totalClients":   connected,
		"activeSessions": s.client.Conversations().Count(),
		"stats":          map[string]int{"totalRequests": totalMsgs / 2},
	})
}

func (s *Server) handleAdminHealth(w http.ResponseWriter, r *http.Request) {
	healthy := s.client.Session().IsConnected()
	if !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"healthy": healthy, "mode": "direct"})
}

func (s *Server) handleSessionClear(w http.ResponseWriter, r *http.Request) {
	s.client.Conversations().Clear()
	log.Println("[Session] all session histories cleared")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":        true,
		"message":        "All session histories cleared",
		"activeSessions": 0,
	})
}

// modelIDs lists known upstream model IDs.
func (s *Server) modelIDs() []string {
	models := s.client.FetchModels()
	ids := make([]string, 0, len(models))
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	return ids
}
