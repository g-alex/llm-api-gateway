// openai.go — OpenAI-compatible endpoints: /v1/chat/completions (stream,
// non-stream, tools) and /v1/models.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/g-alex/llm-api-gateway/internal/tools"
	"github.com/g-alex/llm-api-gateway/internal/zai"
)

// ── OpenAI wire types ──

type openAIDelta struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type openAIMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type openAIChoice struct {
	Index        int            `json:"index"`
	Delta        *openAIDelta   `json:"delta,omitempty"`
	Message      *openAIMessage `json:"message,omitempty"`
	FinishReason *string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

func newChunk(content, model, requestID string, finish *string) openAIResponse {
	return openAIResponse{
		ID:      "chatcmpl-" + requestID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIChoice{{
			Index:        0,
			Delta:        &openAIDelta{Content: content},
			FinishReason: finish,
		}},
	}
}

func newReasoningChunk(reasoning, model, requestID string) openAIResponse {
	return openAIResponse{
		ID:      "chatcmpl-" + requestID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIChoice{{
			Index:        0,
			Delta:        &openAIDelta{ReasoningContent: reasoning},
			FinishReason: nil,
		}},
	}
}

func newCompletion(content, reasoning, prompt, model, requestID string) openAIResponse {
	pt := zai.EstimateTokens(prompt)
	ct := zai.EstimateTokens(content)
	return openAIResponse{
		ID:      "chatcmpl-" + requestID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openAIChoice{{
			Index:        0,
			Message:      &openAIMessage{Role: "assistant", Content: content, ReasoningContent: reasoning},
			FinishReason: strPtr("stop"),
		}},
		Usage: &openAIUsage{
			PromptTokens:     pt,
			CompletionTokens: ct,
			TotalTokens:      pt + ct,
		},
	}
}

func formatOpenAIError(msg, errType string) map[string]interface{} {
	if errType == "" {
		errType = "api_error"
	}
	return map[string]interface{}{
		"error": map[string]interface{}{
			"message": msg,
			"type":    errType,
			"code":    nil,
			"param":   nil,
		},
	}
}

func strPtr(s string) *string { return &s }

// runeChunkEnd returns the largest chunk end ≤ max that does not split a
// UTF-8 rune (backs off up to 3 bytes to a rune start). Byte-sliced
// chunks would cut multi-byte characters (Cyrillic, emoji) in half.
func runeChunkEnd(s string, i, max int) int {
	end := max
	if end > len(s) {
		end = len(s)
	}
	for end < len(s) && end > i+1 && !utf8.RuneStart(s[end]) {
		end--
	}
	return end
}

// statusFromError maps an upstream error string to an HTTP status code.
func statusFromError(errMsg string) int {
	switch {
	case strings.Contains(errMsg, "401"):
		return 401
	case strings.Contains(errMsg, "403"):
		return 403
	case strings.Contains(errMsg, "429"):
		return 429
	case strings.Contains(errMsg, "400"):
		return 400
	default:
		return 500
	}
}

// ── Chat request ──

type chatRequest struct {
	Model           string            `json:"model"`
	Messages        []json.RawMessage `json:"messages"`
	Stream          *bool             `json:"stream"`
	DeepThink       *bool             `json:"deepThink"`
	Search          *bool             `json:"search"`
	WebSearch       *bool             `json:"webSearch"`
	Reasoning       *bool             `json:"reasoning"`
	Thinking        json.RawMessage   `json:"thinking"`
	ReasoningEffort string            `json:"reasoning_effort"`
	Tools           []json.RawMessage `json:"tools,omitempty"`
}

func (r *chatRequest) streamEnabled() bool {
	if r.Stream == nil {
		return true // default stream=true like the original
	}
	return *r.Stream
}

// ── /v1/models ──

// claudeName maps GLM → Anthropic model IDs for Claude Desktop mode.
func claudeName(glmModel string) string {
	switch glmModel {
	case "glm-4.7":
		return "claude-opus-4-7"
	case "glm-5.2":
		return "claude-opus-4-6"
	case "GLM-5-Turbo":
		return "claude-sonnet-4-6"
	case "GLM-5v-Turbo":
		return "claude-3-5-haiku-latest"
	default:
		return glmModel
	}
}

func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	now := time.Now().Unix()
	models := s.client.FetchModels()
	if s.cfg.ClaudeDesktop {
		data := make([]map[string]interface{}, len(models))
		for i, m := range models {
			name := claudeName(m.ID)
			data[i] = map[string]interface{}{
				"id":           name,
				"object":       "model",
				"created":      now,
				"owned_by":     "anthropic",
				"display_name": name,
			}
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data":   data,
		})
		return
	}
	data := make([]map[string]interface{}, len(models))
	for i, m := range models {
		data[i] = map[string]interface{}{
			"id":           m.ID,
			"object":       "model",
			"created":      now,
			"owned_by":     "z-ai",
			"display_name": m.Name,
			"description":  m.Description,
		}
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   data,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	models := s.client.FetchModels()
	ids := make([]string, 0, len(models))
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	currentModel := "glm-4.7"
	if len(ids) > 0 {
		currentModel = ids[0]
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models":       ids,
		"currentModel": currentModel,
	})
}

// ── /v1/chat/completions ──

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Limit body size (50 MB, like the original)
	r.Body = http.MaxBytesReader(w, r.Body, 50*1024*1024)

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(formatOpenAIError("invalid JSON body: "+err.Error(), "invalid_request_error"))
		return
	}

	if len(req.Messages) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(formatOpenAIError("messages is required and must be an array", "invalid_request_error"))
		return
	}

	model := req.Model
	if model == "" {
		model = "glm-4.7"
	}

	sessionID := r.Header.Get("X-Session-Id")
	if sessionID == "" {
		sessionID = "default"
	}
	fresh := r.Header.Get("X-Fresh-Session") == "true"
	conv := s.client.Conversations().GetOrCreate(sessionID, fresh)

	requestID := zai.GenerateID()

	// ── Vision: extract image_url parts, upload them, rewrite the parts
	// to file-id references. req.Messages is byte-identical when the
	// request carries no images (the common case).
	var visionFiles []map[string]interface{}
	if rawMsgs, merr := json.Marshal(req.Messages); merr == nil {
		cleaned, files, vErr := s.client.ProcessVisionMessages(r.Context(), rawMsgs)
		if vErr != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(formatOpenAIError(vErr.Error(), "invalid_request_error"))
			return
		}
		if files != nil {
			if err := json.Unmarshal(cleaned, &req.Messages); err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(formatOpenAIError("image processing: malformed messages", "invalid_request_error"))
				return
			}
			visionFiles = files
		}
	}
	prompt := zai.MessagesToPrompt(req.Messages)

	opts := zai.Request{
		Model:           model,
		WebSearch:       firstNonNilBool(req.WebSearch, req.Search),
		Thinking:        s.thinkingOverride(req),
		ReasoningEffort: req.ReasoningEffort,
		ChatID:          conv.ChatID,
		Messages:        conv.Messages,
		ClientMessages:  req.Messages,
		Files:           visionFiles,
	}

	// Tool calling: the streaming path intercepts <<<TOOL_CALL>>> live,
	// the non-streaming path buffers and parses after the fact.
	if len(req.Tools) > 0 {
		if req.streamEnabled() {
			s.handleChatToolsStream(w, r, req, prompt, model, requestID, conv, opts)
		} else {
			s.handleChatWithTools(w, r, req, prompt, model, requestID, conv, opts)
		}
		return
	}

	if req.streamEnabled() {
		s.handleChatStream(w, r, prompt, model, requestID, opts, conv)
	} else {
		s.handleChatNonStream(w, r, prompt, model, requestID, opts, conv)
	}
}

// firstNonNilBool returns the first non-nil *bool, or nil.
func firstNonNilBool(vals ...*bool) *bool {
	for _, v := range vals {
		if v != nil {
			return v
		}
	}
	return nil
}

// thinkingOverride merges deepThink / reasoning / thinking:{"type":...}
// into a single enable_thinking override (nil = per-model default).
func (s *Server) thinkingOverride(req chatRequest) *bool {
	if req.DeepThink != nil {
		return req.DeepThink
	}
	if req.Reasoning != nil {
		return req.Reasoning
	}
	if len(req.Thinking) > 0 {
		var thinkCfg struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(req.Thinking, &thinkCfg); err == nil {
			enabled := thinkCfg.Type == "enabled"
			return &enabled
		}
	}
	return nil
}

// handleChatWithTools processes non-streaming requests with tools.
// Buffers the full response, parses for tool markers, returns either
// tool_calls or normal text in OpenAI format.
func (s *Server) handleChatWithTools(w http.ResponseWriter, r *http.Request, req chatRequest, prompt, model, requestID string, conv *zai.Conversation, opts zai.Request) {
	// Convert messages: prepend the tools system prompt, convert
	// tool/assistant roles
	convertedMsgs := tools.ConvertMessages(req.Messages, req.Tools)
	// Recompute prompt from converted messages for signature
	convertedPrompt := zai.MessagesToPrompt(convertedMsgs)
	opts.ClientMessages = convertedMsgs

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	// Buffered path: take the authoritative parser content, not deltas
	// (edit_content rewrites would garble delta concatenation).
	content, _, err := s.client.Send(ctx, convertedPrompt, opts, nil, nil)

	if err != nil {
		log.Printf("[Tools] error: %v", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusFromError(err.Error()))
		json.NewEncoder(w).Encode(formatOpenAIError(err.Error(), "api_error"))
		return
	}

	// Parse for tool calls
	toolCalls := tools.ParseCalls(content, req.Tools)
	if len(toolCalls) > 0 {
		cleanContent := tools.StripToolContent(content)

		// Build tool call entries
		type tcFunc struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}
		type tcEntry struct {
			Index    int    `json:"index"`
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function tcFunc `json:"function"`
		}

		entries := make([]tcEntry, len(toolCalls))
		for i, c := range toolCalls {
			entries[i] = tcEntry{Index: i, ID: c.ID, Type: c.Type}
			entries[i].Function.Name = c.Function.Name
			entries[i].Function.Arguments = c.Function.Arguments
		}

		if req.streamEnabled() {
			// SSE streaming format for tool_calls
			flusher, ok := w.(http.Flusher)
			if !ok {
				w.Header().Set("Content-Type", "application/json")
				writeToolCallsJSON(w, requestID, model, cleanContent, entries)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)

			// Chunk 1: role + tool_calls
			chunk1 := map[string]interface{}{
				"id":      "chatcmpl-" + requestID,
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   model,
				"choices": []map[string]interface{}{{
					"index": 0,
					"delta": map[string]interface{}{
						"role":       "assistant",
						"content":    nil,
						"tool_calls": entries,
					},
					"finish_reason": nil,
				}},
			}
			data1, _ := json.Marshal(chunk1)
			fmt.Fprintf(w, "data: %s\n\n", data1)
			flusher.Flush()

			// Chunk 2: finish_reason
			chunk2 := map[string]interface{}{
				"id":      "chatcmpl-" + requestID,
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   model,
				"choices": []map[string]interface{}{{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": "tool_calls",
				}},
			}
			data2, _ := json.Marshal(chunk2)
			fmt.Fprintf(w, "data: %s\n\n", data2)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		} else {
			// Non-streaming JSON
			w.Header().Set("Content-Type", "application/json")
			writeToolCallsJSON(w, requestID, model, cleanContent, entries)
		}
		log.Printf("[Tools] returned %d tool calls", len(toolCalls))
		return
	}

	// No tool calls found — return content as-is (text or raw JSON).
	// Don't leak raw contract markers though: strip complete blocks.
	if strings.Contains(content, tools.MarkerStart) {
		preview := content
		if len(preview) > 1500 {
			preview = preview[:1500] + "...(truncated)"
		}
		log.Printf("[Tools] markers present but no valid tool call parsed — model output:\n%s", preview)
		content = tools.StripBlocks(content)
	}
	if req.streamEnabled() {
		// Client expects SSE — send text as stream chunks
		flusher, ok := w.(http.Flusher)
		if !ok {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(newCompletion(content, "", prompt, model, requestID))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		writeSSE := func(v interface{}) {
			data, _ := json.Marshal(v)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}

		// Split content into chunks to avoid oversized SSE events.
		// Rune-safe: a byte boundary would cut a multi-byte character
		// (Cyrillic, emoji) in half — json.Marshal renders the halves
		// as U+FFFD on the client.
		const chunkSize = 2000
		for i := 0; i < len(content); {
			end := runeChunkEnd(content, i, i+chunkSize)
			writeSSE(newChunk(content[i:end], model, requestID, nil))
			i = end
		}

		// Final chunk with finish_reason
		writeSSE(newChunk("", model, requestID, strPtr("stop")))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	} else {
		// Non-streaming JSON
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(newCompletion(content, "", prompt, model, requestID))
	}
}

// writeToolCallsJSON writes a non-streaming JSON response with
// tool_calls.
func writeToolCallsJSON(w http.ResponseWriter, requestID, model, content string, entries interface{}) {
	resp := map[string]interface{}{
		"id":      "chatcmpl-" + requestID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":       "assistant",
				"content":    content,
				"tool_calls": entries,
			},
			"finish_reason": "tool_calls",
		}},
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request, prompt, model, requestID string, opts zai.Request, conv *zai.Conversation) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	var writeMu sync.Mutex
	writeSSE := func(v interface{}) {
		writeMu.Lock()
		defer writeMu.Unlock()
		data, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Initial chunk
	writeSSE(newChunk("", model, requestID, nil))

	// Keepalive ticker (stopped + joined before return — no writes after
	// close)
	keepAliveStop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// The keepalive is non-essential: writing to a disconnecting
		// client races the handler teardown and can panic on a
		// half-closed connection — an unrecovered panic here kills the
		// whole process. Swallow and stop instead.
		defer func() { recover() }()
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				writeSSE(newChunk("", model, requestID, nil))
			case <-keepAliveStop:
				return
			}
		}
	}()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	finalContent, _, err := s.client.Send(ctx, prompt, opts, func(chunk string) {
		writeSSE(newChunk(chunk, model, requestID, nil))
	}, func(chunk string) {
		writeSSE(newReasoningChunk(chunk, model, requestID))
	})

	if err != nil {
		log.Printf("[Stream] error: %v", err)
		errJSON, _ := json.Marshal(map[string]interface{}{"error": map[string]string{"message": err.Error()}})
		writeMu.Lock()
		fmt.Fprintf(w, "data: %s\n\n", errJSON)
		writeMu.Unlock()
	}

	// Final chunk
	writeSSE(newChunk("", model, requestID, strPtr("stop")))
	writeMu.Lock()
	fmt.Fprintf(w, "data: [DONE]\n\n")
	writeMu.Unlock()
	flusher.Flush()

	close(keepAliveStop)
	wg.Wait()

	// History persistence (authoritative parser content, reasoning
	// excluded)
	if s.cfg.PersistHistory && err == nil && finalContent != "" {
		conv.Messages = append(conv.Messages,
			map[string]interface{}{"role": "user", "content": prompt},
			map[string]interface{}{"role": "assistant", "content": finalContent},
		)
	}
}

// handleChatToolsStream streams a tools request, intercepting
// <<<TOOL_CALL>>> blocks live and rewriting them into tool_calls deltas.
func (s *Server) handleChatToolsStream(w http.ResponseWriter, r *http.Request, req chatRequest, prompt, model, requestID string, conv *zai.Conversation, opts zai.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	convertedMsgs := tools.ConvertMessages(req.Messages, req.Tools)
	convertedPrompt := zai.MessagesToPrompt(convertedMsgs)
	opts.ClientMessages = convertedMsgs

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	var writeMu sync.Mutex
	writeSSE := func(v interface{}) {
		writeMu.Lock()
		defer writeMu.Unlock()
		data, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	emitToolCallDelta := func(tc map[string]interface{}) {
		writeSSE(map[string]interface{}{
			"id":      "chatcmpl-" + requestID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{{
				"index":         0,
				"delta":         map[string]interface{}{"tool_calls": []map[string]interface{}{tc}},
				"finish_reason": nil,
			}},
		})
	}

	writeSSE(newChunk("", model, requestID, nil))

	keepAliveStop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() { recover() }()
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				writeSSE(newChunk("", model, requestID, nil))
			case <-keepAliveStop:
				return
			}
		}
	}()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	interceptor := tools.NewInterceptor()
	toolCallEmitted := false

	finalContent, _, err := s.client.Send(ctx, convertedPrompt, opts, func(chunk string) {
		// edit_content shrinkage is absorbed by the SSE parser (no deltas
		// emitted until caught up), so the interceptor buffer can go
		// stale across a shrink — the end-of-stream ParseCalls fallback
		// below recovers naked JSON, fences and truncated blocks.
		contentDelta, toolCalls, _ := interceptor.Feed(chunk)
		if contentDelta != "" {
			writeSSE(newChunk(contentDelta, model, requestID, nil))
		}
		for _, tc := range toolCalls {
			emitToolCallDelta(tc)
			toolCallEmitted = true
		}
	}, func(chunk string) {
		writeSSE(newReasoningChunk(chunk, model, requestID))
	})

	if err != nil {
		log.Printf("[Stream Tools] error: %v", err)
		writeSSE(formatOpenAIError(err.Error(), "api_error"))
		writeMu.Lock()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		writeMu.Unlock()
		close(keepAliveStop)
		wg.Wait()
		return
	}

	// Flush trailing text (only when no tool calls were emitted)
	if rem := interceptor.FlushFinal(); rem != "" && !toolCallEmitted {
		writeSSE(newChunk(rem, model, requestID, nil))
	}

	// Safety net: full parser over the authoritative content — catches
	// naked JSON, fences and truncated marker blocks.
	if !toolCallEmitted {
		for _, tc := range tools.ParseCalls(finalContent, req.Tools) {
			emitToolCallDelta(map[string]interface{}{
				"index":    0,
				"id":       tc.ID,
				"type":     "function",
				"function": map[string]interface{}{"name": tc.Function.Name, "arguments": tc.Function.Arguments},
			})
			toolCallEmitted = true
		}
	}

	if toolCallEmitted {
		writeSSE(map[string]interface{}{
			"id":      "chatcmpl-" + requestID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": "tool_calls",
			}},
		})
		log.Printf("[Stream Tools] tool_calls emitted")
	} else {
		writeSSE(newChunk("", model, requestID, strPtr("stop")))
	}

	writeMu.Lock()
	fmt.Fprintf(w, "data: [DONE]\n\n")
	writeMu.Unlock()
	flusher.Flush()

	close(keepAliveStop)
	wg.Wait()
}

func (s *Server) handleChatNonStream(w http.ResponseWriter, r *http.Request, prompt, model, requestID string, opts zai.Request, conv *zai.Conversation) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	// Buffered path: authoritative parser content/reasoning, not deltas
	fullContent, fullReasoning, err := s.client.Send(ctx, prompt, opts, nil, nil)

	if err != nil {
		log.Printf("[API] error: %v", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusFromError(err.Error()))
		json.NewEncoder(w).Encode(formatOpenAIError(err.Error(), "api_error"))
		return
	}

	if s.cfg.PersistHistory && fullContent != "" {
		conv.Messages = append(conv.Messages,
			map[string]interface{}{"role": "user", "content": prompt},
			map[string]interface{}{"role": "assistant", "content": fullContent},
		)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(newCompletion(fullContent, fullReasoning, prompt, model, requestID))
}
