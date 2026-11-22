// models.go — live model catalog from the upstream /api/models endpoint
// (cached 5 min) with a static fallback when upstream is unreachable.
package zai

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"strings"
	"time"

	fhttp "github.com/bogdanfinn/fhttp"
)

// ModelInfo describes one upstream model.
type ModelInfo struct {
	ID           string
	Name         string
	Description  string
	Capabilities map[string]interface{}
}

const modelsCacheTTL = 5 * time.Minute

// After a failed live fetch don't hammer the upstream again for this long:
// capability lookups and dashboard polls would otherwise fire a fresh
// request on every call. Var for test injection.
var modelsFailureBackoff = time.Minute

// fallbackModels is used when the upstream is unreachable and the cache is
// empty. Snapshot of the live lineup (HAR 2026-08); keep in sync with the
// upstream. Capabilities here are limited to what feature dispatch actually
// consumes (reasoning_effort on the flagships), so applyReasoningEffort
// keeps working even when the live list cannot be fetched — GET /api/models
// is edge-gated against non-browser clients, so the live path fails closed
// far more often than not.
var fallbackModels = []ModelInfo{
	{ID: "x-preview-l", Name: "GLM-5.3-Flash", Description: "Lightweight flagship model, premium quality, instant response",
		Capabilities: map[string]interface{}{"reasoning_effort": true}},
	{ID: "glm-5.3", Name: "GLM-5.3", Description: "Flagship model, excels at coding and long-horizon tasks",
		Capabilities: map[string]interface{}{"reasoning_effort": true}},
	{ID: "glm-5.2", Name: "GLM-5.2", Description: "Previous flagship model"},
	{ID: "GLM-5-Turbo", Name: "GLM-5-Turbo", Description: "New model for chat, coding, and agentic tasks"},
	{ID: "GLM-5v-Turbo", Name: "GLM-5V-Turbo", Description: "Vision model with evolved intelligence"},
	{ID: "glm-4.7", Name: "GLM-4.7", Description: "Classic high-performance model"},
}

// FetchModels retrieves the model list from the upstream /api/models,
// keeping only models down to glm-4.7 (inclusive; the API returns
// newest-first). The cutoff is intentional: glm-4.7 is the supported floor
// and legacy entries below it are dropped. To expose the full upstream
// list, remove the cutoff in the loop below.
func (c *Client) FetchModels() []ModelInfo {
	c.modelsMu.Lock()
	defer c.modelsMu.Unlock()

	if len(c.modelsCache) > 0 && time.Since(c.modelsCacheTime) < modelsCacheTTL {
		return c.modelsCache
	}

	fail := func(msg string) []ModelInfo {
		c.modelsLastFailure = time.Now()
		if msg != "" {
			log.Printf("[Models] %s", msg)
		}
		if len(c.modelsCache) > 0 {
			return c.modelsCache
		}
		return fallbackModels
	}

	// Inside the failure backoff window: serve cache/fallback silently.
	if !c.modelsLastFailure.IsZero() && time.Since(c.modelsLastFailure) < modelsFailureBackoff {
		return fail("")
	}

	token := c.session.tokenValue()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client, err := zaiHTTP()
	if err != nil {
		return fail("fetchModels client: " + err.Error())
	}
	req, err := fhttp.NewRequestWithContext(ctx, "GET", c.BaseURL+"/api/models", nil)
	if err != nil {
		return fail("fetchModels request: " + err.Error())
	}
	// Mirror the browser's request shape: without the Origin/Referer +
	// Client Hints / Sec-Fetch headers the endpoint answers 403 even when
	// the session token itself is valid.
	req.Header.Set("Origin", c.BaseURL)
	req.Header.Set("Referer", c.BaseURL+"/")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept-Language", "en-US")
	req.Header.Set("authorization", "Bearer "+token)
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("sec-ch-ua", `"Chromium";v="146", "Google Chrome";v="146", "Not)A;Brand";v="24"`)
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-platform", `"Windows"`)
	req.Header.Set("Sec-Fetch-Site", "same-origin")
	req.Header.Set("Sec-Fetch-Mode", "cors")
	req.Header.Set("Sec-Fetch-Dest", "empty")
	resp, err := client.Do(req)
	if err != nil {
		return fail("fetchModels do: " + err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fail("fetchModels status: " + resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fail("fetchModels read: " + err.Error())
	}

	var apiResp struct {
		Data []struct {
			ID   string `json:"id"`
			Name string `json:"name"`
			Info struct {
				Name string `json:"name"`
				Meta struct {
					Description  string                 `json:"description"`
					Capabilities map[string]interface{} `json:"capabilities"`
				} `json:"meta"`
			} `json:"info"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &apiResp); err != nil {
		return fail("fetchModels parse: " + err.Error())
	}

	var filtered []ModelInfo
	for _, m := range apiResp.Data {
		filtered = append(filtered, ModelInfo{
			ID:           m.ID,
			Name:         m.Name,
			Description:  m.Info.Meta.Description,
			Capabilities: m.Info.Meta.Capabilities,
		})
		// glm-4.7 is the cutoff — stop here (inclusive)
		if m.ID == "glm-4.7" {
			break
		}
	}

	if len(filtered) > 0 {
		c.modelsCache = filtered
		c.modelsCacheTime = time.Now()
		log.Printf("[Models] fetched %d models from upstream", len(filtered))
	}

	if len(c.modelsCache) > 0 {
		return c.modelsCache
	}
	return fallbackModels
}

// StartModelRefresher keeps the model cache warm: one fetch shortly after
// startup, then a refresh every modelsCacheTTL. Intended to run as a
// background goroutine.
func (c *Client) StartModelRefresher() {
	time.Sleep(5 * time.Second)
	c.FetchModels()
	ticker := time.NewTicker(modelsCacheTTL)
	defer ticker.Stop()
	for range ticker.C {
		c.FetchModels()
	}
}

// findModelByID returns the model with a case-insensitive ID match
// (upstream mixes lower/upper case in IDs: glm-4.7 vs GLM-5-Turbo).
func findModelByID(models []ModelInfo, id string) *ModelInfo {
	for i := range models {
		if strings.EqualFold(models[i].ID, id) {
			return &models[i]
		}
	}
	return nil
}

// ModelCapabilities returns the raw capabilities map for a model.
func (c *Client) ModelCapabilities(modelID string) map[string]interface{} {
	if m := findModelByID(c.FetchModels(), modelID); m != nil {
		return m.Capabilities
	}
	return nil
}

// ModelSupportsReasoningEffort returns true only when the model's
// capabilities explicitly contain "reasoning_effort": true.
func (c *Client) ModelSupportsReasoningEffort(modelID string) bool {
	if modelID == "" {
		return false
	}
	caps := c.ModelCapabilities(modelID)
	if caps == nil {
		return false
	}
	v, ok := caps["reasoning_effort"].(bool)
	return ok && v
}

// isValidReasoningEffort validates the accepted reasoning_effort values.
func isValidReasoningEffort(value string) bool {
	switch value {
	case "low", "high", "max":
		return true
	default:
		return false
	}
}
