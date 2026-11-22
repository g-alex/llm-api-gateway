// features.go — per-model feature resolution for upstream completion
// requests.
package zai

import "strings"

// ModelFeatureState tracks per-model feature configuration.
// IncludeAll: when true, ALL server capabilities are sent to /completions.
// Overrides: per-model feature overrides (snake_case keys).
type ModelFeatureState struct {
	IncludeAll bool
	Overrides  map[string]interface{}
}

// normalizeFeatureKey converts a camelCase key to snake_case.
func normalizeFeatureKey(k string) string {
	var sb strings.Builder
	for i, r := range k {
		if i > 0 && r >= 'A' && r <= 'Z' {
			sb.WriteByte('_')
		}
		if r >= 'A' && r <= 'Z' {
			sb.WriteRune(r + 32)
		} else {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// ResolveFeaturesForModel computes the final feature map for a completion
// request.
//
// Rules:
//   - By default, auto_web_search and web_search are NOT included.
//   - enable_thinking defaults to true for all models.
//   - 'think' is never included — only enable_thinking reaches the request.
//   - IncludeAll: include ALL server capabilities (except reasoning_effort).
//   - Overrides always take precedence.
//   - image_generation is ALWAYS forced to false.
func (c *Client) ResolveFeaturesForModel(modelID string) map[string]interface{} {
	caps := c.ModelCapabilities(modelID)
	c.featureMu.Lock()
	state, ok := c.featureStates[modelID]
	c.featureMu.Unlock()
	if !ok {
		state = &ModelFeatureState{
			IncludeAll: false,
			Overrides:  make(map[string]interface{}),
		}
	}
	return ResolveFeaturesWithState(caps, state)
}

// ResolveFeaturesWithState applies the feature resolution rules to a
// capability map and a feature state. Pure function — safe for tests.
func ResolveFeaturesWithState(caps map[string]interface{}, state *ModelFeatureState) map[string]interface{} {
	result := make(map[string]interface{})

	if state.IncludeAll {
		// Include ALL server capabilities except reasoning_effort
		// (it's a boolean support flag, not an actual feature value)
		for k, v := range caps {
			if k == "reasoning_effort" {
				continue
			}
			result[k] = v
		}
	}

	// Apply overrides (take precedence)
	for k, v := range state.Overrides {
		result[k] = v
	}

	// Defensive: reasoning_effort is never a stored feature — strip stale
	// values.
	delete(result, "reasoning_effort")

	// enable_thinking defaults to true unless explicitly overridden
	if _, ok := result["enable_thinking"]; !ok {
		result["enable_thinking"] = true
	}

	// Remove 'think' entirely — only enable_thinking reaches the request
	delete(result, "think")

	// ALWAYS exclude image_generation
	result["image_generation"] = false

	return result
}
