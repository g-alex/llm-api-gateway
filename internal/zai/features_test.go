package zai

import "testing"

func TestResolveFeaturesDefaults(t *testing.T) {
	state := &ModelFeatureState{Overrides: map[string]interface{}{}}
	got := ResolveFeaturesWithState(nil, state)

	if v, ok := got["enable_thinking"].(bool); !ok || !v {
		t.Errorf("enable_thinking should default to true, got %v", got["enable_thinking"])
	}
	if v, ok := got["image_generation"].(bool); !ok || v {
		t.Errorf("image_generation must be forced to false, got %v", got["image_generation"])
	}
	if _, ok := got["think"]; ok {
		t.Error("'think' must never reach the request")
	}
	if _, ok := got["web_search"]; ok {
		t.Error("web_search must not be included by default")
	}
}

func TestResolveFeaturesOverridesWin(t *testing.T) {
	state := &ModelFeatureState{
		Overrides: map[string]interface{}{
			"enable_thinking": false,
			"auto_web_search": true,
		},
	}
	got := ResolveFeaturesWithState(nil, state)

	if v, ok := got["enable_thinking"].(bool); !ok || v {
		t.Errorf("override enable_thinking=false ignored: %v", got["enable_thinking"])
	}
	if v, ok := got["auto_web_search"].(bool); !ok || !v {
		t.Errorf("auto_web_search override ignored: %v", got["auto_web_search"])
	}
}

func TestResolveFeaturesIncludeAllExceptReasoningEffort(t *testing.T) {
	caps := map[string]interface{}{
		"thinking":         true,
		"reasoning_effort": true, // support flag, must not leak through
	}
	state := &ModelFeatureState{IncludeAll: true, Overrides: map[string]interface{}{}}
	got := ResolveFeaturesWithState(caps, state)

	if v, ok := got["thinking"].(bool); !ok || !v {
		t.Errorf("IncludeAll should copy capabilities, got %v", got["thinking"])
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Error("reasoning_effort must never be a stored feature value")
	}
}

func TestNormalizeFeatureKey(t *testing.T) {
	cases := map[string]string{
		"enableThinking": "enable_thinking",
		"webSearch":      "web_search",
		"simple":         "simple",
	}
	for in, want := range cases {
		if got := normalizeFeatureKey(in); got != want {
			t.Errorf("normalizeFeatureKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestIsValidReasoningEffort(t *testing.T) {
	for _, ok := range []string{"low", "high", "max"} {
		if !isValidReasoningEffort(ok) {
			t.Errorf("%q should be valid", ok)
		}
	}
	for _, bad := range []string{"", "medium", "MINIMAL"} {
		if isValidReasoningEffort(bad) {
			t.Errorf("%q should be invalid", bad)
		}
	}
}

func TestGetMessageContent(t *testing.T) {
	if got := GetMessageContent([]byte(`"plain"`)); got != "plain" {
		t.Errorf("string content = %q", got)
	}
	parts := `[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"b"}]`
	if got := GetMessageContent([]byte(parts)); got != "a\nb" {
		t.Errorf("parts content = %q, want %q", got, "a\nb")
	}
	if got := GetMessageContent(nil); got != "" {
		t.Errorf("empty content = %q", got)
	}
}
