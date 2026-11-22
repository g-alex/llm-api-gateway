// convert_test.go — unit checks for the Anthropic/Responses message
// converters (pure functions, no network).
package api

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestConvertAnthropicImageBlocks(t *testing.T) {
	msg := json.RawMessage(`{"role":"user","content":[` +
		`{"type":"text","text":"What is on the picture?"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}},` +
		`{"type":"image","source":{"type":"url","url":"https://example.com/cat.jpg"}}]}`)
	out := convertAnthropicMessage(msg)
	if len(out) != 1 {
		t.Fatalf("expected single converted message, got %d", len(out))
	}
	var m struct {
		Role    string                   `json:"role"`
		Content []map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(out[0], &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m.Role != "user" || len(m.Content) != 3 {
		t.Fatalf("role=%s parts=%d", m.Role, len(m.Content))
	}
	if m.Content[0]["type"] != "text" {
		t.Fatalf("part 0 should be text, got %v", m.Content[0])
	}
	if m.Content[1]["type"] != "image_url" {
		t.Fatalf("part 1 should be image_url, got %v", m.Content[1])
	}
	iu := m.Content[1]["image_url"].(map[string]interface{})
	if iu["url"] != "data:image/png;base64,QUJD" {
		t.Fatalf("base64 source should become data URL, got %v", iu["url"])
	}
	iu2 := m.Content[2]["image_url"].(map[string]interface{})
	if iu2["url"] != "https://example.com/cat.jpg" {
		t.Fatalf("url source should pass through, got %v", iu2["url"])
	}
}

func TestConvertAnthropicTextOnlyUnchanged(t *testing.T) {
	msg := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"hello"}]}`)
	out := convertAnthropicMessage(msg)
	if len(out) != 1 {
		t.Fatalf("got %d", len(out))
	}
	if !strings.Contains(string(out[0]), `"content":"hello"`) {
		t.Fatalf("text-only should stay a string content: %s", out[0])
	}
}

func TestResponsesImageParts(t *testing.T) {
	raw := json.RawMessage(`[
		{"type":"input_text","text":"look"},
		{"type":"input_image","image_url":"https://example.com/a.png","detail":"high"},
		{"type":"input_image","file_id":"file-123"},
		{"type":"input_image"}
	]`)
	parts := responsesImageParts(raw)
	if len(parts) != 2 {
		t.Fatalf("expected 2 image parts, got %d", len(parts))
	}
	if got := parts[0]["image_url"].(map[string]interface{})["url"]; got != "https://example.com/a.png" {
		t.Fatalf("part 0 url: %v", got)
	}
	if got := parts[1]["image_url"].(map[string]interface{})["url"]; got != "file-123" {
		t.Fatalf("part 1 should fall back to file_id: %v", got)
	}
	if responsesImageParts(json.RawMessage(`[{"type":"input_text","text":"x"}]`)) != nil {
		t.Fatal("text-only content → nil parts")
	}
	if responsesImageParts(nil) != nil {
		t.Fatal("empty content → nil parts")
	}
}

func TestConvertResponsesInputWithImage(t *testing.T) {
	raw := json.RawMessage(`[{"type":"message","role":"user","content":[` +
		`{"type":"input_text","text":"describe"},` +
		`{"type":"input_image","image_url":"data:image/png;base64,QUJD"}]}]`)
	msgs, err := convertResponsesInput(raw)
	if err != nil {
		t.Fatalf("convertResponsesInput: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("got %d msgs", len(msgs))
	}
	var m struct {
		Role    string                   `json:"role"`
		Content []map[string]interface{} `json:"content"`
	}
	if err := json.Unmarshal(msgs[0], &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m.Role != "user" || len(m.Content) != 2 || m.Content[1]["type"] != "image_url" {
		t.Fatalf("bad conversion: %s", msgs[0])
	}
}
