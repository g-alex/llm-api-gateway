// tools_test.go — parse strategy coverage and stream interceptor checks.
package tools

import (
	"encoding/json"
	"strings"
	"testing"
)

var parseTestTools = []json.RawMessage{
	json.RawMessage(`{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}`),
	json.RawMessage(`{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}`),
}

func decodeCallArgs(t *testing.T, c Call) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(c.Function.Arguments), &m); err != nil {
		t.Fatalf("arguments not a JSON object: %v (%q)", err, c.Function.Arguments)
	}
	return m
}

// Strategy 0 — the streaming contract markers.
func TestParseAgentMarkerBlock(t *testing.T) {
	content := "Checking the file now.\n" +
		"<<<TOOL_CALL>>>\n" +
		`{"name":"write_file","arguments":{"path":"a.txt","content":"hi"}}` + "\n" +
		"<<<END_TOOL_CALL>>>"

	calls := ParseCalls(content, parseTestTools)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "write_file" {
		t.Errorf("tool name = %q, want write_file", calls[0].Function.Name)
	}
	args := decodeCallArgs(t, calls[0])
	if args["path"] != "a.txt" || args["content"] != "hi" {
		t.Errorf("arguments wrong: %+v", args)
	}
}

// Strategy 1 — fenced json with a name field.
func TestParseJSONCodeFenceBlock(t *testing.T) {
	content := "The answer is:\n```json\n" +
		`{"name":"read_file","arguments":{"path":"config.json"}}` +
		"\n```"

	calls := ParseCalls(content, parseTestTools)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Function.Name != "read_file" {
		t.Errorf("tool name = %q, want read_file", calls[0].Function.Name)
	}
	if args := decodeCallArgs(t, calls[0]); args["path"] != "config.json" {
		t.Errorf("arguments wrong: %+v", args)
	}
}

// Strategy 2 (bare object response) and 2b (object wrapped in chatter
// without the json fence).
func TestParseDirectAndEmbeddedJSON(t *testing.T) {
	direct := `{"name":"read_file","arguments":{"path":"t.txt"}}`
	calls := ParseCalls(direct, parseTestTools)
	if len(calls) != 1 || calls[0].Function.Name != "read_file" {
		t.Fatalf("direct JSON not parsed: %+v", calls)
	}

	embedded := "I'll read it right away\n" +
		`{"name":"read_file","arguments":{"path":"x.txt"}}` +
		"\nthat's it"
	calls = ParseCalls(embedded, parseTestTools)
	if len(calls) != 1 || calls[0].Function.Name != "read_file" {
		t.Fatalf("embedded JSON not parsed: %+v", calls)
	}
	if args := decodeCallArgs(t, calls[0]); args["path"] != "x.txt" {
		t.Errorf("arguments wrong: %+v", args)
	}
}

// A parsed call for an unregistered tool must be dropped entirely.
func TestParseDropsUnknownToolName(t *testing.T) {
	content := `<<<TOOL_CALL>>>
{"name":"delete_everything","arguments":{"confirm":true}}
<<<END_TOOL_CALL>>>`

	if calls := ParseCalls(content, parseTestTools); len(calls) != 0 {
		t.Errorf("unknown tool leaked through: %+v", calls)
	}
}

// Regression from a production log: the model emitted a tool call as
// NAKED JSON inside prose — must be caught (strategy 2b).
func TestParseCallsNakedJSONInProse(t *testing.T) {
	content := "Скрипт найден. Теперь прочитаю его для проверки.\n{" +
		`"name":"Read","arguments":{"file_path":"weather.py"}}`
	tools := []json.RawMessage{
		json.RawMessage(`{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}}}}}`),
	}
	calls := ParseCalls(content, tools)
	if len(calls) != 1 || calls[0].Function.Name != "Read" {
		t.Fatalf("naked JSON not caught: %+v", calls)
	}
	var args map[string]interface{}
	json.Unmarshal([]byte(calls[0].Function.Arguments), &args)
	if args["file_path"] != "weather.py" {
		t.Errorf("arguments wrong: %v", args)
	}
}

// StripToolContent must remove the naked JSON span when it parses as a
// named tool call — but leave legitimate JSON without a "name" field
// alone.
func TestStripToolContentNakedJSON(t *testing.T) {
	with := "Скрипт найден. Теперь прочитаю его для проверки.\n{" +
		`"name":"Read","arguments":{"file_path":"weather.py"}}`
	got := StripToolContent(with)
	if strings.Contains(got, "Read") || strings.Contains(got, "weather.py") {
		t.Errorf("naked JSON survived strip: %q", got)
	}
	if !strings.Contains(got, "Скрипт найден") {
		t.Errorf("prose lost: %q", got)
	}
	legit := `Вот JSON: {"result": 42}`
	if got := StripToolContent(legit); !strings.Contains(got, `"result": 42`) {
		t.Errorf("legitimate JSON damaged: %q", got)
	}
}

// ── Message conversion ──

// With tools present, the last user message carrying content-parts must
// survive byte-exact; the tool contract rides along as its own system
// message.
func TestConvertMessagesKeepsMultipartLastUser(t *testing.T) {
	multipart := json.RawMessage(`{"role":"user","content":[` +
		`{"type":"text","text":"What animal is this?"},` +
		`{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}`)
	msgs := []json.RawMessage{multipart}
	out := ConvertMessages(msgs, parseTestTools)

	if len(out) != 2 {
		t.Fatalf("got %d messages, want 2 (system contract + untouched user)", len(out))
	}

	var sys map[string]interface{}
	if err := json.Unmarshal(out[0], &sys); err != nil || sys["role"] != "system" {
		t.Fatalf("first message = %s, want system framing", out[0])
	}
	if s, _ := sys["content"].(string); !strings.Contains(s, toolCallContract[:40]) {
		t.Errorf("system framing missing the tool contract")
	}

	// The user message must be preserved byte-for-byte — including the
	// part.
	var user map[string]interface{}
	if err := json.Unmarshal(out[1], &user); err != nil || user["role"] != "user" {
		t.Fatalf("second message = %s, want original user", out[1])
	}
	parts, ok := user["content"].([]interface{})
	if !ok || len(parts) != 2 {
		t.Fatalf("user content = %s, want the 2-part array preserved", out[1])
	}
	if !strings.Contains(string(out[1]), `"image_url"`) {
		t.Error("image_url part lost during conversion")
	}
	if strings.Contains(string(out[1]), `Input: ""`) {
		t.Error("multipart content collapsed into empty Input string")
	}
}

// A plain-string last user message keeps the embedded framing shape
// (framing + Input: "...").
func TestConvertMessagesStringEmbed(t *testing.T) {
	msgs := []json.RawMessage{
		json.RawMessage(`{"role":"user","content":"plain question"}`),
	}
	out := ConvertMessages(msgs, parseTestTools)

	if len(out) != 1 {
		t.Fatalf("got %d messages, want 1", len(out))
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out[0], &m); err != nil {
		t.Fatal(err)
	}
	content, _ := m["content"].(string)
	if !strings.Contains(content, `Input: "plain question"`) {
		t.Errorf("string embedding changed: %q", content)
	}
	if !strings.Contains(content, toolCallContract[:40]) {
		t.Errorf("framing missing from embedded string: %q", content[:80])
	}
}

// Tool result messages are flattened to user messages.
func TestConvertMessagesFlattensToolRole(t *testing.T) {
	msgs := []json.RawMessage{
		json.RawMessage(`{"role":"user","content":"list files"}`),
		json.RawMessage(`{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a\"}"}}]}`),
		json.RawMessage(`{"role":"tool","tool_call_id":"call_1","content":"ok"}`),
		json.RawMessage(`{"role":"user","content":"thanks"`),
	}
	out := ConvertMessages(msgs, parseTestTools)
	if len(out) != 4 {
		t.Fatalf("got %d messages, want 4", len(out))
	}
	var toolMsg map[string]interface{}
	if err := json.Unmarshal(out[2], &toolMsg); err != nil {
		t.Fatal(err)
	}
	if role, _ := toolMsg["role"].(string); role != "user" {
		t.Errorf("tool role not flattened: %v", toolMsg)
	}
	if c, _ := toolMsg["content"].(string); !strings.Contains(c, "[Tool result for call_1]") || !strings.Contains(c, "ok") {
		t.Errorf("tool result content wrong: %q", c)
	}
}

// ── Stream interceptor ──

// runFeed feeds chunks and collects content, streamed args and flags.
func runFeed(chunks []string) (content string, names []string, args string, finished bool) {
	a := NewInterceptor()
	for _, c := range chunks {
		cd, tcs, fin := a.Feed(c)
		content += cd
		for _, tc := range tcs {
			fn, _ := tc["function"].(map[string]interface{})
			if fn == nil {
				continue
			}
			if n, ok := fn["name"].(string); ok && n != "" {
				names = append(names, n)
			}
			if s, ok := fn["arguments"].(string); ok {
				args += s
			}
		}
		if fin {
			finished = true
		}
	}
	content += a.FlushFinal()
	return
}

// A start marker split across chunk boundaries must still be detected.
func TestMarkerSplitAcrossChunks(t *testing.T) {
	content, names, args, finished := runFeed([]string{
		"Hello <<<TOOL_",
		"CALL>>>\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.txt\"}}",
		"<<<END_TOOL_CALL>>>\nbye",
	})
	if content != "Hello bye" {
		t.Fatalf("content = %q, want %q", content, "Hello bye")
	}
	if len(names) != 1 || names[0] != "write_file" {
		t.Fatalf("names = %v", names)
	}
	if args != `{"path":"a.txt"}` {
		t.Fatalf("args = %q", args)
	}
	if !finished {
		t.Fatal("expected finishToolCalls=true")
	}
}

// Nested braces and escaped quotes inside arguments must not break the
// brace-depth tracking.
func TestNestedBracesAndEscapes(t *testing.T) {
	block := `{"name":"write_file","arguments":{"content":"{\"a\":{}}","n":"}}"}}`
	content, names, args, finished := runFeed([]string{
		"Sure thing.\n<<<TOOL_CALL>>>\n" + block + "\n<<<END_TOOL_CALL>>>",
	})
	if content != "Sure thing.\n" {
		t.Fatalf("content = %q", content)
	}
	if len(names) != 1 || names[0] != "write_file" {
		t.Fatalf("names = %v", names)
	}
	wantArgs := `{"content":"{\"a\":{}}","n":"}}"}`
	if args != wantArgs {
		t.Fatalf("args = %q, want %q", args, wantArgs)
	}
	if !finished {
		t.Fatal("expected finishToolCalls=true")
	}
}

// Non-object arguments fall back to buffered parsing at block end.
func TestFallbackNonObjectArgs(t *testing.T) {
	content, names, args, finished := runFeed([]string{
		`<<<TOOL_CALL>>>{"name":"list_dir","arguments":["x","y"]}`,
		`<<<END_TOOL_CALL>>>done`,
	})
	if !strings.Contains(content, "done") {
		t.Fatalf("content = %q", content)
	}
	// Fallback re-emits the call after the streaming name header — accept
	// both
	if len(names) == 0 || names[0] != "list_dir" {
		t.Fatalf("names = %v", names)
	}
	if args != `["x","y"]` {
		t.Fatalf("args = %q", args)
	}
	if !finished {
		t.Fatal("expected finishToolCalls=true")
	}
}

// Real-world corruption: unescaped quotes inside Python f-strings plus a
// whitespace-padded END marker. Must be salvaged by the repair.
func TestRepairUnescapedQuotesAndLooseEndMarker(t *testing.T) {
	text := `Sure.
<<<TOOL_CALL>>>
{"name":"Write","arguments":{"file_path":"C:/w.py","content":"print(f\"ok\")\nprint(f"bad \" quote")\n"}}
<<<END_TOOL_CALL   >>>
tail`
	calls := ExtractCalls(text)
	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	fn, _ := calls[0]["function"].(map[string]interface{})
	if fn["name"] != "Write" {
		t.Fatalf("name = %v", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	if !strings.Contains(args, `C:/w.py`) {
		t.Fatalf("args lost file_path: %q", args)
	}
	// The repaired content must preserve the literal quotes of the
	// f-string (re-marshaled JSON escapes them again)
	if !strings.Contains(args, `print(f\"bad`) {
		t.Fatalf("args lost unescaped f-string: %q", args)
	}
}

// Stray-quote repair must not break already-valid JSON.
func TestRepairLeavesValidJSONAlone(t *testing.T) {
	in := `{"name":"Bash","arguments":{"command":"echo \"hi\" > f.txt"}}`
	var parsed map[string]interface{}
	if got := parseToolCallRegion(in); got == nil {
		t.Fatal("valid JSON rejected")
	} else if json.Unmarshal([]byte(in), &parsed) != nil {
		t.Fatal("test input itself invalid")
	}
}
