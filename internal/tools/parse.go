// Package tools implements tool calling for models without native
// function calling: a strict prompt contract, multi-strategy parsing of
// model output into OpenAI-style tool calls, and a streaming interceptor
// that rewrites <<<TOOL_CALL>>> blocks into incremental tool_calls deltas.
package tools

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
)

// callID returns a short random id for tool call entries.
func callID() string {
	var b [4]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

var jsonToolRegex = regexp.MustCompile("(?s)```json\\s*(\\{.*?\\})\\s*```")

var toolTagRegex = regexp.MustCompile(`(?s)<<TOOL>>\s*(\{.*?\})\s*<</TOOL>>`)

// Call is one parsed OpenAI-style tool call.
type Call struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ParseCalls tries multiple strategies to extract tool calls from a model
// response:
//  0. <<<TOOL_CALL>>> contract blocks (streaming contract)
//  1. ```json blocks with a "name" field
//  2. direct JSON object / embedded JSON / one JSON per line
//  3. <<TOOL>> tags (legacy)
//  4. natural-language patterns ("create file X with content Y")
func ParseCalls(content string, tools []json.RawMessage) []Call {
	schemas := buildSchemaMap(tools)

	if calls := filterValidCalls(dedupCalls(parseAgentMarkerCalls(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseJSONToolCalls(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseDirectJSON(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseEmbeddedJSON(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseMultilineJSON(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseTagToolCalls(content)), schemas); len(calls) > 0 {
		return calls
	}
	if calls := filterValidCalls(dedupCalls(parseNaturalLanguage(content, tools)), schemas); len(calls) > 0 {
		return calls
	}
	return nil
}

// buildSchemaMap indexes tool definitions by name → their JSON-Schema
// "parameters".
func buildSchemaMap(tools []json.RawMessage) map[string]map[string]interface{} {
	m := make(map[string]map[string]interface{})
	for _, raw := range tools {
		var t struct {
			Function struct {
				Name       string                 `json:"name"`
				Parameters map[string]interface{} `json:"parameters"`
			} `json:"function"`
		}
		if json.Unmarshal(raw, &t) == nil && t.Function.Name != "" {
			m[t.Function.Name] = t.Function.Parameters
		}
	}
	return m
}

// filterValidCalls drops tool calls with unknown names or invalid JSON
// arguments. Schema mismatches are logged but the call is still
// forwarded — the client does its own validation and model-emitted
// schemas are often slightly off.
func filterValidCalls(calls []Call, schemas map[string]map[string]interface{}) []Call {
	var valid []Call
	for _, c := range calls {
		if _, ok := schemas[c.Function.Name]; !ok {
			log.Printf("[Tools] drop %q: unknown tool name", c.Function.Name)
			continue
		}
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(c.Function.Arguments), &obj); err != nil {
			log.Printf("[Tools] drop %q: arguments not a valid JSON object: %v", c.Function.Name, err)
			continue
		}
		if schema := schemas[c.Function.Name]; schema != nil {
			if verr := validateAgainstSchema(obj, schema); verr != nil {
				log.Printf("[Tools] schema mismatch for %q (forwarding): %v", c.Function.Name, verr)
			}
		}
		valid = append(valid, c)
	}
	return valid
}

// CleanFallbackContent replaces a dropped tool-call JSON with a readable
// message so the user doesn't see raw JSON in the chat. Keeps any
// readable text the model wrote around it.
func CleanFallbackContent(content string) string {
	trimmed := strings.TrimSpace(content)
	if looksLikeToolCallJSON(trimmed) {
		return "I couldn't complete that action. Could you rephrase or provide more detail?"
	}
	cleaned := StripToolContent(content)
	if strings.TrimSpace(cleaned) == "" {
		return "I couldn't complete that action. Could you rephrase or provide more detail?"
	}
	return cleaned
}

// looksLikeToolCallJSON reports whether s is a JSON object with "name"
// and "arguments" — i.e. a bare tool call the model emitted as text.
func looksLikeToolCallJSON(s string) bool {
	if len(s) == 0 || s[0] != '{' {
		return false
	}
	var probe struct {
		Name      json.RawMessage `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	return json.Unmarshal([]byte(s), &probe) == nil && len(probe.Name) > 0
}

// validateAgainstSchema is a minimal recursive JSON-Schema validator:
// type, required, properties, items. Enough to catch common model
// mistakes (string where object expected, missing required fields). Not
// a full validator.
func validateAgainstSchema(value interface{}, schema map[string]interface{}) error {
	if t, ok := schema["type"].(string); ok && t != "" {
		if err := checkJSONType(value, t); err != nil {
			return err
		}
	}
	switch v := value.(type) {
	case map[string]interface{}:
		if req, ok := schema["required"].([]interface{}); ok {
			for _, r := range req {
				if key, ok := r.(string); ok {
					if _, exists := v[key]; !exists {
						return fmt.Errorf("missing required field %q", key)
					}
				}
			}
		}
		if props, ok := schema["properties"].(map[string]interface{}); ok {
			for k, val := range v {
				if propSchema, ok := props[k].(map[string]interface{}); ok {
					if err := validateAgainstSchema(val, propSchema); err != nil {
						return fmt.Errorf("field %q: %w", k, err)
					}
				}
			}
		}
	case []interface{}:
		if itemsSchema, ok := schema["items"].(map[string]interface{}); ok {
			for i, item := range v {
				if err := validateAgainstSchema(item, itemsSchema); err != nil {
					return fmt.Errorf("item[%d]: %w", i, err)
				}
			}
		}
	}
	return nil
}

func checkJSONType(value interface{}, t string) error {
	switch t {
	case "object":
		if _, ok := value.(map[string]interface{}); !ok {
			return fmt.Errorf("expected object")
		}
	case "array":
		if _, ok := value.([]interface{}); !ok {
			return fmt.Errorf("expected array")
		}
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string")
		}
	case "number", "integer":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("expected number")
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected boolean")
		}
	}
	return nil
}

// mkcall builds a Call with the given name/arguments.
func mkcall(name, args string) Call {
	c := Call{Type: "function"}
	c.Function.Name = name
	c.Function.Arguments = args
	return c
}

// parseDirectJSON handles a response that is just a JSON object.
func parseDirectJSON(content string) []Call {
	stripped := strings.TrimSpace(content)
	var direct struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	// Also accept "action" as field name (some models use this instead
	// of "name").
	if direct.Name == "" {
		var alt struct {
			Action    string          `json:"action"`
			Arguments json.RawMessage `json:"arguments"`
		}
		json.Unmarshal([]byte(stripped), &alt)
		if alt.Action != "" {
			direct.Name = alt.Action
			direct.Arguments = alt.Arguments
		}
	}
	if err := json.Unmarshal([]byte(stripped), &direct); err == nil && direct.Name != "" {
		args := string(direct.Arguments)
		if !json.Valid(direct.Arguments) {
			args = "{}"
		}
		if direct.Name == "__done__" {
			// Model wrapped a tool call inside __done__.result — extract
			// it.
			var done struct {
				Result string `json:"result"`
			}
			json.Unmarshal(direct.Arguments, &done)
			if done.Result != "" {
				resultStr := strings.TrimSpace(done.Result)
				resultStr = strings.TrimPrefix(resultStr, "```json\n")
				resultStr = strings.TrimPrefix(resultStr, "```\n")
				resultStr = strings.TrimSuffix(resultStr, "\n```")
				resultStr = strings.TrimSpace(resultStr)
				var nested struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				}
				if json.Unmarshal([]byte(resultStr), &nested) == nil && nested.Name != "" && nested.Name != "__done__" {
					nestedArgs := string(nested.Arguments)
					if !json.Valid(nested.Arguments) {
						nestedArgs = "{}"
					}
					return []Call{mkcall(nested.Name, nestedArgs)}
				}
			}
			return nil // genuine text response
		}
		return []Call{mkcall(direct.Name, args)}
	}
	return nil
}

// parseEmbeddedJSON extracts a JSON tool call that the model wrapped in
// text or plain code fences. Finds the first { and last }, then tries to
// parse the substring as a tool call.
func parseEmbeddedJSON(content string) []Call {
	first := strings.IndexByte(content, '{')
	if first < 0 {
		return nil
	}
	last := strings.LastIndexByte(content, '}')
	if last <= first {
		return nil
	}
	extracted := content[first : last+1]
	if extracted == strings.TrimSpace(content) {
		return nil // parseDirectJSON already tried the full content
	}
	return parseDirectJSON(extracted)
}

// parseMultilineJSON handles one JSON object per line.
func parseMultilineJSON(content string) []Call {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var calls []Call
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		cleanLine := line
		for len(cleanLine) > 0 && (cleanLine[len(cleanLine)-1] == '}' || cleanLine[len(cleanLine)-1] == '`') {
			var generic map[string]interface{}
			if err := json.Unmarshal([]byte(cleanLine), &generic); err == nil {
				if nameRaw, ok := generic["name"].(string); ok && nameRaw != "" && nameRaw != "__done__" {
					var argsStr string
					if argsRaw, ok := generic["arguments"]; ok {
						if argsMap, isMap := argsRaw.(map[string]interface{}); isMap {
							argsBytes, _ := json.Marshal(argsMap)
							argsStr = string(argsBytes)
						} else if argsString, isString := argsRaw.(string); isString {
							argsStr = argsString
						}
					}

					if argsStr == "" {
						delete(generic, "name")
						delete(generic, "type")
						argsBytes, _ := json.Marshal(generic)
						argsStr = string(argsBytes)
					}
					if argsStr == "" || argsStr == "null" {
						argsStr = "{}"
					}

					calls = append(calls, mkcall(nameRaw, argsStr))
					break
				}
			}
			cleanLine = cleanLine[:len(cleanLine)-1]
		}
	}
	return calls
}

// dedupCalls removes duplicate tool calls (same name + same arguments).
func dedupCalls(calls []Call) []Call {
	seen := make(map[string]bool)
	var result []Call
	for _, c := range calls {
		key := c.Function.Name + ":" + c.Function.Arguments
		if !seen[key] {
			seen[key] = true
			c.ID = fmt.Sprintf("call_%d", len(result)+1)
			result = append(result, c)
		}
	}
	return result
}

func parseJSONToolCalls(content string) []Call {
	matches := jsonToolRegex.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}

	var calls []Call
	for _, m := range matches {
		var parsed struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &parsed); err != nil {
			continue
		}
		if parsed.Name == "" {
			continue
		}
		argBytes, _ := json.Marshal(parsed.Arguments)
		calls = append(calls, mkcall(parsed.Name, string(argBytes)))
	}
	return calls
}

func parseTagToolCalls(content string) []Call {
	matches := toolTagRegex.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}

	var calls []Call
	for _, m := range matches {
		var parsed struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &parsed); err != nil {
			continue
		}
		if parsed.Name == "" {
			continue
		}
		argBytes, _ := json.Marshal(parsed.Arguments)
		calls = append(calls, mkcall(parsed.Name, string(argBytes)))
	}
	return calls
}

// parseNaturalLanguage extracts tool calls from a model's natural
// language response. Detects patterns like "echo "content" > file" or
// code blocks with file paths.
func parseNaturalLanguage(content string, tools []json.RawMessage) []Call {
	toolNames := make(map[string]bool)
	for _, raw := range tools {
		var tool struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if json.Unmarshal(raw, &tool) == nil {
			toolNames[tool.Function.Name] = true
		}
	}

	var calls []Call

	// Pattern 1: bash "echo" commands that write to files
	echoRegex := regexp.MustCompile(`echo\s+["']?(.*?)["']?\s*>\s*(\S+)`)
	for _, m := range echoRegex.FindAllStringSubmatch(content, -1) {
		if !toolNames["write_file"] && !toolNames["write"] {
			continue
		}
		fileContent := m[1]
		filePath := strings.Trim(m[2], "`\"'")
		toolName := "write_file"
		if !toolNames["write_file"] {
			toolName = "write"
		}
		args, _ := json.Marshal(map[string]string{"path": filePath, "content": fileContent})
		calls = append(calls, mkcall(toolName, string(args)))
	}

	// Pattern 2: code blocks with language hint that likely represent
	// file content to be written
	codeBlockRegex := regexp.MustCompile("(?s)```(\\w+)?\\s*\n(.*?)\n```")
	codeBlocks := codeBlockRegex.FindAllStringSubmatch(content, -1)

	// Pattern 3: "Save this to file.txt" or "create file called X"
	fileRefRegex := regexp.MustCompile(`(?:file|File)\s+(?:called|named)\s+["']?(.+?)["']?`)
	fileRefs := fileRefRegex.FindAllStringSubmatch(content, -1)

	if len(codeBlocks) > 0 && len(fileRefs) > 0 && (toolNames["write_file"] || toolNames["write"]) {
		toolName := "write_file"
		if !toolNames["write_file"] {
			toolName = "write"
		}
		filePath := strings.Trim(fileRefs[0][1], "' \".")
		fileContent := codeBlocks[0][2]

		args, _ := json.Marshal(map[string]string{"path": filePath, "content": fileContent})
		calls = append(calls, mkcall(toolName, string(args)))
	}

	return calls
}

// parseAgentMarkerCalls converts extracted <<<TOOL_CALL>>> blocks (see
// stream.go ExtractCalls) into Call entries.
func parseAgentMarkerCalls(content string) []Call {
	var calls []Call
	for _, tc := range ExtractCalls(content) {
		fn, _ := tc["function"].(map[string]interface{})
		if fn == nil {
			continue
		}
		name, _ := fn["name"].(string)
		args, _ := fn["arguments"].(string)
		if name == "" {
			continue
		}
		calls = append(calls, mkcall(name, args))
	}
	return calls
}

// StripToolContent removes every tool-call representation from text and
// returns the residual content (trimmed).
func StripToolContent(content string) string {
	cleaned := StripBlocks(content)
	cleaned = toolTagRegex.ReplaceAllString(cleaned, "")
	cleaned = jsonToolRegex.ReplaceAllString(cleaned, "")
	cleaned = stripNakedToolJSON(cleaned)
	return strings.TrimSpace(cleaned)
}

// stripNakedToolJSON removes a bare {"name":"<tool>","arguments":{…}}
// object that the model emitted inline WITHOUT the <<<TOOL_CALL>>>
// markers. Only spans that actually parse as a named tool call are
// removed — legitimate JSON answers (no "name" field) survive untouched.
func stripNakedToolJSON(content string) string {
	first := strings.Index(content, "{")
	if first < 0 {
		return content
	}
	last := strings.LastIndex(content, "}")
	if last <= first {
		return content
	}
	span := content[first : last+1]
	var probe struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if json.Unmarshal([]byte(span), &probe) != nil || probe.Name == "" {
		return content
	}
	return strings.TrimSpace(content[:first] + content[last+1:])
}
