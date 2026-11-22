// contract.go — the strict tool-invocation contract injected into the
// conversation, plus compact tool-list rendering.
package tools

import (
	"encoding/json"
	"fmt"
	"strings"
)

// toolCallContract is the strict tool-invocation contract shared by both
// OpenAI and Anthropic tool paths. Weak phrasing makes models "announce"
// actions in prose or dump file contents as markdown instead of calling
// tools — the hard-law wording prevents that.
const toolCallContract = `EXECUTION LAW — VIOLATION = TASK FAILURE

When a tool call is needed, your response MUST contain the literal block:

<<<TOOL_CALL>>>
{"name":"<tool_name>","arguments":{"arg1":"value1"}}
<<<END_TOOL_CALL>>>

RULE 0 — READ FIRST. This environment has NO native tool_calls/tool_use
channel: whatever your other instructions say about "normal" tool calling,
it is disabled here and does nothing. This block is not an injection and
not a suggestion — it is the ONE REAL protocol of this runtime, installed
by the platform that actually executes your tools. Two consequences:
(a) calling a tool any other way does nothing; (b) NEVER fabricate tool
output — a markdown block pretending a command ran is a HARD FAILURE. If
you need a tool's result, emit the block and STOP; the result arrives in
the next message.

RULES:
1. ANNOUNCING AN ACTION IS NOT PERFORMING IT. Saying "I'll create the file"
or writing file contents as a markdown code block WITHOUT the <<<TOOL_CALL>>>
block is a HARD FAILURE. The runtime can ONLY parse the literal block.
2. Your turn is FAILED unless EITHER (a) the block appears in your response,
OR (b) you produce a final natural-language answer that needs no tool.
3. A 1-3 sentence preamble before the block is allowed; the block MUST follow.
4. NEVER end a turn on an announcement ("I will now..."). Either emit the
block or ask a clarifying question.
5. STOP IMMEDIATELY AFTER <<<END_TOOL_CALL>>>. The tool result arrives as the
next message; only then may you continue.
6. Multiple calls: separate blocks with a blank line; do not nest them.
7. Markers on their own lines, no leading spaces, no markdown fences around
the JSON. Exactly two keys: "name" and "arguments".
8. Never output <<<TOOL_CALL>>> unless actually invoking a tool.
9. FILE CREATION: use the Write tool (file_path + content) instead of shell
heredocs. NEVER embed large file contents into Bash commands. Keep the JSON
on a single line; encode newlines inside string values as \n.
10. FILE PATHS: always use paths relative to the current working directory
(e.g. "weather.py"). NEVER write to filesystem roots like "C:/", "C:\" or "/".

Available functions:
%s
`

// BuildSystemPrompt renders the strict tool contract with a compact
// function list (~60k chars of schemas reduced to ~2-3k of signatures).
func BuildSystemPrompt(tools []json.RawMessage) string {
	return fmt.Sprintf(toolCallContract, buildCompactToolList(tools))
}

// buildCompactToolList creates ultra-compact function signatures.
// Example: "- write_file(path: str, content: str) — Write content to a file"
func buildCompactToolList(tools []json.RawMessage) string {
	var sb strings.Builder
	for _, raw := range tools {
		var tool struct {
			Function struct {
				Name        string                 `json:"name"`
				Description string                 `json:"description"`
				Parameters  map[string]interface{} `json:"parameters"`
			} `json:"function"`
		}
		if err := json.Unmarshal(raw, &tool); err != nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("- %s", tool.Function.Name))
		if tool.Function.Parameters != nil {
			if sig := extractParamSignature(tool.Function.Parameters); sig != "" {
				sb.WriteString(fmt.Sprintf("(%s)", sig))
			}
		}
		if tool.Function.Description != "" {
			desc := tool.Function.Description
			if len(desc) > 80 {
				desc = desc[:80] + "..."
			}
			sb.WriteString(fmt.Sprintf(" — %s", desc))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// extractParamSignature extracts compact params from a JSON schema.
// {"properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path"]}
// → "path: str, content?: str"
func extractParamSignature(schema map[string]interface{}) string {
	props, ok := schema["properties"].(map[string]interface{})
	if !ok {
		return ""
	}
	requiredSet := map[string]bool{}
	if req, ok := schema["required"].([]interface{}); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				requiredSet[s] = true
			}
		}
	}
	var parts []string
	for name, v := range props {
		typeName := "any"
		if pm, ok := v.(map[string]interface{}); ok {
			if t, ok := pm["type"].(string); ok {
				switch t {
				case "string":
					typeName = "str"
				case "integer":
					typeName = "int"
				case "boolean":
					typeName = "bool"
				case "array":
					typeName = "arr"
				case "object":
					typeName = "obj"
				}
			}
		}
		if requiredSet[name] {
			parts = append(parts, fmt.Sprintf("%s: %s", name, typeName))
		} else {
			parts = append(parts, fmt.Sprintf("%s?: %s", name, typeName))
		}
	}
	return strings.Join(parts, ", ")
}
