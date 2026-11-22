// convert.go — message conversion for tool-mode conversations: embeds
// the tool contract into the message history and flattens tool role
// messages the model cannot see natively.
package tools

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ConvertMessages rewrites an OpenAI-format message history for a model
// without native tool support:
//   - the strict tool contract is embedded into the last plain user
//     message (or an adjacent/leading system message when that message is
//     multimodal or missing);
//   - "tool" role messages become user messages with a [Tool result]
//     prefix;
//   - assistant tool_calls history is rendered as plain JSON text the
//     model recognizes as its own previous output.
//
// Multimodal (content-parts) user messages are preserved byte-exact — a
// string type-assertion would silently destroy image attachments.
func ConvertMessages(messages []json.RawMessage, tools []json.RawMessage) []json.RawMessage {
	var result []json.RawMessage

	// Build the strict tool-call contract framing
	framing := BuildSystemPrompt(tools)

	// Find the last user message index (where we'll embed the framing)
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		var msg map[string]interface{}
		if json.Unmarshal(messages[i], &msg) == nil {
			if role, _ := msg["role"].(string); role == "user" {
				if _, hasToolCallID := msg["tool_call_id"]; !hasToolCallID {
					lastUserIdx = i
					break
				}
			}
		}
	}

	framingPlaced := false

	for i, raw := range messages {
		var msg map[string]interface{}
		if err := json.Unmarshal(raw, &msg); err != nil {
			result = append(result, raw)
			continue
		}

		role, _ := msg["role"].(string)

		switch role {
		case "system":
			// Keep system messages — they carry the client's instructions
			// (the framing contract rides the last user message)
			result = append(result, raw)

		case "tool":
			content, _ := msg["content"].(string)
			toolCallID, _ := msg["tool_call_id"].(string)
			newMsg := map[string]string{
				"role":    "user",
				"content": fmt.Sprintf("[Tool result for %s]: %s", toolCallID, content),
			}
			b, _ := json.Marshal(newMsg)
			result = append(result, b)

		case "assistant":
			if tc, ok := msg["tool_calls"].([]interface{}); ok && len(tc) > 0 {
				// Convert tool_calls to JSON text (the model sees its
				// previous "output")
				var sb strings.Builder
				for _, call := range tc {
					if c, ok := call.(map[string]interface{}); ok {
						if fn, ok := c["function"].(map[string]interface{}); ok {
							name, _ := fn["name"].(string)
							args, _ := fn["arguments"].(string)
							sb.WriteString(fmt.Sprintf("{\"name\":\"%s\",\"arguments\":%s}\n", name, args))
						}
					}
				}
				origContent, _ := msg["content"].(string)
				newMsg := map[string]string{
					"role":    "assistant",
					"content": origContent + "\n" + sb.String(),
				}
				b, _ := json.Marshal(newMsg)
				result = append(result, b)
			} else {
				result = append(result, raw)
			}

		case "user":
			if i == lastUserIdx {
				// Multipart/multimodal content (e.g. image parts): keep
				// the message untouched and place the contract in an
				// adjacent system message instead.
				origStr, isStr := msg["content"].(string)
				if !isStr {
					sys := map[string]string{"role": "system", "content": framing}
					sysBytes, _ := json.Marshal(sys)
					result = append(result, sysBytes)
					result = append(result, raw)
					framingPlaced = true
					continue
				}
				// Embed framing into the last user message
				trimmed := strings.TrimSpace(origStr)
				if trimmed == "" || trimmed == "(no content)" {
					// CLI continuation nudge — direct the model to EXECUTE
					// the announced action instead of narrating again.
					newContent := framing + "\n\nUser: (empty continuation nudge)\n\n" +
						"Continue the CURRENT task NOW: you already announced the next action — emit its tool call block immediately. Do not answer in prose."
					b, _ := json.Marshal(map[string]string{"role": "user", "content": newContent})
					result = append(result, b)
					framingPlaced = true
					continue
				}
				newContent := framing + "Input: \"" + origStr + "\""
				b, _ := json.Marshal(map[string]string{"role": "user", "content": newContent})
				result = append(result, b)
				framingPlaced = true
			} else {
				result = append(result, raw)
			}

		default:
			result = append(result, raw)
		}
	}

	// If no user message was found AND nothing else carried the framing,
	// prepend it as a standalone system message.
	if !framingPlaced {
		sysMsg := map[string]string{"role": "system", "content": framing}
		sysBytes, _ := json.Marshal(sysMsg)
		result = append([]json.RawMessage{sysBytes}, result...)
	}

	return result
}
