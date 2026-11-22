// sse_test.go — self-checks for the stateful SSE parser
// (edit_index semantics, <details> reasoning split, shrinkage, errors).
package sse

import (
	"encoding/json"
	"io"
	"strconv"
	"strings"
	"testing"
	"unicode/utf16"
)

// feedAll runs SSE lines through the parser, collecting emitted deltas.
func feedAll(p *Parser, lines []string) (content, reasoning string, err error) {
	var c, r strings.Builder
	for _, l := range lines {
		e := p.FeedLine(l, func(d string) { c.WriteString(d) }, func(d string) { r.WriteString(d) })
		if e != nil {
			return c.String(), r.String(), e
		}
	}
	p.FlushFinal(func(d string) { c.WriteString(d) }, func(d string) { r.WriteString(d) })
	return c.String(), r.String(), nil
}

func deltaLine(payload string) string {
	return `data: {"data":{"delta_content":"` + payload + `"}}`
}

func editLine(payload string, index int) string {
	return `data: {"data":{"edit_content":"` + payload + `","edit_index":` + strconv.Itoa(index) + `}}`
}

// edit_index is UTF-16-based: replace "мир" (unit 7) in "привет мир".
func TestEditIndexRuneReplacement(t *testing.T) {
	p := NewParser("")
	content, _, err := feedAll(p, []string{
		deltaLine("привет мир"),
		editLine("ВСЕЛЕННАЯ", 7),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// After shrink+regrow only the regrown tail is re-emitted; final state
	// is what matters — check the parser's accumulated content.
	if got := p.Content(); got != "привет ВСЕЛЕННАЯ" {
		t.Fatalf("content = %q, want %q", got, "привет ВСЕЛЕННАЯ")
	}
	_ = content
}

// edit_index beyond current length truncates at the end (JS substring
// semantics) — plain concatenation, no space padding.
func TestEditIndexPadding(t *testing.T) {
	p := NewParser("")
	p.applyEdit("X", 3)
	if got := p.Content(); got != "X" {
		t.Fatalf("content = %q, want %q", got, "X")
	}
}

// <details> block is split into reasoning; surrounding text stays content.
func TestDetailsSplit(t *testing.T) {
	p := NewParser("")
	content, reasoning, err := feedAll(p, []string{
		deltaLine("<details>step by step</details>The answer is 42."),
		deltaLine(" More."),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "The answer is 42. More." {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "step by step" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

// Unclosed <details> mid-stream: content frozen, reasoning keeps growing.
func TestDetailsUnclosed(t *testing.T) {
	p := NewParser("")
	content, reasoning, err := feedAll(p, []string{
		deltaLine("<details>thinking..."),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if reasoning != "thinking..." {
		t.Fatalf("reasoning = %q", reasoning)
	}

	// Closing the tag releases the content
	content, _, err = feedAll(p, []string{deltaLine("</details>Done")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "Done" {
		t.Fatalf("content = %q, want Done", content)
	}
}

// edit_content truncation under the monotonic emitter: the client keeps
// what it already received (a stale tail cannot be retracted), the emitter
// never RE-EMITS the overlapping region, and the authoritative final
// content is correct.
func TestShrinkageRewind(t *testing.T) {
	p := NewParser("")
	content, _, err := feedAll(p, []string{
		deltaLine("Hello world"),
		editLine("Hi", 0),
		deltaLine(" there"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Content() != "Hi there" {
		t.Fatalf("content = %q, want %q", p.Content(), "Hi there")
	}
	// The rewritten "Hello world" never reached the client: the whole
	// first snapshot fit inside the holdback window and was absorbed.
	if content != "Hi there" {
		t.Fatalf("client stream = %q, want %q", content, "Hi there")
	}
}

func TestInlineErrorAndDone(t *testing.T) {
	p := NewParser("")
	_, _, err := feedAll(p, []string{
		`data: {"data":{"error":{"detail":"Model not available for current user level"}}}`,
	})
	if err == nil || !strings.Contains(err.Error(), "Model not available") {
		t.Fatalf("expected inline error, got %v", err)
	}

	p2 := NewParser("")
	_, _, err = feedAll(p2, []string{deltaLine("ok"), "data: [DONE]"})
	if err != io.EOF {
		t.Fatalf("expected io.EOF on [DONE], got %v", err)
	}
}

// ── UTF-16 edit_index semantics ──

func TestUtf16IndexToByteIndex(t *testing.T) {
	cases := []struct {
		s    string
		idx  int
		want int
		note string
	}{
		{"abcdef", 0, 0, "zero → start"},
		{"abcdef", 3, 3, "ascii passthrough"},
		{"abcdef", 10, 6, "beyond length clamps to end"},
		{"привет мир", 7, 13, "BMP cyrillic: 6 cyr runes + space = 7 units → byte 13"},
		{"🙂ab", 1, 0, "inside surrogate pair → clamp to rune start"},
		{"🙂ab", 2, 4, "second unit after pair → after rune"},
		{"🙂ab", 3, 5, "third unit → after 'a'"},
		{"a𝕏b", 2, 1, "astral mid-pair clamp"},
		{"", 5, 0, "empty string"},
	}
	for _, c := range cases {
		if got := utf16IndexToByteIndex(c.s, c.idx); got != c.want {
			t.Errorf("%s: utf16IndexToByteIndex(%q,%d) = %d, want %d", c.note, c.s, c.idx, got, c.want)
		}
	}
}

func TestApplyEditUtf16EmojiClamp(t *testing.T) {
	// "🙂ab": emoji occupies UTF-16 units 0-1. JS substring(0,3)="🙂a";
	// a naive rune-based index would produce "🙂ab" (whole string).
	p := NewParser("info")
	p.fullText.WriteString("🙂ab")
	p.applyEdit("X", 3)
	if got := p.fullText.String(); got != "🙂aX" {
		t.Errorf("emoji clamp edit = %q, want %q", got, "🙂aX")
	}
}

func TestApplyEditBeyondLengthNoPadding(t *testing.T) {
	// substring(0, bigIndex) === whole string → plain concatenation.
	p := NewParser("info")
	p.fullText.WriteString("abc")
	p.applyEdit("!", 10)
	if got := p.fullText.String(); got != "abc!" {
		t.Errorf("beyond-length edit = %q, want %q", got, "abc!")
	}
}

func TestApplyEditLegacyAppendOnMissingIndex(t *testing.T) {
	// editIndex -1 (field absent) keeps the append behavior.
	p := NewParser("info")
	p.fullText.WriteString("keep")
	p.applyEdit("+tail", -1)
	if got := p.fullText.String(); got != "keep+tail" {
		t.Errorf("legacy append = %q, want %q", got, "keep+tail")
	}
}

func TestFeedLineUtf16RewriteEndToEnd(t *testing.T) {
	// End-to-end: delta builds a string containing an emoji, then an
	// edit_content lands INSIDE that emoji's surrogate pair (UTF-16 index).
	p := NewParser("info")
	_ = p.FeedLine(`data: {"data":{"delta_content":"ab🙂cd"}}`, nil, nil)
	// units: a0 b1 🙂2-3 c4 d5 — edit at 4 = right after the pair
	_ = p.FeedLine(`data: {"data":{"edit_content":"Z!","edit_index":4}}`, nil, nil)
	if got := p.Content(); got != "ab🙂Z!" {
		t.Errorf("e2e utf16 rewrite = %q, want %q", got, "ab🙂Z!")
	}
}

// ── reasoning duplication / quote-marker holdback ──

// sink adapts strings.Builder to the parser callback signature.
func sink(b *strings.Builder) func(string) { return func(s string) { b.WriteString(s) } }

// streamFixture feeds SSE lines through a live parser, accumulating client-
// visible deltas exactly like a client would (append-only), then performs
// the final flush. Returns (clientContent, clientReasoning).
func streamFixture(t *testing.T, lines []string) (string, string) {
	t.Helper()
	p := NewParser("")
	var content, reasoning strings.Builder
	for _, l := range lines {
		if err := p.FeedLine(l, sink(&content), sink(&reasoning)); err != nil && err != io.EOF {
			t.Fatalf("FeedLine: %v", err)
		}
	}
	p.FlushFinal(sink(&content), sink(&reasoning))
	return content.String(), reasoning.String()
}

// editEvent builds an edit_content SSE line whose edit_index is a UTF-16
// code-unit offset.
func editEvent(cur, addition string) string {
	idx := len([]rune(cur))
	for _, r := range cur {
		if utf16.RuneLen(r) > 1 {
			idx++ // astral runes take two UTF-16 units
		}
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"data": map[string]interface{}{"edit_content": cur + addition, "edit_index": float64(idx)},
	})
	return "data: " + string(payload)
}

// TestSSEReasoningQuotedLinesNoDuplication: <details> body streamed one char
// at a time with "> "-quoted lines — reasoning must be emitted exactly once,
// with no stray ">" and no re-emitted prefixes.
func TestSSEReasoningQuotedLinesNoDuplication(t *testing.T) {
	full := "<details type=\"reasoning\" open><summary>Thinking</summary>\n> First line.\n> Second line.</details>The answer."
	runes := []rune(full)
	var lines []string
	for i := 1; i <= len(runes); i++ {
		lines = append(lines, editEvent("", string(runes[:i])))
	}
	lines = append(lines, "data: [DONE]")

	_, reasoning := streamFixture(t, lines)
	want := "Thinking\nFirst line.\nSecond line." // summary header stays in reasoning
	if reasoning != want {
		t.Errorf("reasoning = %q, want %q", reasoning, want)
	}
	if strings.Count(reasoning, "First") != 1 || strings.Count(reasoning, ">") != 0 {
		t.Errorf("duplication/quote leak: %q", reasoning)
	}
}

// TestSSEReasoningQuoteMarkerPartialArrival: transient bare ">" at snapshot
// tail must never reach the client.
func TestSSEReasoningQuoteMarkerPartialArrival(t *testing.T) {
	p := NewParser("")
	var reasoning strings.Builder
	_ = p.FeedLine(`data: {"data":{"content":"<details>"}}`, nil, sink(&reasoning))
	_ = p.FeedLine(`data: {"data":{"edit_content":"<details>>","edit_index":9}}`, nil, sink(&reasoning))
	// snapshot reasoning (stripped) is now ">" — must have been withheld
	if strings.Contains(reasoning.String(), ">") {
		t.Fatalf("transient bare \">\" leaked: %q", reasoning.String())
	}
	_ = p.FeedLine(`data: {"data":{"edit_content":"<details>> abc","edit_index":9}}`, nil, sink(&reasoning))
	p.FlushFinal(nil, sink(&reasoning))
	if got := strings.TrimLeft(reasoning.String(), " "); got != "abc" {
		t.Errorf("reasoning = %q, want %q", got, " abc")
	}
}

// TestSSEReasoningAppendOnlyInvariant: every partial concatenation of
// reasoning deltas must be a prefix of the final reasoning text.
func TestSSEReasoningAppendOnlyInvariant(t *testing.T) {
	full := "<details>\" > one\n> two tails\n> three</details>done"
	runes := []rune(full)
	var lines []string
	for i := 1; i <= len(runes); i++ {
		lines = append(lines, editEvent("", string(runes[:i])))
	}
	p := NewParser("")
	var acc strings.Builder
	for _, l := range lines {
		_ = p.FeedLine(l, nil, sink(&acc))
	}
	p.FlushFinal(nil, sink(&acc))
	finalText := p.Reasoning()
	// append-only: acc must equal finalText exactly (no re-emissions)
	if acc.String() != finalText {
		t.Errorf("client view %q != parser final %q — re-emission happened", acc.String(), finalText)
	}
}

// TestSSEReasoningTailBacktrackAbsorbed: an edit_content tail rewrite inside
// the holdback window leaves no stale "wrong tail" in the client view.
func TestSSEReasoningTailBacktrackAbsorbed(t *testing.T) {
	p := NewParser("")
	var reasoning strings.Builder
	base := "<details>> correct text "
	_ = p.FeedLine(editEvent("", base), nil, sink(&reasoning))
	// upstream rewrote the tail: same prefix, "wrong tail" appended, then
	// corrected within the holdback window
	_ = p.FeedLine(editEvent("", base+"wrong"), nil, sink(&reasoning))
	_ = p.FeedLine(editEvent("", base+"tail</details>"), nil, sink(&reasoning))
	p.FlushFinal(nil, sink(&reasoning))

	if strings.Contains(reasoning.String(), "wrong") {
		t.Errorf("stale backtrack tail leaked: %q", reasoning.String())
	}
	if !strings.HasSuffix(reasoning.String(), "correct text tail") {
		t.Errorf("final reasoning wrong: %q", reasoning.String())
	}
}

func TestHoldBackPartialQuoteMarker(t *testing.T) {
	cases := []struct{ in, want string }{
		{">", ""},
		{"abc\n>", "abc\n"},
		{"", ""},
		{"no marker", "no marker"},
		{"ends with > char >", "ends with > char >"}, // not a bare last-line marker
		{"a > b", "a > b"},
		{"line1\n>x", "line1\n>x"},
		{"x\n>\n", "x\n>\n"},
		{"<details>", "<details>"},
		{"summary >", "summary >"}, // ditto
	}
	for _, c := range cases {
		if got := holdBackPartialQuoteMarker(c.in); got != c.want {
			t.Errorf("holdBackPartialQuoteMarker(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
