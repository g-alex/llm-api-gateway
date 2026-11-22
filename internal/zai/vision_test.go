// vision_test.go — pure (no-network) checks for the vision pipeline.
package zai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/url"
	"testing"
)

func TestExtractImageURL(t *testing.T) {
	part := map[string]interface{}{
		"type":      "image_url",
		"image_url": map[string]interface{}{"url": " https://x/y.png "},
	}
	if got := extractImageURL(part); got != "https://x/y.png" {
		t.Fatalf("got %q", got)
	}
	if got := extractImageURL(map[string]interface{}{"type": "text", "text": "hi"}); got != "" {
		t.Fatalf("non-image part should be empty, got %q", got)
	}
	if got := extractImageURL(map[string]interface{}{"type": "image_url"}); got != "" {
		t.Fatalf("missing image_url object should be empty, got %q", got)
	}
}

func TestResolveDataURL(t *testing.T) {
	payload := base64.StdEncoding.EncodeToString([]byte("fakepng"))
	img, err := resolveDataURL("data:image/png;base64," + payload)
	if err != nil {
		t.Fatalf("resolveDataURL: %v", err)
	}
	if img.contentType != "image/png" || img.filename != "image.png" || string(img.data) != "fakepng" {
		t.Fatalf("bad image: %+v", img)
	}

	if _, err := resolveDataURL("data:image/png," + payload); err == nil {
		t.Fatal("non-base64 data URL should fail")
	}
	if _, err := resolveDataURL("data:image/png;base64"); err == nil {
		t.Fatal("malformed data URL (no comma) should fail")
	}
}

func TestResolveDataURLLenientPadding(t *testing.T) {
	// StdEncoding without padding — must fall through to base64Decode.
	raw := base64.RawStdEncoding.EncodeToString([]byte("abc123"))
	img, err := resolveDataURL("data:;base64," + raw)
	if err != nil {
		t.Fatalf("lenient decode failed: %v", err)
	}
	if string(img.data) != "abc123" {
		t.Fatalf("got %q", img.data)
	}
}

func TestBase64DecodeVariants(t *testing.T) {
	cases := map[string]string{
		base64.RawStdEncoding.EncodeToString([]byte("hi")): "hi",
		base64.RawURLEncoding.EncodeToString([]byte("hi")): "hi",
		base64.StdEncoding.EncodeToString([]byte("hi")):    "hi",
		base64.URLEncoding.EncodeToString([]byte("hi")):    "hi",
	}
	for in, want := range cases {
		got, err := base64Decode(in)
		if err != nil || string(got) != want {
			t.Fatalf("base64Decode(%q) = %q, %v", in, got, err)
		}
	}
	if _, err := base64Decode("!!!not-base64!!!"); err == nil {
		t.Fatal("garbage should fail")
	}
}

func TestExtFromMime(t *testing.T) {
	if extFromMime("image/png") != ".png" || extFromMime("IMAGE/JPEG") != ".jpg" || extFromMime("application/zip") != ".bin" {
		t.Fatal("extFromMime mapping broken")
	}
}

func TestFilenameFromURL(t *testing.T) {
	u, _ := url.Parse("https://example.com/path/photo 2024!.png?a=b")
	if got := filenameFromURL(u); got != "photo_2024_.png" {
		t.Fatalf("got %q", got)
	}
	u2, _ := url.Parse("https://example.com/")
	if got := filenameFromURL(u2); got != "" {
		t.Fatalf("empty path should give empty filename, got %q", got)
	}
}

func TestProcessVisionMessagesPassthrough(t *testing.T) {
	c := &Client{}
	raw := json.RawMessage(`[{"role":"user","content":"plain text"},{"role":"user","content":[{"type":"text","text":"typed"}]}]`)
	rewritten, files, err := c.ProcessVisionMessages(context.Background(), raw)
	if err != nil {
		t.Fatalf("passthrough: %v", err)
	}
	if string(rewritten) != string(raw) {
		t.Fatalf("text-only input must be byte-identical:\n%s\n%s", raw, rewritten)
	}
	if files != nil {
		t.Fatalf("no images → nil files, got %v", files)
	}
}

// Multipart (image part) survival through the pure transforms:
// GetMessageContent must keep text parts only, messagesToPrompt must
// flatten them without binary garbage.
func TestGetMessageContentExtractsTextOnly(t *testing.T) {
	got := GetMessageContent(json.RawMessage(`[{"type":"text","text":"hello "},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"world"}]`))
	if got != "hello \nworld" { // parts are joined with "\n"
		t.Errorf("GetMessageContent = %q, want %q", got, "hello \nworld")
	}
	multipart := json.RawMessage(`{"role":"user","content":[` +
		`{"type":"text","text":"What animal is this?"},` +
		`{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}`)
	if messagesToPrompt([]json.RawMessage{multipart}) != "What animal is this?" {
		t.Errorf("messagesToPrompt flattening broken: %q", messagesToPrompt([]json.RawMessage{multipart}))
	}
}
