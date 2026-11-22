// httpclient.go — browser-fingerprinted HTTP clients (bogdanfinn/tls-client).
// All outgoing Z.AI traffic goes through a Chrome TLS profile so edge
// filters see a real browser fingerprint, not Go's default.
package zai

import (
	"fmt"
	"log"
	"sync"

	fhttp "github.com/bogdanfinn/fhttp"
	tls_client "github.com/bogdanfinn/tls-client"
	"github.com/bogdanfinn/tls-client/profiles"
)

// httpDoer is everything we need from an HTTP client.
type httpDoer interface {
	Do(*fhttp.Request) (*fhttp.Response, error)
}

// userAgent — latest Chrome UA matching the TLS profile below.
const userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"

var (
	clientCache   = map[int]httpDoer{}
	clientCacheMu sync.Mutex
)

// browserClient returns a Chrome-fingerprinted client, cached per timeout
// (a couple of fixed keys). timeoutSecs 0 = no timeout (streaming
// responses need long-lived connections).
func browserClient(timeoutSecs int) (httpDoer, error) {
	clientCacheMu.Lock()
	defer clientCacheMu.Unlock()
	if c, ok := clientCache[timeoutSecs]; ok {
		return c, nil
	}
	c, err := newBrowserClient(timeoutSecs)
	if err != nil {
		return nil, err
	}
	clientCache[timeoutSecs] = c
	log.Printf("[TLS] browser client ready (profile=chrome_146 timeout=%ds)", timeoutSecs)
	return c, nil
}

// newBrowserClient builds a fresh tls-client instance (uncached).
func newBrowserClient(timeoutSecs int) (httpDoer, error) {
	// tls-client quirk: not passing the timeout option at all defaults to
	// a hard 30s whole-request deadline that silently kills thinking-heavy
	// streams. 600s clears every real bound (per-attempt ctx, watchdog).
	if timeoutSecs <= 0 {
		timeoutSecs = 600
	}
	c, err := tls_client.NewHttpClient(nil,
		tls_client.WithClientProfile(profiles.Chrome_146),
		tls_client.WithTimeoutSeconds(timeoutSecs),
	)
	if err != nil {
		return nil, fmt.Errorf("tls client: %w", err)
	}
	return c, nil
}

// zaiHTTP returns the browser-fingerprinted client for Z.AI traffic.
func zaiHTTP() (httpDoer, error) {
	return browserClient(0)
}
