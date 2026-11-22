// pool.go — account pool: a token file with one JWT per line.
//
// Policy: a single ZAI_TOKEN may become invalid mid-life (revocation or
// expiry), killing every chat. With a pool, the gateway picks the next
// healthy account instead:
//   - at session init: candidates are validated against GET /api/v1/auths/.
//     Definitively invalid ones (401/403/410) are skipped; the first alive
//     one wins.
//   - if EVERY provided token is invalid, init falls back to the guest flow
//     so the service keeps working (limited model set) rather than refusing.
//   - mid-session auth failures rotate to the next account
//     (see Client.Send's unauthorized branch).
package zai

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
)

// Pool is a round-robin set of Z.AI account tokens.
type Pool struct {
	mu     sync.Mutex
	tokens []string
	idx    int // current candidate
}

// LoadPool parses the token file: one JWT per line, '#' comments and
// blank lines ignored, duplicates dropped, whitespace trimmed.
func LoadPool(path string) (*Pool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read token file: %w", err)
	}
	var tokens []string
	seen := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if seen[line] {
			continue
		}
		seen[line] = true
		tokens = append(tokens, line)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no usable tokens in %s", path)
	}
	log.Printf("[Pool] loaded %d accounts from %s", len(tokens), path)
	return &Pool{tokens: tokens}, nil
}

// Size reports how many accounts the pool holds (0 for nil).
func (p *Pool) Size() int {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.tokens)
}

// Current returns the selected candidate ("" for an absent pool).
func (p *Pool) Current() string {
	if p == nil || len(p.tokens) == 0 {
		return ""
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tokens[p.idx%len(p.tokens)]
}

// Rotate advances the round-robin pointer and returns the new candidate.
func (p *Pool) Rotate() string {
	if p == nil || len(p.tokens) == 0 {
		return ""
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.idx = (p.idx + 1) % len(p.tokens)
	return p.tokens[p.idx]
}

// ClassifyAuthCheck turns an auths/ probe result into an account verdict:
// "alive", "dead", or "ambiguous". A clean 2xx with a JSON id proves the
// account is alive; 401/403/410 mean it is definitively unusable; anything
// else (network flake, odd response) must NOT kill an otherwise-plausible
// token, so callers treat ambiguous verdicts as alive to preserve an
// offline-tolerant boot.
func ClassifyAuthCheck(status int, body string) string {
	switch {
	case status >= 200 && status < 300 && strings.Contains(body, "\"id\""):
		return "alive"
	case status == 401 || status == 403 || status == 410:
		return "dead"
	default:
		return "ambiguous"
	}
}

// ShortToken makes a long JWT safe to print in logs.
func ShortToken(t string) string {
	if len(t) > 16 {
		return t[:16] + "..."
	}
	return t
}
