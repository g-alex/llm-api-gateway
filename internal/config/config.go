// Package config loads llm-api-gateway configuration from an optional
// .env file and environment variables, and validates it before startup.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds every runtime setting of the gateway.
type Config struct {
	Host string // listen address
	Port string // listen port

	// AuthToken is the Bearer token API clients must present.
	// Required: the server refuses to start without it.
	AuthToken string

	ZAIToken       string // single Z.AI JWT ("" = guest session)
	ZAITokensFile  string // account pool file, one JWT per line
	UseAccountPool bool   // enable the account pool
	LogLevel       string // debug | info | warn | error
	Timeout        time.Duration
	ClaudeDesktop  bool // expose claude-* aliases on /v1/models and /v1/messages
	AnthropicMinMS int  // min gap between upstream sends for /v1/messages
	PersistHistory bool // append request/response turns to the conversation store
}

// validLogLevels lists accepted LOG_LEVEL values.
var validLogLevels = map[string]bool{
	"debug": true, "info": true, "warn": true, "error": true,
}

// Load reads the optional .env file (real environment variables take
// precedence), builds the Config and validates it.
func Load() (*Config, error) {
	LoadEnvFile(".env")
	cfg := &Config{
		Host:           envOr("HOST", "0.0.0.0"),
		Port:           envOr("PORT", "5082"),
		AuthToken:      os.Getenv("AUTH_TOKEN"),
		ZAIToken:       os.Getenv("ZAI_TOKEN"),
		ZAITokensFile:  envOr("ZAI_TOKENS_FILE", "zai_tokens.txt"),
		UseAccountPool: envOr("USE_ACCOUNT_POOL", "") == "true",
		LogLevel:       envOr("LOG_LEVEL", "info"),
		Timeout:        time.Duration(envIntOr("TIMEOUT", 300000)) * time.Millisecond,
		ClaudeDesktop:  envOr("CLAUDE_DESKTOP", "") == "true",
		AnthropicMinMS: envIntOr("ANTHROPIC_ZAI_MIN_MS", 1500),
		PersistHistory: envOr("PERSIST_HISTORY", "") == "true",
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// Validate reports configuration errors before the server starts.
func (c *Config) Validate() error {
	if strings.TrimSpace(c.AuthToken) == "" {
		return fmt.Errorf("AUTH_TOKEN is required: set it in the environment or .env")
	}
	if _, err := strconv.Atoi(c.Port); err != nil {
		return fmt.Errorf("PORT must be numeric, got %q", c.Port)
	}
	if !validLogLevels[c.LogLevel] {
		return fmt.Errorf("LOG_LEVEL must be one of debug|info|warn|error, got %q", c.LogLevel)
	}
	if c.Timeout <= 0 {
		return fmt.Errorf("TIMEOUT must be positive, got %v", c.Timeout)
	}
	return nil
}

// LoadEnvFile applies KEY=VALUE pairs from a .env-style file to the
// process environment. Values already set in the real environment win,
// so the file only fills gaps. A missing file is not an error.
func LoadEnvFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return // no .env file — that's fine
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), "`\"'")
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}
