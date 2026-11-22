package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// withEnv sets env vars for the test and restores the old values after.
func withEnv(t *testing.T, kv map[string]string) {
	t.Helper()
	for k, v := range kv {
		old, had := os.LookupEnv(k)
		os.Setenv(k, v)
		t.Cleanup(func() {
			if had {
				os.Setenv(k, old)
			} else {
				os.Unsetenv(k)
			}
		})
	}
}

func TestLoadDefaults(t *testing.T) {
	withEnv(t, map[string]string{
		"AUTH_TOKEN": "test-token", "ZAI_TOKEN": "", "HOST": "", "PORT": "",
		"ZAI_TOKENS_FILE": "", "USE_ACCOUNT_POOL": "", "LOG_LEVEL": "",
		"TIMEOUT": "", "CLAUDE_DESKTOP": "", "ANTHROPIC_ZAI_MIN_MS": "",
	})
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Host != "0.0.0.0" || cfg.Port != "5082" {
		t.Errorf("defaults: got host=%q port=%q", cfg.Host, cfg.Port)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel: got %q, want info", cfg.LogLevel)
	}
	if cfg.Timeout != 300*time.Second {
		t.Errorf("Timeout: got %v, want 300s", cfg.Timeout)
	}
	if cfg.UseAccountPool || cfg.ClaudeDesktop {
		t.Error("boolean flags should default to false")
	}
	if cfg.AnthropicMinMS != 1500 {
		t.Errorf("AnthropicMinMS: got %d, want 1500", cfg.AnthropicMinMS)
	}
}

func TestLoadRequiresAuthToken(t *testing.T) {
	withEnv(t, map[string]string{"AUTH_TOKEN": ""})
	if _, err := Load(); err == nil {
		t.Fatal("expected error when AUTH_TOKEN is empty")
	}
}

func TestValidateRejectsBadValues(t *testing.T) {
	base := func() *Config {
		return &Config{AuthToken: "t", Port: "5082", LogLevel: "info", Timeout: time.Second}
	}
	if err := base().Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	cases := map[string]func(*Config){
		"empty auth token": func(c *Config) { c.AuthToken = "  " },
		"non-numeric port": func(c *Config) { c.Port = "http" },
		"bad log level":    func(c *Config) { c.LogLevel = "verbose" },
		"zero timeout":     func(c *Config) { c.Timeout = 0 },
	}
	for name, mutate := range cases {
		c := base()
		mutate(c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
}

func TestLoadEnvFileFillsGapsOnly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".env")
	content := "# comment\n\nPORT=9999\nAUTH_TOKEN=\"from-file\"\nBADLINE\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	withEnv(t, map[string]string{"PORT": "7777", "AUTH_TOKEN": ""})

	LoadEnvFile(path)

	if got := os.Getenv("PORT"); got != "7777" {
		t.Errorf("real env should win: PORT=%q, want 7777", got)
	}
	if got := os.Getenv("AUTH_TOKEN"); got != "from-file" {
		t.Errorf("gap should be filled from file: AUTH_TOKEN=%q, want from-file", got)
	}
}

func TestLoadEnvFileMissingIsFine(t *testing.T) {
	LoadEnvFile(filepath.Join(t.TempDir(), "does-not-exist.env"))
}
