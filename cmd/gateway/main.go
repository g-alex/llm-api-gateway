// cmd/gateway — entry point: config, account pool, upstream client,
// HTTP server and graceful shutdown.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/g-alex/llm-api-gateway/internal/api"
	"github.com/g-alex/llm-api-gateway/internal/config"
	"github.com/g-alex/llm-api-gateway/internal/zai"
)

// version is set at build time via -ldflags "-X main.version=…".
var version = "dev"

func main() {
	var tokensFile string
	flag.StringVar(&tokensFile, "zai-tokens", "", "Z.AI token file, one JWT per line (overrides ZAI_TOKENS_FILE)")
	flag.Parse()

	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "configuration error: %v\n", err)
		os.Exit(1)
	}
	if tokensFile != "" {
		cfg.ZAITokensFile = tokensFile
	}

	// Account pool (optional): one JWT per line, validated at startup.
	var pool *zai.Pool
	if cfg.UseAccountPool {
		pool, err = zai.LoadPool(cfg.ZAITokensFile)
		if err != nil {
			log.Printf("[Startup] account pool: %v — continuing with single-token/guest mode", err)
		}
	}

	client := zai.New(zai.DefaultBaseURL, cfg.LogLevel, cfg.ZAIToken, pool)
	client.PersistHistory = cfg.PersistHistory

	// Warm the model catalog in the background.
	go client.StartModelRefresher()

	srv := api.New(cfg, client)

	httpServer := &http.Server{
		Addr:              cfg.Host + ":" + cfg.Port,
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 30 * time.Second,
	}

	go func() {
		printBanner(cfg)
		log.Printf("[Startup] listening on %s", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	// Graceful shutdown on SIGINT/SIGTERM.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Println("[Shutdown] signal received, shutting down…")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("[Shutdown] forced: %v", err)
	}
	log.Println("[Shutdown] done")
}

func printBanner(cfg *config.Config) {
	cyan, green, reset := "\033[36m", "\033[32m", "\033[0m"
	mode := "guest"
	if cfg.ZAIToken != "" {
		mode = "single-token"
	}
	if cfg.UseAccountPool {
		mode = "account-pool"
	}
	fmt.Println()
	fmt.Println(cyan + "  llm-api-gateway — multi-protocol LLM gateway" + reset)
	fmt.Println(green + "  mode:" + reset + " " + mode + cyan + "  claude-desktop:" + reset + " " + fmt.Sprint(cfg.ClaudeDesktop) + cyan + "  version:" + reset + " " + version)
	fmt.Println()
}
