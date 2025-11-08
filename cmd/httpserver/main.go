package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/contenox/ollamatokenizer"
)

type tokenizeRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type tokenizeResponse struct {
	Tokens []int `json:"tokens"`
	Count  int   `json:"count"`
}

type countRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type countResponse struct {
	Count int `json:"count"`
}

func main() {
	addr := os.Getenv("ADDR")
	if addr == "" {
		addr = ":8080"
	}

	// Get fallback model (default to empty)
	fallbackModel := os.Getenv("FALLBACK_MODEL")

	// Get preload models
	preloadModels := strings.Split(os.Getenv("PRELOAD_MODELS"), ",")

	// Use default URLs if requested
	useDefaultURLs := os.Getenv("USE_DEFAULT_URLS") == "true"

	// Parse model URLs if provided
	var tokenizerOpts []ollamatokenizer.TokenizerOption

	// Only use custom models if USE_DEFAULT_URLS is not "true"
	if !useDefaultURLs {
		modelsEnv := os.Getenv("TOKENIZER_MODELS")
		modelMap := make(map[string]string)
		for _, kv := range strings.Split(modelsEnv, ",") {
			parts := strings.SplitN(kv, "=", 2)
			if len(parts) == 2 {
				modelMap[parts[0]] = parts[1]
			}
		}
		tokenizerOpts = append(tokenizerOpts, ollamatokenizer.TokenizerWithModelMap(modelMap))
	}

	// Add fallback model option if specified
	if fallbackModel != "" {
		tokenizerOpts = append(tokenizerOpts, ollamatokenizer.TokenizerWithFallbackModel(fallbackModel))
	}

	// Preload models if specified
	if len(preloadModels) > 0 && preloadModels[0] != "" {
		tokenizerOpts = append(tokenizerOpts, ollamatokenizer.TokenizerWithPreloadedModels(preloadModels...))
	}

	tokenizer, err := ollamatokenizer.NewTokenizer(tokenizerOpts...)
	if err != nil {
		log.Fatalf("Failed to init tokenizer: %v", err)
	}

	http.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var req tokenizeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}

		tokens, err := tokenizer.Tokenize(req.Model, req.Prompt)
		if err != nil {
			http.Error(w, "tokenize failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		resp := tokenizeResponse{Tokens: tokens, Count: len(tokens)}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	// Add the /count endpoint for direct token counting
	http.HandleFunc("/count", func(w http.ResponseWriter, r *http.Request) {
		var req countRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}

		count, err := tokenizer.CountTokens(req.Model, req.Prompt)
		if err != nil {
			http.Error(w, "count tokens failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		resp := countResponse{Count: count}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	log.Println("Tokenizer HTTP server listening on ", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
