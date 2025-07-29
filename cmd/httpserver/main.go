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

func main() {
	modelsEnv := os.Getenv("TOKENIZER_MODELS") // e.g. "llama3=https://.../llama3.model,tiny=https://.../tiny.model"
	modelMap := make(map[string]string)
	for _, kv := range strings.Split(modelsEnv, ",") {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 2 {
			modelMap[parts[0]] = parts[1]
		}
	}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithModelMap(modelMap),
	)
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
		_ = json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	log.Println("Tokenizer HTTP server listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
