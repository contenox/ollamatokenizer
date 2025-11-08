package ollamatokenizer_test

import (
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/contenox/ollamatokenizer"
	"github.com/stretchr/testify/require"
)

func TestTokenize(t *testing.T) {
	defer quiet()()

	// Set up a tokenizer with fast timeout (optional)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("tiny"), // makes invalid models fallback
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer: %v", err)
	}

	testCases := []struct {
		model   string
		input   string
		wantErr bool
	}{
		{
			model:   "tiny",
			input:   "Hello world!",
			wantErr: false,
		},
		{
			model:   "invalid-model",
			input:   "Test input",
			wantErr: true,
		},
		{
			model:   "tiny",
			input:   "",
			wantErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.model+"/"+tc.input, func(t *testing.T) {
			tokens, err := tokenizer.Tokenize(tc.model, tc.input)
			if tc.wantErr {
				if err == nil {
					t.Error("expected error but got none")
				} else {
					t.Logf("got expected error: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(tokens) == 0 && tc.input != "" {
				t.Error("expected non-empty tokens for non-empty input")
			}
			t.Logf("Tokens (%d): %+v", len(tokens), tokens)
		})
	}
}

func TestCountTokens(t *testing.T) {
	defer quiet()()

	httpClient := &http.Client{Timeout: 30 * time.Second}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("tiny"),
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer: %v", err)
	}

	testCases := []struct {
		model     string
		input     string
		wantErr   bool
		wantCount int
	}{
		{
			model:     "tiny",
			input:     "Hello world!",
			wantErr:   false,
			wantCount: 3,
		},
		{
			model:     "tiny",
			input:     "",
			wantErr:   false,
			wantCount: 0, // Empty input should return 0 tokens
		},
		{
			model:     "invalid-model",
			input:     "Test input",
			wantErr:   true,
			wantCount: 0, // Error expected, no count
		},
	}

	for _, tc := range testCases {
		t.Run(tc.model+"/"+tc.input, func(t *testing.T) {
			count, err := tokenizer.CountTokens(tc.model, tc.input)
			if tc.wantErr {
				if err == nil {
					t.Error("expected error but got none")
				} else {
					t.Logf("got expected error: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if count != tc.wantCount {
				t.Errorf("expected token count %d, got %d", tc.wantCount, count)
			}
		})
	}
}

func TestAvailableModels(t *testing.T) {
	defer quiet()()
	httpClient := &http.Client{Timeout: 30 * time.Second}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("tiny"),
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer: %v", err)
	}

	availableModels := tokenizer.AvailableModels()
	if len(availableModels) == 0 {
		t.Fatal("expected available models, got none")
	}

	// Check if the "tiny" model is in the list of available models
	modelFound := slices.Contains(availableModels, "tiny")

	if !modelFound {
		t.Errorf("expected 'tiny' to be in the list of available models")
	}
}

func TestOptimalTokenizerModel(t *testing.T) {
	defer quiet()()
	httpClient := &http.Client{Timeout: 30 * time.Second}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("tiny"),
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer: %v", err)
	}

	testCases := []struct {
		basedOnModel  string
		expectedModel string
	}{
		{
			basedOnModel:  "granite-embedding-30m",
			expectedModel: "granite-embedding-30m",
		},
		{
			basedOnModel:  "llama3.2",
			expectedModel: "llama-3.2", // "llama3.2" should map to "llama-3.2"
		},
		{
			basedOnModel:  "phi3",
			expectedModel: "phi-3", // "phi3" should map to "phi-3"
		},
		{
			basedOnModel:  "nonexistent-model",
			expectedModel: "tiny",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.basedOnModel, func(t *testing.T) {
			model, err := tokenizer.OptimalTokenizerModel(tc.basedOnModel)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if model != tc.expectedModel {
				t.Errorf("expected model %s, got %s", tc.expectedModel, model)
			}
		})
	}
}

func TestPreloadOption(t *testing.T) {
	defer quiet()()
	httpClient := &http.Client{Timeout: 30 * time.Second}

	// Test with a small model (e.g., "tiny") preloaded
	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithPreloadedModels("tiny"),
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer with preload: %v", err)
	}

	// Test that preloaded model can be used for tokenization without crashing
	_, err = tokenizer.Tokenize("tiny", "Hello, preloaded world!")
	if err != nil {
		t.Fatalf("tokenization failed with preloaded model: %v", err)
	}

	// Test that preloaded model does not cause any crashes during CountTokens
	count, err := tokenizer.CountTokens("tiny", "Hello, preloaded world!")
	if err != nil {
		t.Fatalf("counting tokens failed with preloaded model: %v", err)
	}
	if count == 0 {
		t.Fatal("expected non-zero token count after using preloaded model")
	}

	// Ensure the model is loaded properly and doesn't crash with AvailableModels
	availableModels := tokenizer.AvailableModels()
	modelFound := slices.Contains(availableModels, "tiny")
	if !modelFound {
		t.Fatalf("expected 'tiny' to be in the list of available models after preload")
	}
}

func TestConcurrentTokenization(t *testing.T) {
	defer quiet()()

	httpClient := &http.Client{Timeout: 30 * time.Second}

	// Preload models to avoid network delays during concurrent testing
	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithPreloadedModels("tiny", "granite-embedding-30m"),
		ollamatokenizer.TokenizerWithFallbackModel("granite-embedding-30m"),
	)
	if err != nil {
		t.Fatalf("failed to initialize tokenizer: %v", err)
	}

	const numGoroutines = 500
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := range numGoroutines {
		go func(id int) {
			defer wg.Done()
			model := "granite-embedding-30m"
			if id%2 == 0 {
				model = "tiny"
			}
			prompt := fmt.Sprintf("Goroutine %d: 🚀 Concurrent test! %s", id, strings.Repeat("na ", id%5))

			// Test both Tokenize and CountTokens
			tokens, err := tokenizer.Tokenize(model, prompt)
			if err != nil {
				t.Errorf("goroutine %d: Tokenize(%s) failed: %v", id, model, err)
				return
			}
			if len(tokens) == 0 {
				t.Errorf("goroutine %d: got 0 tokens for non-empty input", id)
			}

			count, err := tokenizer.CountTokens(model, prompt)
			if err != nil {
				t.Errorf("goroutine %d: CountTokens(%s) failed: %v", id, model, err)
				return
			}
			if count != len(tokens) {
				t.Errorf("goroutine %d: CountTokens=%d ≠ len(Tokenize)=%d", id, count, len(tokens))
			}
		}(i)
	}

	wg.Wait()
}

func TestTokenCount(t *testing.T) {
	httpClient := &http.Client{Timeout: 30 * time.Second}
	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("granite-embedding-30m"),
	)
	require.NoError(t, err)

	testText := "This is a benchmark test string for measuring embedding performance"
	tokenCount, err := tokenizer.CountTokens("granite-embedding-30m", testText)
	require.NoError(t, err)

	t.Logf("Token count for benchmark text: %d", tokenCount)
	require.Equal(t, 13, tokenCount)
}

func TestCountTokensLargeInput(t *testing.T) {
	defer quiet()()

	httpClient := &http.Client{Timeout: 30 * time.Second}

	tokenizer, err := ollamatokenizer.NewTokenizer(
		ollamatokenizer.TokenizerWithHTTPClient(httpClient),
		ollamatokenizer.TokenizerWithFallbackModel("tiny"),
	)
	require.NoError(t, err, "failed to initialize tokenizer")

	// Create a large input that definitely exceeds the 16KB limit
	// Using a repeating pattern to ensure we have enough data
	baseText := "This is a large text input that will be repeated to exceed the 16KB limit. "
	largeInput := strings.Repeat(baseText, 500) // Should be well over 16KB

	// Verify it's actually large enough
	require.Greater(t, len(largeInput), 16*1024, "test input should exceed 16KB limit")

	// Test that CountTokens doesn't error with large input
	count, err := tokenizer.CountTokens("tiny", largeInput)

	// We only care that it doesn't error
	require.NoError(t, err, "CountTokens should not error with large input")
	require.Greater(t, count, 0, "should have tokenized at least some tokens")

	t.Logf("Successfully tokenized large input (%d bytes) into %d tokens", len(largeInput), count)
}
