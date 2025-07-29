module github.com/contenox/ollamatokenizer

go 1.24.1

// why? that's why: github.com/apache/arrow/go/arrow -> github.com/ollama/ollama/server
replace google.golang.org/genproto => google.golang.org/genproto v0.0.0-20250303144028-a0af3efb3deb

require (
	github.com/ollama/ollama v0.6.5
	github.com/stretchr/testify v1.10.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/kr/pretty v0.3.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/rogpeppe/go-internal v1.13.1 // indirect
	gopkg.in/check.v1 v1.0.0-20201130134442-10cb98267c6c // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
