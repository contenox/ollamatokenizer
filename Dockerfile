FROM golang:1.24-alpine AS builder
WORKDIR /app
RUN apk add --no-cache git gcc g++ musl-dev

COPY go.mod go.sum ./
RUN go mod download

COPY . .
WORKDIR /app/cmd/httpserver
RUN go build -o /app/tokenizer-api-server .

FROM alpine:3.19
RUN apk add --no-cache libstdc++
COPY --from=builder /app/tokenizer-api-server /tokenizer-api-server
ENTRYPOINT ["/tokenizer-api-server"]
