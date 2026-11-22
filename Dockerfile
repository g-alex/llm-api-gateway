# Build stage
FROM golang:1.26-alpine AS build
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/gateway ./cmd/gateway

# Run stage
FROM alpine:3.21
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 gateway
USER gateway
EXPOSE 5082
ENTRYPOINT ["/out/gateway"]
