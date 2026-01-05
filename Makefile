.DEFAULT_GOAL := help
.PHONY: fmt clippy test build check run help

PROJECT_NAME := kernelport
PROJECT_VERSION := $(shell cargo pkgid -p kernelport-server 2>/dev/null | sed -E 's/.*@//')
ifeq ($(strip $(PROJECT_VERSION)),)
PROJECT_VERSION := unknown
endif

fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all

build:
	cargo build --all

check: fmt clippy test build

run:
	RUST_LOG=info cargo run -p kernelport-server

help:
	@printf "%s\n" \
		"$(PROJECT_NAME) $(PROJECT_VERSION)" \
		"Targets:" \
		"  fmt    - Run rustfmt" \
		"  clippy - Run clippy with warnings denied" \
		"  test   - Run tests" \
		"  build  - Build all crates" \
		"  check  - Run fmt, clippy, test, build" \
		"  run    - Run kernelport-server"
