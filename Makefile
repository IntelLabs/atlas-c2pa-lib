# Variables
PROJECT_NAME = atlas-c2pa-lib
CARGO = cargo
CARGO_DOC_OPTS = --no-deps
CARGO_TEST_OPTS =
CARGO_CLIPPY_OPTS = -- -D warnings
CARGO_BENCH_OPTS =

# Default target
.DEFAULT_GOAL := help

# Build targets
all: build

build:
	@echo "Building $(PROJECT_NAME)..."
	$(CARGO) build

release:
	@echo "Building release version of $(PROJECT_NAME)..."
	$(CARGO) build --release

# Documentation targets
doc:
	@echo "Generating documentation..."
	$(CARGO) doc $(CARGO_DOC_OPTS)

doc-open: doc
	@echo "Opening documentation in browser..."
	$(CARGO) doc --open $(CARGO_DOC_OPTS)

# Test targets
test:
	@echo "Running tests..."
	$(CARGO) test $(CARGO_TEST_OPTS)

test-verbose:
	@echo "Running tests with verbose output..."
	$(CARGO) test $(CARGO_TEST_OPTS) -- --nocapture

test-doc:
	@echo "Testing documentation examples..."
	$(CARGO) test --doc

# Code quality targets
fmt:
	@echo "Formatting code..."
	$(CARGO) fmt

fmt-check:
	@echo "Checking code format..."
	$(CARGO) fmt -- --check

lint:
	@echo "Linting code with Clippy..."
	$(CARGO) clippy $(CARGO_CLIPPY_OPTS)

# Performance targets
bench:
	@echo "Running benchmarks..."
	$(CARGO) bench $(CARGO_BENCH_OPTS)

# Run and clean targets
run:
	@echo "Running $(PROJECT_NAME)..."
	$(CARGO) run

clean:
	@echo "Cleaning $(PROJECT_NAME)..."
	$(CARGO) clean

# CI and verification targets
check: fmt-check lint test
	@echo "All checks passed!"

verify: check doc test-doc bench
	@echo "All verification steps completed successfully!"

# Publishing targets
publish-dry-run:
	@echo "Performing a dry run of cargo publish..."
	$(CARGO) publish --dry-run

publish:
	@echo "Publishing $(PROJECT_NAME)..."
	$(CARGO) publish

# Help target
help:
	@echo "Makefile commands for $(PROJECT_NAME):"
	@echo ""
	@echo "Build Commands:"
	@echo "  make              - Build the project (same as 'make build')"
	@echo "  make build        - Build the project in debug mode"
	@echo "  make release      - Build the project in release mode"
	@echo ""
	@echo "Documentation:"
	@echo "  make doc          - Generate documentation"
	@echo "  make doc-open     - Generate documentation and open in browser"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run the tests"
	@echo "  make test-verbose - Run the tests with verbose output"
	@echo "  make test-doc     - Test documentation examples"
	@echo "  make bench        - Run benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt          - Format the code"
	@echo "  make fmt-check    - Check code formatting without changing files"
	@echo "  make lint         - Lint the code using Clippy"
	@echo ""
	@echo "Utility:"
	@echo "  make run          - Run the project"
	@echo "  make clean        - Clean the project"
	@echo ""
	@echo "CI and Verification:"
	@echo "  make check        - Run format check, lint, and tests (for CI)"
	@echo "  make verify       - Full verification including docs and benchmarks"
	@echo ""
	@echo "Publishing:"
	@echo "  make publish-dry-run - Test the publishing process"
	@echo "  make publish      - Publish the crate to crates.io"

# Phony targets
.PHONY: all build release doc doc-open test test-verbose test-doc fmt fmt-check lint bench run clean check verify publish-dry-run publish help
