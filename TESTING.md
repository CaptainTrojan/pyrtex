# Testing Guide for pyrtex

This document describes how to run the comprehensive test suite for pyrtex.

## Quick Start

Run the default test suite (linting + unit tests + coverage):
```bash
./test_runner.sh
```

## Test Runner Options

### Basic Usage
```bash
./test_runner.sh [OPTIONS]
```

### Common Commands

**Run all tests including integration and e2e:**
```bash
./test_runner.sh --all
```

**Run only linting (black, flake8, mypy):**
```bash
./test_runner.sh --lint-only
```

**Run only unit tests with verbose output:**
```bash
./test_runner.sh --unit-only -v
```

**Run tests with integration tests:**
```bash
./test_runner.sh --integration
```

**Clean all generated files:**
```bash
./test_runner.sh --clean
```

**Check if all tools are available:**
```bash
./test_runner.sh --check
```

### Advanced Options

- `-v, --verbose`: Enable verbose output
- `-f, --fail-fast`: Stop on first failure
- `-j, --jobs N`: Number of parallel jobs
- `--no-lint`: Skip linting
- `--no-unit`: Skip unit tests
- `--no-coverage`: Skip coverage reporting

## Test Structure

```
tests/
├── conftest.py          # Test fixtures and configuration
├── unit/                # Fast unit tests (no network calls)
│   ├── test_client.py   # Client class tests
│   └── test_payload.py  # Payload generation tests
└── integration/         # Integration tests
    └── test_full_run.py # End-to-end workflow tests
```

## Test Categories

### Unit Tests
- Fast execution (< 1 second each)
- Mock all external dependencies
- Test individual components in isolation
- Achieve 90%+ code coverage

### Integration Tests
- Test component interactions
- Use simulation mode to avoid real API calls
- Test the full workflow without costs

### E2E Tests
- Require real GCP authentication
- Will incur small costs
- Test against real Vertex AI endpoints
- Marked with `@pytest.mark.e2e`

## Running Tests in CI

The test runner is designed to work well in CI environments:

```bash
# Install dependencies
pip install -e .[dev]

# Run tests suitable for CI
./test_runner.sh --no-e2e -v

# Run with coverage reporting
./test_runner.sh --coverage-only
```

## Linting Tools

The test runner includes several linting tools:

- **black**: Code formatting
- **flake8**: Style checking
- **mypy**: Type checking

Configure these tools in `pyproject.toml`.

## Coverage Reports

Coverage reports are generated in multiple formats:
- Terminal output (with --cov-report=term-missing)
- HTML report in `reports/coverage/index.html`
- XML report in `reports/coverage.xml`

## Tips

1. **Use simulation mode for fast testing**: When developing, use `simulation_mode=True` to avoid API calls
2. **Run linting frequently**: Use `./test_runner.sh --lint-only` during development
3. **Check coverage**: Aim for 90%+ coverage on new code
4. **Use dry run**: Test your prompt templates with `job.submit(dry_run=True)`

## Troubleshooting

### Missing Tools
If you get "Missing required tools" errors:
```bash
pip install -e .[dev]
```

### GCP Authentication for E2E Tests
```bash
gcloud auth application-default login
```

### Python Path Issues
The test runner automatically sets up the Python path, but if you encounter import errors:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```
