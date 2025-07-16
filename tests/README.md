# Testing Guide for Pyrtex

This directory contains comprehensive tests for the Pyrtex library.

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and test utilities
├── unit/                # Fast unit tests (no network calls)
│   ├── test_client.py   # Tests for Job class functionality
│   └── test_payload.py  # Tests for JSONL payload generation
└── integration/         # Integration tests (some may require GCP)
    └── test_full_run.py # End-to-end tests and real-world scenarios
```

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast, no GCP required)
python test_runner.py unit

# Run integration tests (excluding costly e2e tests)
python test_runner.py integration

# Run all tests with coverage
python test_runner.py coverage

# Format code
python test_runner.py format

# Run linting
python test_runner.py lint
```

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Fast execution** (< 1 second per test)
- **No network calls** - all GCP services are mocked
- **100% coverage goal** - these tests should cover all code paths
- **No authentication required**

#### Integration Tests (`tests/integration/`)
- **Dry run tests** - Test payload generation without submission
- **Simulation mode tests** - Test dummy data generation
- **Real GCP tests** - Marked with `@pytest.mark.incurs_costs` and skipped by default

### Test Markers

- `@pytest.mark.e2e` - End-to-end tests that may require GCP setup
- `@pytest.mark.incurs_costs` - Tests that will incur small GCP costs (skipped by default)

### Running Specific Test Types

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests (excluding costly ones)
pytest tests/integration/ -v -m "not incurs_costs"

# Run only e2e tests (including costly ones) - requires GCP setup
pytest tests/integration/ -v -m "e2e and incurs_costs"

# Run tests with coverage
pytest tests/ --cov=src/pyrtex --cov-report=html -m "not incurs_costs"
```

## Test Features

### 1. Comprehensive Mocking
- All GCP services (Storage, BigQuery, Vertex AI) are mocked in unit tests
- No real network calls or authentication required
- Fast execution suitable for CI/CD

### 2. Simulation Mode Testing
- Tests the `simulation_mode=True` feature
- Verifies dummy data generation works correctly
- Ensures schema compliance of generated dummy data

### 3. Dry Run Testing
- Tests the `dry_run=True` feature
- Verifies JSONL payload generation
- Ensures proper template rendering and file handling

### 4. Real-World Scenarios
- End-to-end tests that can run against real GCP (when enabled)
- Error handling tests
- Batch processing tests
- File upload tests

### 5. Schema Validation
- Tests with simple and complex Pydantic schemas
- Validates proper JSON schema generation
- Tests error handling for invalid schemas

## Development Workflow

### Before Committing
```bash
# Format code
python test_runner.py format

# Run linting
python test_runner.py lint

# Run type checking
python test_runner.py type

# Run all tests (excluding costly e2e)
python test_runner.py all
```

### Before Releasing
```bash
# Run full test suite including coverage
python test_runner.py coverage

# Manually run costly e2e tests if GCP is set up
python test_runner.py e2e
```

## GCP Setup for E2E Tests

To run the full end-to-end tests that actually hit GCP:

1. **Authentication**: Run `gcloud auth application-default login`
2. **Project Setup**: Ensure you have a GCP project with Vertex AI API enabled
3. **Environment Variables**: Set `GOOGLE_PROJECT_ID` if needed
4. **Run Tests**: `pytest tests/integration/ -v -m "e2e and incurs_costs"`

⚠️ **Warning**: E2E tests marked with `incurs_costs` will create small charges on your GCP account.

## Coverage Goals

We aim for **95%+ coverage** on unit tests. The coverage report will show:
- Line coverage for all source files
- Missing coverage areas
- HTML report in `htmlcov/` directory

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. **GCP Authentication Errors**: Run `gcloud auth application-default login`

3. **Mock Failures**: Check that you're using the `mock_gcp_clients` fixture

4. **Coverage Too Low**: Add more unit tests to cover missing code paths

### Test-Specific Issues

- **Simulation Mode**: Ensure `simulation_mode=True` is set in the Job constructor
- **Dry Run**: Check that `dry_run=True` is passed to the `submit()` method
- **File Tests**: Temporary files should be cleaned up properly

## Contributing

When adding new features:

1. **Add unit tests** for all new functionality
2. **Add integration tests** for user-facing features
3. **Maintain coverage** above 95%
4. **Follow existing patterns** in test structure and naming
5. **Use appropriate fixtures** from `conftest.py`

For questions about testing, refer to the test files themselves for examples of proper usage.
