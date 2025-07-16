#!/bin/bash
# comprehensive_test_runner.sh - Full-featured test runner for pyrtex

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
TESTS_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$PROJECT_ROOT/reports"

# Default values
RUN_LINTING=true
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=false
RUN_E2E_TESTS=false
RUN_COVERAGE=true
VERBOSE=false
FAIL_FAST=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

A comprehensive test runner for the pyrtex library.

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -f, --fail-fast         Stop on first failure
    
    --no-lint              Skip linting (black, flake8, mypy)
    --no-unit              Skip unit tests
    --no-coverage          Skip coverage reporting
    --integration          Run integration tests
    --e2e                  Run end-to-end tests (requires GCP auth)
    --all                  Run all tests including e2e
    
    --lint-only            Only run linting, skip tests
    --unit-only            Only run unit tests
    
    --clean                Clean all generated files and reports
    --check                Check if all tools are available

EXAMPLES:
    $0                     # Run default tests (lint + unit + coverage)
    $0 --integration       # Run integration tests too
    $0 --all              # Run everything including e2e tests
    $0 --lint-only        # Only run linting
    $0 --unit-only -v     # Only unit tests with verbose output
    $0 --clean            # Clean all reports and cache files

EOF
}

# Function to check if tools are available
check_tools() {
    print_header "Checking Required Tools"
    
    local missing_tools=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_tools+=("pip3")
    fi
    
    # Check Python packages
    local python_packages=("pytest" "black" "flake8" "mypy")
    for package in "${python_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_tools+=("python3-$package")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_status "Install missing tools with: pip install -e .[dev]"
        exit 1
    fi
    
    print_success "All required tools are available"
}

# Function to clean generated files
clean_files() {
    print_header "Cleaning Generated Files"
    
    # Remove Python cache
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove coverage files
    rm -f "$PROJECT_ROOT/.coverage"
    rm -rf "$PROJECT_ROOT/htmlcov"
    rm -f "$PROJECT_ROOT/coverage.xml"
    
    # Remove pytest cache
    rm -rf "$PROJECT_ROOT/.pytest_cache"
    
    # Remove mypy cache
    rm -rf "$PROJECT_ROOT/.mypy_cache"
    
    # Remove reports directory
    rm -rf "$REPORTS_DIR"
    
    # Remove build artifacts
    rm -rf "$PROJECT_ROOT/build"
    rm -rf "$PROJECT_ROOT/dist"
    rm -rf "$PROJECT_ROOT"/*.egg-info
    
    print_success "Cleaned all generated files"
}

# Function to run linting
run_linting() {
    if [ "$RUN_LINTING" = false ]; then
        return 0
    fi
    
    print_header "Running Code Linting"
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    local lint_failed=false
    
    # Run Black
    print_status "Running Black (code formatting)..."
    if [ "$VERBOSE" = true ]; then
        if ! black --check --diff "$SRC_DIR" "$TESTS_DIR"; then
            print_error "Black formatting check failed"
            lint_failed=true
        fi
    else
        if ! black --check --diff "$SRC_DIR" "$TESTS_DIR" > "$REPORTS_DIR/black.log" 2>&1; then
            print_error "Black formatting check failed (see $REPORTS_DIR/black.log)"
            lint_failed=true
        fi
    fi
    
    # Run Flake8
    print_status "Running Flake8 (style checking)..."
    if [ "$VERBOSE" = true ]; then
        if ! flake8 "$SRC_DIR" "$TESTS_DIR"; then
            print_error "Flake8 style check failed"
            lint_failed=true
        fi
    else
        if ! flake8 "$SRC_DIR" "$TESTS_DIR" > "$REPORTS_DIR/flake8.log" 2>&1; then
            print_error "Flake8 style check failed (see $REPORTS_DIR/flake8.log)"
            lint_failed=true
        fi
    fi
    
    # Run MyPy
    print_status "Running MyPy (type checking)..."
    if [ "$VERBOSE" = true ]; then
        if ! mypy "$SRC_DIR"; then
            print_error "MyPy type check failed"
            lint_failed=true
        fi
    else
        if ! mypy "$SRC_DIR" > "$REPORTS_DIR/mypy.log" 2>&1; then
            print_error "MyPy type check failed (see $REPORTS_DIR/mypy.log)"
            lint_failed=true
        fi
    fi
    
    if [ "$lint_failed" = true ]; then
        print_error "Linting failed!"
        if [ "$FAIL_FAST" = true ]; then
            exit 1
        fi
        return 1
    else
        print_success "All linting checks passed!"
        return 0
    fi
}

# Function to run unit tests
run_unit_tests() {
    if [ "$RUN_UNIT_TESTS" = false ]; then
        return 0
    fi
    
    print_header "Running Unit Tests"
    
    local pytest_args=()
    
    # Add basic arguments
    pytest_args+=("$TESTS_DIR/unit")
    
    # Add verbosity
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    # Add fail fast
    if [ "$FAIL_FAST" = true ]; then
        pytest_args+=("-x")
    fi
    
    # Add coverage if enabled
    if [ "$RUN_COVERAGE" = true ]; then
        pytest_args+=("--cov=src/pyrtex")
        pytest_args+=("--cov-report=term-missing")
        pytest_args+=("--cov-report=html:$REPORTS_DIR/coverage")
        pytest_args+=("--cov-report=xml:$REPORTS_DIR/coverage.xml")
    fi
    
    # Add markers to exclude e2e tests
    pytest_args+=("-m" "not e2e and not incurs_costs")
    
    print_status "Running: pytest ${pytest_args[*]}"
    
    if ! python3 -m pytest "${pytest_args[@]}"; then
        print_error "Unit tests failed!"
        return 1
    fi
    
    print_success "Unit tests passed!"
    return 0
}

# Function to run integration tests
run_integration_tests() {
    if [ "$RUN_INTEGRATION_TESTS" = false ]; then
        return 0
    fi
    
    print_header "Running Integration Tests"
    
    local pytest_args=()
    pytest_args+=("$TESTS_DIR/integration")
    
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    if [ "$FAIL_FAST" = true ]; then
        pytest_args+=("-x")
    fi
    
    # Exclude e2e tests unless explicitly enabled
    if [ "$RUN_E2E_TESTS" = false ]; then
        pytest_args+=("-m" "not e2e and not incurs_costs")
    fi
    
    print_status "Running: pytest ${pytest_args[*]}"
    
    if ! python3 -m pytest "${pytest_args[@]}"; then
        print_error "Integration tests failed!"
        return 1
    fi
    
    print_success "Integration tests passed!"
    return 0
}

# Function to run e2e tests
run_e2e_tests() {
    if [ "$RUN_E2E_TESTS" = false ]; then
        return 0
    fi
    
    print_header "Running End-to-End Tests"
    print_warning "E2E tests require GCP authentication and will incur small costs"
    
    # Check for GCP authentication
    if ! python3 -c "import google.auth; google.auth.default()" &> /dev/null; then
        print_error "GCP authentication not found. Run 'gcloud auth application-default login'"
        return 1
    fi
    
    local pytest_args=()
    pytest_args+=("$TESTS_DIR")
    pytest_args+=("-m" "e2e")
    
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    if [ "$FAIL_FAST" = true ]; then
        pytest_args+=("-x")
    fi
    
    print_status "Running: pytest ${pytest_args[*]}"
    
    if ! python3 -m pytest "${pytest_args[@]}"; then
        print_error "E2E tests failed!"
        return 1
    fi
    
    print_success "E2E tests passed!"
    return 0
}

# Main function
main() {
    local exit_code=0
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--fail-fast)
                FAIL_FAST=true
                shift
                ;;
            --no-lint)
                RUN_LINTING=false
                shift
                ;;
            --no-unit)
                RUN_UNIT_TESTS=false
                shift
                ;;
            --no-coverage)
                RUN_COVERAGE=false
                shift
                ;;
            --integration)
                RUN_INTEGRATION_TESTS=true
                shift
                ;;
            --e2e)
                RUN_E2E_TESTS=true
                shift
                ;;
            --all)
                RUN_INTEGRATION_TESTS=true
                RUN_E2E_TESTS=true
                shift
                ;;
            --lint-only)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=false
                RUN_E2E_TESTS=false
                RUN_COVERAGE=false
                shift
                ;;
            --unit-only)
                RUN_LINTING=false
                RUN_INTEGRATION_TESTS=false
                RUN_E2E_TESTS=false
                shift
                ;;
            --clean)
                clean_files
                exit 0
                ;;
            --check)
                check_tools
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Set up Python path
    export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
    
    # Check tools availability
    check_tools
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    print_header "Starting Test Suite for pyrtex"
    print_status "Project root: $PROJECT_ROOT"
    print_status "Reports directory: $REPORTS_DIR"
    
    # Run linting
    if ! run_linting; then
        exit_code=1
    fi
    
    # Run unit tests
    if ! run_unit_tests; then
        exit_code=1
    fi
    
    # Run integration tests
    if ! run_integration_tests; then
        exit_code=1
    fi
    
    # Run e2e tests
    if ! run_e2e_tests; then
        exit_code=1
    fi
    
    # Final summary
    print_header "Test Suite Complete"
    
    if [ $exit_code -eq 0 ]; then
        print_success "All tests passed! 🎉"
    else
        print_error "Some tests failed! 😞"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
