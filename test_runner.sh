#!/bin/bash
# test_runner.sh - Enhanced test runner for pyrtex with linting support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Function to run flake8 linting
run_flake8() {
    print_header "Running Flake8 Linting"
    
    if ! command -v flake8 &> /dev/null; then
        print_error "flake8 not found. Install with: pip install -e .[dev]"
        return 1
    fi
    
    print_status "Checking code style with flake8..."
    
    # Run flake8 on source and test directories
    if flake8 src/ tests/ examples/; then
        print_success "Flake8 checks passed!"
        return 0
    else
        print_error "Flake8 found style issues"
        return 1
    fi
}

# Function to run tests
run_tests() {
    local test_type=$1
    local project_id=$2
    local verbose=$3
    
    # Set verbose flag for pytest
    local pytest_verbose=""
    if [[ "$verbose" == "true" ]]; then
        pytest_verbose="-v"
    fi
    
    if [[ "$test_type" == "real" ]]; then
        print_header "Running Real GCP Tests"
        print_status "Project ID: $project_id"
        print_warning "This will incur small GCP costs"
        
        # Set environment variable and run only real tests
        export GOOGLE_PROJECT_ID="$project_id"
        pytest tests/integration/ -m "incurs_costs" $pytest_verbose --cov=src --cov-report=xml:reports/coverage.xml
    elif [[ "$test_type" == "unit" ]]; then
        print_header "Running Unit Tests Only"
        pytest tests/unit/ $pytest_verbose --cov=src --cov-report=xml:reports/coverage.xml --cov-fail-under=100
    elif [[ "$test_type" == "integration" ]]; then
        print_header "Running Integration Tests (Mocked)"
        pytest tests/integration/ -m "not incurs_costs" $pytest_verbose --cov=src --cov-report=xml:reports/coverage.xml
    else
        print_header "Running All Mocked Tests"
        print_status "Skipping tests that incur GCP costs"
        
        # Run all tests except those that incur costs
        pytest -m "not incurs_costs" $pytest_verbose --cov=src --cov-report=xml:reports/coverage.xml
    fi
}

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test Runner Options:"
    echo "  --unit                 Run unit tests only"
    echo "  --integration          Run integration tests (mocked)"
    echo "  --real                 Run tests that incur GCP costs"
    echo "  --project-id <id>      Set GCP project ID for real tests"
    echo ""
    echo "Linting Options:"
    echo "  --flake                Run flake8 linting only"
    echo "  --flake-fix            Run flake8 and attempt to fix issues"
    echo ""
    echo "General Options:"
    echo "  -v, --verbose          Enable verbose output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all mocked tests"
    echo "  $0 --unit                             # Run unit tests only"
    echo "  $0 --integration                      # Run integration tests (mocked)"
    echo "  $0 --real --project-id my-project     # Run real tests with project ID"
    echo "  $0 --flake                            # Run linting only"
    echo "  $0 --flake-fix                        # Run linting with auto-fix"
    echo "  $0 --flake --unit                     # Run linting then unit tests"
}

# Default values
RUN_TESTS=true
TEST_TYPE="all"
RUN_FLAKE=false
FLAKE_FIX=false
VERBOSE=false
PROJECT_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --real)
            TEST_TYPE="real"
            shift
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --flake)
            RUN_FLAKE=true
            shift
            ;;
        --flake-fix)
            RUN_FLAKE=true
            FLAKE_FIX=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create reports directory if it doesn't exist
mkdir -p reports

# Validate arguments
if [[ "$TEST_TYPE" == "real" ]]; then
    if [[ -z "$PROJECT_ID" ]]; then
        print_error "Project ID is required when running real tests. Use --project-id <id>"
        exit 1
    fi
fi

# If only flake8 requested and no test flags, don't run tests
if [[ "$RUN_FLAKE" == "true" ]] && [[ "$TEST_TYPE" == "all" ]] && [[ $# -eq 0 ]]; then
    # Check if --flake was the only argument (or with --flake-fix)
    RUN_TESTS=false
fi

exit_code=0

# Run flake8 if requested
if [[ "$RUN_FLAKE" == "true" ]]; then
    if [[ "$FLAKE_FIX" == "true" ]]; then
        print_header "Running Flake8 with Auto-fix"
        print_status "Attempting to fix issues with black and isort..."
        
        # Run black for code formatting
        if command -v black &> /dev/null; then
            print_status "Running black formatter..."
            black src/ tests/ examples/
        else
            print_warning "black not found, skipping auto-formatting"
        fi
        
        # Run isort for import sorting
        if command -v isort &> /dev/null; then
            print_status "Running isort for imports..."
            isort src/ tests/ examples/
        else
            print_warning "isort not found, skipping import sorting"
        fi
        
        # Run flake8 after fixes
        if ! run_flake8; then
            exit_code=1
        fi
    else
        if ! run_flake8; then
            exit_code=1
        fi
    fi
fi

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    if ! run_tests "$TEST_TYPE" "$PROJECT_ID" "$VERBOSE"; then
        exit_code=1
    fi
fi

# Final status
if [[ $exit_code -eq 0 ]]; then
    print_success "All checks passed!"
else
    print_error "Some checks failed!"
fi

exit $exit_code
