#!/bin/bash
# test_runner.sh - Simple test runner for pyrtex

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

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Default values
RUN_REAL_TESTS=false
VERBOSE=false
PROJECT_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --real)
            RUN_REAL_TESTS=true
            shift
            ;;
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --real                 Run tests that incur GCP costs"
            echo "  --project-id <id>      Set GCP project ID for real tests"
            echo "  -v, --verbose          Enable verbose output"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run mocked tests only"
            echo "  $0 --real --project-id nakuptady     # Run real tests with project ID"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set verbose flag for pytest
PYTEST_VERBOSE=""
if [[ "$VERBOSE" == "true" ]]; then
    PYTEST_VERBOSE="-v"
fi

# Validate arguments
if [[ "$RUN_REAL_TESTS" == "true" ]]; then
    if [[ -z "$PROJECT_ID" ]]; then
        print_error "Project ID is required when running real tests. Use --project-id <id>"
        exit 1
    fi
    print_header "Running Real GCP Tests"
    print_status "Project ID: $PROJECT_ID"
    print_status "This will incur small GCP costs"
    
    # Set environment variable and run only real tests
    export GOOGLE_PROJECT_ID="$PROJECT_ID"
    exec pytest tests/integration/ -m "incurs_costs" $PYTEST_VERBOSE
else
    print_header "Running Mocked Tests"
    print_status "Skipping tests that incur GCP costs"
    
    # Run all tests except those that incur costs
    exec pytest -m "not incurs_costs" $PYTEST_VERBOSE
fi
