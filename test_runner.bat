@echo off
REM test_runner.bat - Enhanced test runner for pyrtex with linting support

setlocal EnableDelayedExpansion

REM Default values
set "RUN_TESTS=true"
set "TEST_TYPE=all"
set "RUN_FLAKE=false"
set "FLAKE_FIX=false"
set "VERBOSE=false"
set "PROJECT_ID="
set "exit_code=0"
set "TEST_FLAGS_PROVIDED=false"

REM Function to print colored output (Windows doesn't support colors easily in batch, using echo)
goto :parse_args

:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_header
echo.
echo ========================================
echo  %~1
echo ========================================
echo.
goto :eof

REM Function to run flake8 linting
:run_flake8
call :print_header "Running Flake8 Linting"

REM Check if flake8 is available
where flake8 >nul 2>&1
if errorlevel 1 (
    call :print_error "flake8 not found. Install with: pip install -e .[dev]"
    exit /b 1
)

call :print_status "Checking code style with flake8..."

REM Run flake8 on source and test directories
flake8 src\ tests\ examples\
if errorlevel 1 (
    call :print_error "Flake8 found style issues"
    exit /b 1
) else (
    call :print_success "Flake8 checks passed!"
    exit /b 0
)

REM Function to run tests
:run_tests
set "test_type=%~1"
set "project_id=%~2"
set "verbose=%~3"

REM Set verbose flag for pytest
set "pytest_verbose="
if "%verbose%"=="true" set "pytest_verbose=-v"

if "%test_type%"=="real" (
    call :print_header "Running Real GCP Tests"
    call :print_status "Project ID: %project_id%"
    call :print_warning "This will incur small GCP costs"
    
    REM Set environment variable and run only real tests
    set "GOOGLE_PROJECT_ID=%project_id%"
    pytest tests\integration\ -m "incurs_costs" %pytest_verbose%
) else if "%test_type%"=="unit" (
    call :print_header "Running Unit Tests Only"
    pytest tests\unit\ %pytest_verbose% --cov=src --cov-report=xml:reports\coverage.xml --cov-fail-under=100
) else if "%test_type%"=="integration" (
    call :print_header "Running Integration Tests (Mocked)"
    pytest tests\integration\ -m "not incurs_costs" %pytest_verbose%
) else (
    call :print_header "Running All Mocked Tests"
    call :print_status "Skipping tests that incur GCP costs"
    
    REM Run all tests except those that incur costs
    pytest -m "not incurs_costs" %pytest_verbose% --cov=src --cov-report=xml:reports\coverage.xml
)

goto :eof

REM Function to show usage
:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Test Runner Options:
echo   --unit                 Run unit tests only
echo   --integration          Run integration tests (mocked)
echo   --real                 Run tests that incur GCP costs
echo   --project-id ^<id^>      Set GCP project ID for real tests
echo.
echo Linting Options:
echo   --flake                Run flake8 linting only
echo   --flake-fix            Run flake8 and attempt to fix issues
echo.
echo General Options:
echo   -v, --verbose          Enable verbose output
echo   -h, --help             Show this help message
echo.
echo Examples:
echo   %~nx0                                    # Run all mocked tests
echo   %~nx0 --unit                             # Run unit tests only
echo   %~nx0 --integration                      # Run integration tests (mocked)
echo   %~nx0 --real --project-id my-project     # Run real tests with project ID
echo   %~nx0 --flake                            # Run linting only
echo   %~nx0 --flake-fix                        # Run linting with auto-fix
echo   %~nx0 --flake --unit                     # Run linting then unit tests
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :validate_args

if "%~1"=="--unit" (
    set "TEST_TYPE=unit"
    set "TEST_FLAGS_PROVIDED=true"
    shift
    goto :parse_args
) else if "%~1"=="--integration" (
    set "TEST_TYPE=integration"
    set "TEST_FLAGS_PROVIDED=true"
    shift
    goto :parse_args
) else if "%~1"=="--real" (
    set "TEST_TYPE=real"
    set "TEST_FLAGS_PROVIDED=true"
    shift
    goto :parse_args
) else if "%~1"=="--project-id" (
    set "PROJECT_ID=%~2"
    shift
    shift
    goto :parse_args
) else if "%~1"=="--flake" (
    set "RUN_FLAKE=true"
    shift
    goto :parse_args
) else if "%~1"=="--flake-fix" (
    set "RUN_FLAKE=true"
    set "FLAKE_FIX=true"
    shift
    goto :parse_args
) else if "%~1"=="-v" (
    set "VERBOSE=true"
    shift
    goto :parse_args
) else if "%~1"=="--verbose" (
    set "VERBOSE=true"
    shift
    goto :parse_args
) else if "%~1"=="-h" (
    call :show_help
    exit /b 0
) else if "%~1"=="--help" (
    call :show_help
    exit /b 0
) else (
    call :print_error "Unknown option: %~1"
    call :show_help
    exit /b 1
)

:validate_args
REM Create reports directory if it doesn't exist
if not exist reports mkdir reports

REM Validate arguments
if "%TEST_TYPE%"=="real" (
    if "%PROJECT_ID%"=="" (
        call :print_error "Project ID is required when running real tests. Use --project-id <id>"
        exit /b 1
    )
)

REM If only flake8 requested and no test flags, don't run tests
if "%RUN_FLAKE%"=="true" if "%TEST_FLAGS_PROVIDED%"=="false" (
    set "RUN_TESTS=false"
)

REM Run flake8 if requested
if "%RUN_FLAKE%"=="true" (
    if "%FLAKE_FIX%"=="true" (
        call :print_header "Running Flake8 with Auto-fix"
        call :print_status "Attempting to fix issues with black and isort..."
        
        REM Run black for code formatting
        where black >nul 2>&1
        if not errorlevel 1 (
            call :print_status "Running black formatter..."
            black src\ tests\ examples\
        ) else (
            call :print_warning "black not found, skipping auto-formatting"
        )
        
        REM Run isort for import sorting
        where isort >nul 2>&1
        if not errorlevel 1 (
            call :print_status "Running isort for imports..."
            isort src\ tests\ examples\
        ) else (
            call :print_warning "isort not found, skipping import sorting"
        )
        
        REM Run flake8 after fixes
        call :run_flake8
        if errorlevel 1 set "exit_code=1"
    ) else (
        call :run_flake8
        if errorlevel 1 set "exit_code=1"
    )
)

REM Run tests if requested
if "%RUN_TESTS%"=="true" (
    call :run_tests "%TEST_TYPE%" "%PROJECT_ID%" "%VERBOSE%"
    if errorlevel 1 set "exit_code=1"
)

REM Final status
if "%exit_code%"=="0" (
    call :print_success "All checks passed!"
) else (
    call :print_error "Some checks failed!"
)

exit /b %exit_code%
