#!/usr/bin/env python3
"""
Test script to verify the package works correctly before publishing.
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_import():
    """Test that the package imports correctly."""
    try:
        import pyrtex
        print(f"✅ Package import successful!")
        print(f"   Version: {pyrtex.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_core_classes():
    """Test that core classes can be imported."""
    try:
        from pyrtex import Job, InfrastructureConfig, GenerationConfig
        print("✅ Core classes import successful!")
        return True
    except ImportError as e:
        print(f"❌ Core classes import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without GCP calls."""
    try:
        from pydantic import BaseModel
        from pyrtex import Job
        
        class TestOutput(BaseModel):
            result: str
        
        job = Job[TestOutput](
            model="gemini-2.0-flash-lite-001",
            output_schema=TestOutput,
            prompt_template="Test: {{ value }}",
            simulation_mode=True
        )
        
        print("✅ Basic job creation successful!")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing PyRTex package...")
    print()
    
    tests = [
        test_basic_import,
        test_core_classes,
        test_basic_functionality,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Package is ready for publishing.")
        return 0
    else:
        print("💥 Some tests failed. Please fix before publishing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
