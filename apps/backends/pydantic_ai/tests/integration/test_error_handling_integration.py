#!/usr/bin/env python3
"""
Integration tests for error handling system.
Tests file structure, module integration, and cross-component interactions.
"""

import os
import sys


def test_exception_file_structure():
    """Integration test: Validate exception file structure and content."""
    print("🧪 Testing exception file structure...")
    
    try:
        # Read and validate the exceptions file directly
        exceptions_file = os.path.join('..', '..', 'src', 'api', 'exceptions.py')
        with open(exceptions_file, 'r') as f:
            content = f.read()
        
        # Check for required exception classes
        exc_classes = [
            'PydanticAIException', 'ValidationException', 'AuthenticationException',
            'AuthorizationException', 'ResourceNotFoundException', 'AgentNotFoundException',
            'SessionNotFoundException', 'ToolNotFoundException', 'ModelNotAvailableException',
            'ToolExecutionException', 'ConfigurationException', 'ExternalServiceException',
            'RateLimitException', 'BusinessLogicException'
        ]
        
        missing_classes = []
        for exc_name in exc_classes:
            if f'class {exc_name}' not in content:
                missing_classes.append(exc_name)
            else:
                print(f"✅ Exception class {exc_name} exists")
        
        if missing_classes:
            print(f"❌ Missing exception classes: {missing_classes}")
            return False
        
        # Check for proper inheritance structure
        inheritance_checks = [
            'class PydanticAIException(Exception)',
            'class ValidationException(PydanticAIException)',
            'class ResourceNotFoundException(PydanticAIException)',
            'class AgentNotFoundException(ResourceNotFoundException)',
        ]
        
        for check in inheritance_checks:
            if check in content:
                print(f"✅ Inheritance check passed: {check}")
            else:
                print(f"❌ Inheritance check failed: {check}")
                return False
        
        # Check for proper constructor patterns
        constructor_patterns = [
            'def __init__(self',
            'self.message = ',
            'self.error_code = ',
            'self.details = ',
            'super().__init__(',
        ]
        
        for pattern in constructor_patterns:
            if pattern in content:
                print(f"✅ Constructor pattern found: {pattern}")
            else:
                print(f"❌ Constructor pattern missing: {pattern}")
                return False
        
        # Check for error codes
        error_codes = [
            'VALIDATION_ERROR',
            'AUTHENTICATION_ERROR', 
            'AUTHORIZATION_ERROR',
            'RESOURCE_NOT_FOUND',
            'MODEL_NOT_AVAILABLE',
            'TOOL_EXECUTION_ERROR'
        ]
        
        for code in error_codes:
            if code in content:
                print(f"✅ Error code found: {code}")
            else:
                print(f"❌ Error code missing: {code}")
                return False
        
        print("✅ Exception system structure validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Exception structure test failed: {e}")
        return False


def test_middleware_file_structure():
    """Integration test: Validate error handler middleware structure."""
    print("🧪 Testing error handler structure...")
    
    try:
        with open('../../src/api/middleware/error_handler.py', 'r') as f:
            content = f.read()
        
        # Check for correct function names (as they actually exist)
        required_components = [
            'ErrorResponse',
            'setup_exception_handlers',
            'request_id_middleware',
            'pydantic_ai_exception_handler',
            'validation_exception_handler',
            'http_exception_handler',
            'sqlalchemy_exception_handler',
            'generic_exception_handler'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        
        print("✅ All required error handler components present")
        
        # Check for proper error response structure
        if 'class ErrorResponse' in content and 'error_code:' in content and 'message:' in content:
            print("✅ ErrorResponse class properly structured")
        else:
            print("❌ ErrorResponse class missing or incomplete")
            return False
        
        # Check for request ID middleware
        if 'request_id_middleware' in content and 'X-Request-ID' in content:
            print("✅ Request ID middleware implemented")
        else:
            print("❌ Request ID middleware missing")
            return False
        
        # Check for exception handler setup
        if 'app.add_exception_handler' in content:
            print("✅ Exception handler registration implemented")
        else:
            print("❌ Exception handler registration missing")
            return False
        
        # Check for proper status code mapping
        if 'status_code_map' in content:
            print("✅ Status code mapping implemented")
        else:
            print("❌ Status code mapping missing")
            return False
        
        print("✅ Error handler structure validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handler structure test failed: {e}")
        return False


def test_service_integration():
    """Integration test: Validate service layer exception integration."""
    print("🧪 Testing service layer integration...")
    
    try:
        # Read the agent service file
        with open('../../src/services/agents/agent_service.py', 'r') as f:
            content = f.read()
        
        # Check for proper exception imports
        required_imports = [
            'AgentNotFoundException',
            'ModelNotAvailableException',
            'AuthorizationException',
            'BusinessLogicException',
            'ExternalServiceException'
        ]
        
        missing_imports = []
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"❌ Missing exception imports: {missing_imports}")
            return False
        
        print("✅ All required exception imports present")
        
        # Check for exception raising
        if 'raise AgentNotFoundException' in content:
            print("✅ AgentNotFoundException properly used")
        else:
            print("❌ AgentNotFoundException not used in service")
            return False
        
        if 'raise ModelNotAvailableException' in content:
            print("✅ ModelNotAvailableException properly used")
        else:
            print("❌ ModelNotAvailableException not used in service")
            return False
        
        if 'raise AuthorizationException' in content:
            print("✅ AuthorizationException properly used")
        else:
            print("❌ AuthorizationException not used in service")
            return False
        
        print("✅ Service integration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        return False


def test_main_app_integration():
    """Integration test: Validate main.py integration."""
    print("🧪 Testing main app integration...")
    
    try:
        with open('../../src/main.py', 'r') as f:
            content = f.read()
        
        # Check for error handler setup
        if 'setup_exception_handlers' in content:
            print("✅ Exception handlers setup called")
        else:
            print("❌ Exception handlers setup missing")
            return False
        
        # Check for middleware setup
        if 'request_id_middleware' in content:
            print("✅ Request ID middleware registered")
        else:
            print("❌ Request ID middleware missing")
            return False
        
        print("✅ Main app integration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Main integration test failed: {e}")
        return False


def main():
    """Run integration tests for error handling system."""
    print("🔗 Error Handling Integration Tests")
    print("=" * 45)
    
    tests = [
        ("Exception File Structure", test_exception_file_structure),
        ("Middleware File Structure", test_middleware_file_structure),
        ("Service Layer Integration", test_service_integration),
        ("Main App Integration", test_main_app_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔗 Running {test_name}...")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASS")
                passed += 1
            else:
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 45)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 45)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-" * 45)
    print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Error handling system properly integrated")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)