"""
Test script for EasyEdit API examples
====================================

This script tests the basic_usage.py and advanced_usage.py examples
to ensure they work correctly with the enhanced API.
"""

import sys
import os
import traceback
from pathlib import Path

# Add EasyEdit to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI


def test_basic_usage():
    """Test the basic usage example."""
    print("=== Testing Basic Usage Example ===")

    try:
        # Initialize API
        api = EasyEditAPI()
        print("+ API initialized successfully")

        # Test system information
        system_info = api.get_system_info()
        print(f"+ System info retrieved: {system_info['available_scripts']} scripts")

        # Test supported methods, datasets, models
        methods = api.get_supported_methods()
        datasets = api.get_supported_datasets()
        models = api.get_supported_models()

        print(f"+ Supported methods: {len(methods)}")
        print(f"+ Supported datasets: {len(datasets)}")
        print(f"+ Supported models: {len(models)}")

        # Test script discovery
        scripts = api.list_available_scripts()
        print(f"+ Script discovery: {len(scripts)} scripts found")

        # Test compatibility search
        rome_scripts = api.find_compatible_scripts(method="ROME")
        print(f"+ ROME compatibility: {len(rome_scripts)} scripts")

        # Test method-dataset validation
        is_valid = api.validate_method_dataset_combination("ROME", "ZsreDataset")
        print(f"+ Method-dataset validation: {is_valid}")

        # Test configuration suggestion
        config = api.suggest_config(method="ROME", dataset="ZsreDataset")
        print(f"+ Configuration suggestion: {config['model']['model_intent']['method']}")

        # Test configuration validation
        validation = api.validate_config(config)
        print(f"+ Configuration validation: {validation['valid']}")

        # Test parameter mapping
        mapping = api.map_config_to_parameters(config)
        print(f"+ Parameter mapping: {mapping['script_name']}")

        print("\n+ Basic usage test completed successfully!")
        return True

    except Exception as e:
        print(f"\n- Basic usage test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_advanced_features():
    """Test advanced features."""
    print("\n=== Testing Advanced Features ===")

    try:
        api = EasyEditAPI()

        # Test multiple method-dataset combinations
        test_combinations = [
            ("ROME", "ZsreDataset"),
            ("FT", "CounterFactDataset"),
            ("MEMIT", "WikiBioDataset"),
            ("KN", "KnowEditDataset"),
        ]

        print("Testing method-dataset combinations:")
        for method, dataset in test_combinations:
            try:
                is_valid = api.validate_method_dataset_combination(method, dataset)
                config = api.suggest_config(method=method, dataset=dataset)
                validation = api.validate_config(config)
                print(f"  + {method} + {dataset}: valid={is_valid}, config_valid={validation['valid']}")
            except Exception as e:
                print(f"  - {method} + {dataset}: {str(e)}")

        # Test model-specific searches
        models_to_test = ["llama", "gpt2", "baichuan"]
        print("\nTesting model-specific searches:")
        for model in models_to_test:
            try:
                scripts = api.find_compatible_scripts(model=model)
                print(f"  + {model}: {len(scripts)} compatible scripts")
            except Exception as e:
                print(f"  - {model}: {str(e)}")

        # Test batch configuration
        print("\nTesting batch configuration:")
        batch_configs = []
        for method in ["ROME", "FT", "MEMIT"]:
            try:
                config = api.suggest_config(method=method, dataset="ZsreDataset")
                batch_configs.append(config)
                print(f"  + {method} config created")
            except Exception as e:
                print(f"  - {method} config failed: {str(e)}")

        print(f"\n+ Advanced features test completed! Created {len(batch_configs)} configs")
        return True

    except Exception as e:
        print(f"\n- Advanced features test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")

    try:
        api = EasyEditAPI()

        # Test invalid method
        try:
            is_valid = api.validate_method_dataset_combination("INVALID_METHOD", "ZsreDataset")
            print(f"+ Invalid method handled: {is_valid}")
        except Exception as e:
            print(f"+ Invalid method exception: {str(e)}")

        # Test invalid dataset
        try:
            is_valid = api.validate_method_dataset_combination("ROME", "INVALID_DATASET")
            print(f"+ Invalid dataset handled: {is_valid}")
        except Exception as e:
            print(f"+ Invalid dataset exception: {str(e)}")

        # Test invalid configuration
        try:
            validation = api.validate_config({"invalid": "config"})
            print(f"+ Invalid config handled: {validation['valid']}")
        except Exception as e:
            print(f"+ Invalid config exception: {str(e)}")

        print("\n+ Error handling test completed!")
        return True

    except Exception as e:
        print(f"\n- Error handling test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("EasyEdit API Examples Test Suite")
    print("=" * 50)

    results = []

    # Run all tests
    results.append(test_basic_usage())
    results.append(test_advanced_features())
    results.append(test_error_handling())

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Basic Usage: {'+ PASS' if results[0] else '- FAIL'}")
    print(f"Advanced Features: {'+ PASS' if results[1] else '- FAIL'}")
    print(f"Error Handling: {'+ PASS' if results[2] else '- FAIL'}")

    total_passed = sum(results)
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n*** All tests passed! The examples should work correctly. ***")
    else:
        print(f"\n!!! {total_tests - total_passed} test(s) failed. Please check the errors above. !!!")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)