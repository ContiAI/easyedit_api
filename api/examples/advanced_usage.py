"""
Advanced Usage Example
====================

Demonstrates advanced features of the EasyEdit API including:
- Complex configuration scenarios
- Asynchronous execution
- Error handling and recovery
- Performance optimization
- Custom script discovery
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add EasyEdit to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI


def demonstrate_advanced_discovery(api):
    """Demonstrate advanced script discovery capabilities."""
    print("=== Advanced Script Discovery ===")

    # Get all supported capabilities
    methods = api.get_supported_methods()
    datasets = api.get_supported_datasets()
    models = api.get_supported_models()

    print(f"Total Methods: {len(methods)}")
    print(f"Total Datasets: {len(datasets)}")
    print(f"Total Models: {len(models)}")

    # Find most versatile scripts (support most methods)
    most_versatile = sorted(
        api.script_registry.scripts.values(),
        key=lambda x: len(x.supported_methods),
        reverse=True
    )[:3]

    print("\nMost versatile scripts:")
    for script in most_versatile:
        print(f"  {script['name']}: {len(script['supported_methods'])} methods")

    # Find method compatibility matrix
    print("\nMethod compatibility matrix (sample):")
    sample_methods = methods[:5]
    for method in sample_methods:
        compatible_scripts = api.find_compatible_scripts(method=method)
        compatible_datasets = set()
        for script in compatible_scripts:
            compatible_datasets.update(script['supported_datasets'])
        print(f"  {method}: {len(compatible_scripts)} scripts, {len(compatible_datasets)} datasets")


def demonstrate_configuration_scenarios(api):
    """Demonstrate various configuration scenarios."""
    print("\n=== Configuration Scenarios ===")

    # Scenario 1: Knowledge editing with ROME
    print("\n1. Knowledge Editing Scenario:")
    config = api.suggest_config(method="ROME", dataset="ZsreDataset")
    config['model']['model_intent'].update({
        'architecture_preference': 'llama',
        'size_preference': '7b'
    })
    config['dataset']['data_intent'].update({
        'ds_size': 100,
        'required_fields': ['prompt', 'target_new', 'ground_truth']
    })
    validation = api.validate_config(config)
    print(f"   Valid: {validation['valid']}")
    print(f"   Script: {validation.get('script_name', 'None')}")

    # Scenario 2: Batch editing with MEMIT
    print("\n2. Batch Editing Scenario:")
    config = api.suggest_config(method="MEMIT", dataset="CounterFactDataset")
    config['editing']['intent'].update({
        'strategy': 'batch',
        'constraints': {
            'locality_preservation': 'high',
            'generalization': 'medium'
        }
    })
    validation = api.validate_config(config)
    print(f"   Valid: {validation['valid']}")
    print(f"   Script: {validation.get('script_name', 'None')}")

    # Scenario 3: Safety editing
    print("\n3. Safety Editing Scenario:")
    config = api.suggest_config(method="SafeEdit", dataset="ZsreDataset")
    validation = api.validate_config(config)
    print(f"   Valid: {validation['valid']}")
    print(f"   Script: {validation.get('script_name', 'None')}")

    # Scenario 4: Model-specific configuration
    print("\n4. Model-Specific Configuration:")
    model_configs = [
        ("llama-7b", "ROME"),
        ("gpt2-xl", "FT"),
        ("baichuan-7b", "KN"),
    ]

    for model, method in model_configs[:2]:
        config = api.suggest_config(method=method, dataset="ZsreDataset")
        config['model']['model_intent']['hparams_dir'] = f"../hparams/{method}/{model}"
        validation = api.validate_config(config)
        print(f"   {model} + {method}: {'+' if validation['valid'] else '-'}")


def demonstrate_async_execution(api):
    """Demonstrate asynchronous execution capabilities."""
    print("\n=== Asynchronous Execution Demo ===")

    # Prepare multiple experiment configurations
    experiments = [
        api.suggest_config(method="ROME", dataset="ZsreDataset"),
        api.suggest_config(method="FT", dataset="CounterFactDataset"),
        api.suggest_config(method="MEMIT", dataset="WikiBioDataset"),
    ]

    print(f"Preparing {len(experiments)} experiments for async execution")

    # Submit async tasks (commented out for safety)
    print("Submitting async tasks...")
    task_ids = []
    for i, config in enumerate(experiments):
        print(f"  Experiment {i+1}: {config['model']['model_intent']['method']}")
        # task_info = api.execute_experiment(config, async_execution=True)
        # task_ids.append(task_info['task_id'])
        # print(f"    Task ID: {task_info['task_id']}")

    # Monitor task progress (simulated)
    print("\nMonitoring task progress:")
    # for task_id in task_ids:
    #     status = api.get_task_status(task_id)
    #     print(f"  Task {task_id}: {status['status']}")

    print("  (Tasks would run in background with progress tracking)")


def demonstrate_error_handling(api):
    """Demonstrate error handling and recovery."""
    print("\n=== Error Handling Demonstration ===")

    # Test invalid configurations
    invalid_configs = [
        # Invalid method
        {"model": {"model_intent": {"method": "INVALID_METHOD"}}},
        # Missing required sections
        {"model": {"model_intent": {"method": "ROME"}}},
        # Incompatible method-dataset combination
        api.suggest_config(method="ROME", dataset="InvalidDataset"),
    ]

    for i, config in enumerate(invalid_configs):
        print(f"\nTesting invalid config {i+1}:")
        try:
            validation = api.validate_config(config)
            print(f"  Valid: {validation['valid']}")
            if not validation['valid']:
                print(f"  Errors: {validation['errors'][:2]}")  # Show first 2 errors
        except Exception as e:
            print(f"  Exception: {str(e)}")

    # Test recovery mechanisms
    print("\nRecovery mechanisms:")
    try:
        # Get valid configurations for recovery
        valid_configs = []
        for method in ["ROME", "FT", "MEMIT"]:
            try:
                config = api.suggest_config(method=method, dataset="ZsreDataset")
                if api.validate_config(config)['valid']:
                    valid_configs.append(config)
            except Exception as e:
                print(f"  Failed to create config for {method}: {str(e)}")

        print(f"  Successfully created {len(valid_configs)} fallback configurations")
    except Exception as e:
        print(f"  Recovery failed: {str(e)}")


def demonstrate_performance_optimization(api):
    """Demonstrate performance optimization features."""
    print("\n=== Performance Optimization ===")

    # Test script selection optimization
    print("Script selection optimization:")
    import time

    # Time different search strategies
    start_time = time.time()
    scripts_by_method = api.find_compatible_scripts(method="ROME")
    method_time = time.time() - start_time

    start_time = time.time()
    scripts_by_dataset = api.find_compatible_scripts(dataset="ZsreDataset")
    dataset_time = time.time() - start_time

    start_time = time.time()
    scripts_combined = api.find_compatible_scripts(method="ROME", dataset="ZsreDataset")
    combined_time = time.time() - start_time

    print(f"  Method-only search: {method_time:.4f}s ({len(scripts_by_method)} results)")
    print(f"  Dataset-only search: {dataset_time:.4f}s ({len(scripts_by_dataset)} results)")
    print(f"  Combined search: {combined_time:.4f}s ({len(scripts_combined)} results)")

    # Show caching effectiveness
    print("\nCaching effectiveness:")
    start_time = time.time()
    for _ in range(5):
        api.find_compatible_scripts(method="ROME")
    cached_time = time.time() - start_time
    print(f"  5 repeated searches: {cached_time:.4f}s (benefiting from caching)")


def demonstrate_custom_patterns(api):
    """Demonstrate custom usage patterns."""
    print("\n=== Custom Usage Patterns ===")

    # Pattern 1: Method comparison experiment
    print("1. Method Comparison Pattern:")
    methods = ["ROME", "FT", "MEMIT"]
    base_config = api.suggest_config(method="ROME", dataset="ZsreDataset")

    comparison_configs = []
    for method in methods:
        config = base_config.copy()
        config['model']['model_intent']['method'] = method
        validation = api.validate_config(config)
        if validation['valid']:
            comparison_configs.append((method, config))

    print(f"  Valid method comparisons: {len(comparison_configs)}")
    for method, _ in comparison_configs:
        print(f"    {method}: +")

    # Pattern 2: Dataset scalability testing
    print("\n2. Dataset Scalability Pattern:")
    base_config = api.suggest_config(method="ROME", dataset="ZsreDataset")
    sizes = [10, 50, 100, 200]

    for size in sizes:
        config = base_config.copy()
        config['dataset']['data_intent']['ds_size'] = size
        validation = api.validate_config(config)
        print(f"  Size {size}: {'+' if validation['valid'] else '-'}")

    # Pattern 3: Model family testing
    print("\n3. Model Family Testing Pattern:")
    model_families = [
        ("llama", ["7b", "13b"]),
        ("gpt2", ["xl", "large"]),
        ("baichuan", ["7b", "13b"]),
    ]

    for family, sizes in model_families[:2]:  # Show first 2 families
        print(f"  {family} family:")
        for size in sizes:
            model = f"{family}-{size}"
            scripts = api.find_compatible_scripts(model=model)
            print(f"    {model}: {len(scripts)} compatible scripts")


def main():
    """Main advanced demonstration function."""
    print("=== EasyEdit API Advanced Usage Example ===\n")

    # Initialize API
    api = EasyEditAPI()

    # Run all demonstrations
    try:
        demonstrate_advanced_discovery(api)
        demonstrate_configuration_scenarios(api)
        demonstrate_async_execution(api)
        demonstrate_error_handling(api)
        demonstrate_performance_optimization(api)
        demonstrate_custom_patterns(api)

        print("\n=== Advanced Usage Example Complete ===")
        print("\nNote: Actual execution is commented out for safety.")
        print("Uncomment execution lines to run real experiments.")

    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()