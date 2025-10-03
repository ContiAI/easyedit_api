"""
Basic Usage Example
==================

Demonstrates basic usage of the EasyEdit API for experiment execution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI, get_api


def main():
    """Basic usage example"""
    print("=== EasyEdit API Basic Usage Example ===\n")

    # Initialize API
    api = EasyEditAPI()

    # 1. System Information and Capabilities
    print("1. Getting system information and capabilities...")
    system_info = api.get_system_info()
    print(f"   Available scripts: {system_info['available_scripts']}")
    print(f"   Examples directory: {system_info['examples_dir']}")

    # Show supported methods, datasets, and models
    methods = api.get_supported_methods()
    datasets = api.get_supported_datasets()
    models = api.get_supported_models()

    print(f"   Supported editing methods: {len(methods)}")
    print(f"   Supported datasets: {len(datasets)}")
    print(f"   Supported models: {len(models)}")
    print()

    # 2. Comprehensive Script Discovery
    print("2. Discovering available scripts...")
    scripts = api.list_available_scripts()
    print(f"   Found {len(scripts)} scripts:")
    for script in scripts[:5]:  # Show first 5
        print(f"   - {script['name']}: {len(script['supported_methods'])} methods, "
              f"{len(script['supported_datasets'])} datasets")
    if len(scripts) > 5:
        print(f"   ... and {len(scripts) - 5} more")
    print()

    # 3. Get Script Information
    if scripts:
        print("3. Getting detailed script information...")
        script_name = scripts[0]['name']
        script_info = api.get_script_info(script_name)
        print(f"   Script: {script_info['name']}")
        print(f"   Description: {script_info['description']}")
        print(f"   Supported methods: {script_info['supported_methods']}")
        print(f"   Supported datasets: {script_info['supported_datasets']}")
        print(f"   Required parameters: {script_info['required_parameters']}")
        print()

    # 4. Advanced Compatibility Search
    print("4. Advanced compatibility search...")
    # Search by method
    rome_scripts = api.find_compatible_scripts(method="ROME")
    print(f"   Scripts supporting ROME: {len(rome_scripts)}")
    # Search by dataset
    zsre_scripts = api.find_compatible_scripts(dataset="ZsreDataset")
    print(f"   Scripts supporting ZsreDataset: {len(zsre_scripts)}")
    # Search by model
    llama_scripts = api.find_compatible_scripts(model="llama")
    print(f"   Scripts supporting llama models: {len(llama_scripts)}")
    # Combined search
    compatible_scripts = api.find_compatible_scripts(method="ROME", dataset="ZsreDataset")
    print(f"   Scripts compatible with ROME + ZsreDataset: {len(compatible_scripts)}")
    print()

    # 5. Method-Dataset Validation
    print("5. Method-Dataset validation...")
    validation_pairs = [
        ("ROME", "ZsreDataset"),
        ("FT", "CounterFactDataset"),
        ("MEMIT", "WikiBioDataset"),
        ("KN", "KnowEditDataset"),
        ("SERAC", "WikiBioDataset"),
        ("WISE", "ZsreDataset"),
        ("SafeEdit", "ZsreDataset"),
        ("HalluEditBench", "WikiBioDataset"),
    ]

    for method, dataset in validation_pairs[:3]:  # Show first 3
        is_valid = api.validate_method_dataset_combination(method, dataset)
        print(f"   {method} + {dataset}: {'+ Valid' if is_valid else '- Invalid'}")
    print()

    # 6. Configuration Suggestion and Creation
    print("6. Configuration suggestion and creation...")
    # Get suggested configuration
    suggested_config = api.suggest_config(method="ROME", dataset="ZsreDataset")
    print("   Suggested configuration for ROME + ZsreDataset:")
    print(f"   Method: {suggested_config['model']['model_intent']['method']}")
    print(f"   Dataset: {suggested_config['dataset']['data_intent'].get('dataset_name', 'ZsreDataset')}")
    print(f"   Model preference: {suggested_config['model']['model_intent'].get('architecture_preference', 'llama')}")

    # Create default configuration
    default_config = api.create_default_config()
    print("   Default configuration structure:")
    print("   model:")
    print("     model_intent:")
    print(f"       purpose: {default_config['model']['model_intent']['purpose']}")
    print(f"       architecture_preference: {default_config['model']['model_intent']['architecture_preference']}")
    print("   editing:")
    print("     intent:")
    print(f"       goal: {default_config['editing']['intent']['goal']}")
    print(f"       strategy: {default_config['editing']['intent']['strategy']}")
    print()

    # 7. Advanced Configuration Validation
    print("7. Advanced configuration validation...")
    validation_result = api.validate_config(default_config)
    print(f"   Configuration valid: {validation_result['valid']}")
    if validation_result['script_name']:
        print(f"   Selected script: {validation_result['script_name']}")
    if validation_result['errors']:
        print(f"   Errors: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"   Warnings: {validation_result['warnings']}")

    # Test multiple method-dataset combinations
    print("   Testing multiple configurations:")
    test_configs = [
        ("ROME", "ZsreDataset"),
        ("FT", "CounterFactDataset"),
        ("MEMIT", "WikiBioDataset"),
        ("KN", "KnowEditDataset"),
    ]

    for method, dataset in test_configs[:2]:  # Show first 2
        config = api.suggest_config(method=method, dataset=dataset)
        validation = api.validate_config(config)
        print(f"   {method} + {dataset}: {'+' if validation['valid'] else '-'} "
              f"(Script: {validation.get('script_name', 'None')})")
    print()

    # 8. Parameter Mapping
    print("8. Parameter mapping...")
    mapping_result = api.map_config_to_parameters(default_config)
    print(f"   Mapped to script: {mapping_result['script_name']}")
    print(f"   Parameters: {list(mapping_result['parameters'].keys())}")
    print("   Sample parameters:")
    for key, value in list(mapping_result['parameters'].items())[:3]:
        print(f"     {key}: {value}")
    print()

    # 9. Quick Experiment Configuration
    print("9. Quick experiment configuration...")
    print("   Preparing ROME experiment on llama-7b with ZsreDataset")
    print("   (Actual execution commented out for safety)")

    # Show what parameters would be used
    quick_params = {
        "method": "ROME",
        "model": "llama-7b",
        "dataset": "zsre",
        "hparams_dir": "../hparams/ROME/llama-7b",
        "data_dir": "./data",
        "ds_size": 50
    }
    print("   Parameters for quick experiment:")
    for key, value in quick_params.items():
        print(f"     {key}: {value}")

    # Uncomment to actually run:
    # result = api.quick_experiment(**quick_params)
    # print(f"   Result: {result}")

    print()

    # 10. Batch Experiment Example
    print("10. Batch experiment example...")
    batch_configs = [
        api.suggest_config(method="ROME", dataset="ZsreDataset"),
        api.suggest_config(method="FT", dataset="CounterFactDataset"),
        api.suggest_config(method="MEMIT", dataset="WikiBioDataset"),
    ]

    print(f"   Prepared {len(batch_configs)} batch configurations")
    for i, config in enumerate(batch_configs[:2]):  # Show first 2
        method = config['model']['model_intent']['method']
        dataset = config['dataset']['data_intent'].get('dataset_name', 'Unknown')
        print(f"     Config {i+1}: {method} + {dataset}")

    # Uncomment to actually run batch:
    # results = api.batch_execute(batch_configs, max_concurrent=2)
    # print(f"   Batch execution results: {len(results)} experiments")

    print()

    # 11. Advanced Script Registry Operations
    print("11. Advanced script registry operations...")
    print("   Saving script registry...")
    api.save_script_registry("./script_registry.json")
    print("   Registry saved to ./script_registry.json")

    # Show registry statistics
    print("   Registry statistics:")
    print(f"     Total scripts: {len(api.script_registry.scripts)}")
    print(f"     Unique methods: {len(set(m for s in api.script_registry.scripts.values() for m in s.supported_methods))}")
    print(f"     Unique datasets: {len(set(d for s in api.script_registry.scripts.values() for d in s.supported_datasets))}")

    print("\n=== Basic Usage Example Complete ===")


if __name__ == "__main__":
    main()