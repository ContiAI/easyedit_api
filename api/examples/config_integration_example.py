"""
Config Integration Example
=========================

Demonstrates how to integrate declarative YAML configurations with the EasyEdit API.
Shows loading, converting, and executing experiments from config files.
"""

import sys
import os
from pathlib import Path

# Add EasyEdit to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI
from config_loader import ConfigLoader


def main():
    """Main demonstration function."""
    print("=== EasyEdit Config Integration Example ===\n")

    # Initialize API and config loader
    api = EasyEditAPI()
    loader = ConfigLoader(api)

    # 1. List available configurations
    print("1. Discovering available configurations...")
    config_files = loader.get_available_configs()
    print(f"   Found {len(config_files)} configuration files")

    if not config_files:
        print("   No configuration files found. Creating sample...")
        loader.create_sample_config("./sample_config.yaml")
        config_files = ["./sample_config.yaml"]

    # Show first few configs
    for i, config_file in enumerate(config_files[:3]):
        print(f"   {i+1}. {Path(config_file).name}")
    if len(config_files) > 3:
        print(f"   ... and {len(config_files) - 3} more")
    print()

    # 2. Load and validate configurations
    print("2. Loading and validating configurations...")
    valid_configs = []
    for config_file in config_files[:3]:  # Test first 3 configs
        try:
            print(f"   Testing {Path(config_file).name}...")
            validation = loader.validate_config_compatibility(config_file)

            print(f"     Valid: {validation.get('valid', False)}")
            if validation.get('script_name'):
                print(f"     Script: {validation['script_name']}")

            # Check method compatibility
            if 'method_compatibility' in validation:
                method_comp = validation['method_compatibility']
                print(f"     Method '{method_comp['method']}': {'+ Available' if method_comp['available'] else '- Not found'}")

            # Check dataset compatibility
            if 'dataset_compatibility' in validation:
                dataset_comp = validation['dataset_compatibility']
                print(f"     Dataset '{dataset_comp['dataset']}': {'+ Available' if dataset_comp['available'] else '- Not found'}")

            if validation.get('valid', False):
                valid_configs.append(config_file)

        except Exception as e:
            print(f"     Error: {str(e)}")

        print()

    # 3. Convert declarative to API format
    print("3. Converting declarative to API format...")
    if valid_configs:
        config_file = valid_configs[0]
        print(f"   Converting {Path(config_file).name}...")

        # Load declarative config
        declarative_config = loader.load_declarative_config(config_file)
        print(f"     Loaded config sections: {list(declarative_config.keys())}")

        # Convert to API format
        api_config = loader.load_and_convert_config(config_file)
        print(f"     API config sections: {list(api_config.keys())}")

        # Show mapping details
        if 'model' in api_config and 'model_intent' in api_config['model']:
            method = api_config['model']['model_intent'].get('method', 'Unknown')
            print(f"     Mapped method: {method}")

        if 'dataset' in api_config and 'data_intent' in api_config['dataset']:
            dataset = api_config['dataset']['data_intent'].get('dataset_name', 'Unknown')
            print(f"     Mapped dataset: {dataset}")

    else:
        print("   No valid configurations found")
        # Create a simple API config for demonstration
        api_config = api.suggest_config(method="ROME", dataset="ZsreDataset")
        print("     Created sample API config")

    print()

    # 4. Execute experiment (commented out for safety)
    print("4. Experiment execution...")
    if valid_configs:
        config_file = valid_configs[0]
        print(f"   Would execute experiment from {Path(config_file).name}")
        print("   (Actual execution commented out for safety)")

        # Uncomment to execute:
        # try:
        #     result = loader.execute_declarative_experiment(config_file)
        #     print(f"   Success: {result.get('success', False)}")
        #     print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
        # except Exception as e:
        #     print(f"   Execution failed: {str(e)}")

    else:
        print("   No valid configurations to execute")

    print()

    # 5. Create custom configuration
    print("5. Creating custom configuration...")
    custom_config_path = "./custom_config.yaml"

    # Generate config from method and dataset
    suggested_config = loader.suggest_config_from_template(
        method="FT",
        dataset="CounterFactDataset"
    )

    print("   Generated configuration for FT + CounterFactDataset:")
    print(f"     Method: {suggested_config['model']['model_intent'].get('method')}")
    print(f"     Dataset: {suggested_config['dataset']['data_intent'].get('dataset_name')}")

    # Save as configuration file
    loader.create_sample_config(custom_config_path, method="FT", dataset="CounterFactDataset")
    print(f"   Saved custom config: {custom_config_path}")

    # Validate the created config
    validation = loader.validate_config_compatibility(custom_config_path)
    print(f"   Validation: {'+ Valid' if validation.get('valid', False) else '- Invalid'}")

    print()

    # 6. Configuration batch processing
    print("6. Configuration batch processing...")
    batch_results = []

    for config_file in valid_configs[:2]:  # Process first 2 valid configs
        try:
            # Load API config
            api_config = loader.load_and_convert_config(config_file)

            # Validate
            validation = api.validate_config(api_config)
            if validation['valid']:
                batch_results.append({
                    'config_file': Path(config_file).name,
                    'script': validation['script_name'],
                    'method': api_config['model']['model_intent'].get('method', 'Unknown'),
                    'dataset': api_config['dataset']['data_intent'].get('dataset_name', 'Unknown')
                })

        except Exception as e:
            print(f"   Error processing {Path(config_file).name}: {str(e)}")

    print(f"   Successfully processed {len(batch_results)} configurations:")
    for result in batch_results:
        print(f"     {result['config_file']}: {result['method']} + {result['dataset']} -> {result['script']}")

    print()

    # 7. System integration summary
    print("7. System integration summary...")
    print("   + Config loader successfully bridges declarative configs and API")
    print("   + Supports both YAML and JSON configurations")
    print("   + Provides validation and compatibility checking")
    print("   + Enables batch processing of multiple configurations")
    print("   + Maintains backward compatibility with existing API")

    print("\n=== Config Integration Example Complete ===")

    # Cleanup
    if os.path.exists("./sample_config.yaml"):
        os.remove("./sample_config.yaml")
    if os.path.exists("./custom_config.yaml"):
        os.remove("./custom_config.yaml")


if __name__ == "__main__":
    main()