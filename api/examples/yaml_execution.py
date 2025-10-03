"""
YAML Configuration Execution Example
===================================

Demonstrates how to use YAML configuration files for experiment execution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI
import json


def create_sample_config():
    """Create a sample YAML configuration file"""
    config = {
        "model": {
            "model_intent": {
                "purpose": "knowledge_editing",
                "architecture_preference": "auto",
                "size_preference": "medium",
                "hparams_dir": "../hparams/ROME/llama-7b"
            }
        },
        "editing": {
            "intent": {
                "goal": "knowledge_editing",
                "strategy": "precise_localization",
                "method": "ROME",
                "constraints": {
                    "locality_preservation": "high",
                    "generalization": "medium"
                }
            }
        },
        "dataset": {
            "data_intent": {
                "purpose": "knowledge_editing",
                "domain": "general_knowledge",
                "data_type": "structured_factual",
                "data_dir": "./data",
                "ds_size": 100  # Limit dataset size for demo
            }
        },
        "execution": {
            "execution_intent": {
                "goal": "successful_editing",
                "strategy": "adaptive",
                "metrics_save_dir": "./output",
                "quality_requirements": {
                    "success_rate_threshold": 0.9
                },
                "efficiency_requirements": {
                    "max_execution_time": 3600
                }
            }
        }
    }

    # Save to JSON file
    with open("sample_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Created sample_config.json")
    return "sample_config.json"


def main():
    """YAML configuration execution example"""
    print("=== YAML Configuration Execution Example ===\n")

    # Initialize API
    api = EasyEditAPI()

    # 1. Create Sample Configuration
    print("1. Creating sample JSON configuration...")
    config_file = create_sample_config()
    print()

    # 2. Load and Display Configuration
    print("2. Loading and displaying configuration...")
    with open(config_file, "r") as f:
        config_content = f.read()
    print("Configuration content:")
    print(config_content)
    print()

    # 3. Validate Configuration
    print("3. Validating JSON configuration...")
    validation_result = api.validate_config(config_file)
    print(f"   Valid: {validation_result['valid']}")
    print(f"   Selected script: {validation_result['script_name']}")
    if validation_result['errors']:
        print(f"   Errors: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"   Warnings: {validation_result['warnings']}")
    print()

    # 4. Map Configuration
    print("4. Mapping configuration to script parameters...")
    mapping_result = api.map_config_to_parameters(config_file)
    print(f"   Script: {mapping_result['script_name']}")
    print("   Parameters:")
    for key, value in mapping_result['parameters'].items():
        print(f"     {key}: {value}")
    print()

    # 5. Execute Experiment (Commented for safety)
    print("5. Executing experiment from YAML configuration...")
    print("   (Actual execution commented out for safety)")
    print("   To execute, uncomment the following lines:")

    # Synchronous execution
    print("   # Synchronous execution:")
    print("   # result = api.execute_experiment(config_file)")
    print("   # print(f'Success: {result[\"success\"]}')")
    print("   # print(f'Execution time: {result[\"execution_time\"]} seconds')")

    # Asynchronous execution
    print("   # Asynchronous execution:")
    print("   # task_id = api.execute_experiment(config_file, async_execution=True)")
    print("   # print(f'Task ID: {task_id}')")
    print("   #")
    print("   # # Wait for completion")
    print("   # success = api.wait_for_completion(task_id, timeout=300)")
    print("   # if success:")
    print("   #     result = api.get_task_result(task_id)")
    print("   #     print(f'Result: {result}')")
    print()

    # 6. Batch Execution Example
    print("6. Batch execution example...")
    configs = [
        config_file,  # Use the config we just created
        # You could add more config files here
    ]

    print("   Batch execution code (commented for safety):")
    print("   # results = api.batch_execute(configs, max_concurrent=2)")
    print("   # for i, result in enumerate(results):")
    print("   #     print(f'Experiment {i+1}: Success={result.get(\"success\", False)}')")
    print()

    # 7. Configuration Variations
    print("7. Configuration variations...")
    variations = [
        {
            "name": "FT Method",
            "config": {
                "model": {
                    "model_intent": {
                        "purpose": "knowledge_editing",
                        "method": "FT",
                        "hparams_dir": "../hparams/FT/llama-7b"
                    }
                },
                "editing": {
                    "intent": {
                        "goal": "knowledge_editing",
                        "method": "FT"
                    }
                },
                "dataset": {
                    "data_intent": {
                        "data_dir": "./data",
                        "ds_size": 50
                    }
                }
            }
        },
        {
            "name": "MEMIT Method",
            "config": {
                "model": {
                    "model_intent": {
                        "purpose": "knowledge_editing",
                        "method": "MEMIT",
                        "hparams_dir": "../hparams/MEMIT/llama-7b"
                    }
                },
                "editing": {
                    "intent": {
                        "goal": "knowledge_editing",
                        "method": "MEMIT"
                    }
                },
                "dataset": {
                    "data_intent": {
                        "data_dir": "./data",
                        "ds_size": 50
                    }
                }
            }
        }
    ]

    for variation in variations:
        print(f"   {variation['name']}:")
        validation = api.validate_config(variation['config'])
        print(f"     Valid: {validation['valid']}")
        print(f"     Script: {validation['script_name']}")
    print()

    # 8. Environment Variable Example
    print("8. Environment variable substitution...")
    env_config = {
        "model": {
            "model_intent": {
                "purpose": "${EXPERIMENT_PURPOSE}",
                "hparams_dir": "${HPARAMS_DIR}/ROME/llama-7b"
            }
        },
        "dataset": {
            "data_intent": {
                "data_dir": "${DATA_DIR}"
            }
        }
    }

    # Set environment variables
    os.environ["EXPERIMENT_PURPOSE"] = "knowledge_editing"
    os.environ["HPARAMS_DIR"] = "../hparams"
    os.environ["DATA_DIR"] = "./data"

    mapping_result = api.map_config_to_parameters(env_config)
    print(f"   Environment variables processed successfully: {mapping_result['script_name']}")
    print()

    # 9. Cleanup
    print("9. Cleaning up...")
    if os.path.exists(config_file):
        os.remove(config_file)
        print(f"   Removed {config_file}")

    print("\n=== JSON Configuration Example Complete ===")


if __name__ == "__main__":
    main()