"""
Configuration Examples
======================

Demonstrates various configuration options and patterns.
"""

import sys
import os
import json
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.api_client import EasyEditClient
from api.utils.config_loader import ConfigLoader
from api.utils.logging_utils import setup_logging, get_logger


def main():
    """Configuration examples"""
    # Setup logging
    setup_logging({
        'level': 'INFO',
        'console_output': True,
        'colored_output': True
    })

    logger = get_logger("config_examples")

    print("=== EasyEdit API Configuration Examples ===\n")

    # 1. Basic intent configuration
    print("1. Basic intent configuration...")
    basic_config = {
        "model": {
            "model_intent": {
                "purpose": "knowledge_editing",
                "architecture_preference": "auto",
                "size_preference": "medium"
            }
        },
        "editing": {
            "intent": {
                "goal": "knowledge_editing",
                "strategy": "precise_localization",
                "constraints": {
                    "locality_preservation": "high",
                    "generalization": "medium"
                }
            }
        },
        "execution": {
            "execution_intent": {
                "goal": "successful_editing",
                "strategy": "adaptive",
                "quality_requirements": {
                    "success_rate_threshold": 0.9
                }
            }
        }
    }

    print("   YAML configuration:")
    print(yaml.dump(basic_config, default_flow_style=False, indent=2))
    print()

    # 2. Advanced configuration with multiple models
    print("2. Advanced configuration with multiple models...")
    advanced_config = {
        "experiment": {
            "name": "multi_model_comparison",
            "description": "Compare editing methods across multiple models"
        },
        "models": [
            {
                "model_intent": {
                    "purpose": "knowledge_editing",
                    "architecture_preference": "decoder_only",
                    "size_preference": "small",
                    "performance_requirements": {
                        "editing_compatibility": "high",
                        "inference_speed": "medium"
                    }
                }
            },
            {
                "model_intent": {
                    "purpose": "knowledge_editing",
                    "architecture_preference": "decoder_only",
                    "size_preference": "medium",
                    "performance_requirements": {
                        "editing_compatibility": "high",
                        "inference_speed": "fast"
                    }
                }
            }
        ],
        "editing_methods": [
            {
                "intent": {
                    "goal": "knowledge_editing",
                    "strategy": "precise_localization",
                    "constraints": {
                        "locality_preservation": "high",
                        "generalization": "high",
                        "computational_cost": "low"
                    }
                }
            },
            {
                "intent": {
                    "goal": "knowledge_editing",
                    "strategy": "global_adjustment",
                    "constraints": {
                        "locality_preservation": "medium",
                        "generalization": "high",
                        "computational_cost": "medium"
                    }
                }
            }
        ],
        "execution": {
            "execution_intent": {
                "goal": "comprehensive_analysis",
                "strategy": "parallel",
                "quality_requirements": {
                    "success_rate_threshold": 0.85,
                    "locality_threshold": 0.8,
                    "generalization_threshold": 0.8
                },
                "efficiency_requirements": {
                    "max_execution_time": 7200,
                    "max_memory_usage": "32gb"
                },
                "robustness_requirements": {
                    "fault_tolerance": "high",
                    "retry_attempts": 3
                }
            }
        }
    }

    print("   Advanced YAML configuration:")
    print(yaml.dump(advanced_config, default_flow_style=False, indent=2))
    print()

    # 3. Configuration validation
    print("3. Configuration validation...")
    config_loader = ConfigLoader()

    # Define a simple schema for validation
    schema = {
        "required": ["model", "editing"],
        "fields": {
            "model": {
                "type": "object",
                "required": ["model_intent"]
            },
            "model.intent": {
                "type": "object",
                "required": ["purpose", "architecture_preference", "size_preference"],
                "fields": {
                    "purpose": {
                        "type": "string",
                        "enum": ["knowledge_editing", "behavior_modification", "capability_removal"]
                    },
                    "architecture_preference": {
                        "type": "string",
                        "enum": ["auto", "decoder_only", "encoder_decoder", "hybrid"]
                    },
                    "size_preference": {
                        "type": "string",
                        "enum": ["tiny", "small", "medium", "large", "any"]
                    }
                }
            }
        }
    }

    # Test validation
    print("   Testing valid configuration...")
    valid_result = config_loader.validate_config(basic_config, "intent")
    print(f"   Valid: {valid_result.is_valid}")
    if valid_result.errors:
        print(f"   Errors: {valid_result.errors}")
    if valid_result.warnings:
        print(f"   Warnings: {valid_result.warnings}")

    print("\n   Testing invalid configuration...")
    invalid_config = basic_config.copy()
    invalid_config["model"]["model_intent"]["purpose"] = "invalid_purpose"
    invalid_result = config_loader.validate_config(invalid_config, "intent")
    print(f"   Valid: {invalid_result.is_valid}")
    if invalid_result.errors:
        print(f"   Errors: {invalid_result.errors}")
    print()

    # 4. Environment variable substitution
    print("4. Environment variable substitution...")
    env_config = {
        "model": {
            "model_intent": {
                "purpose": "${EDITING_PURPOSE}",
                "size_preference": "${MODEL_SIZE}"
            }
        },
        "execution": {
            "execution_intent": {
                "quality_requirements": {
                    "success_rate_threshold": ${SUCCESS_THRESHOLD}
                }
            }
        }
    }

    print("   Configuration with environment variables:")
    print(yaml.dump(env_config, default_flow_style=False, indent=2))

    # Set some environment variables for demonstration
    os.environ["EDITING_PURPOSE"] = "knowledge_editing"
    os.environ["MODEL_SIZE"] = "medium"
    os.environ["SUCCESS_THRESHOLD"] = "0.9"

    resolved_config = config_loader.resolve_environment_variables(env_config)
    print("\n   After environment variable resolution:")
    print(yaml.dump(resolved_config, default_flow_style=False, indent=2))
    print()

    # 5. Configuration merging
    print("5. Configuration merging...")
    base_config = {
        "model": {
            "model_intent": {
                "purpose": "knowledge_editing",
                "architecture_preference": "auto"
            }
        },
        "execution": {
            "execution_intent": {
                "strategy": "adaptive"
            }
        }
    }

    override_config = {
        "model": {
            "model_intent": {
                "size_preference": "medium"
            }
        },
        "execution": {
            "execution_intent": {
                "quality_requirements": {
                    "success_rate_threshold": 0.95
                }
            }
        }
    }

    print("   Base configuration:")
    print(yaml.dump(base_config, default_flow_style=False, indent=2))

    print("\n   Override configuration:")
    print(yaml.dump(override_config, default_flow_style=False, indent=2))

    merged_config = config_loader.merge_configs(base_config, override_config)
    print("\n   Merged configuration:")
    print(yaml.dump(merged_config, default_flow_style=False, indent=2))
    print()

    # 6. Save and load configurations
    print("6. Save and load configurations...")
    config_dir = "./temp_configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Save as YAML
    yaml_path = os.path.join(config_dir, "example_config.yaml")
    config_loader.save_config(basic_config, yaml_path, "yaml")
    print(f"   Saved YAML configuration: {yaml_path}")

    # Save as JSON
    json_path = os.path.join(config_dir, "example_config.json")
    config_loader.save_config(basic_config, json_path, "json")
    print(f"   Saved JSON configuration: {json_path}")

    # Load configurations
    loaded_yaml = config_loader.load_config(yaml_path)
    loaded_json = config_loader.load_config(json_path)

    print(f"\n   Loaded YAML matches original: {loaded_yaml == basic_config}")
    print(f"   Loaded JSON matches original: {loaded_json == basic_config}")

    # Clean up
    import shutil
    if os.path.exists(config_dir):
        shutil.rmtree(config_dir)
    print(f"   Cleaned up temporary directory: {config_dir}")
    print()

    # 7. Default configurations
    print("7. Default configurations...")
    default_types = ["intent_driven", "method_profile", "execution_config"]

    for config_type in default_types:
        default_config = config_loader.get_default_config(config_type)
        print(f"   {config_type}:")
        print(yaml.dump(default_config, default_flow_style=False, indent=2))
        print()

    # 8. Configuration for EasyEdit client
    print("8. EasyEdit client configuration...")
    client_config = {
        "plugin_directories": [
            "./plugins",
            "./custom_methods",
            "./extensions"
        ],
        "discovery": {
            "auto_discover_methods": True,
            "method_search_paths": [
                "easyeditor/models",
                "./custom_methods"
            ],
            "auto_discover_datasets": True,
            "dataset_search_paths": [
                "./data",
                "./datasets"
            ]
        },
        "execution": {
            "default_mode": "adaptive",
            "resource_monitoring": True,
            "optimization": {
                "auto_optimize": True,
                "optimization_goals": ["speed", "memory", "accuracy"]
            }
        },
        "logging": {
            "level": "INFO",
            "console_output": True,
            "file_output": True,
            "log_file": "./logs/easyedit_api.log",
            "json_format": True
        }
    }

    print("   Client configuration:")
    print(yaml.dump(client_config, default_flow_style=False, indent=2))

    # Initialize client with configuration
    try:
        client = EasyEditClient(config=client_config)
        print("\n   Client initialized successfully")
        print(f"   Plugin system enabled: {client.enable_plugins}")
        if client.plugin_manager:
            print(f"   Plugin directories: {client.plugin_manager.plugin_dirs}")
    except Exception as e:
        print(f"\n   Client initialization failed: {e}")

    print("\n=== Configuration Examples Complete ===")


if __name__ == "__main__":
    main()