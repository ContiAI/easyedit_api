"""
EasyEdit Configuration Loader
=============================

Bridges the declarative configuration system with the EasyEdit API.
Provides methods to load and convert declarative YAML configurations
to API-compatible formats.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from easyedit_api import EasyEditAPI

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and converts declarative configurations to API format."""

    def __init__(self, easyedit_api: Optional[EasyEditAPI] = None):
        """Initialize config loader with optional API instance."""
        self.api = easyedit_api or EasyEditAPI()
        self.config_cache = {}

    def load_declarative_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a declarative YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary containing loaded configuration
        """
        try:
            # Try to load YAML if available
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded YAML config from {config_path}")
                return config
            except ImportError:
                logger.warning("pyyaml not available, attempting basic YAML parsing")
                return self._basic_yaml_load(config_path)

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise ValueError(f"Configuration loading failed: {e}")

    def _basic_yaml_load(self, config_path: str) -> Dict[str, Any]:
        """Basic YAML loader for when pyyaml is not available."""
        config = {}
        current_section = None
        current_indent = 0

        with open(config_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip()

                # Skip empty lines and comments
                if not line.strip() or line.strip().startswith('#'):
                    continue

                # Calculate indentation
                indent = len(line) - len(line.lstrip())

                # Handle section headers (top-level keys)
                if indent == 0 and ':' in line:
                    key = line.split(':')[0].strip()
                    current_section = key
                    config[current_section] = {}
                    current_indent = indent
                    continue

                # Handle nested key-value pairs
                if current_section and ':' in line:
                    key_part = line.split(':', 1)[0].strip()
                    value_part = line.split(':', 1)[1].strip()

                    # Convert value to appropriate type
                    if value_part.startswith('[') and value_part.endswith(']'):
                        # List value
                        try:
                            value = json.loads(value_part.replace("'", '"'))
                        except:
                            value = [item.strip().strip("'\"") for item in value_part[1:-1].split(',')]
                    elif value_part.lower() in ('true', 'false'):
                        value = value_part.lower() == 'true'
                    elif value_part.isdigit():
                        value = int(value_part)
                    elif value_part.replace('.', '').isdigit():
                        value = float(value_part)
                    else:
                        # String value
                        value = value_part.strip('"\'')

                    # Handle nested structure
                    if indent > current_indent:
                        if key_part not in config[current_section]:
                            config[current_section][key_part] = {}
                        config[current_section][key_part] = value
                    else:
                        config[current_section][key_part] = value

        return config

    def convert_to_api_config(self, declarative_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert declarative configuration to API-compatible format.

        Args:
            declarative_config: Declarative YAML configuration

        Returns:
            API-compatible configuration dictionary
        """
        api_config = {
            "model": {
                "model_intent": {}
            },
            "editing": {
                "intent": {}
            },
            "dataset": {
                "data_intent": {}
            }
        }

        # Extract model configuration
        if "model" in declarative_config:
            model_config = declarative_config["model"]

            # Handle model intent
            if "model_intent" in model_config:
                api_config["model"]["model_intent"].update(model_config["model_intent"])
            elif "method" in model_config:
                # Handle direct method specification
                api_config["model"]["model_intent"]["method"] = model_config["method"]

            # Handle other model properties
            for key, value in model_config.items():
                if key != "model_intent" and key not in ["name", "path", "type"]:
                    api_config["model"]["model_intent"][key] = value

        # Extract editing configuration
        if "editing" in declarative_config:
            editing_config = declarative_config["editing"]

            if "method" in editing_config:
                api_config["editing"]["intent"]["method"] = editing_config["method"]
                api_config["model"]["model_intent"]["method"] = editing_config["method"]

            if "hyperparameters" in editing_config:
                api_config["editing"]["intent"].update(editing_config["hyperparameters"])

            # Map other editing properties
            for key, value in editing_config.items():
                if key not in ["method", "hyperparameters"]:
                    api_config["editing"]["intent"][key] = value

        # Extract dataset configuration
        if "dataset" in declarative_config:
            dataset_config = declarative_config["dataset"]

            api_config["dataset"]["data_intent"]["dataset_name"] = dataset_config.get("name", "ZsreDataset")
            api_config["dataset"]["data_intent"]["data_dir"] = str(Path(dataset_config.get("path", "./data")).parent)
            api_config["dataset"]["data_intent"]["ds_size"] = dataset_config.get("preprocessing", {}).get("max_length", 512)

            # Map field configurations
            if "field_mapping" in dataset_config:
                api_config["dataset"]["data_intent"]["required_fields"] = list(dataset_config["field_mapping"].values())

        # Extract execution configuration
        if "execution" in declarative_config:
            exec_config = declarative_config["execution"]

            api_config["execution"] = {
                "execution_intent": {
                    "metrics_save_dir": declarative_config.get("environment", {}).get("results_dir", "./results"),
                    "timeout": exec_config.get("resources", {}).get("timeout", 3600)
                }
            }

            # Add quality requirements
            if "evaluation" in declarative_config:
                eval_config = declarative_config["evaluation"]
                api_config["execution"]["execution_intent"]["quality_requirements"] = {
                    "success_rate_threshold": eval_config.get("success_threshold", 0.8)
                }

        return api_config

    def load_and_convert_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load declarative config and convert to API format.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            API-compatible configuration
        """
        # Load declarative configuration
        declarative_config = self.load_declarative_config(config_path)

        # Convert to API format
        api_config = self.convert_to_api_config(declarative_config)

        # Cache the result
        self.config_cache[config_path] = {
            "declarative": declarative_config,
            "api": api_config
        }

        logger.info(f"Successfully converted {config_path} to API format")
        return api_config

    def validate_config_compatibility(self, config_path: str) -> Dict[str, Any]:
        """
        Validate that a declarative config is compatible with available scripts.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validation result dictionary
        """
        try:
            # Load and convert configuration
            api_config = self.load_and_convert_config(config_path)

            # Validate using API
            validation_result = self.api.validate_config(api_config)

            # Add declarative-specific validation
            declarative_config = self.load_declarative_config(config_path)

            # Check method availability
            method = declarative_config.get("editing", {}).get("method")
            if method:
                compatible_scripts = self.api.find_compatible_scripts(method=method)
                validation_result["method_compatibility"] = {
                    "method": method,
                    "compatible_scripts": len(compatible_scripts),
                    "available": len(compatible_scripts) > 0
                }

            # Check dataset availability
            dataset_name = declarative_config.get("dataset", {}).get("name")
            if dataset_name:
                compatible_scripts = self.api.find_compatible_scripts(dataset=dataset_name)
                validation_result["dataset_compatibility"] = {
                    "dataset": dataset_name,
                    "compatible_scripts": len(compatible_scripts),
                    "available": len(compatible_scripts) > 0
                }

            return validation_result

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "config_path": config_path
            }

    def execute_declarative_experiment(self, config_path: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an experiment from declarative configuration.

        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional execution parameters

        Returns:
            Execution result
        """
        # Load and convert configuration
        api_config = self.load_and_convert_config(config_path)

        # Execute using API
        result = self.api.execute_experiment(api_config, **kwargs)

        logger.info(f"Successfully executed experiment from {config_path}")
        return result

    def suggest_config_from_template(self, method: str, dataset: str, **kwargs) -> Dict[str, Any]:
        """
        Generate API configuration based on method and dataset.

        Args:
            method: Editing method name
            dataset: Dataset name
            **kwargs: Additional configuration options

        Returns:
            API configuration dictionary
        """
        # Use API's suggest_config method
        base_config = self.api.suggest_config(method=method, dataset=dataset, **kwargs)

        # Enhance with declarative-style metadata
        base_config["metadata"] = {
            "source": "generated_from_api",
            "method": method,
            "dataset": dataset,
            "generated_at": str(Path(__file__).stat().st_mtime)
        }

        return base_config

    def get_available_configs(self, config_dir: str = None) -> List[str]:
        """
        Get list of available configuration files.

        Args:
            config_dir: Directory to search for config files

        Returns:
            List of configuration file paths
        """
        if config_dir is None:
            config_dir = str(Path(__file__).parent.parent / "config")

        config_path = Path(config_dir)
        if not config_path.exists():
            logger.warning(f"Config directory not found: {config_dir}")
            return []

        config_files = []
        for pattern in ["*.yaml", "*.yml"]:
            config_files.extend(config_path.glob(pattern))

        return [str(f) for f in config_files]

    def create_sample_config(self, output_path: str, method: str = "ROME", dataset: str = "ZsreDataset") -> None:
        """
        Create a sample configuration file.

        Args:
            output_path: Path to save the configuration file
            method: Editing method for the sample
            dataset: Dataset for the sample
        """
        # Generate API configuration
        api_config = self.suggest_config_from_template(method, dataset)

        # Convert to declarative format
        declarative_config = {
            "experiment": {
                "name": f"{method.lower()}_{dataset.lower()}_experiment",
                "description": f"Sample {method} experiment on {dataset}",
                "version": "1.0",
                "tags": [method, dataset]
            },
            "model": {
                "model_intent": api_config["model"]["model_intent"]
            },
            "editing": {
                "method": method,
                "intent": api_config["editing"]["intent"]
            },
            "dataset": {
                "name": dataset,
                "path": "./data",
                "type": "json"
            },
            "execution": {
                "mode": "sync",
                "resources": {
                    "timeout": 3600
                }
            }
        }

        # Save as YAML if possible, otherwise as JSON
        try:
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(declarative_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created YAML config: {output_path}")
        except ImportError:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(declarative_config, f, indent=2)
            logger.info(f"Created JSON config: {output_path} (YAML requires pyyaml)")


# Convenience functions
def load_config(config_path: str, api: EasyEditAPI = None) -> Dict[str, Any]:
    """Load and convert a declarative configuration."""
    loader = ConfigLoader(api)
    return loader.load_and_convert_config(config_path)


def execute_config(config_path: str, api: EasyEditAPI = None, **kwargs) -> Dict[str, Any]:
    """Execute an experiment from declarative configuration."""
    loader = ConfigLoader(api)
    return loader.execute_declarative_experiment(config_path, **kwargs)


def validate_config(config_path: str, api: EasyEditAPI = None) -> Dict[str, Any]:
    """Validate a declarative configuration."""
    loader = ConfigLoader(api)
    return loader.validate_config_compatibility(config_path)


if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader()

    # List available configs
    configs = loader.get_available_configs()
    print(f"Available configurations: {len(configs)}")

    if configs:
        # Validate first config
        validation = loader.validate_config_compatibility(configs[0])
        print(f"Validation result: {validation}")

        # Create sample config
        loader.create_sample_config("./sample_config.yaml")
        print("Created sample configuration")