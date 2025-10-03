"""
Configuration Bridge
===================

Bridges the gap between API and YAML configuration layers.
Provides unified configuration management and ensures consistency.
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigBridge:
    """
    Bridge between API and YAML configuration layers.

    This class ensures consistency between API expectations and YAML structure,
    providing unified configuration management.
    """

    def __init__(self, bridge_config_path: str = None):
        """
        Initialize the configuration bridge.

        Args:
            bridge_config_path: Path to the bridge configuration file
        """
        if bridge_config_path is None:
            bridge_config_path = os.path.join(
                os.path.abspath("."),
                "config",
                "unified_config_bridge.yaml"
            )

        self.bridge_config_path = bridge_config_path
        self.bridge_config = self._load_bridge_config()
        self.config_cache = {}

    def _load_bridge_config(self) -> Dict[str, Any]:
        """Load the bridge configuration."""
        try:
            with open(self.bridge_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load bridge config: {e}")
            return self._get_default_bridge_config()

    def _get_default_bridge_config(self) -> Dict[str, Any]:
        """Get default bridge configuration."""
        return {
            "config_mappings": {
                "api_to_yaml": {},
                "yaml_to_api": {}
            },
            "parameter_validation": {
                "required_parameters": {
                    "experiment_execution": [
                        "editing.method",
                        "dataset.data_dir"
                    ]
                }
            },
            "default_values": {
                "common": {
                    "device": "0",
                    "metrics_save_dir": "./results/metrics"
                }
            }
        }

    def api_to_yaml_config(self, api_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert API configuration to YAML configuration format.

        Args:
            api_config: API configuration dictionary

        Returns:
            YAML-compatible configuration dictionary
        """
        yaml_config = {
            "experiment": {
                "name": api_config.get("experiment_name", "api_generated_experiment"),
                "description": "Generated from API configuration",
                "version": "1.0"
            },
            "model": {
                "model_intent": {}
            },
            "editing": {
                "intent": {}
            },
            "dataset": {
                "data_intent": {}
            },
            "execution": {
                "execution_intent": {}
            }
        }

        # Apply mappings from bridge configuration
        mappings = self.bridge_config.get("config_mappings", {}).get("api_to_yaml", {})

        # Map model configuration
        if "model" in api_config:
            self._map_api_to_yaml_section(
                api_config["model"],
                yaml_config["model"]["model_intent"],
                mappings.get("model_intent", {})
            )

        # Map editing configuration
        if "editing" in api_config:
            self._map_api_to_yaml_section(
                api_config["editing"],
                yaml_config["editing"]["intent"],
                mappings.get("editing_intent", {})
            )

        # Map dataset configuration
        if "dataset" in api_config:
            self._map_api_to_yaml_section(
                api_config["dataset"],
                yaml_config["dataset"]["data_intent"],
                mappings.get("dataset_intent", {})
            )

        # Map execution configuration
        if "execution" in api_config:
            self._map_api_to_yaml_section(
                api_config["execution"],
                yaml_config["execution"]["execution_intent"],
                mappings.get("execution_intent", {})
            )

        return yaml_config

    def yaml_to_api_config(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert YAML configuration to API configuration format.

        Args:
            yaml_config: YAML configuration dictionary

        Returns:
            API-compatible configuration dictionary
        """
        api_config = {}

        # Apply mappings from bridge configuration
        mappings = self.bridge_config.get("config_mappings", {}).get("yaml_to_api", {})

        # Extract model configuration
        if "model" in yaml_config and "model_intent" in yaml_config["model"]:
            api_config["model"] = self._map_yaml_to_api_section(
                yaml_config["model"]["model_intent"],
                mappings.get("model", {}).get("model_intent", {})
            )

        # Extract editing configuration
        if "editing" in yaml_config and "intent" in yaml_config["editing"]:
            api_config["editing"] = self._map_yaml_to_api_section(
                yaml_config["editing"]["intent"],
                mappings.get("editing", {}).get("intent", {})
            )

        # Extract dataset configuration
        if "dataset" in yaml_config and "data_intent" in yaml_config["dataset"]:
            api_config["dataset"] = self._map_yaml_to_api_section(
                yaml_config["dataset"]["data_intent"],
                mappings.get("dataset", {}).get("data_intent", {})
            )

        # Extract execution configuration
        if "execution" in yaml_config and "execution_intent" in yaml_config["execution"]:
            api_config["execution"] = self._map_yaml_to_api_section(
                yaml_config["execution"]["execution_intent"],
                mappings.get("execution", {}).get("execution_intent", {})
            )

        return api_config

    def _map_api_to_yaml_section(self,
                                api_section: Dict[str, Any],
                                yaml_section: Dict[str, Any],
                                mappings: Dict[str, str]):
        """Map a single section from API to YAML format."""
        for api_key, yaml_key in mappings.items():
            if api_key in api_section:
                # Handle nested mappings
                if '.' in yaml_key:
                    keys = yaml_key.split('.')
                    current = yaml_section
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = api_section[api_key]
                else:
                    yaml_section[yaml_key] = api_section[api_key]

    def _map_yaml_to_api_section(self,
                                yaml_section: Dict[str, Any],
                                mappings: Dict[str, str]):
        """Map a single section from YAML to API format."""
        api_section = {}

        for yaml_key, api_key in mappings.items():
            if isinstance(api_key, dict):
                # Handle nested mappings
                for nested_key, nested_api_key in api_key.items():
                    if nested_key in yaml_section:
                        api_section[nested_api_key] = yaml_section[nested_key]
            else:
                if yaml_key in yaml_section:
                    api_section[api_key] = yaml_section[yaml_key]

        return api_section

    def validate_configuration(self,
                              config: Dict[str, Any],
                              config_type: str = "yaml") -> Dict[str, Any]:
        """
        Validate configuration against bridge rules.

        Args:
            config: Configuration to validate
            config_type: Type of configuration ("yaml" or "api")

        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []

        # Get required parameters for configuration type
        required_params = self.bridge_config.get("parameter_validation", {}).get("required_parameters", {})
        required = required_params.get(f"{config_type}_execution", [])

        # Check required parameters
        for param_path in required:
            if not self._check_parameter_exists(config, param_path):
                errors.append(f"Missing required parameter: {param_path}")

        # Apply default values
        config = self._apply_default_values(config)

        # Validate parameter types
        type_errors = self._validate_parameter_types(config)
        errors.extend(type_errors)

        # Validate paths
        path_errors = self._validate_paths(config)
        errors.extend(path_errors)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_config": config
        }

    def _check_parameter_exists(self, config: Dict[str, Any], param_path: str) -> bool:
        """Check if a parameter exists in the configuration using dot notation."""
        keys = param_path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False

        return True

    def _apply_default_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values from bridge configuration."""
        defaults = self.bridge_config.get("default_values", {})

        # Apply common defaults
        for key, value in defaults.get("common", {}).items():
            if not self._check_parameter_exists(config, key):
                self._set_parameter_value(config, key, value)

        return config

    def _set_parameter_value(self, config: Dict[str, Any], param_path: str, value: Any):
        """Set a parameter value using dot notation."""
        keys = param_path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _validate_parameter_types(self, config: Dict[str, Any]) -> List[str]:
        """Validate parameter types."""
        errors = []
        type_validation = self.bridge_config.get("parameter_validation", {}).get("type_validation", {})

        # This is a simplified type validation
        # In a full implementation, you would check each parameter against its expected type

        return errors

    def _validate_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate file paths in configuration."""
        errors = []

        # Common path parameters to validate
        path_params = [
            "model.model_intent.hparams_dir",
            "dataset.data_intent.data_dir",
            "execution.execution_intent.metrics_save_dir"
        ]

        for param_path in path_params:
            if self._check_parameter_exists(config, param_path):
                param_value = self._get_parameter_value(config, param_path)
                if isinstance(param_value, str):
                    # Check if path exists or can be created
                    path = Path(param_value)
                    if not path.parent.exists():
                        errors.append(f"Parent directory does not exist for path: {param_value}")

        return errors

    def _get_parameter_value(self, config: Dict[str, Any], param_path: str) -> Any:
        """Get a parameter value using dot notation."""
        keys = param_path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def get_script_parameters(self,
                             script_name: str,
                             yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get script-specific parameters from YAML configuration.

        Args:
            script_name: Name of the script
            yaml_config: YAML configuration

        Returns:
            Script-specific parameters
        """
        script_params = {}

        # Extract common parameters based on script patterns
        if "zsre" in script_name:
            script_params = self._extract_zsre_params(yaml_config)
        elif "knowedit" in script_name:
            script_params = self._extract_knowedit_params(yaml_config)
        elif "wise" in script_name.lower():
            script_params = self._extract_wise_params(yaml_config)
        else:
            script_params = self._extract_generic_params(yaml_config)

        return script_params

    def _extract_zsre_params(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for ZsRE scripts."""
        params = {}

        # Common parameters
        if "editing" in yaml_config and "intent" in yaml_config["editing"]:
            params["editing_method"] = yaml_config["editing"]["intent"].get("method", "ROME")

        if "dataset" in yaml_config and "data_intent" in yaml_config["dataset"]:
            params["data_dir"] = yaml_config["dataset"]["data_intent"].get("data_dir", "./data")
            params["ds_size"] = yaml_config["dataset"]["data_intent"].get("ds_size", 1000)

        if "model" in yaml_config and "model_intent" in yaml_config["model"]:
            params["hparams_dir"] = yaml_config["model"]["model_intent"].get("hparams_dir", "./hparams/ROME/llama-7b")

        if "execution" in yaml_config and "execution_intent" in yaml_config["execution"]:
            params["metrics_save_dir"] = yaml_config["execution"]["execution_intent"].get("metrics_save_dir", "./results/metrics")

        return params

    def _extract_knowedit_params(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for KnowEdit scripts."""
        params = self._extract_zsre_params(yaml_config)  # KnowEdit uses similar parameters
        return params

    def _extract_wise_params(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for WISE scripts."""
        params = self._extract_zsre_params(yaml_config)  # WISE uses similar parameters
        params["editing_method"] = "WISE"  # Override method for WISE
        return params

    def _extract_generic_params(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generic parameters for any script."""
        return self._extract_zsre_params(yaml_config)  # Use ZsRE as baseline

    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve relative paths in configuration.

        Args:
            config: Configuration with potentially relative paths

        Returns:
            Configuration with resolved paths
        """
        path_resolution = self.bridge_config.get("path_resolution", {})
        base_paths = path_resolution.get("base_paths", {})

        # Resolve common paths
        path_params = [
            ("model.model_intent.hparams_dir", "hparams"),
            ("dataset.data_intent.data_dir", "data"),
            ("execution.execution_intent.metrics_save_dir", "results")
        ]

        for param_path, base_key in path_params:
            if self._check_parameter_exists(config, param_path):
                param_value = self._get_parameter_value(config, param_path)
                if isinstance(param_value, str) and not os.path.isabs(param_value):
                    # Resolve relative to base path
                    base_path = base_paths.get(base_key, ".")
                    resolved_path = os.path.abspath(os.path.join(base_path, param_value))
                    self._set_parameter_value(config, param_path, resolved_path)

        return config

    def get_fallback_strategy(self,
                             method: str,
                             error_type: str) -> Dict[str, Any]:
        """
        Get fallback strategy for method and error type.

        Args:
            method: Editing method
            error_type: Type of error

        Returns:
            Fallback strategy dictionary
        """
        error_handling = self.bridge_config.get("error_handling", {})

        if error_type == "method_failure":
            fallbacks = error_handling.get("method_fallbacks", {})
            return {
                "fallback_methods": fallbacks.get(method, ["FT"]),
                "strategy": "try_alternative_method"
            }
        elif error_type == "script_failure":
            fallbacks = error_handling.get("script_fallbacks", {})
            return {
                "fallback_scripts": fallbacks.get(method, ["run_zsre_llama2"]),
                "strategy": "try_alternative_script"
            }
        else:
            recovery_strategies = error_handling.get("recovery_strategies", {})
            return {
                "strategy": recovery_strategies.get(error_type, "log_error_and_continue")
            }

    def get_version_compatibility(self, config_version: str) -> Dict[str, Any]:
        """
        Check version compatibility for configuration.

        Args:
            config_version: Version of the configuration

        Returns:
            Version compatibility information
        """
        version_compat = self.bridge_config.get("version_compatibility", {})
        supported_versions = version_compat.get("supported_versions", [])

        return {
            "compatible": config_version in supported_versions,
            "supported_versions": supported_versions,
            "features": version_compat.get("version_features", {}).get(config_version, []),
            "migration_path": version_compat.get("migrations", {}).get(f"{config_version}_to_latest", [])
        }