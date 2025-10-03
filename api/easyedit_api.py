"""
EasyEdit API
============

Lightweight API for executing EasyEdit experiments through YAML configuration.
Provides unified interface for discovering, configuring, and executing run scripts.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from script_registry import ScriptRegistry, ScriptInfo
from config_mapper import ConfigMapper, MappingResult
from script_executor import ScriptExecutor, ExecutionResult, ExecutionTask
from config_bridge import ConfigBridge

logger = logging.getLogger(__name__)


class EasyEditAPI:
    """
    Main API interface for EasyEdit experiment execution.

    This API provides a simple interface for:
    - Auto-discovering available run scripts
    - Mapping YAML configurations to script parameters
    - Executing experiments with monitoring
    - Collecting and parsing results
    """

    def __init__(self,
                 examples_dir: str = None,
                 base_dir: str = None,
                 enable_async: bool = True):
        """
        Initialize EasyEdit API.

        Args:
            examples_dir: Directory containing run scripts (default: EasyEdit/examples)
            base_dir: Base directory for EasyEdit project
            enable_async: Whether to enable asynchronous execution
        """
        self.base_dir = base_dir or os.path.abspath(".")
        self.examples_dir = examples_dir or os.path.join(self.base_dir, "examples")

        # Initialize components
        self.script_registry = ScriptRegistry(self.examples_dir)
        self.config_mapper = ConfigMapper(self.script_registry)
        self.executor = ScriptExecutor(self.base_dir)
        self.config_bridge = ConfigBridge()  # Add configuration bridge

        self.enable_async = enable_async

        logger.info(f"EasyEdit API initialized with examples_dir: {self.examples_dir}")

    # Discovery Methods

    def list_available_scripts(self) -> List[Dict[str, Any]]:
        """
        List all available run scripts.

        Returns:
            List of script information dictionaries
        """
        scripts = self.script_registry.list_scripts()
        return [script.to_dict() for script in scripts]

    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific script.

        Args:
            script_name: Name of the script

        Returns:
            Script information or None if not found
        """
        script_info = self.script_registry.get_script(script_name)
        return script_info.to_dict() if script_info else None

    def find_compatible_scripts(self,
                              method: str = None,
                              dataset: str = None,
                              model: str = None) -> List[Dict[str, Any]]:
        """
        Find scripts compatible with given requirements.

        Args:
            method: Editing method name (e.g., "ROME", "FT")
            dataset: Dataset name (e.g., "ZsreDataset")
            model: Model name or pattern

        Returns:
            List of compatible script information
        """
        scripts = self.script_registry.find_compatible_scripts(
            method=method,
            dataset=dataset,
            model=model
        )
        return [script.to_dict() for script in scripts]

    # Configuration Methods

    def validate_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate YAML configuration without execution.

        Args:
            config: YAML file path or configuration dictionary

        Returns:
            Validation result with errors and warnings
        """
        try:
            # Load configuration if it's a file path
            if isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as f:
                    import yaml
                    config_dict = yaml.safe_load(f)
            else:
                config_dict = config

            # Validate using configuration bridge
            validation_result = self.config_bridge.validate_configuration(config_dict, "yaml")

            if not validation_result["valid"]:
                return {
                    "valid": False,
                    "script_name": None,
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                    "parameters": {}
                }

            # Map to script
            mapping_result = self.config_mapper.map_config_to_script(validation_result["validated_config"])
            return {
                "valid": mapping_result.success,
                "script_name": mapping_result.script_name,
                "errors": mapping_result.errors,
                "warnings": mapping_result.warnings,
                "parameters": mapping_result.script_parameters
            }
        except Exception as e:
            return {
                "valid": False,
                "script_name": None,
                "errors": [str(e)],
                "warnings": [],
                "parameters": {}
            }

    def map_config_to_parameters(self,
                                config: Union[str, Dict[str, Any]],
                                intent_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Map YAML configuration to script-specific parameters.

        Args:
            config: YAML file path or configuration dictionary
            intent_requirements: Additional intent requirements

        Returns:
            Mapped parameters dictionary
        """
        mapping_result = self.config_mapper.map_config_to_script(config, intent_requirements)

        if not mapping_result.success:
            raise ValueError(f"Configuration mapping failed: {'; '.join(mapping_result.errors)}")

        return {
            "script_name": mapping_result.script_name,
            "parameters": mapping_result.script_parameters,
            "errors": mapping_result.errors,
            "warnings": mapping_result.warnings
        }

    def create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration template.

        Returns:
            Default configuration dictionary
        """
        return self.config_mapper.create_default_config()

    # Execution Methods

    def execute_experiment(self,
                           config: Union[str, Dict[str, Any]],
                           async_execution: bool = None,
                           timeout: float = None) -> Union[Dict[str, Any], str]:
        """
        Execute an experiment from YAML configuration.

        This is the main method for running experiments.

        Args:
            config: YAML file path or configuration dictionary
            async_execution: Whether to execute asynchronously (default: API setting)
            timeout: Execution timeout in seconds

        Returns:
            Execution result if sync, task_id if async
        """
        try:
            # Map configuration to script parameters
            mapping_result = self.config_mapper.map_config_to_script(config)

            if not mapping_result.success:
                raise ValueError(f"Configuration mapping failed: {'; '.join(mapping_result.errors)}")

            # Determine execution mode
            if async_execution is None:
                async_execution = self.enable_async

            # Execute script
            result = self.executor.execute_script(
                script_name=mapping_result.script_name,
                parameters=mapping_result.script_parameters,
                async_execution=async_execution,
                timeout=timeout
            )

            # Format result
            if isinstance(result, str):
                # Async execution - return task_id
                return {
                    "task_id": result,
                    "script_name": mapping_result.script_name,
                    "status": "queued",
                    "async": True
                }
            else:
                # Sync execution - return result
                return {
                    "task_id": result.task_id,
                    "script_name": result.script_name,
                    "status": "completed",
                    "async": False,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "return_code": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_files": result.output_files,
                    "metrics": result.metrics,
                    "errors": mapping_result.errors,
                    "warnings": mapping_result.warnings
                }

        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            return {
                "task_id": f"error_{int(os.times()[4] * 1000)}",
                "status": "error",
                "success": False,
                "error": str(e)
            }

    def execute_script(self,
                      script_name: str,
                      parameters: Dict[str, Any],
                      async_execution: bool = None,
                      timeout: float = None) -> Union[Dict[str, Any], str]:
        """
        Execute a specific script with parameters.

        Args:
            script_name: Name of the script to execute
            parameters: Script parameters
            async_execution: Whether to execute asynchronously
            timeout: Execution timeout in seconds

        Returns:
            Execution result if sync, task_id if async
        """
        try:
            # Validate script exists
            script_info = self.script_registry.get_script(script_name)
            if not script_info:
                raise ValueError(f"Script not found: {script_name}")

            # Validate parameters
            is_valid, errors = self.script_registry.validate_parameters(script_name, parameters)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")

            # Execute script
            result = self.executor.execute_script(
                script_name=script_name,
                parameters=parameters,
                async_execution=async_execution if async_execution is not None else self.enable_async,
                timeout=timeout
            )

            # Format result
            if isinstance(result, str):
                return {
                    "task_id": result,
                    "script_name": script_name,
                    "status": "queued",
                    "async": True
                }
            else:
                return {
                    "task_id": result.task_id,
                    "script_name": script_name,
                    "status": "completed",
                    "async": False,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "return_code": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_files": result.output_files,
                    "metrics": result.metrics
                }

        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return {
                "task_id": f"error_{int(os.times()[4] * 1000)}",
                "script_name": script_name,
                "status": "error",
                "success": False,
                "error": str(e)
            }

    # Task Management Methods

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of an execution task.

        Args:
            task_id: Task ID

        Returns:
            Task status information
        """
        return self.executor.get_task_status(task_id)

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get result of a completed task.

        Args:
            task_id: Task ID

        Returns:
            Task result or None if not found
        """
        result = self.executor.get_task_result(task_id)
        return result.to_dict() if result else None

    def wait_for_completion(self, task_id: str, timeout: float = None) -> bool:
        """
        Wait for task completion.

        Args:
            task_id: Task ID
            timeout: Maximum time to wait (seconds)

        Returns:
            True if task completed successfully
        """
        return self.executor.wait_for_completion(task_id, timeout)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled
        """
        return self.executor.cancel_task(task_id)

    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """
        List all active tasks.

        Returns:
            List of active task information
        """
        return self.executor.list_active_tasks()

    # Utility Methods

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and capabilities.

        Returns:
            System information dictionary
        """
        return {
            "base_dir": self.base_dir,
            "examples_dir": self.examples_dir,
            "available_scripts": len(self.script_registry.scripts),
            "async_execution_enabled": self.enable_async,
            "active_tasks": len(self.executor.active_tasks),
            "completed_tasks": len(self.executor.task_results)
        }

    def save_script_registry(self, output_path: str):
        """
        Save script registry to file.

        Args:
            output_path: Path to save registry
        """
        self.script_registry.save_registry(output_path)

    def load_script_registry(self, input_path: str):
        """
        Load script registry from file.

        Args:
            input_path: Path to load registry from
        """
        self.script_registry.load_registry(input_path)

    def cleanup(self):
        """
        Cleanup resources.
        """
        self.executor.cleanup()
        logger.info("EasyEdit API cleaned up")

    # Enhanced Convenience Methods for Common Use Cases

    def quick_experiment(self,
                        method: str = "ROME",
                        model: str = "llama-7b",
                        dataset: str = "zsre",
                        hparams_dir: str = None,
                        data_dir: str = "./data",
                        **kwargs) -> Dict[str, Any]:
        """
        Quick experiment execution with common parameters.

        Args:
            method: Editing method (ROME, FT, MEMIT, etc.)
            model: Model name/path
            dataset: Dataset name
            hparams_dir: Hyperparameters directory
            data_dir: Data directory
            **kwargs: Additional parameters

        Returns:
            Execution result
        """
        # Build configuration
        config = {
            "model": {
                "model_intent": {
                    "purpose": "knowledge_editing",
                    "method": method,
                    "hparams_dir": hparams_dir or f"../hparams/{method}/llama-7b"
                }
            },
            "editing": {
                "intent": {
                    "goal": "knowledge_editing",
                    "method": method
                }
            },
            "dataset": {
                "data_intent": {
                    "data_dir": data_dir,
                    "dataset_type": dataset
                }
            }
        }

        # Add additional parameters
        if kwargs:
            config["execution"] = {
                "execution_intent": {
                    "additional_args": kwargs
                }
            }

        return self.execute_experiment(config)

    def batch_execute(self,
                     configs: List[Union[str, Dict[str, Any]]],
                     max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Execute multiple experiments in batch.

        Args:
            configs: List of configurations (file paths or dicts)
            max_concurrent: Maximum concurrent executions

        Returns:
            List of task results
        """
        if not self.enable_async:
            logger.warning("Async execution disabled, running experiments sequentially")
            return [self.execute_experiment(config) for config in configs]

        # Submit all tasks
        task_ids = []
        for config in configs:
            result = self.execute_experiment(config, async_execution=True)
            if isinstance(result, dict) and "task_id" in result:
                task_ids.append(result["task_id"])

        # Wait for completion with concurrency limit
        results = []
        completed = 0

        while completed < len(task_ids):
            active_count = 0
            for task_id in task_ids:
                status = self.get_task_status(task_id)
                if status["status"] in ["running", "queued"]:
                    active_count += 1
                elif status["status"] in ["completed", "failed"]:
                    if task_id not in [r.get("task_id") for r in results]:
                        result = self.get_task_result(task_id)
                        if result:
                            results.append(result)
                        completed += 1

            # Limit concurrent execution
            if active_count >= max_concurrent:
                import time
                time.sleep(1)

        return results

    def get_supported_methods(self) -> List[str]:
        """
        Get list of all supported editing methods.

        Returns:
            List of supported method names
        """
        methods = set()
        for script in self.script_registry.scripts.values():
            methods.update(script.supported_methods)
        return sorted(list(methods))

    def get_supported_datasets(self) -> List[str]:
        """
        Get list of all supported datasets.

        Returns:
            List of supported dataset names
        """
        datasets = set()
        for script in self.script_registry.scripts.values():
            datasets.update(script.supported_datasets)
        return sorted(list(datasets))

    def get_supported_models(self) -> List[str]:
        """
        Get list of all supported models.

        Returns:
            List of supported model names
        """
        models = set()
        for script in self.script_registry.scripts.values():
            models.update(script.supported_models)
        return sorted(list(models))

    def suggest_config(self,
                      method: str = None,
                      model: str = None,
                      dataset: str = None,
                      purpose: str = "knowledge_editing") -> Dict[str, Any]:
        """
        Suggest a configuration based on requirements.

        Args:
            method: Preferred editing method
            model: Preferred model
            dataset: Preferred dataset
            purpose: Experiment purpose

        Returns:
            Suggested configuration dictionary
        """
        # Find compatible scripts
        compatible_scripts = self.script_registry.find_compatible_scripts(
            method=method,
            dataset=dataset,
            model=model
        )

        if not compatible_scripts:
            # Fallback to any script
            compatible_scripts = self.script_registry.list_scripts()

        if compatible_scripts:
            best_script = compatible_scripts[0]

            # Build suggested configuration
            config = {
                "model": {
                    "model_intent": {
                        "purpose": purpose,
                        "method": method or (best_script.supported_methods[0] if best_script.supported_methods else "ROME"),
                        "architecture_preference": "auto",
                        "size_preference": "medium"
                    }
                },
                "editing": {
                    "intent": {
                        "goal": purpose,
                        "method": method or (best_script.supported_methods[0] if best_script.supported_methods else "ROME"),
                        "strategy": "precise_localization"
                    }
                },
                "dataset": {
                    "data_intent": {
                        "data_dir": "./data",
                        "ds_size": 1000
                    }
                },
                "execution": {
                    "execution_intent": {
                        "metrics_save_dir": "./output",
                        "quality_requirements": {
                            "success_rate_threshold": 0.8
                        },
                        "efficiency_requirements": {
                            "max_execution_time": 3600
                        }
                    }
                }
            }

            return config

        return self.create_default_config()

    def validate_method_dataset_combination(self,
                                          method: str,
                                          dataset: str,
                                          model: str = None) -> Dict[str, Any]:
        """
        Validate if a method-dataset-model combination is supported.

        Args:
            method: Editing method
            dataset: Dataset name
            model: Model name (optional)

        Returns:
            Validation result
        """
        compatible_scripts = self.script_registry.find_compatible_scripts(
            method=method,
            dataset=dataset,
            model=model
        )

        return {
            "valid": len(compatible_scripts) > 0,
            "compatible_scripts": [script.name for script in compatible_scripts],
            "suggested_script": compatible_scripts[0].name if compatible_scripts else None,
            "warnings": [] if compatible_scripts else ["No compatible scripts found"]
        }

    def get_script_parameters(self, script_name: str) -> Dict[str, Any]:
        """
        Get parameter information for a script.

        Args:
            script_name: Name of the script

        Returns:
            Parameter information
        """
        return self.script_registry.get_script_parameters(script_name)


# Convenience function for quick initialization
def get_api(examples_dir: str = None, base_dir: str = None) -> EasyEditAPI:
    """
    Get EasyEdit API instance with default configuration.

    Args:
        examples_dir: Examples directory path
        base_dir: Base directory path

    Returns:
        EasyEditAPI instance
    """
    return EasyEditAPI(examples_dir=examples_dir, base_dir=base_dir)