"""
Script Registry
===============

Auto-discovers and manages run scripts in the examples directory.
Provides script metadata and parameter mapping capabilities.
"""

import os
import re
import importlib.util
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScriptInfo:
    """Information about a run script"""
    name: str
    file_path: str
    description: str = ""
    supported_methods: List[str] = None
    supported_datasets: List[str] = None
    supported_models: List[str] = None
    required_parameters: List[str] = None
    optional_parameters: Dict[str, Any] = None
    examples_dir: str = ""

    def __post_init__(self):
        if self.supported_methods is None:
            self.supported_methods = []
        if self.supported_datasets is None:
            self.supported_datasets = []
        if self.supported_models is None:
            self.supported_models = []
        if self.required_parameters is None:
            self.required_parameters = []
        if self.optional_parameters is None:
            self.optional_parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "description": self.description,
            "supported_methods": self.supported_methods,
            "supported_datasets": self.supported_datasets,
            "supported_models": self.supported_models,
            "required_parameters": self.required_parameters,
            "optional_parameters": self.optional_parameters
        }


class ScriptRegistry:
    """
    Registry for managing run scripts.

    Auto-discovers scripts in the examples directory and provides
    metadata and parameter mapping capabilities.
    """

    def __init__(self, examples_dir: str = None):
        self.examples_dir = examples_dir or os.path.join(os.path.abspath("."), "examples")
        self.scripts: Dict[str, ScriptInfo] = {}
        self._script_patterns = [
            r"run_(.+?)\.py$",  # run_*.py scripts
        ]
        self._load_scripts()

    def _load_scripts(self):
        """Auto-discover and load script information"""
        try:
            examples_path = Path(self.examples_dir)
            if not examples_path.exists():
                logger.warning(f"Examples directory not found: {self.examples_dir}")
                return

            # Find all Python scripts matching patterns with more comprehensive discovery
            script_files = []

            # Use patterns to find scripts
            for pattern in self._script_patterns:
                # Handle glob patterns properly
                if '*' in pattern:
                    script_files.extend(examples_path.glob(pattern))
                else:
                    script_files.extend(examples_path.glob(f"*{pattern}"))

            # Specifically look for run_*.py files (most common pattern)
            run_scripts = examples_path.glob("run_*.py")
            script_files.extend(run_scripts)

            # Also look for edit_*.py files
            edit_scripts = examples_path.glob("edit_*.py")
            script_files.extend(edit_scripts)

            # Look for test_*.py files that might be runnable examples
            test_scripts = examples_path.glob("test_*.py")
            script_files.extend(test_scripts)

            # Look for demo_*.py files
            demo_scripts = examples_path.glob("demo_*.py")
            script_files.extend(demo_scripts)

            # Also scan for any Python files that might be examples
            all_py_files = examples_path.glob("*.py")
            for py_file in all_py_files:
                # Skip common non-script files
                if py_file.name not in ["__init__.py", "setup.py", "config.py", "utils.py"]:
                    script_files.append(py_file)

            # Remove duplicates while preserving order
            seen_files = set()
            unique_script_files = []
            for script_file in script_files:
                if script_file not in seen_files:
                    seen_files.add(script_file)
                    unique_script_files.append(script_file)

            # Analyze each script
            loaded_count = 0
            for script_file in unique_script_files:
                if script_file.is_file():
                    try:
                        script_info = self._analyze_script(script_file)
                        if script_info:
                            self.scripts[script_info.name] = script_info
                            loaded_count += 1
                            logger.info(f"Discovered script: {script_info.name}")
                    except Exception as e:
                        logger.warning(f"Failed to analyze script {script_file}: {e}")

            logger.info(f"Loaded {loaded_count} scripts from {self.examples_dir}")

        except Exception as e:
            logger.error(f"Failed to load scripts: {e}")
            # Try to continue with empty scripts rather than crashing
            self.scripts = {}

    def _analyze_script(self, script_file: Path) -> Optional[ScriptInfo]:
        """Analyze a script file to extract metadata"""
        try:
            script_name = script_file.stem
            # Try multiple encodings for robustness
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            script_content = None

            for encoding in encodings:
                try:
                    script_content = script_file.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use errors='ignore' as last resort
            if script_content is None:
                script_content = script_file.read_text(encoding='utf-8', errors='ignore')

            # Extract basic information
            script_info = ScriptInfo(
                name=script_name,
                file_path=str(script_file),
                examples_dir=self.examples_dir
            )

            # Parse script content for metadata
            script_info.supported_methods = self._extract_supported_methods(script_content)
            script_info.supported_datasets = self._extract_supported_datasets(script_content)
            script_info.supported_models = self._extract_supported_models(script_content)
            script_info.required_parameters = self._extract_required_parameters(script_content)
            script_info.optional_parameters = self._extract_optional_parameters(script_content)
            script_info.description = self._extract_description(script_content)

            return script_info

        except Exception as e:
            logger.error(f"Failed to analyze script {script_file}: {e}")
            return None

    def _extract_supported_methods(self, content: str) -> List[str]:
        """Extract supported editing methods from script content"""
        methods = []

        # Look for method imports and references
        method_patterns = [
            r"from easyeditor import \((.*?)\)",  # Import statements
            r"editing_hparams = (\w+)HyperParams",  # HyperParams assignments
            r"elif.*editing_method == ['\"](\w+)['\"]",  # Method conditionals
            r"ALG_DICT\s*=\s*{([^}]+)}",  # Algorithm dictionaries
            r"editing_method.*default.*?['\"](\w+)['\"]",  # Default editing methods
            r"methods\s*=\s*\[(.*?)\]",  # Method lists
            r"available_methods\s*=\s*\[(.*?)\]",  # Available method lists
        ]

        # Common method names in EasyEdit
        common_methods = [
            "ROME", "FT", "MEMIT", "KN", "PMET", "DINM", "SERAC", "IKE",
            "GRACE", "MELO", "WISE", "MEND", "InstructEdit", "MALMEN",
            "AdaLoRA", "AlphaEdit", "ConvsEnt", "SafeEdit", "UltraEdit",
            "CKnowEdit", "HalluEditBench", "LLMEval", "WikiBigEdit", "AdsEdit",
            "ConceptEdit", "PersonalityEdit", "SafetyEdit", "WISEEdit"
        ]

        for pattern in method_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if isinstance(match, str):
                    # Extract method names from the match
                    method_names = re.findall(r"(\w+)HyperParams|['\"](\w+)['\"]", match)
                    for mn in method_names:
                        method_name = mn[0] or mn[1]
                        if method_name and method_name not in methods:
                            methods.append(method_name)

        # Also look for direct method name mentions
        for method in common_methods:
            if method in content and method not in methods:
                methods.append(method)

        return list(set(methods))

    def _extract_supported_datasets(self, content: str) -> List[str]:
        """Extract supported datasets from script content"""
        datasets = []

        # Look for dataset imports and references
        dataset_patterns = [
            r"from easyeditor import (\w+Dataset)",  # Dataset imports
            r"(\w+)Dataset\(",  # Dataset usage
            r"args\.data_dir.*['\"]([^'\"]+)['\"]",  # Data directory references
            r"dataset\s*=\s*(\w+Dataset)\(",  # Dataset assignments
            r"train_dataset\s*=\s*(\w+Dataset)\(",  # Training dataset assignments
            r"test_dataset\s*=\s*(\w+Dataset)\(",  # Test dataset assignments
        ]

        # Common dataset names in EasyEdit
        common_datasets = [
            "ZsreDataset", "KnowEditDataset", "WikiBioDataset", "CounterFactDataset",
            "ConvSentDataset", "LongformDataset", "PersonalityDataset", "SafetyDataset",
            "ConceptDataset", "WikiBigEditDataset", "AdsEditDataset", "CKnowEditDataset",
            "HalluDataset", "LLMDataset", "WISEDataset", "UltraEditDataset"
        ]

        for pattern in dataset_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    dataset_name = match[0] or match[1]
                else:
                    dataset_name = match

                if dataset_name and dataset_name not in datasets:
                    datasets.append(dataset_name)

        # Also look for direct dataset name mentions
        for dataset in common_datasets:
            if dataset in content and dataset not in datasets:
                datasets.append(dataset)

        return list(set(datasets))

    def _extract_supported_models(self, content: str) -> List[str]:
        """Extract supported models from script content"""
        models = []

        # Look for model references
        model_patterns = [
            r"args\.model_name.*['\"]([^'\"]+)['\"]",  # Model name arguments
            r"model_path.*['\"]([^'\"]+(?:gpt|llama|gpt-j|bert|t5|mistral|baichuan|chatglm|internlm|qwen)[^'\"]*)['\"]",  # Model paths
            r"args\.model.*['\"]([^'\"]+)['\"]",  # Model arguments
            r"(gpt-2|gpt2|llama|gpt-j|bert|t5|mistral|baichuan|chatglm|internlm|qwen)",  # Model name patterns
            r"(vicuna|alpaca|koala|wizard)",  # Fine-tuned model names
        ]

        # Common model names and patterns
        common_models = [
            "gpt2", "gpt2-xl", "gpt-j", "gpt-j-6b", "llama", "llama-7b", "llama-13b", "llama-30b", "llama-65b",
            "mistral", "mistral-7b", "baichuan", "baichuan-7b", "chatglm", "chatglm-6b", "chatglm2-6b",
            "internlm", "internlm-7b", "qwen", "qwen-7b", "qwen-14b", "vicuna", "vicuna-7b", "vicuna-13b",
            "alpaca", "alpaca-7b", "koala", "koala-13b", "wizard", "wizard-13b"
        ]

        for pattern in model_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match and match not in models:
                    models.append(match.lower())

        # Also look for direct model name mentions
        for model in common_models:
            if model.lower() in content.lower() and model.lower() not in models:
                models.append(model.lower())

        return list(set(models))

    def _extract_required_parameters(self, content: str) -> List[str]:
        """Extract required parameters from script content"""
        required_params = []

        # Look for argparse required arguments with more comprehensive patterns
        arg_patterns = [
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?required=True",  # Required args
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default=None",  # Args with None default
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?required=\s*True",  # Required args with spaces
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?help=.*?required",  # Help text mentions required
            r"args\.(\w+)\s*=\s*args\.\w+\s*or\s*None",  # Pattern indicating required fallback
        ]

        # Common required parameters in EasyEdit scripts (expanded)
        common_required = [
            "editing_method", "hparams_dir", "data_dir", "model_name", "model_path",
            "editing_name", "dataset_name", "output_dir", "results_dir", "config",
            "model_path", "editing_method", "hparams_dir", "data_dir", "output_dir",
            "model_path", "pretrain_path", "editing_method", "hparams_dir", "data_dir",
            "model_path", "editing_method", "hparams_dir", "data_dir", "output_dir",
            "model_path", "editing_method", "hparams_dir", "data_dir", "output_dir"
        ]

        for pattern in arg_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                param_name = match.replace("--", "")
                if param_name not in required_params:
                    required_params.append(param_name)

        # Also check for commonly required parameters that might not be explicitly marked
        for param in common_required:
            if param in content and param not in required_params:
                # Check if it's used in a way that suggests it's required
                if f"args.{param}" in content:
                    required_params.append(param)

        # Look for parameter usage patterns that indicate requirement
        usage_patterns = [
            r"if\s+args\.(\w+)\s+is\s+None:",  # None check pattern
            r"args\.(\w+)\s*=\s*args\.(\w+)\s*or\s*['\"]([^'\"]+)['\"]",  # Fallback pattern
        ]

        for pattern in usage_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    param_name = match[0]
                else:
                    param_name = match
                if param_name not in required_params:
                    required_params.append(param_name)

        # Remove duplicates and return
        return list(set(required_params))

    def _extract_optional_parameters(self, content: str) -> Dict[str, Any]:
        """Extract optional parameters from script content"""
        optional_params = {}

        # Look for argparse optional arguments with more comprehensive patterns
        arg_patterns = [
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default=([^,\)]+)",  # Standard default pattern
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default\s*=\s*([^,\)]+)",  # Default with spaces
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default\s*=\s*['\"]([^'\"]+)['\"]",  # String defaults
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default\s*=\s*(\d+)",  # Numeric defaults
            r"parser\.add_argument\(['\"](--[^'\"]+)['\"].*?default\s*=\s*(True|False)",  # Boolean defaults
        ]

        # Common optional parameters in EasyEdit scripts with their typical defaults (expanded)
        common_optional = {
            # Dataset and training parameters
            "ds_size": 10000,
            "batch_size": 1,
            "eval_batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "max_length": 512,
            "sequential_edit": False,

            # Device and execution parameters
            "device": "cuda",
            "gpu_id": 0,
            "cpu_only": False,
            "use_fp16": False,
            "use_bf16": False,

            # Output and logging parameters
            "metrics_save_dir": "./results",
            "output_dir": "./output",
            "results_dir": "./results",
            "logging_level": "INFO",
            "log_dir": "./logs",
            "save_ckpt": False,
            "ckpt_dir": "./checkpoints",
            "eval_only": False,
            "use_cache": True,
            "cache_dir": "./cache",

            # Model parameters
            "model_parallel": False,
            "load_in_8bit": False,
            "load_in_4bit": False,
            "trust_remote_code": False,

            # Experiment parameters
            "seed": 42,
            "verbose": False,
            "debug": False,
            "dry_run": False,

            # Generation parameters
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 100,
            "do_sample": True,
            "num_beams": 1,

            # Evaluation parameters
            "use_wandb": False,
            "wandb_project": "easyedit",
            "eval_steps": 100,
            "save_steps": 1000,

            # Memory and optimization
            "gradient_checkpointing": False,
            "use_flash_attention": False,
            "memory_efficient": False
        }

        for pattern in arg_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) >= 2:
                    param_name = match[0].replace("--", "")
                    default_value = match[1]

                    # Clean up default value
                    default_value = default_value.strip().strip("'\"")

                    # Type conversion
                    if default_value.lower() in ['true', 'false']:
                        default_value = default_value.lower() == 'true'
                    elif default_value.isdigit():
                        default_value = int(default_value)
                    elif default_value.replace('.', '', 1).isdigit():
                        default_value = float(default_value)
                    elif default_value == 'None':
                        continue  # Skip None values as they indicate required parameters

                    if param_name not in optional_params:
                        optional_params[param_name] = default_value

        # Add common optional parameters if they appear in the script but weren't found
        for param, default_val in common_optional.items():
            if param in content and param not in optional_params:
                # Check if the parameter is actually used in the script
                if f"args.{param}" in content or f"config.{param}" in content:
                    optional_params[param] = default_val

        return optional_params

    def _extract_description(self, content: str) -> str:
        """Extract description from script content"""
        # Look for docstring or comments with more comprehensive patterns
        docstring_patterns = [
            r'"""(.*?)"""',  # Triple double quotes
            r"'''(.*?)'''",  # Triple single quotes
            r'"""([^"]+)"""',  # Triple double quotes without newlines
            r"'''([^']+)'''",  # Triple single quotes without newlines
        ]

        for pattern in docstring_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Clean up the docstring
                description = matches[0].strip()
                # Remove common docstring elements like parameters, returns, etc.
                description = re.sub(r'\n\s*[Aa]rgs?:.*?\n', '\n', description)
                description = re.sub(r'\n\s*[Rr]eturns?:.*?\n', '\n', description)
                description = re.sub(r'\n\s*[Nn]otes?:.*?\n', '\n', description)
                description = re.sub(r'\n\s*[Ee]xamples?:.*?\n', '\n', description)
                description = re.sub(r'\n\s*-{3,}.*?\n', '\n', description)
                description = ' '.join(line.strip() for line in description.split('\n') if line.strip())
                if description and len(description) > 10:  # Minimum length check
                    return description

        # Look for file header comments
        header_patterns = [
            r"#.*?([A-Z][^#]*)",  # Capital letter after comment
            r"#.*?([a-zA-Z][^#]{20,})",  # Longer text after comment
            r"#\s*([A-Z][a-zA-Z\s]{10,})",  # Multi-word description
            r"#\s*(.*?run.*?script)",  # Common pattern
            r"#\s*(.*?example.*)",  # Common pattern
            r"#\s*(.*?demo.*)",  # Common pattern
        ]

        for pattern in header_patterns:
            matches = re.findall(pattern, content)
            if matches:
                description = matches[0].strip()
                # Clean up common patterns
                description = re.sub(r'^#+\s*', '', description)
                description = re.sub(r'\s*#+\s*$', '', description)
                if description and len(description) > 10:
                    return description

        # Look for specific comment patterns that indicate purpose
        purpose_patterns = [
            r"#\s*(This script|Script to|Example of|Demonstrates|Shows how to)\s*([^#]+)",
            r"#\s*(Run|Execute|Perform|Test|Evaluate)\s*([^#]+)",
            r"#\s*(Knowledge|Model|Editing|Experiment)\s*([^#]+)",
        ]

        for pattern in purpose_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                description = ' '.join(matches[0]).strip()
                if description and len(description) > 10:
                    return description

        # Generate a description based on script content if nothing else found
        methods = self._extract_supported_methods(content)
        if methods:
            return f"Script for {', '.join(methods[:3])} editing methods"

        return ""

    def get_script(self, script_name: str) -> Optional[ScriptInfo]:
        """Get script information by name"""
        return self.scripts.get(script_name)

    def list_scripts(self) -> List[ScriptInfo]:
        """List all discovered scripts"""
        return list(self.scripts.values())

    def find_compatible_scripts(self,
                              method: str = None,
                              dataset: str = None,
                              model: str = None) -> List[ScriptInfo]:
        """Find scripts compatible with given requirements"""
        compatible_scripts = []

        for script_info in self.scripts.values():
            is_compatible = True

            # Check method compatibility
            if method and method not in script_info.supported_methods:
                is_compatible = False

            # Check dataset compatibility
            if dataset and dataset not in script_info.supported_datasets:
                # Some scripts might support generic datasets
                if "ZsreDataset" not in script_info.supported_datasets and \
                   "KnowEditDataset" not in script_info.supported_datasets:
                    is_compatible = False

            # Check model compatibility
            if model:
                model_lower = model.lower()
                script_models = [m.lower() for m in script_info.supported_models]
                if not any(model_m in script_model for model_m in [model_lower] for script_model in script_models):
                    is_compatible = False

            if is_compatible:
                compatible_scripts.append(script_info)

        return compatible_scripts

    def get_script_parameters(self, script_name: str) -> Dict[str, Any]:
        """Get parameter information for a script"""
        script_info = self.get_script(script_name)
        if not script_info:
            return {}

        return {
            "required": script_info.required_parameters,
            "optional": script_info.optional_parameters
        }

    def validate_parameters(self,
                           script_name: str,
                           parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameters for a script"""
        script_info = self.get_script(script_name)
        if not script_info:
            return False, [f"Script '{script_name}' not found"]

        errors = []

        # Check required parameters (but be lenient for optional ones)
        for required_param in script_info.required_parameters:
            # Skip commonly optional parameters
            if required_param not in parameters:
                # These parameters are often optional in practice
                optional_params = ['ds_size', 'metrics_save_dir', 'result_path']
                if required_param not in optional_params:
                    errors.append(f"Required parameter '{required_param}' is missing")

        # Check parameter types and values
        for param_name, param_value in parameters.items():
            # Basic validation
            if param_value is None and param_name in script_info.required_parameters:
                errors.append(f"Required parameter '{param_name}' cannot be None")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary"""
        return {
            "examples_dir": self.examples_dir,
            "scripts_count": len(self.scripts),
            "scripts": {name: info.to_dict() for name, info in self.scripts.items()}
        }

    def save_registry(self, output_path: str):
        """Save registry to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved script registry to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def load_registry(self, input_path: str):
        """Load registry from JSON file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.scripts = {}
            for script_name, script_data in data.get("scripts", {}).items():
                self.scripts[script_name] = ScriptInfo(**script_data)

            logger.info(f"Loaded script registry from {input_path}")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")