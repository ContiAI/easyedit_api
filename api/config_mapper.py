"""
Configuration Mapper
===================

Maps YAML configuration to script-specific parameters.
Handles configuration validation, normalization, and conversion.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    """Result of configuration mapping"""
    success: bool
    script_name: str
    script_parameters: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class ConfigMapper:
    """
    Maps declarative YAML configuration to script-specific parameters.

    Features:
    - Intent to script selection
    - Universal parameter mapping
    - Environment variable substitution
    - Configuration validation
    """

    def __init__(self, script_registry=None):
        self.script_registry = script_registry

    def map_config_to_script(self,
                            yaml_config: Union[str, Dict[str, Any]],
                            intent_requirements: Dict[str, Any] = None) -> MappingResult:
        """
        Map YAML configuration to script-specific parameters.

        Args:
            yaml_config: YAML file path or configuration dictionary
            intent_requirements: Additional intent-based requirements

        Returns:
            MappingResult with script parameters and validation info
        """
        errors = []
        warnings = []

        try:
            # Load configuration
            if isinstance(yaml_config, str):
                config = self._load_config_file(yaml_config)
            else:
                config = yaml_config

            # Process environment variables
            config = self._process_environment_variables(config)

            # Validate configuration structure
            validation_result = self._validate_config_structure(config)
            errors.extend(validation_result.get("errors", []))
            warnings.extend(validation_result.get("warnings", []))

            if errors:
                return MappingResult(False, "", {}, errors, warnings)

            # Extract experimental intent
            intent = self._extract_experimental_intent(config)
            if intent_requirements:
                intent.update(intent_requirements)

            # Select appropriate script
            script_name = self._select_script(intent)
            if not script_name:
                errors.append("No compatible script found for the given configuration")
                return MappingResult(False, "", {}, errors, warnings)

            # Map universal parameters to script-specific parameters
            script_params = self._map_parameters(config, intent, script_name)

            # Validate script parameters
            if self.script_registry:
                is_valid, param_errors = self.script_registry.validate_parameters(script_name, script_params)
                errors.extend(param_errors)
                if not is_valid:
                    return MappingResult(False, script_name, script_params, errors, warnings)

            return MappingResult(True, script_name, script_params, errors, warnings)

        except Exception as e:
            errors.append(f"Configuration mapping failed: {str(e)}")
            return MappingResult(False, "", {}, errors, warnings)

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        import yaml
                        config = yaml.safe_load(f)
                    except ImportError:
                        raise ValueError("YAML support requires pyyaml package. Install with: pip install pyyaml")
                else:
                    # Try to determine format from content
                    content = f.read()
                    f.seek(0)  # Reset file pointer
                    try:
                        # Try JSON first
                        config = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            # Try YAML if available
                            import yaml
                            config = yaml.safe_load(content)
                        except ImportError:
                            raise ValueError("Cannot determine file format. Use .json or install pyyaml for .yaml support")
                        except Exception:
                            raise ValueError("Invalid configuration file format")

            logger.info(f"Loaded config from {config_path}")
            return config

        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variables in configuration"""
        def process_value(value):
            if isinstance(value, str):
                # Match ${VAR_NAME} or $VAR_NAME patterns
                matches = re.findall(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)', value)
                for match in matches:
                    var_name = match[0] or match[1]
                    if var_name in os.environ:
                        value = value.replace(f'${{{var_name}}}' if match[0] else f'${var_name}',
                                           os.environ[var_name])
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        return process_value(config)

    def _validate_config_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic configuration structure"""
        errors = []
        warnings = []

        # Check required top-level sections
        required_sections = ["model", "editing"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Required section '{section}' is missing")

        # Validate model intent
        if "model" in config and "model_intent" in config["model"]:
            model_intent = config["model"]["model_intent"]
            if "purpose" not in model_intent:
                warnings.append("Model purpose not specified, using default")
            if "architecture_preference" not in model_intent:
                model_intent["architecture_preference"] = "auto"
            if "size_preference" not in model_intent:
                model_intent["size_preference"] = "medium"

        # Validate editing intent
        if "editing" in config and "intent" in config["editing"]:
            editing_intent = config["editing"]["intent"]
            if "goal" not in editing_intent:
                warnings.append("Editing goal not specified, using default")
            if "strategy" not in editing_intent:
                editing_intent["strategy"] = "precise_localization"

        return {"errors": errors, "warnings": warnings}

    def _extract_experimental_intent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract experimental intent from configuration"""
        intent = {}

        # Model intent
        if "model" in config and "model_intent" in config["model"]:
            intent["model"] = config["model"]["model_intent"]

        # Editing intent
        if "editing" in config and "intent" in config["editing"]:
            intent["editing"] = config["editing"]["intent"]

        # Dataset intent
        if "dataset" in config and "data_intent" in config["dataset"]:
            intent["dataset"] = config["dataset"]["data_intent"]

        # Execution intent
        if "execution" in config and "execution_intent" in config["execution"]:
            intent["execution"] = config["execution"]["execution_intent"]

        return intent

    def _select_script(self, intent: Dict[str, Any]) -> Optional[str]:
        """Select appropriate script based on intent"""
        if not self.script_registry:
            logger.warning("No script registry available, using default selection")
            return "run_zsre_llama2"

        # Extract selection criteria from intent
        method = None
        dataset = None
        model = None

        if "editing" in intent:
            # Extract method from editing goal or strategy
            editing_intent = intent["editing"]
            if "method" in editing_intent:
                method = editing_intent["method"]
            elif "goal" in editing_intent:
                goal = editing_intent["goal"]
                # Enhanced goal to method mapping
                goal_to_method = {
                    "knowledge_editing": "ROME",
                    "behavior_modification": "FT",
                    "capability_removal": "MEMIT",
                    "fact_update": "ROME",
                    "parameter_update": "FT",
                    "batch_editing": "MEMIT",
                    "single_edit": "ROME",
                    "multi_edit": "KN",
                    "safety_editing": "SafeEdit",
                    "hallucination_editing": "HalluEditBench",
                    "concept_editing": "ConceptEdit",
                    "personality_editing": "PersonalityEdit",
                    "wisdom_editing": "WISE",
                    "ultra_editing": "UltraEdit",
                    "grace_editing": "GRACE",
                    "melo_editing": "MELO",
                    "mend_editing": "MEND",
                    "serac_editing": "SERAC",
                    "ike_editing": "IKE",
                    "pmet_editing": "PMET",
                    "dinm_editing": "DINM",
                    "alphedit_editing": "AlphaEdit",
                    "convsent_editing": "ConvsEnt",
                    "instruct_editing": "InstructEdit",
                    "malmem_editing": "MALMEN",
                    "adlora_editing": "AdaLoRA"
                }
                method = goal_to_method.get(goal)

        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            # Extract dataset type
            if "data_type" in dataset_intent:
                data_type = dataset_intent["data_type"]
                # Enhanced data type to dataset mapping
                data_type_to_dataset = {
                    "structured_factual": "ZsreDataset",
                    "conversational": "KnowEditDataset",
                    "instructional": "InstructDataset",
                    "biographical": "WikiBioDataset",
                    "counterfactual": "CounterFactDataset",
                    "conversation_sentiment": "ConvSentDataset",
                    "longform": "LongformDataset",
                    "personality": "PersonalityDataset",
                    "safety": "SafetyDataset",
                    "concept": "ConceptDataset",
                    "hallucination": "HalluDataset",
                    "llm": "LLMDataset",
                    "wisdom": "WISEDataset",
                    "ultra_edit": "UltraEditDataset",
                    "cknow_edit": "CKnowEditDataset",
                    "wiki_big_edit": "WikiBigEditDataset",
                    "ads_edit": "AdsEditDataset"
                }
                dataset = data_type_to_dataset.get(data_type)
            elif "dataset_name" in dataset_intent:
                # Direct dataset name specification
                dataset = dataset_intent["dataset_name"]

        if "model" in intent:
            model_intent = intent["model"]
            if "size_preference" in model_intent:
                # Enhanced size to model mapping
                size_to_model = {
                    "tiny": "gpt2-small",
                    "small": "gpt2-medium",
                    "medium": "llama-7b",
                    "large": "llama-13b",
                    "xlarge": "llama-30b",
                    "xxlarge": "llama-65b",
                    "mistral_tiny": "mistral-7b",
                    "mistral_small": "mistral-7b",
                    "mistral_medium": "mistral-7b",
                    "baichuan_tiny": "baichuan-7b",
                    "baichuan_small": "baichuan-7b",
                    "baichuan_medium": "baichuan-13b",
                    "chatglm_tiny": "chatglm-6b",
                    "chatglm_small": "chatglm-6b",
                    "chatglm_medium": "chatglm2-6b",
                    "internlm_tiny": "internlm-7b",
                    "internlm_small": "internlm-7b",
                    "internlm_medium": "internlm-20b",
                    "qwen_tiny": "qwen-7b",
                    "qwen_small": "qwen-7b",
                    "qwen_medium": "qwen-14b",
                    "qwen_large": "qwen-72b",
                    "vicuna_tiny": "vicuna-7b",
                    "vicuna_small": "vicuna-7b",
                    "vicuna_medium": "vicuna-13b",
                    "alpaca_tiny": "alpaca-7b",
                    "alpaca_small": "alpaca-7b",
                    "koala_small": "koala-13b",
                    "koala_medium": "koala-13b",
                    "wizard_tiny": "wizard-7b",
                    "wizard_small": "wizard-13b",
                    "wizard_medium": "wizard-13b"
                }
                model = size_to_model.get(model_intent["size_preference"])
            elif "model_name" in model_intent:
                model = model_intent["model_name"]
            elif "model_path" in model_intent:
                # Extract model name from path
                model_path = model_intent["model_path"]
                model_patterns = [
                    r"(gpt2|llama|mistral|baichuan|chatglm|internlm|qwen|vicuna|alpaca|koala|wizard)[^/]*",
                    r"(gpt-2|gpt-j|bert|t5)[^/]*"
                ]
                for pattern in model_patterns:
                    match = re.search(pattern, model_path, re.IGNORECASE)
                    if match:
                        model = match.group(1)
                        break

        # Find compatible scripts
        compatible_scripts = self.script_registry.find_compatible_scripts(
            method=method,
            dataset=dataset,
            model=model
        )

        if not compatible_scripts:
            logger.warning("No compatible scripts found, using default")
            return "run_zsre_llama2"

        # Select the best matching script with enhanced scoring
        best_script = compatible_scripts[0]
        best_score = self._score_script_compatibility(best_script, method, dataset, model)

        for script in compatible_scripts[1:]:
            score = self._score_script_compatibility(script, method, dataset, model)
            if score > best_score:
                best_script = script
                best_score = score

        logger.info(f"Selected script: {best_script.name}")
        return best_script.name

    def _score_script_compatibility(self,
                                 script_info,
                                 method: str = None,
                                 dataset: str = None,
                                 model: str = None) -> int:
        """Score script compatibility with given criteria"""
        score = 0

        if method and method in script_info.supported_methods:
            score += 3

        if dataset and dataset in script_info.supported_datasets:
            score += 2

        if model:
            model_lower = model.lower()
            script_models = [m.lower() for m in script_info.supported_models]
            if any(model_m in script_model for model_m in [model_lower] for script_model in script_models):
                score += 1

        return score

    def _map_parameters(self,
                        config: Dict[str, Any],
                        intent: Dict[str, Any],
                        script_name: str) -> Dict[str, Any]:
        """Map universal configuration to script-specific parameters"""
        script_params = {}

        # Enhanced parameter mapping for all script patterns
        if script_name.startswith("run_"):
            # Most run scripts use these common parameters
            self._map_common_parameters(config, intent, script_params)

            # Enhanced script-specific parameter mappings
            if "zsre" in script_name:
                self._map_zsre_parameters(config, intent, script_params)
            elif "knowedit" in script_name:
                self._map_knowedit_parameters(config, intent, script_params)
            elif "akew" in script_name.upper():
                self._map_akew_parameters(config, intent, script_params)
            elif "wise" in script_name.lower():
                self._map_wise_parameters(config, intent, script_params)
            elif "safe" in script_name.lower():
                self._map_safe_parameters(config, intent, script_params)
            elif "hallu" in script_name.lower():
                self._map_hallu_parameters(config, intent, script_params)
            elif "concept" in script_name.lower():
                self._map_concept_parameters(config, intent, script_params)
            elif "personality" in script_name.lower():
                self._map_personality_parameters(config, intent, script_params)
            elif "ultra" in script_name.lower():
                self._map_ultra_parameters(config, intent, script_params)
            elif "cknow" in script_name.lower():
                self._map_cknow_parameters(config, intent, script_params)
            elif "wiki" in script_name.lower():
                self._map_wiki_parameters(config, intent, script_params)
            elif "ads" in script_name.lower():
                self._map_ads_parameters(config, intent, script_params)
            elif "llm" in script_name.lower():
                self._map_llm_parameters(config, intent, script_params)
            elif "conv" in script_name.lower():
                self._map_conv_parameters(config, intent, script_params)
            elif "grace" in script_name.lower():
                self._map_grace_parameters(config, intent, script_params)
            elif "melo" in script_name.lower():
                self._map_melo_parameters(config, intent, script_params)
            elif "mend" in script_name.lower():
                self._map_mend_parameters(config, intent, script_params)
            elif "serac" in script_name.lower():
                self._map_serac_parameters(config, intent, script_params)
            elif "ike" in script_name.lower():
                self._map_ike_parameters(config, intent, script_params)
            elif "pmet" in script_name.lower():
                self._map_pmet_parameters(config, intent, script_params)
            elif "dinm" in script_name.lower():
                self._map_dinm_parameters(config, intent, script_params)
            elif "alpha" in script_name.lower():
                self._map_alpha_parameters(config, intent, script_params)
            elif "instruct" in script_name.lower():
                self._map_instruct_parameters(config, intent, script_params)
            elif "malmem" in script_name.lower():
                self._map_malmem_parameters(config, intent, script_params)
            elif "adlora" in script_name.lower():
                self._map_adlora_parameters(config, intent, script_params)
            else:
                # Generic mapping for other scripts
                self._map_generic_parameters(config, intent, script_params)

        # Add execution parameters
        if "execution" in intent:
            self._map_execution_parameters(intent["execution"], script_params)

        # Add any additional parameters from the config
        self._map_additional_parameters(config, intent, script_params)

        return script_params

    def _map_common_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters common to most scripts"""
        # Map method selection - FIXED to match YAML structure
        if "editing" in intent:
            editing_intent = intent["editing"]
            if "method" in editing_intent:
                script_params["editing_method"] = editing_intent["method"]
            else:
                # Infer method from goal
                goal = editing_intent.get("goal", "knowledge_editing")
                method_map = {
                    "knowledge_editing": "ROME",
                    "behavior_modification": "FT",
                    "capability_removal": "MEMIT"
                }
                script_params["editing_method"] = method_map.get(goal, "ROME")

        # Map hparams directory - FIXED to match YAML structure
        if "model" in intent:
            model_intent = intent["model"]
            if "hparams_dir" in model_intent:
                script_params["hparams_dir"] = model_intent["hparams_dir"]
            else:
                # Default hparams directory based on method
                method = script_params.get("editing_method", "ROME")
                script_params["hparams_dir"] = f"./hparams/{method}/llama-7b"

        # Map data directory - FIXED to match YAML structure
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            else:
                script_params["data_dir"] = "./data"

        # Map output directory - FIXED to match YAML structure
        if "execution" in intent:
            execution_intent = intent["execution"]
            if "metrics_save_dir" in execution_intent:
                script_params["metrics_save_dir"] = execution_intent["metrics_save_dir"]
            else:
                script_params["metrics_save_dir"] = "./results/metrics"

    def _map_zsre_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to ZsRE scripts"""
        # Dataset size
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

    def _map_knowedit_parameters(self,
                               config: Dict[str, Any],
                               intent: Dict[str, Any],
                               script_params: Dict[str, Any]):
        """Map parameters specific to KnowEdit scripts"""
        # KnowEdit specific parameters
        if "execution" in intent:
            execution_intent = intent["execution"]
            if "result_path" in execution_intent:
                script_params["result_path"] = execution_intent["result_path"]

    def _map_akew_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to AKEW scripts"""
        # AKEW specific parameters
        script_params["structured"] = True
        script_params["unstructured"] = True

    def _map_execution_parameters(self,
                                execution_intent: Dict[str, Any],
                                script_params: Dict[str, Any]):
        """Map execution-related parameters"""
        # Enhanced execution parameter mapping
        execution_mappings = {
            "metrics_save_dir": "metrics_save_dir",
            "output_dir": "output_dir",
            "results_dir": "results_dir",
            "log_dir": "log_dir",
            "cache_dir": "cache_dir",
            "ckpt_dir": "ckpt_dir",
            "save_ckpt": "save_ckpt",
            "eval_only": "eval_only",
            "use_cache": "use_cache",
            "use_wandb": "use_wandb",
            "wandb_project": "wandb_project",
            "eval_steps": "eval_steps",
            "save_steps": "save_steps",
            "seed": "seed",
            "verbose": "verbose",
            "debug": "debug",
            "dry_run": "dry_run"
        }

        for config_key, script_key in execution_mappings.items():
            if config_key in execution_intent:
                script_params[script_key] = execution_intent[config_key]

        # Additional execution parameters
        if "additional_args" in execution_intent:
            script_params.update(execution_intent["additional_args"])

        # Quality and efficiency requirements
        if "quality_requirements" in execution_intent:
            quality_req = execution_intent["quality_requirements"]
            if "success_rate_threshold" in quality_req:
                script_params["success_threshold"] = quality_req["success_rate_threshold"]

        if "efficiency_requirements" in execution_intent:
            efficiency_req = execution_intent["efficiency_requirements"]
            if "max_execution_time" in efficiency_req:
                script_params["timeout"] = efficiency_req["max_execution_time"]

    def _map_wise_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to WISE scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # WISE specific parameters
        script_params["wisdom_model"] = True
        script_params["editing_method"] = "WISE"

    def _map_safe_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to SafeEdit scripts"""
        if "editing" in intent:
            editing_intent = intent["editing"]
            if "constraints" in editing_intent:
                constraints = editing_intent["constraints"]
                if "safety_level" in constraints:
                    script_params["safety_level"] = constraints["safety_level"]

        # SafeEdit specific parameters
        script_params["safety_check"] = True
        script_params["editing_method"] = "SafeEdit"

    def _map_hallu_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to HalluEditBench scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # HalluEditBench specific parameters
        script_params["hallucination_detection"] = True
        script_params["editing_method"] = "HalluEditBench"

    def _map_concept_parameters(self,
                                config: Dict[str, Any],
                                intent: Dict[str, Any],
                                script_params: Dict[str, Any]):
        """Map parameters specific to ConceptEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # ConceptEdit specific parameters
        script_params["concept_editing"] = True
        script_params["editing_method"] = "ConceptEdit"

    def _map_personality_parameters(self,
                                   config: Dict[str, Any],
                                   intent: Dict[str, Any],
                                   script_params: Dict[str, Any]):
        """Map parameters specific to PersonalityEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # PersonalityEdit specific parameters
        script_params["personality_editing"] = True
        script_params["editing_method"] = "PersonalityEdit"

    def _map_ultra_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to UltraEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # UltraEdit specific parameters
        script_params["ultra_editing"] = True
        script_params["editing_method"] = "UltraEdit"

    def _map_cknow_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to CKnowEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # CKnowEdit specific parameters
        script_params["cknowledge_editing"] = True
        script_params["editing_method"] = "CKnowEdit"

    def _map_wiki_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to WikiBigEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # WikiBigEdit specific parameters
        script_params["wiki_editing"] = True
        script_params["editing_method"] = "WikiBigEdit"

    def _map_ads_parameters(self,
                           config: Dict[str, Any],
                           intent: Dict[str, Any],
                           script_params: Dict[str, Any]):
        """Map parameters specific to AdsEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # AdsEdit specific parameters
        script_params["ads_editing"] = True
        script_params["editing_method"] = "AdsEdit"

    def _map_llm_parameters(self,
                           config: Dict[str, Any],
                           intent: Dict[str, Any],
                           script_params: Dict[str, Any]):
        """Map parameters specific to LLMEval scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # LLMEval specific parameters
        script_params["llm_evaluation"] = True
        script_params["editing_method"] = "LLMEval"

    def _map_conv_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to ConvSent scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # ConvSent specific parameters
        script_params["conversation_sentiment"] = True
        script_params["editing_method"] = "ConvsEnt"

    def _map_grace_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to GRACE scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # GRACE specific parameters
        script_params["grace_editing"] = True
        script_params["editing_method"] = "GRACE"

    def _map_melo_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to MELO scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # MELO specific parameters
        script_params["melo_editing"] = True
        script_params["editing_method"] = "MELO"

    def _map_mend_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to MEND scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # MEND specific parameters
        script_params["mend_editing"] = True
        script_params["editing_method"] = "MEND"

    def _map_serac_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to SERAC scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # SERAC specific parameters
        script_params["serac_editing"] = True
        script_params["editing_method"] = "SERAC"

    def _map_ike_parameters(self,
                           config: Dict[str, Any],
                           intent: Dict[str, Any],
                           script_params: Dict[str, Any]):
        """Map parameters specific to IKE scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # IKE specific parameters
        script_params["ike_editing"] = True
        script_params["editing_method"] = "IKE"

    def _map_pmet_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to PMET scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # PMET specific parameters
        script_params["pmet_editing"] = True
        script_params["editing_method"] = "PMET"

    def _map_dinm_parameters(self,
                            config: Dict[str, Any],
                            intent: Dict[str, Any],
                            script_params: Dict[str, Any]):
        """Map parameters specific to DINM scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # DINM specific parameters
        script_params["dinm_editing"] = True
        script_params["editing_method"] = "DINM"

    def _map_alpha_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to AlphaEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # AlphaEdit specific parameters
        script_params["alpha_editing"] = True
        script_params["editing_method"] = "AlphaEdit"

    def _map_instruct_parameters(self,
                                config: Dict[str, Any],
                                intent: Dict[str, Any],
                                script_params: Dict[str, Any]):
        """Map parameters specific to InstructEdit scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # InstructEdit specific parameters
        script_params["instruct_editing"] = True
        script_params["editing_method"] = "InstructEdit"

    def _map_malmem_parameters(self,
                             config: Dict[str, Any],
                             intent: Dict[str, Any],
                             script_params: Dict[str, Any]):
        """Map parameters specific to MALMEN scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # MALMEN specific parameters
        script_params["malmem_editing"] = True
        script_params["editing_method"] = "MALMEN"

    def _map_adlora_parameters(self,
                              config: Dict[str, Any],
                              intent: Dict[str, Any],
                              script_params: Dict[str, Any]):
        """Map parameters specific to AdaLoRA scripts"""
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        # AdaLoRA specific parameters
        script_params["adlora_editing"] = True
        script_params["editing_method"] = "AdaLoRA"

    def _map_generic_parameters(self,
                               config: Dict[str, Any],
                               intent: Dict[str, Any],
                               script_params: Dict[str, Any]):
        """Map parameters for generic scripts"""
        # Basic fallback mapping
        if "dataset" in intent:
            dataset_intent = intent["dataset"]
            if "data_dir" in dataset_intent:
                script_params["data_dir"] = dataset_intent["data_dir"]
            if "ds_size" in dataset_intent:
                script_params["ds_size"] = dataset_intent["ds_size"]

        if "editing" in intent:
            editing_intent = intent["editing"]
            if "method" in editing_intent:
                script_params["editing_method"] = editing_intent["method"]

    def _map_additional_parameters(self,
                                  config: Dict[str, Any],
                                  intent: Dict[str, Any],
                                  script_params: Dict[str, Any]):
        """Map additional parameters from config"""
        # Map any additional parameters that might be in the config
        additional_mappings = {
            "batch_size": "batch_size",
            "eval_batch_size": "eval_batch_size",
            "num_workers": "num_workers",
            "pin_memory": "pin_memory",
            "max_length": "max_length",
            "sequential_edit": "sequential_edit",
            "device": "device",
            "gpu_id": "gpu_id",
            "cpu_only": "cpu_only",
            "use_fp16": "use_fp16",
            "use_bf16": "use_bf16",
            "model_parallel": "model_parallel",
            "load_in_8bit": "load_in_8bit",
            "load_in_4bit": "load_in_4bit",
            "trust_remote_code": "trust_remote_code",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_new_tokens": "max_new_tokens",
            "do_sample": "do_sample",
            "num_beams": "num_beams",
            "gradient_checkpointing": "gradient_checkpointing",
            "use_flash_attention": "use_flash_attention",
            "memory_efficient": "memory_efficient"
        }

        # Check for additional parameters in all sections
        for section in ["model", "editing", "dataset", "execution"]:
            if section in config:
                section_config = config[section]
                for config_key, script_key in additional_mappings.items():
                    if config_key in section_config:
                        script_params[script_key] = section_config[config_key]

        # Also check intent sections
        for section_name, section_intent in intent.items():
            for config_key, script_key in additional_mappings.items():
                if config_key in section_intent:
                    script_params[script_key] = section_intent[config_key]

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration template"""
        return {
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
                    "required_fields": ["prompt", "target_new", "ground_truth"]
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