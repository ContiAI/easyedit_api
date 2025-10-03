"""
EasyEdit API
============

A lightweight, extensible API for executing EasyEdit experiments through
configuration files. Provides automatic script discovery, configuration mapping,
and unified execution capabilities.

This API serves as a bridge between declarative YAML configurations and
the EasyEdit framework's execution system.
"""

# Main API interface
from .easyedit_api import EasyEditAPI, get_api

# Configuration loader
from .config_loader import ConfigLoader, load_config, execute_config, validate_config

# Supporting modules
from .script_registry import ScriptRegistry, ScriptInfo
from .config_mapper import ConfigMapper, MappingResult
from .script_executor import ScriptExecutor, ExecutionResult

# Version
__version__ = "2.0.0"

__all__ = [
    # Main API
    "EasyEditAPI",
    "get_api",

    # Configuration
    "ConfigLoader",
    "load_config",
    "execute_config",
    "validate_config",

    # Core components
    "ScriptRegistry",
    "ScriptInfo",
    "ConfigMapper",
    "MappingResult",
    "ScriptExecutor",
    "ExecutionResult",

    # Version
    "__version__"
]