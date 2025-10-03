"""
Plugin Development Example
=========================

Demonstrates how to create and use custom plugins.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.api_client import EasyEditClient
from api.plugins.plugin_manager import MethodPlugin, PluginManager
from api.utils.logging_utils import setup_logging, get_logger


class CustomEditingMethod(MethodPlugin):
    """Example custom editing method plugin"""

    @property
    def name(self) -> str:
        return "custom_editing_method"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Custom editing method for demonstration"

    def initialize(self, config):
        """Initialize the plugin"""
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.001)
        self.max_iterations = config.get("max_iterations", 100)
        return True

    def is_compatible(self, context):
        """Check compatibility with context"""
        model_type = context.get("model", {}).get("type", "")
        return model_type in ["gpt2", "llama", "gpt-j", "any"]

    def get_method_info(self):
        """Get method information"""
        return {
            "name": self.name,
            "type": "editing_method",
            "supported_models": ["gpt2", "llama", "gpt-j"],
            "capabilities": ["knowledge_editing", "behavior_modification"],
            "parameters": {
                "learning_rate": {
                    "type": "float",
                    "default": 0.001,
                    "range": [0.0001, 0.01],
                    "description": "Learning rate for optimization"
                },
                "max_iterations": {
                    "type": "int",
                    "default": 100,
                    "range": [10, 1000],
                    "description": "Maximum number of iterations"
                },
                "batch_size": {
                    "type": "int",
                    "default": 1,
                    "range": [1, 32],
                    "description": "Batch size for processing"
                }
            },
            "requirements": {
                "min_memory_gb": 4,
                "gpu_required": False,
                "training_required": False
            }
        }

    def validate_parameters(self, parameters):
        """Validate method parameters"""
        required_params = ["learning_rate", "max_iterations"]
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        # Validate parameter ranges
        lr = parameters["learning_rate"]
        if not (0.0001 <= lr <= 0.01):
            return False, "Learning rate must be between 0.0001 and 0.01"

        max_iter = parameters["max_iterations"]
        if not (10 <= max_iter <= 1000):
            return False, "Max iterations must be between 10 and 1000"

        return True, "Parameters are valid"

    def execute_edit(self, model, parameters):
        """Execute the editing method (simulated)"""
        try:
            # Simulate editing process
            import time
            import random

            learning_rate = parameters.get("learning_rate", self.learning_rate)
            max_iterations = parameters.get("max_iterations", self.max_iterations)

            print(f"Executing custom editing with lr={learning_rate}, max_iter={max_iterations}")

            # Simulate editing iterations
            for i in range(min(max_iterations, 5)):  # Limit for demo
                time.sleep(0.1)  # Simulate computation
                progress = (i + 1) / max_iterations * 100
                print(f"  Iteration {i+1}/{max_iterations} ({progress:.1f}%)")

            # Simulate results
            success = random.random() > 0.1  # 90% success rate
            reliability = random.uniform(0.8, 0.98)
            generalization = random.uniform(0.7, 0.95)
            locality = random.uniform(0.85, 0.99)

            return {
                "success": success,
                "edited_model": model,
                "metrics": {
                    "reliability": reliability,
                    "generalization": generalization,
                    "locality": locality,
                    "iterations_used": min(max_iterations, 5)
                },
                "parameters_used": parameters
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "edited_model": model
            }


def main():
    """Plugin development example"""
    # Setup logging
    setup_logging({
        'level': 'INFO',
        'console_output': True,
        'colored_output': True
    })

    logger = get_logger("plugin_development")

    print("=== EasyEdit API Plugin Development Example ===\n")

    # 1. Create custom plugin instance
    print("1. Creating custom plugin...")
    custom_plugin = CustomEditingMethod()
    print(f"   Plugin name: {custom_plugin.name}")
    print(f"   Plugin version: {custom_plugin.version}")
    print(f"   Plugin description: {custom_plugin.description}")

    # 2. Initialize plugin
    print("\n2. Initializing plugin...")
    plugin_config = {
        "learning_rate": 0.005,
        "max_iterations": 50,
        "debug_mode": True
    }
    success = custom_plugin.initialize(plugin_config)
    print(f"   Initialization: {'Success' if success else 'Failed'}")

    # 3. Get method information
    print("\n3. Getting method information...")
    method_info = custom_plugin.get_method_info()
    print(f"   Supported models: {method_info['supported_models']}")
    print(f"   Capabilities: {method_info['capabilities']}")
    print("   Parameters:")
    for param_name, param_info in method_info['parameters'].items():
        print(f"     - {param_name}: {param_info['description']} (default: {param_info['default']})")

    # 4. Validate parameters
    print("\n4. Validating parameters...")
    test_params = {
        "learning_rate": 0.003,
        "max_iterations": 75,
        "batch_size": 4
    }
    is_valid, message = custom_plugin.validate_parameters(test_params)
    print(f"   Parameter validation: {'Valid' if is_valid else 'Invalid'} - {message}")

    # 5. Test compatibility
    print("\n5. Testing compatibility...")
    test_contexts = [
        {"model": {"type": "gpt2"}},
        {"model": {"type": "llama"}},
        {"model": {"type": "unknown"}}
    ]

    for context in test_contexts:
        compatible = custom_plugin.is_compatible(context)
        model_type = context["model"]["type"]
        print(f"   Model {model_type}: {'Compatible' if compatible else 'Not compatible'}")

    # 6. Simulate execution
    print("\n6. Simulating execution...")
    model = "gpt2-xl"  # Simulated model
    result = custom_plugin.execute_edit(model, test_params)

    print(f"   Execution result: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print("   Metrics:")
        for metric, value in result['metrics'].items():
            print(f"     {metric}: {value}")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")

    # 7. Create plugin template
    print("\n7. Creating plugin template...")
    template_dir = "./plugin_templates"
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

    # Use a simple plugin manager for template creation
    plugin_manager = PluginManager()
    template_created = plugin_manager.create_plugin_template(
        "my_custom_method",
        "method",
        template_dir
    )

    if template_created:
        print(f"   Plugin template created in: {template_dir}/my_custom_method/")
        print("   Files created:")
        for file in os.listdir(f"{template_dir}/my_custom_method"):
            print(f"     - {file}")
    else:
        print("   Failed to create plugin template")

    # 8. Plugin registration with EasyEdit client
    print("\n8. Plugin registration with EasyEdit client...")
    client = EasyEditClient()

    # Manually register our custom plugin
    if client.plugin_manager:
        from api.plugins.plugin_manager import PluginMetadata

        metadata = PluginMetadata(
            name="custom_editing_method",
            version="1.0.0",
            description="Custom editing method for demonstration",
            plugin_type="method",
            entry_point="custom_editing_method:CustomEditingMethod"
        )

        registered = client.plugin_manager.register(custom_plugin, metadata)
        print(f"   Plugin registration: {'Success' if registered else 'Failed'}")

        # Check if plugin is discoverable
        methods = client.discover_methods()
        custom_methods = [m for m in methods if m['name'] == 'custom_editing_method']
        print(f"   Plugin discoverable: {'Yes' if custom_methods else 'No'}")

    print("\n=== Plugin Development Example Complete ===")


if __name__ == "__main__":
    main()