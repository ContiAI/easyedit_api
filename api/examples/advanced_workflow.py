"""
Advanced Workflow Example
========================

Demonstrates advanced workflow with task execution and monitoring.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.api_client import EasyEditClient
from api.utils.logging_utils import setup_logging, get_logger


def main():
    """Advanced workflow example"""
    # Setup logging
    setup_logging({
        'level': 'INFO',
        'console_output': True,
        'colored_output': True
    })

    logger = get_logger("advanced_workflow")

    # Initialize client
    client = EasyEditClient()

    print("=== EasyEdit API Advanced Workflow Example ===\n")

    # 1. Load intent configuration from file
    print("1. Loading intent configuration...")
    config_file = "../../config/example_intent_driven_experiment.yaml"
    if os.path.exists(config_file):
        intents = client.declare_intent(config_file)
        print(f"   Loaded intents from: {config_file}")
    else:
        # Use default configuration
        intent_config = {
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
                    },
                    "efficiency_requirements": {
                        "max_execution_time": 3600
                    }
                }
            }
        }
        intents = client.declare_intent(intent_config)
        print("   Using default intent configuration")

    print()

    # 2. Component discovery with filtering
    print("2. Advanced component discovery...")
    requirements = client.translate_intent_to_requirements(intents)

    # Discover methods with specific capabilities
    method_requirements = {
        **requirements.get("method_requirements", {}),
        "capabilities": ["knowledge_editing"],
        "max_memory_usage": "16gb"
    }
    methods = client.discover_methods(method_requirements)
    print(f"   Found {len(methods)} methods matching requirements")

    # Discover models with size constraints
    model_requirements = {
        **requirements.get("model_requirements", {}),
        "max_parameters": "10b"
    }
    models = client.discover_models(model_requirements)
    print(f"   Found {len(models)} models matching size constraints")

    print()

    # 3. Parameter optimization
    print("3. Advanced parameter optimization...")
    editing_goals = {
        "editing_type": "knowledge_editing",
        "quality_targets": {
            "reliability": 0.95,
            "generalization": 0.85,
            "locality": 0.90
        },
        "constraints": {
            "locality_preservation": "high",
            "computational_cost": "medium",
            "execution_time": "fast"
        }
    }

    universal_params = client.create_universal_parameters(editing_goals)
    print(f"   Created universal parameters with {len(universal_params)} settings")

    # Map to first compatible method
    if methods and models:
        method_info = methods[0]
        model_info = models[0]

        method_params = client.map_to_method_parameters(
            universal_params, method_info, {"model": model_info}
        )
        print(f"   Mapped to {method_info['name']} method")

        # Optimize for multiple objectives
        optimization_goals = ["speed", "memory", "accuracy"]
        optimized_params = client.optimize_parameters(
            method_params, method_info, optimization_goals
        )
        print(f"   Optimized for goals: {optimization_goals}")

    print()

    # 4. Task submission and monitoring
    print("4. Task submission and monitoring...")
    if methods and models:
        # Create a sample dataset path (would normally exist)
        dataset_path = "./data/sample_dataset.json"

        # Submit editing task
        task_id = client.submit_editing_task(
            method_name=methods[0]["name"],
            model_name=models[0]["name"],
            dataset_path=dataset_path,
            parameters=optimized_params if 'optimized_params' in locals() else method_params,
            execution_config={
                "mode": "adaptive",
                "resource_allocation": {
                    "gpu_memory_limit_gb": 8,
                    "cpu_cores": 4
                },
                "optimization": {
                    "parallel_execution": True,
                    "adaptive_parameters": True
                }
            }
        )

        print(f"   Submitted task: {task_id}")

        # Monitor task progress
        print("   Monitoring task progress...")
        timeout = 30  # 30 seconds for demonstration
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = client.monitor_task(task_id)
            print(f"     Status: {status['state']} - {status.get('message', '')}")

            if status['state'] in ['completed', 'failed', 'cancelled']:
                break

            time.sleep(2)

        # Get final result
        result = client.get_task_result(task_id)
        if result:
            print(f"   Task result: {result.get('status', 'Unknown')}")
            if 'metrics' in result:
                print("   Metrics:")
                for metric, value in result['metrics'].items():
                    print(f"     {metric}: {value}")

    print()

    # 5. Batch processing example
    print("5. Batch processing example...")
    if methods:
        # Create multiple similar tasks
        task_ids = []
        sample_datasets = [
            "./data/dataset1.json",
            "./data/dataset2.json",
            "./data/dataset3.json"
        ]

        for i, dataset_path in enumerate(sample_datasets):
            try:
                task_id = client.submit_editing_task(
                    method_name=methods[0]["name"],
                    model_name=models[0]["name"] if models else "gpt2-xl",
                    dataset_path=dataset_path,
                    parameters=method_params,
                    execution_config={
                        "mode": "parallel",
                        "priority": "normal"
                    }
                )
                task_ids.append(task_id)
                print(f"   Submitted batch task {i+1}: {task_id}")

            except Exception as e:
                print(f"   Failed to submit task {i+1}: {e}")

        # Monitor all batch tasks
        print("   Monitoring batch tasks...")
        for task_id in task_ids:
            try:
                success = client.wait_for_completion(task_id, timeout=10)
                status = "completed" if success else "failed/timeout"
                print(f"     Task {task_id}: {status}")
            except Exception as e:
                print(f"     Task {task_id}: error - {e}")

    print()

    # 6. System optimization suggestions
    print("6. System optimization analysis...")
    status = client.get_system_status()
    print(f"   Active tasks: {status['active_tasks']}")
    print(f"   Available plugins: {status['plugins']['loaded_count']}")

    # Get resource allocation suggestions
    if hasattr(client.execution_engine, 'resource_manager'):
        try:
            active_tasks = [{"id": "demo", "requirements": {"cpu_percent": 50, "memory_gb": 4}}]
            suggestions = client.execution_engine.resource_manager.optimize_resource_allocation(active_tasks)
            if suggestions['reallocations']:
                print("   Resource reallocation suggestions:")
                for realloc in suggestions['reallocations']:
                    print(f"     - {realloc}")
        except Exception as e:
            print(f"   Could not get optimization suggestions: {e}")

    print("\n=== Advanced Workflow Example Complete ===")


if __name__ == "__main__":
    main()