"""
Advanced Execution Example
========================

Demonstrates advanced features including async execution, task management, and monitoring.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyedit_api import EasyEditAPI


def main():
    """Advanced execution example"""
    print("=== Advanced Execution Example ===\n")

    # Initialize API with async execution enabled
    api = EasyEditAPI(enable_async=True)

    # 1. System Overview
    print("1. System overview...")
    system_info = api.get_system_info()
    print(f"   Available scripts: {system_info['available_scripts']}")
    print(f"   Async execution: {system_info['async_execution_enabled']}")
    print(f"   Currently active tasks: {system_info['active_tasks']}")
    print()

    # 2. Prepare Multiple Experiments
    print("2. Preparing multiple experiments...")
    experiments = [
        {
            "name": "ROME Experiment",
            "config": {
                "model": {
                    "model_intent": {
                        "purpose": "knowledge_editing",
                        "method": "ROME",
                        "hparams_dir": "../hparams/ROME/llama-7b"
                    }
                },
                "editing": {
                    "intent": {
                        "goal": "knowledge_editing",
                        "method": "ROME"
                    }
                },
                "dataset": {
                    "data_intent": {
                        "data_dir": "./data",
                        "ds_size": 10  # Small size for demo
                    }
                }
            }
        },
        {
            "name": "FT Experiment",
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
                        "ds_size": 10
                    }
                }
            }
        }
    ]

    print(f"   Prepared {len(experiments)} experiments")
    print()

    # 3. Asynchronous Execution
    print("3. Demonstrating asynchronous execution...")
    print("   (Actual execution commented out for safety)")
    print()

    task_ids = []

    # Code for async execution (commented for safety)
    async_code = '''
    # Submit all tasks asynchronously
    for exp in experiments:
        print(f"   Submitting {exp['name']}...")
        result = api.execute_experiment(exp['config'], async_execution=True)
        if isinstance(result, dict) and 'task_id' in result:
            task_ids.append(result['task_id'])
            print(f"   Task ID: {result['task_id']}")
        else:
            print(f"   Error: {result}")

    print(f"\\n   Submitted {len(task_ids)} tasks")
    '''

    print("   Async execution code:")
    print(async_code)

    # 4. Task Monitoring
    print("4. Task monitoring example...")
    print("   (Simulated monitoring since no real tasks)")

    # Simulate task monitoring
    simulated_task_ids = ["task_123456_run_zsre_llama2", "task_123457_run_knowedit_llama2"]

    print("   Monitoring code:")
    print('''
    # Monitor task status
    for task_id in task_ids:
        status = api.get_task_status(task_id)
        print(f"   Task {task_id}: {status['status']}")

        if status['status'] == 'running':
            print(f"     Running since: {status.get('start_time', 'unknown')}")
        elif status['status'] == 'completed':
            print(f"     Success: {status.get('success', False)}")
            print(f"     Execution time: {status.get('execution_time', 0)} seconds")
    ''')

    # Simulate status display
    for task_id in simulated_task_ids:
        print(f"   Task {task_id}: simulated_running")
        time.sleep(0.5)  # Simulate monitoring delay

    print()

    # 5. Batch Execution with Concurrency Control
    print("5. Batch execution with concurrency control...")
    print("   (Code example - actual execution commented out)")

    batch_code = '''
    # Execute experiments with max concurrency
    configs = [exp['config'] for exp in experiments]
    results = api.batch_execute(configs, max_concurrent=2)

    print(f"   Batch execution completed: {len(results)} results")
    for i, result in enumerate(results):
        print(f"     Experiment {i+1}: Success={result.get('success', False)}, "
              f"Time={result.get('execution_time', 0):.2f}s")
    '''

    print(batch_code)
    print()

    # 6. Direct Script Execution
    print("6. Direct script execution...")
    print("   Execute specific scripts with custom parameters")

    direct_execution_code = '''
    # Execute specific script directly
    script_params = {
        'editing_method': 'ROME',
        'hparams_dir': '../hparams/ROME/llama-7b',
        'data_dir': './data',
        'ds_size': 10,
        'metrics_save_dir': './output'
    }

    # Synchronous execution
    result = api.execute_script(
        script_name='run_zsre_llama2',
        parameters=script_params,
        async_execution=False
    )

    print(f"   Direct execution result: {result.get('success', False)}")
    '''

    print(direct_execution_code)
    print()

    # 7. Quick Experiment Method
    print("7. Quick experiment method...")
    print("   Simplified interface for common experiments")

    quick_experiment_code = '''
    # Quick experiment with common parameters
    result = api.quick_experiment(
        method='ROME',
        model='llama-7b',
        dataset='zsre',
        hparams_dir='../hparams/ROME/llama-7b',
        data_dir='./data',
        ds_size=10
    )

    print(f"   Quick experiment result: {result.get('success', False)}")
    '''

    print(quick_experiment_code)
    print()

    # 8. Error Handling and Recovery
    print("8. Error handling and recovery...")

    error_handling_code = '''
    # Error handling example
    try:
        result = api.execute_experiment(invalid_config)
    except ValueError as e:
        print(f"   Configuration error: {e}")

    # Validate before execution
    validation = api.validate_config(config)
    if not validation['valid']:
        print(f"   Configuration errors: {validation['errors']}")
        # Fix configuration and retry
    else:
        result = api.execute_experiment(config)
    '''

    print(error_handling_code)
    print()

    # 9. Resource Management
    print("9. Resource management...")

    resource_code = '''
    # Monitor active tasks
    active_tasks = api.list_active_tasks()
    print(f"   Active tasks: {len(active_tasks)}")

    # Cancel specific task if needed
    if active_tasks:
        task_id = active_tasks[0]['task_id']
        cancelled = api.cancel_task(task_id)
        print(f"   Cancelled task {task_id}: {cancelled}")

    # Wait for all tasks to complete
    for task in active_tasks:
        api.wait_for_completion(task['task_id'], timeout=300)
    '''

    print(resource_code)
    print()

    # 10. Cleanup
    print("10. System cleanup...")
    cleanup_code = '''
    # Cleanup resources
    api.cleanup()
    print("   API resources cleaned up")
    '''

    print(cleanup_code)

    print("\n=== Advanced Execution Example Complete ===")
    print("\nNote: Actual execution is commented out for safety.")
    print("Uncomment the relevant sections to run real experiments.")


if __name__ == "__main__":
    main()