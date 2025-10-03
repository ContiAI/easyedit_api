"""
Script Executor
===============

Unified script execution engine for EasyEdit run scripts.
Handles script execution, monitoring, and result collection.
"""

import subprocess
import sys
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import queue
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of script execution"""
    task_id: str
    script_name: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    start_time: datetime
    end_time: datetime
    output_files: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "script_name": self.script_name,
            "success": self.success,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "output_files": self.output_files,
            "metrics": self.metrics
        }


@dataclass
class ExecutionTask:
    """Task to be executed"""
    task_id: str
    script_path: str
    parameters: Dict[str, Any]
    working_dir: str
    timeout: Optional[float] = None
    capture_output: bool = True
    env_vars: Dict[str, str] = None

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}


class ScriptExecutor:
    """
    Unified script execution engine.

    Features:
    - Subprocess-based script execution
    - Real-time monitoring and logging
    - Timeout handling
    - Result collection and parsing
    - Output file detection
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or "E:/ContiAI/EasyEdit"
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.task_results: Dict[str, ExecutionResult] = {}
        self.execution_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None

        # Start worker thread for async execution
        self._start_worker()

    def _start_worker(self):
        """Start worker thread for processing execution queue"""
        def worker():
            while not self._stop_event.is_set():
                try:
                    task = self.execution_queue.get(timeout=1.0)
                    if task:
                        self._execute_task(task)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker thread error: {e}")

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        logger.info("Script executor worker started")

    def execute_script(self,
                      script_name: str,
                      parameters: Dict[str, Any],
                      async_execution: bool = False,
                      timeout: float = None) -> Union[ExecutionResult, str]:
        """
        Execute a script with given parameters.

        Args:
            script_name: Name of the script to execute
            parameters: Script parameters
            async_execution: Whether to execute asynchronously
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult if sync, task_id if async
        """
        try:
            # Build script path
            script_path = self._get_script_path(script_name)
            if not script_path:
                raise ValueError(f"Script not found: {script_name}")

            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{script_name}"

            # Create execution task
            task = ExecutionTask(
                task_id=task_id,
                script_path=script_path,
                parameters=parameters,
                working_dir=self.base_dir,
                timeout=timeout,
                env_vars=self._build_environment_variables()
            )

            if async_execution:
                # Add to queue for async execution
                self.execution_queue.put(task)
                self.active_tasks[task_id] = task
                logger.info(f"Queued async task: {task_id}")
                return task_id
            else:
                # Execute synchronously
                result = self._execute_task(task)
                return result

        except Exception as e:
            logger.error(f"Failed to execute script {script_name}: {e}")
            if async_execution:
                return f"error_{int(time.time() * 1000)}"
            else:
                return ExecutionResult(
                    task_id=f"error_{int(time.time() * 1000)}",
                    script_name=script_name,
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=str(e),
                    execution_time=0.0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    output_files=[],
                    metrics={}
                )

    def _get_script_path(self, script_name: str) -> Optional[str]:
        """Get full path to script with enhanced search"""
        # Try direct path
        if os.path.exists(script_name):
            return script_name

        # Try in examples directory
        examples_dir = Path(self.base_dir) / "examples"

        # Try different naming patterns
        script_patterns = [
            f"{script_name}.py",
            f"run_{script_name}.py",
            f"run_{script_name.lower()}.py",
            f"run_{script_name.upper()}.py",
            f"edit_{script_name}.py",
            f"test_{script_name}.py",
            f"demo_{script_name}.py",
            f"{script_name}_edit.py",
            f"{script_name}_run.py",
        ]

        for pattern in script_patterns:
            script_path = examples_dir / pattern
            if script_path.exists():
                return str(script_path)

        # Try case-insensitive search in examples directory
        try:
            for file_path in examples_dir.glob("*.py"):
                if script_name.lower() in file_path.stem.lower():
                    return str(file_path)
        except Exception as e:
            logger.warning(f"Failed to search in examples directory: {e}")

        # Try recursive search in examples directory
        try:
            for file_path in examples_dir.rglob("*.py"):
                if script_name.lower() in file_path.stem.lower():
                    return str(file_path)
        except Exception as e:
            logger.warning(f"Failed to recursively search in examples directory: {e}")

        return None

    def _build_environment_variables(self) -> Dict[str, str]:
        """Build environment variables for script execution"""
        env = os.environ.copy()

        # Add EasyEdit specific environment variables
        env["PYTHONPATH"] = self.base_dir
        env["EASYEDIT_HOME"] = self.base_dir

        return env

    def _execute_task(self, task: ExecutionTask) -> ExecutionResult:
        """Execute a single task"""
        start_time = datetime.now()
        stdout = ""
        stderr = ""
        return_code = -1
        success = False

        try:
            logger.info(f"Executing task: {task.task_id} - {task.script_path}")

            # Build command line arguments
            cmd = self._build_command(task)

            # Execute script
            process = subprocess.Popen(
                cmd,
                cwd=task.working_dir,
                env=task.env_vars,
                stdout=subprocess.PIPE if task.capture_output else None,
                stderr=subprocess.PIPE if task.capture_output else None,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )

            # Monitor execution
            if task.capture_output:
                stdout, stderr = process.communicate(timeout=task.timeout)
            else:
                process.wait(timeout=task.timeout)

            return_code = process.returncode
            success = return_code == 0

            execution_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Task {task.task_id} completed with return code: {return_code}")

            # Collect output files and metrics
            output_files = self._collect_output_files(task)
            metrics = self._parse_metrics(stdout, stderr, output_files)

            result = ExecutionResult(
                task_id=task.task_id,
                script_name=Path(task.script_path).stem,
                success=success,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now(),
                output_files=output_files,
                metrics=metrics
            )

            # Store result
            self.task_results[task.task_id] = result

            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            return result

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task.task_id} timed out after {task.timeout} seconds")
            return ExecutionResult(
                task_id=task.task_id,
                script_name=Path(task.script_path).stem,
                success=False,
                return_code=-1,
                stdout=stdout,
                stderr=f"Execution timed out after {task.timeout} seconds",
                execution_time=(datetime.now() - start_time).total_seconds(),
                start_time=start_time,
                end_time=datetime.now(),
                output_files=[],
                metrics={}
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return ExecutionResult(
                task_id=task.task_id,
                script_name=Path(task.script_path).stem,
                success=False,
                return_code=-1,
                stdout=stdout,
                stderr=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                start_time=start_time,
                end_time=datetime.now(),
                output_files=[],
                metrics={}
            )

    def _build_command(self, task: ExecutionTask) -> List[str]:
        """Build command line for script execution with enhanced parameter handling"""
        cmd = [sys.executable, task.script_path]

        # Add parameters as command line arguments with enhanced type handling
        for param_name, param_value in task.parameters.items():
            if param_value is None:
                continue

            # Convert parameter name to command line argument
            arg_name = f"--{param_name}"

            # Handle different parameter types comprehensively
            if isinstance(param_value, bool):
                if param_value:
                    cmd.append(arg_name)
            elif isinstance(param_value, (list, tuple)):
                # Handle list parameters
                for item in param_value:
                    if item is not None:
                        cmd.extend([arg_name, str(item)])
            elif isinstance(param_value, dict):
                # Handle dictionary parameters - convert to JSON string
                import json
                cmd.extend([arg_name, json.dumps(param_value)])
            elif isinstance(param_value, (int, float, str)):
                # Handle basic types
                cmd.extend([arg_name, str(param_value)])
            elif isinstance(param_value, Path):
                # Handle Path objects
                cmd.extend([arg_name, str(param_value)])
            else:
                # Handle other types with string conversion
                try:
                    cmd.extend([arg_name, str(param_value)])
                except Exception as e:
                    logger.warning(f"Failed to convert parameter {param_name}: {e}")
                    continue

        return cmd

    def _collect_output_files(self, task: ExecutionTask) -> List[str]:
        """Collect output files generated by the script with enhanced search"""
        output_files = []

        try:
            working_path = Path(task.working_dir)

            # Extended list of common output directories
            output_dirs = [
                working_path / "output",
                working_path / "outputs",
                working_path / "results",
                working_path / "logs",
                working_path / "metrics",
                working_path / "checkpoints",
                working_path / "cache",
                working_path / "models",
                working_path / "eval_results",
                working_path / "edit_results",
                working_path / "predictions",
                working_path / "saved_models",
                working_path / "wandb",
                working_path / "tensorboard",
            ]

            # Search in output directories
            for output_dir in output_dirs:
                if output_dir.exists():
                    for file_path in output_dir.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            output_files.append(str(file_path))

            # Look for files with task_id or script_name in name
            search_patterns = [
                f"*{task.task_id}*",
                f"*{task.script_name}*",
                f"*editing*",
                f"*result*",
                f"*output*",
                f"*metrics*",
                f"*log*",
                f"*ckpt*",
                f"*model*",
                f"*eval*",
            ]

            for pattern in search_patterns:
                for file_path in working_path.rglob(pattern):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        if str(file_path) not in output_files:  # Avoid duplicates
                            output_files.append(str(file_path))

            # Look for common output file extensions
            output_extensions = [
                ".json", ".txt", ".log", ".csv", ".pkl", ".pt", ".pth",
                ".bin", ".h5", ".npz", ".npy", ".yaml", ".yml", ".toml",
                ".png", ".jpg", ".jpeg", ".pdf", ".html", ".md"
            ]

            for ext in output_extensions:
                for file_path in working_path.rglob(f"*{ext}"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        if str(file_path) not in output_files:  # Avoid duplicates
                            output_files.append(str(file_path))

            # Remove duplicates and sort
            output_files = sorted(list(set(output_files)))

        except Exception as e:
            logger.warning(f"Failed to collect output files: {e}")

        return output_files

    def _parse_metrics(self,
                      stdout: str,
                      stderr: str,
                      output_files: List[str]) -> Dict[str, Any]:
        """Parse metrics from script output"""
        metrics = {}

        try:
            # Parse stdout for common metrics
            stdout_lines = stdout.split('\n')

            for line in stdout_lines:
                line = line.strip()

                # Look for common metric patterns
                if "Edit_Succ:" in line:
                    try:
                        metrics["edit_success"] = float(line.split("Edit_Succ:")[1].strip().split()[0])
                    except:
                        pass

                elif "Overall_portability:" in line:
                    try:
                        metrics["overall_portability"] = float(line.split("Overall_portability:")[1].strip().split()[0])
                    except:
                        pass

                elif "accuracy" in line.lower() or "acc" in line.lower():
                    try:
                        # Extract percentage values
                        import re
                        percentages = re.findall(r'(\d+\.?\d*)%', line)
                        if percentages:
                            metrics["accuracy"] = float(percentages[0])
                    except:
                        pass

                elif "loss" in line.lower():
                    try:
                        loss_value = re.findall(r'loss[:\s]+(\d+\.?\d*)', line.lower())
                        if loss_value:
                            metrics["loss"] = float(loss_value[0])
                    except:
                        pass

            # Parse output files for additional metrics
            for file_path in output_files:
                if file_path.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            file_data = json.load(f)
                            if isinstance(file_data, dict):
                                metrics.update(file_data)
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Failed to parse metrics: {e}")

        return metrics

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        if task_id in self.active_tasks:
            return {
                "task_id": task_id,
                "status": "running",
                "start_time": self.active_tasks[task_id].task_id.split('_')[1]
            }
        elif task_id in self.task_results:
            result = self.task_results[task_id]
            return {
                "task_id": task_id,
                "status": "completed" if result.success else "failed",
                "success": result.success,
                "execution_time": result.execution_time,
                "return_code": result.return_code
            }
        else:
            return {
                "task_id": task_id,
                "status": "not_found"
            }

    def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get result of a completed task"""
        return self.task_results.get(task_id)

    def wait_for_completion(self, task_id: str, timeout: float = None) -> bool:
        """Wait for task completion"""
        start_time = time.time()

        while True:
            status = self.get_task_status(task_id)

            if status["status"] in ["completed", "failed"]:
                return status["status"] == "completed"

            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for task {task_id}")
                return False

            time.sleep(0.5)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        # Note: This is a simplified implementation
        # In practice, you might need to terminate the subprocess
        if task_id in self.active_tasks:
            logger.info(f"Cancelling task: {task_id}")
            del self.active_tasks[task_id]
            return True
        return False

    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks"""
        return [
            self.get_task_status(task_id)
            for task_id in self.active_tasks
        ]

    def cleanup(self):
        """Cleanup resources"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # Clear active tasks and results
        self.active_tasks.clear()
        self.task_results.clear()

        logger.info("Script executor cleaned up")