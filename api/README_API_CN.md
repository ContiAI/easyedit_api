# EasyEdit API 文档

## 概述

EasyEdit API 提供了一个轻量级、可扩展的接口，用于通过配置文件执行 EasyEdit 实验。它通过自动发现脚本、将配置映射到参数、提供统一的执行能力，简化了运行编辑实验的过程。

## 目录

1. [架构](#架构)
2. [核心概念](#核心概念)
3. [快速开始](#快速开始)
4. [API 参考](#api-参考)
5. [配置](#配置)
6. [示例](#示例)
7. [高级用法](#高级用法)
8. [故障排除](#故障排除)

## 架构

### 核心组件

```
api/
├── script_registry.py    # 脚本发现和管理
├── config_mapper.py      # 配置到参数的映射
├── script_executor.py    # 统一脚本执行
├── easyedit_api.py       # 主要 API 接口
└── examples/            # 使用示例
```

### 组件交互

1. **脚本注册表**: 自动发现 examples 目录中的运行脚本并提取元数据
2. **配置映射器**: 将 JSON 配置映射到脚本特定参数
3. **脚本执行器**: 具有异步支持的统一执行引擎
4. **EasyEdit API**: 提供对所有功能的统一访问的主要接口

## 核心概念

### 1. 脚本自动发现

API 自动分析运行脚本以提取：
- 支持的编辑方法（ROME、FT、MEMIT 等）
- 兼容的数据集和模型
- 必需和可选参数
- 脚本描述和元数据

### 2. 配置映射

JSON 配置通过意图分析映射到脚本参数：
- **模型意图**: 指定编辑方法和模型配置
- **编辑意图**: 定义编辑目标和约束
- **数据集意图**: 配置数据源和参数
- **执行意图**: 设置执行参数和要求

### 3. 统一执行

支持多种执行模式的单一接口：
- **同步**: 阻塞执行，立即获得结果
- **异步**: 非阻塞执行，具有任务管理
- **批量**: 具有并发控制的多个实验
- **直接**: 使用自定义参数执行特定脚本

## 快速开始

### 安装

```python
# 确保 EasyEdit 已安装（如果尚未安装）
pip install -r requirements.txt

# 将 API 添加到 Python 路径
import sys
sys.path.append('E:/ContiAI/EasyEdit')

# 导入 API
from easyedit_api import EasyEditAPI
```

### 基本用法

```python
# 初始化 API
api = EasyEditAPI()

# 列出可用脚本
scripts = api.list_available_scripts()
print(f"找到 {len(scripts)} 个脚本")

# 使用配置执行实验
config = {
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
            "ds_size": 100
        }
    }
}

# 执行实验
result = api.execute_experiment(config)
print(f"成功: {result['success']}")
print(f"执行时间: {result['execution_time']:.2f}秒")
```

### 快速实验

```python
# 常见实验的简化接口
result = api.quick_experiment(
    method="ROME",
    model="llama-7b",
    dataset="zsre",
    hparams_dir="../hparams/ROME/llama-7b",
    data_dir="./data",
    ds_size=50
)
```

### 获取支持的方法、数据集和模型

```python
# 获取所有支持的编辑方法
methods = api.get_supported_methods()
print(f"支持的方法: {len(methods)}")
print(f"方法列表: {methods[:10]}...")  # 显示前10个

# 获取所有支持的数据集
datasets = api.get_supported_datasets()
print(f"支持的数据集: {len(datasets)}")

# 获取所有支持的模型
models = api.get_supported_models()
print(f"支持的模型: {len(models)}")
```

### 配置建议

```python
# 获取特定方法和数据集的建议配置
config = api.suggest_config(method="ROME", dataset="ZsreDataset")
print(f"建议的配置: {config}")

# 验证方法-数据集组合
is_valid = api.validate_method_dataset_combination("ROME", "ZsreDataset")
print(f"ROME + ZsreDataset 是否有效: {is_valid}")
```

## API 参考

### EasyEditAPI

EasyEdit 框架的主要接口。

#### 构造函数

```python
EasyEditAPI(examples_dir=None, base_dir=None, enable_async=True)
```

- `examples_dir`: 包含运行脚本的目录（默认：EasyEdit/examples）
- `base_dir`: EasyEdit 项目的基础目录
- `enable_async`: 是否启用异步执行

#### 发现方法

##### `list_available_scripts()`

列出所有发现的运行脚本。

**返回值:** 脚本信息字典列表

**示例：**
```python
scripts = api.list_available_scripts()
for script in scripts:
    print(f"{script['name']}: {script['description']}")
    print(f"  方法: {script['supported_methods']}")
    print(f"  数据集: {script['supported_datasets']}")
```

##### `get_script_info(script_name)`

获取特定脚本的信息。

**参数：**
- `script_name`: 脚本名称

**返回值:** 脚本信息字典或 None

##### `find_compatible_scripts(method=None, dataset=None, model=None)`

查找与给定要求兼容的脚本。

**参数：**
- `method`: 编辑方法名称（例如："ROME"、"FT"）
- `dataset`: 数据集名称（例如："ZsreDataset"）
- `model`: 模型名称或模式

**返回值:** 兼容脚本信息列表

**示例：**
```python
# 查找支持 ROME 方法的脚本
rome_scripts = api.find_compatible_scripts(method="ROME")

# 查找特定模型和数据集的脚本
compatible_scripts = api.find_compatible_scripts(
    method="ROME",
    dataset="ZsreDataset",
    model="llama"
)
```

#### 配置方法

##### `validate_config(config)`

验证 JSON 配置而不执行。

**参数：**
- `config`: JSON 文件路径或配置字典

**返回值:** 包含错误和警告的验证结果

**示例：**
```python
validation = api.validate_config(config)
if validation['valid']:
    print("配置有效")
    print(f"选择的脚本: {validation['script_name']}")
else:
    print("配置错误:")
    for error in validation['errors']:
        print(f"  - {error}")
```

##### `map_config_to_parameters(config, intent_requirements=None)`

将 JSON 配置映射到脚本特定参数。

**参数：**
- `config`: JSON 文件路径或配置字典
- `intent_requirements`: 额外的意图要求

**返回值:** 映射的参数字典

**示例：**
```python
mapping = api.map_config_to_parameters(config)
print(f"脚本: {mapping['script_name']}")
print("参数:")
for key, value in mapping['parameters'].items():
    print(f"  {key}: {value}")
```

##### `create_default_config()`

创建默认配置模板。

**返回值:** 默认配置字典

#### 执行方法

##### `execute_experiment(config, async_execution=None, timeout=None)`

从 JSON 配置执行实验。

**参数：**
- `config`: JSON 文件路径或配置字典
- `async_execution`: 是否异步执行（默认：API 设置）
- `timeout`: 执行超时时间（秒）

**返回值:** 同步时为执行结果，异步时为任务信息

**示例：**
```python
# 同步执行
result = api.execute_experiment(config)
print(f"成功: {result['success']}")
print(f"时间: {result['execution_time']:.2f}秒")

# 异步执行
task_info = api.execute_experiment(config, async_execution=True)
task_id = task_info['task_id']
print(f"任务已提交: {task_id}")
```

##### `execute_script(script_name, parameters, async_execution=None, timeout=None)`

使用参数执行特定脚本。

**参数：**
- `script_name`: 要执行的脚本名称
- `parameters`: 脚本参数字典
- `async_execution`: 是否异步执行
- `timeout`: 执行超时时间（秒）

**返回值:** 同步时为执行结果，异步时为任务信息

**示例：**
```python
# 执行特定脚本
result = api.execute_script(
    script_name="run_zsre_llama2",
    parameters={
        "editing_method": "ROME",
        "hparams_dir": "../hparams/ROME/llama-7b",
        "data_dir": "./data",
        "ds_size": 100
    }
)
```

##### `quick_experiment(method, model, dataset, hparams_dir=None, data_dir="./data", **kwargs)`

使用常见参数快速执行实验。

**参数：**
- `method`: 编辑方法（ROME、FT、MEMIT 等）
- `model`: 模型名称/路径
- `dataset`: 数据集名称
- `hparams_dir`: 超参数目录
- `data_dir`: 数据目录
- `**kwargs`: 额外参数

**返回值:** 执行结果

#### 任务管理方法

##### `get_task_status(task_id)`

获取执行任务的状态。

**参数：**
- `task_id`: 任务 ID

**返回值:** 任务状态信息

##### `get_task_result(task_id)`

获取已完成任务的结果。

**参数：**
- `task_id`: 任务 ID

**返回值:** 任务结果字典或 None

##### `wait_for_completion(task_id, timeout=None)`

等待任务完成。

**参数：**
- `task_id`: 任务 ID
- `timeout`: 最大等待时间（秒）

**返回值:** 如果任务成功完成则为 True

##### `cancel_task(task_id)`

取消运行中的任务。

**参数：**
- `task_id`: 任务 ID

**返回值:** 如果任务已取消则为 True

##### `list_active_tasks()`

列出所有活动任务。

**返回值:** 活动任务信息列表

#### 批量执行

##### `batch_execute(configs, max_concurrent=3)`

批量执行多个实验。

**参数：**
- `configs`: 配置列表（文件路径或字典）
- `max_concurrent`: 最大并发执行数

**返回值:** 任务结果列表

**示例：**
```python
# 执行多个实验
configs = [config1, config2, config3]
results = api.batch_execute(configs, max_concurrent=2)

for i, result in enumerate(results):
    print(f"实验 {i+1}: 成功={result.get('success', False)}")
```

#### 工具方法

##### `get_system_info()`

获取系统信息和功能。

**返回值:** 系统信息字典

**示例：**
```python
info = api.get_system_info()
print(f"可用脚本: {info['available_scripts']}")
print(f"异步执行: {info['async_execution_enabled']}")
print(f"活动任务: {info['active_tasks']}")
```

##### `get_supported_methods()`

获取所有支持的编辑方法。

**返回值:** 方法名称列表

**示例：**
```python
methods = api.get_supported_methods()
print(f"支持的方法: {methods}")
```

##### `get_supported_datasets()`

获取所有支持的数据集。

**返回值:** 数据集名称列表

**示例：**
```python
datasets = api.get_supported_datasets()
print(f"支持的数据集: {datasets}")
```

##### `get_supported_models()`

获取所有支持的模型。

**返回值:** 模型名称列表

**示例：**
```python
models = api.get_supported_models()
print(f"支持的模型: {models}")
```

##### `suggest_config(method, dataset, **kwargs)`

为特定方法和数据集建议配置。

**参数：**
- `method`: 编辑方法名称
- `dataset`: 数据集名称
- `**kwargs`: 额外配置参数

**返回值:** 建议的配置字典

**示例：**
```python
config = api.suggest_config(method="ROME", dataset="ZsreDataset")
print(f"建议的配置: {config}")
```

##### `validate_method_dataset_combination(method, dataset)`

验证方法-数据集组合是否有效。

**参数：**
- `method`: 编辑方法名称
- `dataset`: 数据集名称

**返回值:** 如果组合有效则为 True

**示例：**
```python
is_valid = api.validate_method_dataset_combination("ROME", "ZsreDataset")
print(f"组合有效: {is_valid}")
```

##### `save_script_registry(output_path)`

将脚本注册表保存到文件。

##### `load_script_registry(input_path)`

从文件加载脚本注册表。

##### `cleanup()`

清理资源。

## 配置

### JSON 配置结构

API 使用 JSON 配置文件来指定实验参数：

```json
{
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
      "method": "ROME",
      "constraints": {
        "locality_preservation": "high",
        "generalization": "medium"
      }
    }
  },
  "dataset": {
    "data_intent": {
      "data_dir": "./data",
      "ds_size": 100,
      "required_fields": ["prompt", "target_new", "ground_truth"]
    }
  },
  "execution": {
    "execution_intent": {
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
```

### 配置部分

#### 模型意图
- `purpose`: 总体目的（例如："knowledge_editing"）
- `method`: 要使用的编辑方法
- `hparams_dir`: 包含超参数的目录
- `architecture_preference`: 模型架构偏好
- `size_preference`: 模型大小偏好

#### 编辑意图
- `goal`: 编辑目标（例如："knowledge_editing"）
- `method`: 特定编辑方法
- `strategy`: 编辑策略
- `constraints`: 编辑约束和要求

#### 数据集意图
- `data_dir`: 包含数据集文件的目录
- `ds_size`: 数据集大小限制
- `required_fields`: 必需的数据字段
- `data_type`: 数据类型（结构化、对话式等）

#### 执行意图
- `metrics_save_dir`: 保存指标的目录
- `quality_requirements`: 质量阈值和要求
- `efficiency_requirements`: 时间和资源约束

### 环境变量

配置文件支持环境变量替换：

```json
{
  "model": {
    "model_intent": {
      "hparams_dir": "${HPARAMS_DIR}/ROME/llama-7b"
    }
  },
  "dataset": {
    "data_intent": {
      "data_dir": "${DATA_DIR}"
    }
  }
}
```

可以在 API 使用前设置环境变量：
```python
import os
os.environ["HPARAMS_DIR"] = "../hparams"
os.environ["DATA_DIR"] = "./data"
```

## 示例

### 基本用法示例

```python
from easyedit_api import EasyEditAPI

# 初始化 API
api = EasyEditAPI()

# 1. 获取系统概览和功能
print("=== 系统概览 ===")
system_info = api.get_system_info()
print(f"可用脚本: {system_info['available_scripts']}")
print(f"支持的编辑方法: {len(api.get_supported_methods())}")
print(f"支持的数据集: {len(api.get_supported_datasets())}")
print(f"支持的模型: {len(api.get_supported_models())}")

# 2. 查看支持的方法、数据集和模型
print("\n=== 支持的功能 ===")
methods = api.get_supported_methods()
datasets = api.get_supported_datasets()
models = api.get_supported_models()

print(f"编辑方法 ({len(methods)}): {', '.join(methods[:10])}...")
print(f"数据集 ({len(datasets)}): {', '.join(datasets[:8])}...")
print(f"模型 ({len(models)}): {', '.join(models[:8])}...")

# 3. 发现可用脚本
print("\n=== 脚本发现 ===")
scripts = api.list_available_scripts()
print(f"找到 {len(scripts)} 个脚本")

# 4. 高级兼容性搜索
print("\n=== 兼容性搜索 ===")
rome_scripts = api.find_compatible_scripts(method="ROME")
print(f"支持 ROME 的脚本: {len(rome_scripts)}")
zsre_scripts = api.find_compatible_scripts(dataset="ZsreDataset")
print(f"支持 ZsreDataset 的脚本: {len(zsre_scripts)}")
llama_scripts = api.find_compatible_scripts(model="llama")
print(f"支持 llama 的脚本: {len(llama_scripts)}")

# 5. 方法-数据集验证
print("\n=== 方法-数据集验证 ===")
validation_pairs = [
    ("ROME", "ZsreDataset"),
    ("FT", "CounterFactDataset"),
    ("MEMIT", "WikiBioDataset"),
    ("KN", "KnowEditDataset"),
]

for method, dataset in validation_pairs:
    is_valid = api.validate_method_dataset_combination(method, dataset)
    print(f"{method} + {dataset}: {'✓ 有效' if is_valid else '✗ 无效'}")

# 6. 配置建议和创建
print("\n=== 配置管理 ===")
# 获取建议配置
config = api.suggest_config(method="ROME", dataset="ZsreDataset")
print(f"为 ROME + ZsreDataset 建议的配置")
print(f"方法: {config['model']['model_intent']['method']}")
print(f"模型偏好: {config['model']['model_intent'].get('architecture_preference', 'llama')}")

# 7. 验证配置
validation = api.validate_config(config)
if validation['valid']:
    print("配置有效")
    print(f"选择的脚本: {validation['script_name']}")

    # 8. 映射配置到参数
    mapping = api.map_config_to_parameters(config)
    print(f"映射到脚本: {mapping['script_name']}")
    print(f"参数数量: {len(mapping['parameters'])}")

    # 9. 执行实验（注释掉以安全运行）
    # result = api.execute_experiment(config)
    # print(f"实验完成: {result['success']}")
else:
    print("配置错误:", validation['errors'])
```

### 配置文件示例

```python
# 保存配置到文件
import json

config = {
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
            "ds_size": 100
        }
    }
}

with open("my_experiment.json", "w") as f:
    json.dump(config, f, indent=2)

# 从文件执行
result = api.execute_experiment("my_experiment.json")
```

### 异步执行示例

```python
# 启用异步执行
api = EasyEditAPI(enable_async=True)

# 提交多个任务
task_ids = []
configs = [config1, config2, config3]

for config in configs:
    task_info = api.execute_experiment(config, async_execution=True)
    task_ids.append(task_info['task_id'])
    print(f"已提交任务: {task_info['task_id']}")

# 监控任务
for task_id in task_ids:
    status = api.get_task_status(task_id)
    print(f"任务 {task_id}: {status['status']}")

# 等待完成
for task_id in task_ids:
    success = api.wait_for_completion(task_id, timeout=300)
    if success:
        result = api.get_task_result(task_id)
        print(f"任务 {task_id} 成功完成")
    else:
        print(f"任务 {task_id} 失败或超时")
```

### 批量执行示例

```python
# 使用并发控制执行多个实验
configs = [
    {"model": {"model_intent": {"method": "ROME", "hparams_dir": "../hparams/ROME/llama-7b"}},
     "dataset": {"data_intent": {"data_dir": "./data", "ds_size": 50}}},
    {"model": {"model_intent": {"method": "FT", "hparams_dir": "../hparams/FT/llama-7b"}},
     "dataset": {"data_intent": {"data_dir": "./data", "ds_size": 50}}},
    {"model": {"model_intent": {"method": "MEMIT", "hparams_dir": "../hparams/MEMIT/llama-7b"}},
     "dataset": {"data_intent": {"data_dir": "./data", "ds_size": 50}}}
]

# 最多并发执行 2 个任务
results = api.batch_execute(configs, max_concurrent=2)

for i, result in enumerate(results):
    print(f"实验 {i+1}:")
    print(f"  成功: {result.get('success', False)}")
    print(f"  时间: {result.get('execution_time', 0):.2f}秒")
    print(f"  脚本: {result.get('script_name', 'unknown')}")
```

## 高级用法

### 自定义脚本发现

API 自动发现 examples 目录中的脚本。脚本被分析以获取：
- 方法支持（ROME、FT、MEMIT 等）
- 数据集兼容性（ZsreDataset、KnowEditDataset 等）
- 模型支持（llama、gpt2 等）
- 参数要求

### 参数映射

API 将通用配置参数映射到脚本特定参数：

```python
# 通用配置
config = {
    "model": {"model_intent": {"method": "ROME"}},
    "dataset": {"data_intent": {"data_dir": "./data"}}
}

# 映射到脚本特定参数
# 对于 run_zsre_llama2.py:
# {"editing_method": "ROME", "data_dir": "./data", ...}
# 对于 run_knowedit_llama2.py:
# {"editing_method": "ROME", "data_dir": "./data", ...}
```

### 错误处理和验证

```python
# 执行前验证
validation = api.validate_config(config)
if not validation['valid']:
    print("配置问题:")
    for error in validation['errors']:
        print(f"  - {error}")
    # 修复配置并重试
else:
    result = api.execute_experiment(config)
```

### 资源管理

```python
# 监控系统资源
info = api.get_system_info()
print(f"活动任务: {info['active_tasks']}")
print(f"已完成任务: {info['completed_tasks']}")

# 列出活动任务
active_tasks = api.list_active_tasks()
for task in active_tasks:
    print(f"任务 {task['task_id']}: {task['status']}")

# 如果需要，取消任务
if active_tasks:
    cancelled = api.cancel_task(active_tasks[0]['task_id'])
    print(f"已取消任务: {cancelled}")
```

## 故障排除

### 常见问题

#### 1. 脚本发现问题

**症状：** 未找到脚本或脚本信息不正确

**解决方案：**
- 验证 examples 目录路径
- 检查脚本文件权限
- 确保脚本遵循命名约定（run_*.py）
- 检查脚本是否可读且包含有效的 Python 代码

#### 2. 配置验证错误

**症状：** 配置验证失败

**解决方案：**
- 检查必需的部分（model、editing）
- 验证方法名称是否受支持
- 确保文件路径正确
- 使用 `validate_config()` 获取详细的错误信息

#### 3. 执行失败

**症状：** 脚本无法执行

**解决方案：**
- 验证脚本路径和权限
- 检查是否提供了所有必需的参数
- 确保模型和数据集文件存在
- 查看脚本特定要求

#### 4. 导入错误

**症状：** 模块导入错误

**解决方案：**
- 确保 EasyEdit 已正确安装
- 检查 Python 路径是否包含 EasyEdit 目录
- 验证所有依赖项已安装

### 调试模式

启用调试日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或检查脚本注册表内容
api = EasyEditAPI()
scripts = api.list_available_scripts()
for script in scripts:
    print(f"脚本: {script['name']}")
    print(f"  方法: {script['supported_methods']}")
    print(f"  参数: {script['required_parameters']}")
```

### 获取帮助

- 查看 `api/examples/` 中的示例
- 使用 `validate_config()` 调试配置问题
- 使用 `list_available_scripts()` 查看脚本信息
- 使用 `get_system_info()` 检查系统状态

## 最佳实践

1. **先验证**: 始终在执行前验证配置
2. **从小开始**: 使用小的数据集大小进行测试
3. **监控资源**: 跟踪活动任务和系统资源
4. **使用异步**: 对长时间运行的实验使用异步执行
5. **处理错误**: 实现适当的错误处理和重试逻辑
6. **批量处理**: 对多个实验使用批量执行
7. **清理**: 完成后记得清理资源

## 贡献

要为 EasyEdit API 做出贡献：

1. 遵循现有的代码结构和模式
2. 添加适当的错误处理
3. 为新功能更新文档
4. 使用多个脚本和配置进行测试
5. 确保向后兼容性

## 许可证

此 API 是 EasyEdit 框架的一部分。有关许可信息，请参见主仓库。