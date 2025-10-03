# EasyEdit API Examples

本目录包含 EasyEdit API 的使用示例，从基础用法到高级特性。

## 示例文件

### 1. basic_usage.py - 基础用法示例
演示 EasyEdit API 的基本功能：
- 系统信息获取
- 脚本自动发现
- 兼容性搜索
- 方法-数据集验证
- 配置建议和创建
- 参数映射
- 快速实验配置
- 批量实验示例
- 脚本注册表操作

**运行方式：**
```bash
cd E:\ContiAI\EasyEdit\api\examples
python basic_usage.py
```

### 2. advanced_usage.py - 高级用法示例
演示高级功能和复杂场景：
- 高级脚本发现
- 复杂配置场景
- 异步执行
- 错误处理和恢复
- 性能优化
- 自定义使用模式

**运行方式：**
```bash
cd E:\ContiAI\EasyEdit\api\examples
python advanced_usage.py
```

### 3. test_examples.py - 示例测试脚本
验证示例代码是否正常工作：
- 基础功能测试
- 高级特性测试
- 错误处理测试

**运行方式：**
```bash
cd E:\ContiAI\EasyEdit\api\examples
python test_examples.py
```

## 核心功能演示

### 系统概览
```python
from easyedit_api import EasyEditAPI

api = EasyEditAPI()

# 获取系统信息
system_info = api.get_system_info()
print(f"可用脚本: {system_info['available_scripts']}")

# 获取支持的方法、数据集、模型
methods = api.get_supported_methods()
datasets = api.get_supported_datasets()
models = api.get_supported_models()

print(f"支持的方法: {len(methods)}")
print(f"支持的数据集: {len(datasets)}")
print(f"支持的模型: {len(models)}")
```

### 兼容性搜索
```python
# 按方法搜索
rome_scripts = api.find_compatible_scripts(method="ROME")

# 按数据集搜索
zsre_scripts = api.find_compatible_scripts(dataset="ZsreDataset")

# 按模型搜索
llama_scripts = api.find_compatible_scripts(model="llama")

# 组合搜索
compatible_scripts = api.find_compatible_scripts(
    method="ROME",
    dataset="ZsreDataset",
    model="llama"
)
```

### 配置管理
```python
# 获取建议配置
config = api.suggest_config(method="ROME", dataset="ZsreDataset")

# 验证方法-数据集组合
is_valid = api.validate_method_dataset_combination("ROME", "ZsreDataset")

# 验证配置
validation = api.validate_config(config)

# 映射到脚本参数
mapping = api.map_config_to_parameters(config)
```

### 实验执行
```python
# 快速实验
result = api.quick_experiment(
    method="ROME",
    model="llama-7b",
    dataset="zsre",
    hparams_dir="../hparams/ROME/llama-7b",
    data_dir="./data"
)

# 批量实验
configs = [config1, config2, config3]
results = api.batch_execute(configs, max_concurrent=2)

# 异步实验
task_info = api.execute_experiment(config, async_execution=True)
task_id = task_info['task_id']
status = api.get_task_status(task_id)
```

## 常见使用模式

### 1. 方法比较
```python
methods = ["ROME", "FT", "MEMIT"]
results = []

for method in methods:
    config = api.suggest_config(method=method, dataset="ZsreDataset")
    result = api.execute_experiment(config)
    results.append((method, result))
```

### 2. 数据集扩展性测试
```python
sizes = [10, 50, 100, 200]
results = []

for size in sizes:
    config = api.suggest_config(method="ROME", dataset="ZsreDataset")
    config['dataset']['data_intent']['ds_size'] = size
    result = api.execute_experiment(config)
    results.append((size, result))
```

### 3. 模型家族测试
```python
model_families = [
    ("llama", ["7b", "13b"]),
    ("gpt2", ["xl", "large"]),
    ("baichuan", ["7b", "13b"])
]

for family, sizes in model_families:
    for size in sizes:
        model = f"{family}-{size}"
        scripts = api.find_compatible_scripts(model=model)
        print(f"{model}: {len(scripts)} 兼容脚本")
```

## 支持的编辑方法

- **ROME**: Rank-One Model Editing
- **FT**: Fine-Tuning
- **MEMIT**: Mass Editing Memory in Transformer
- **KN**: Knowledge Neurons
- **PMET**: Parameter-Efficient Model Editing
- **DINM**: Dynamic Integration of Neural Modules
- **SERAC**: SElective Retrieval and Augmentation Cache
- **IKE**: In-Context Knowledge Editing
- **GRACE**: Graph-based Representation Alteration for Concept Editing
- **MELO**: Multilingual Editing with Locality preservation
- **WISE**: Wikipedia-based Integrated Semantic Editing
- **MEND**: Model Editor Networks with Gradient Decomposition
- **InstructEdit**: Instruction-based Editing
- **MALMEN**: Meta-Learning for Model Editing
- **AdaLoRA**: Adaptive Low-Rank Adaptation
- **AlphaEdit**: Alpha-based Editing
- **ConvsEnt**: Conversational Entity Editing
- **SafeEdit**: Safety-focused Editing
- **UltraEdit**: Ultra-efficient Editing
- **CKnowEdit**: Concept Knowledge Editing
- **HalluEditBench**: Hallucination Editing Benchmark
- **LLMEval**: LLM Evaluation Editing
- **WikiBigEdit**: Wikipedia Large-scale Editing
- **AdsEdit**: Adversarial Defense Editing
- **ConceptEdit**: Concept-specific Editing
- **PersonalityEdit**: Personality Editing
- **SafetyEdit**: Safety Editing
- **WISEEdit**: WISE-based Editing

## 支持的数据集

- **ZsreDataset**: Zero-shot Relation Extraction Dataset
- **KnowEditDataset**: Knowledge Editing Dataset
- **WikiBioDataset**: Wikipedia Biography Dataset
- **CounterFactDataset**: Counterfactual Dataset
- **CFDataset**: Counterfactual Dataset (alternative)
- **RecentDataset**: Recent Events Dataset
- **MQuAKEDataset**: Multi-hop Question Answering for Knowledge Editing
- **MQuAKE_CF_Dataset**: MQuAKE Counterfactual Dataset
- **MQuAKE_T_Dataset**: MQuAKE Temporal Dataset
- **HalluEditBench_Dataset**: Hallucination Editing Benchmark Dataset
- **SafetyDataset**: Safety Editing Dataset
- **CoTDataset**: Chain of Thought Dataset
- **MathDataset**: Mathematical Reasoning Dataset
- **CodeDataset**: Code Generation Dataset
- **MultiTurnDataset**: Multi-turn Conversation Dataset

## 支持的模型

- **llama**: LLaMA family (7B, 13B, 30B, 65B)
- **llama2**: LLaMA 2 family
- **llama3**: LLaMA 3 family
- **gpt2**: GPT-2 family (small, medium, large, xl)
- **gptj**: GPT-J 6B
- **gpt_neox**: GPT-NeoX 20B
- **baichuan**: Baichuan family (7B, 13B)
- **chatglm**: ChatGLM family (6B, 130B)
- **internlm**: InternLM family (7B, 20B)
- **qwen**: Qwen family (7B, 14B, 72B)
- **mistral**: Mistral family (7B, 8x7B)
- **mixtral**: Mixtral 8x7B
- **opt**: OPT family (125M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B)
- **bloom**: BLOOM family (560M, 3B, 7.1B, 176B)
- **roberta**: RoBERTa family
- **t5**: T5 family
- **gemma**: Gemma family (2B, 7B)

## 错误处理

```python
try:
    config = api.suggest_config(method="ROME", dataset="ZsreDataset")
    result = api.execute_experiment(config)
    print(f"实验成功: {result['success']}")
except Exception as e:
    print(f"实验失败: {str(e)}")

    # 获取有效的备选配置
    fallback_configs = []
    for method in ["FT", "MEMIT", "KN"]:
        try:
            config = api.suggest_config(method=method, dataset="ZsreDataset")
            if api.validate_config(config)['valid']:
                fallback_configs.append(config)
        except:
            continue
```

## 性能优化

1. **使用缓存**: API 内部缓存了脚本搜索结果
2. **批量执行**: 使用 `batch_execute()` 进行并发实验
3. **异步执行**: 使用 `async_execution=True` 进行长时间运行的实验
4. **精确搜索**: 指具体的方法、数据集、模型参数以缩小搜索范围

## 故障排除

### 常见问题

1. **脚本未找到**: 检查 examples 目录路径是否正确
2. **配置验证失败**: 使用 `validate_config()` 获取详细错误信息
3. **执行失败**: 检查模型和数据集文件是否存在
4. **导入错误**: 确保 EasyEdit 已正确安装

### 调试技巧

```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查脚本注册表
api = EasyEditAPI()
scripts = api.list_available_scripts()
for script in scripts:
    print(f"脚本: {script['name']}")
    print(f"  方法: {script['supported_methods']}")
    print(f"  数据集: {script['supported_datasets']}")
```

## 贡献

欢迎贡献新的示例和改进建议！

1. 创建新的示例文件
2. 更新此 README 文档
3. 添加测试用例
4. 确保代码风格一致

## 许可证

本示例代码遵循 EasyEdit 项目的许可证。