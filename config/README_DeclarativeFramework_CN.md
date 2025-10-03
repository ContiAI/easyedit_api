# EasyEdit 声明式框架

## 介绍

EasyEdit 声明式框架是一种革命性的模型编辑方法，它将**实验意图**与**执行过程**彻底分离。研究者不再需要手动构造复杂的命令行参数和选择脚本，而是可以通过结构化的YAML配置文件来声明他们的实验目标。

## 理念

### 当前痛点

- **手动脚本选择**: 研究者必须为每种编辑方法手动选择合适的脚本
- **复杂命令行**: 冗长且容易出错的命令行参数
- **刚性配置**: 硬编码的参数限制了实验和扩展性
- **管理开销**: 随着编辑方法增加，管理成本呈指数级增长
- **迭代缓慢**: 手动流程拖慢研究和实验速度
- **扩展性有限**: 添加新方法/模型/数据集需要手动配置更新

### 我们的解决方案

我们正在构建一个智能的**意图驱动框架**（Core API），它能够：
- 接受YAML格式的实验意图声明
- 基于用户目标自动发现最优的方法、模型和数据集
- 将通用参数映射到任何特定的方法实现
- 提供智能的执行优化和自适应
- 实现真正扩展性的零配置插件架构

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                       意图接口层                               │
├─────────────────────────────────────────────────────────────────┤
│  用户意图声明 → 智能发现 → 自适应执行                          │
│  (想要实现什么) → &映射     → (自动优化和执行)                  │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                    插件架构                                   │
├─────────────────────────────────────────────────────────────────┤
│  方法插件 │ 模型插件 │ 数据集插件 │ 执行插件                   │
│  (自动发现)│(自动适配) │(自动检测) │ (智能调度)                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心工作流

### 1. 意图声明阶段（用户侧）
用户在YAML配置文件中声明他们的实验意图和目标：

```yaml
# 模型意图 - 需要什么模型能力
model:
  model_intent:
    purpose: "knowledge_editing"
    architecture_preference: "auto"
    size_preference: "medium"
    performance_requirements:
      editing_compatibility: "high"

  model_discovery:
    auto_discover: true
    discovery_sources:
      local_paths: ["./hugging_cache/", "./models/"]
    selection_criteria:
      compatibility_score: 0.8

# 编辑意图 - 想要实现什么编辑目标
editing:
  intent:
    goal: "knowledge_editing"
    strategy: "precise_localization"
    constraints:
      locality_preservation: "high"
      generalization: "medium"
      training_required: false

  method_selection:
    preferred_method: "auto"  # 让系统发现最优方法
    selection_criteria:
      accuracy_weight: 0.7
      speed_weight: 0.2
      memory_weight: 0.1

# 数据集意图 - 需要什么数据
dataset:
  data_intent:
    purpose: "knowledge_editing"
    domain: "general_knowledge"
    data_type: "structured_factual"
    required_fields: ["prompt", "target_new", "ground_truth"]

  dataset_discovery:
    auto_discover: true
    discovery_sources:
      local_paths: ["./data/", "./datasets/"]
    selection_criteria:
      compatibility_score: 0.8
```

### 2. 智能发现和映射阶段（Core API）
Core API：
- 分析用户意图和需求
- 自动发现兼容的模型、方法和数据集
- 将通用参数映射到特定方法实现
- 生成优化的执行计划
- 处理架构和格式适配

### 3. 自适应执行阶段（智能引擎）
智能执行引擎：
- 自动选择最优执行策略
- 监控进度并适应条件变化
- 处理资源优化和容错
- 提供实时反馈和控制
- 从执行历史中学习以优化未来执行

## 核心特性

### 意图驱动架构
- **声明式意图**: 定义想要实现什么，而不是如何实现
- **目标导向**: 指定结果和约束，让系统处理实现细节
- **通用抽象**: 通过统一接口适用于任何方法、模型或数据集
- **智能推荐**: 基于目标系统建议最优配置

### 智能发现与适配
- **自动发现**: 自动查找兼容的模型、方法和数据集
- **架构适配**: 自动适配任何模型架构，无需手动配置
- **格式推断**: 自动检测和适配任何数据集格式
- **参数映射**: 通用参数与特定方法参数之间的智能转换

### 通用参数系统
- **基于意图的参数**: 指定目标（精度、速度、内存）而非技术值
- **自动优化**: 自动调整参数以获得最佳性能
- **多目标优化**: 平衡竞争目标（精度vs速度vs内存）
- **上下文感知映射**: 参数根据方法、模型、数据集和硬件上下文自适应

### 可扩展插件架构
- **零配置插件**: 添加新方法、模型、数据集无需更新核心系统
- **动态加载**: 插件在运行时自动发现和加载
- **接口标准化**: 一致的插件接口确保兼容性
- **生命周期管理**: 自动插件验证、依赖解析和更新

### 自适应执行引擎
- **智能优化**: 基于性能和资源使用进行实时优化
- **预测扩展**: 预测资源需求并相应扩展
- **容错性**: 自动错误恢复和回退策略
- **学习系统**: 基于历史执行数据持续改进

### 未来就绪设计
- **AI驱动优化**: 机器学习用于参数调整和方法选择
- **分布式执行**: 内置对集群和云部署的支持
- **多目标决策制定**: 自动平衡竞争目标
- **持续学习**: 每次执行后系统都会改进

## 配置文件

### 1. 主配置模板（`config_template.yaml`）
革命性的意图驱动配置结构：
- **意图声明**: 想要实现什么（目标、约束、偏好）
- **发现配置**: 如何找到最优组件（模型、方法、数据集）
- **通用参数**: 适用于任何方法的目标导向参数
- **插件架构**: 任何未来组件的可扩展系统
- **智能执行**: 自适应执行优化和监控
- **AI驱动特性**: 机器学习优化和预测能力

### 2. 方法配置（`method_profiles_template.yaml`）
动态方法发现和注册：
- **自动发现**: 从`easyeditor/models/`目录自动发现方法
- **插件接口**: 任何编辑方法的标准接口
- **动态映射**: 通用参数映射到特定方法实现
- **性能特征**: 自动收集的性能指标和能力
- **扩展性**: 新方法自动集成，无需配置更改

### 3. 数据集配置（`dataset_profiles_template.yaml`)
智能数据处理：
- **格式推断**: 自动检测和适配任何数据集格式
- **模式映射**: 基于语义理解的智能字段映射
- **质量评估**: 自动数据质量评估和过滤
- **兼容性匹配**: 自动匹配数据集与编辑方法
- **通用预处理**: 适用于任何数据集结构和格式

### 4. 脚本注册表（`scripts_registry_template.yaml`）
动态脚本执行系统：
- **自动发现**: 在多个目录中自动查找执行脚本
- **模式匹配**: 脚本组织和命名的灵活模式
- **参数映射**: 配置与脚本参数之间的自动转换
- **验证**: 脚本兼容性和需求的运行时验证
- **回退策略**: 失败时自动选择替代脚本

### 5. 意图驱动示例（`example_intent_driven_experiment.yaml`）
展示意图驱动配置功能的完整示例：
- **意图声明**: 指定目标和约束
- **自动发现**: 让系统找到最优组件
- **通用参数**: 目标导向参数规范
- **智能优化**: 自动参数调整和方法选择

## 示例配置

### 意图驱动实验（`example_intent_driven_experiment.yaml`）
```yaml
# 模型意图 - 需要什么模型能力
model:
  model_intent:
    purpose: "knowledge_editing"
    architecture_preference: "auto"
    size_preference: "medium"

  model_discovery:
    auto_discover: true
    discovery_sources:
      local_paths: ["./hugging_cache/", "./models/"]
    selection_criteria:
      compatibility_score: 0.8

# 编辑意图 - 想要实现什么编辑目标
editing:
  intent:
    goal: "knowledge_editing"
    strategy: "precise_localization"
    constraints:
      locality_preservation: "high"
      generalization: "medium"

  method_selection:
    preferred_method: "auto"  # 让系统发现最优方法
    selection_criteria:
      accuracy_weight: 0.7
      speed_weight: 0.2
      memory_weight: 0.1

# 数据集意图 - 需要什么数据
dataset:
  data_intent:
    purpose: "knowledge_editing"
    domain: "general_knowledge"
    required_fields: ["prompt", "target_new", "ground_truth"]

  dataset_discovery:
    auto_discover: true
    discovery_sources:
      local_paths: ["./data/", "./datasets/"]
```

### 传统兼容示例（`example_rome_experiment.yaml`）
为偏好传统配置的用户提供：
```yaml
experiment:
  name: "rome_zsre_llama2_experiment"
  description: "使用ROME在LLaMA-2上编辑ZsRE数据集"

model:
  name: "llama-2-7b"
  path: "./hugging_cache/llama-2-7b-chat"

editing:
  method: "ROME"
  hyperparameters:
    layers: [5]
    v_lr: 0.5

dataset:
  name: "ZsRE"
  path: "./data/zsre_mend_eval.json"
```

### 意图驱动的批量处理（`example_batch_experiment.yaml`）
```yaml
# 意图驱动的批量处理
editing_tasks:
  - task_name: "auto_knowledge_editing"
    intent:
      goal: "knowledge_editing"
      strategy: "precise_localization"
    auto_discover: true  # 系统找到最优方法和数据集

  - task_name: "efficient_behavior_modification"
    intent:
      goal: "behavior_modification"
      constraints:
        computational_cost: "low"
        execution_time: "fast"
    auto_discover: true
```

## 优势

### 对研究者
- **专注研究**: 专注于实验设计，而非执行细节
- **快速原型**: 快速测试不同的方法和配置
- **可重现科学**: 精确重现任何实验
- **协作**: 轻松分享和复制实验

### 对系统管理员
- **资源管理**: 高效的计算资源分配
- **监控**: 集中监控和日志记录
- **可扩展性**: 从单个实验轻松扩展到大批量
- **维护**: 简化的维护和更新

### 对框架开发者
- **可扩展性**: 轻松添加新方法和特性
- **模块化**: 清晰的关注点分离
- **测试**: 标准化的测试框架
- **文档**: 自文档化的配置系统

## 路线图

### 第一阶段：基础
- [x] 配置模板设计
- [x] 方法和数据集配置
- [x] 脚本注册系统
- [ ] Core API实现

### 第二阶段：核心功能
- [ ] 基础执行引擎
- [ ] 参数映射和验证
- [ ] 进度监控
- [ ] 错误处理和恢复

### 第三阶段：高级功能
- [ ] 批量处理
- [ ] 序列化编辑工作流
- [ ] 资源管理
- [ ] 分布式执行

### 第四阶段：生态系统
- [ ] Web界面
- [ ] 可视化工具
- [ ] 与ML平台集成
- [ ] 社区插件

## 开始使用

1. **克隆仓库**
   ```bash
   git clone https://github.com/zjunlp/EasyEdit.git
   cd EasyEdit
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置实验**
   - 将`config/config_template.yaml`复制到你的实验目录
   - 根据需要修改配置
   - 使用方法和数据集配置作为参考

4. **执行实验**
   ```bash
   # 当Core API实现后
   python easyedit/core_api.py --config your_config.yaml
   ```

## 贡献

我们欢迎对声明式框架的贡献：

1. **新方法**: 在方法配置中添加你的编辑方法
2. **新数据集**: 在数据集配置中注册你的数据集
3. **新脚本**: 在脚本注册表中映射你的执行脚本
4. **改进**: 建议对配置系统的改进

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 致谢

- EasyEdit团队的基础工作
- 各种编辑方法的贡献者
- 提供反馈和建议的研究社区