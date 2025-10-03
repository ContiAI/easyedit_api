# EasyEdit Declarative Framework

## Introduction

The EasyEdit Declarative Framework is a revolutionary approach to model editing that separates **experimental intent** from **execution process**. Instead of manually constructing complex command-line arguments and selecting scripts, researchers can now declare their experimental goals through structured YAML configuration files.

## Philosophy

### Current Pain Points

- **Manual Script Selection**: Researchers must manually choose the appropriate script for each editing method
- **Complex Command Lines**: Long, error-prone command-line arguments
- **Rigid Configuration**: Hard-coded parameters limit experimentation and extensibility
- **Management Overhead**: As editing methods increase, management costs rise exponentially
- **Slow Iteration**: Manual processes slow down research and experimentation
- **Limited Extensibility**: Adding new methods/models/datasets requires manual configuration updates

### Our Solution

We're building an intelligent **intent-driven framework** (Core API) that:
- Accepts high-level experimental intent declarations in YAML format
- Auto-discovers optimal methods, models, and datasets based on user goals
- Maps universal parameters to any method-specific implementation
- Provides intelligent execution optimization and adaptation
- Enables true extensibility with zero-configuration plugin architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Intent Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  User Intent Declaration → Intelligent Discovery → Adaptive    │
│  (WHAT to achieve)        → & Mapping        → Execution        │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│                    Plugin Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│  Method Plugins │ Model Plugins │ Dataset Plugins │ Execution   │
│  (Auto-discover)│(Auto-adapt)   │(Auto-detect)   │ Plugins      │
└─────────────────────────────────────────────────────────────────┘
```

## Core Workflow

### 1. Intent Declaration Phase (User Side)
Users declare their experimental intent and goals:

```yaml
# Model intent - WHAT model capabilities you need
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

# Editing intent - WHAT editing you want to achieve
editing:
  intent:
    goal: "knowledge_editing"
    strategy: "precise_localization"
    constraints:
      locality_preservation: "high"
      generalization: "medium"
      training_required: false

  method_selection:
    preferred_method: "auto"  # Let system discover optimal method
    selection_criteria:
      accuracy_weight: 0.7
      speed_weight: 0.2
      memory_weight: 0.1

# Dataset intent - WHAT data you need
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

### 2. Intelligent Discovery and Mapping Phase (Core API)
The Core API:
- Analyzes user intent and requirements
- Auto-discovers compatible models, methods, and datasets
- Maps universal parameters to method-specific implementations
- Generates optimized execution plans
- Handles architecture and format adaptation

### 3. Adaptive Execution Phase (Intelligent Engine)
The Intelligent Execution Engine:
- Automatically selects optimal execution strategy
- Monitors progress and adapts to conditions
- Handles resource optimization and fault tolerance
- Provides real-time feedback and control
- Learns from execution history for future optimization

## Key Features

### Intent-Driven Architecture
- **Declarative Intent**: Define WHAT you want to achieve, not HOW to achieve it
- **Goal-Oriented**: Specify outcomes and constraints, let system handle implementation
- **Universal Abstraction**: Works with any method, model, or dataset through unified interfaces
- **Smart Recommendations**: System suggests optimal configurations based on goals

### Intelligent Discovery & Adaptation
- **Auto-Discovery**: Automatically find compatible models, methods, and datasets
- **Architecture Adaptation**: Auto-adapt to any model architecture without manual configuration
- **Format Inference**: Automatically detect and adapt to any dataset format
- **Parameter Mapping**: Intelligent translation between universal and method-specific parameters

### Universal Parameter System
- **Intent-Based Parameters**: Specify goals (precision, speed, memory) rather than technical values
- **Auto-Optimization**: Automatically tune parameters for optimal performance
- **Multi-Objective Optimization**: Balance competing objectives (accuracy vs speed vs memory)
- **Context-Aware Mapping**: Parameters adapt to method, model, dataset, and hardware context

### Extensible Plugin Architecture
- **Zero-Configuration Plugins**: Add new methods, models, datasets without updating core system
- **Dynamic Loading**: Plugins auto-discovered and loaded at runtime
- **Interface Standardization**: Consistent plugin interfaces ensure compatibility
- **Lifecycle Management**: Automatic plugin validation, dependency resolution, and updates

### Adaptive Execution Engine
- **Intelligent Optimization**: Real-time optimization based on performance and resource usage
- **Predictive Scaling**: Anticipate resource needs and scale accordingly
- **Fault Tolerance**: Automatic error recovery and fallback strategies
- **Learning System**: Improves over time based on historical execution data

### Future-Ready Design
- **AI-Driven Optimization**: Machine learning for parameter tuning and method selection
- **Distributed Execution**: Built-in support for cluster and cloud deployment
- **Multi-Objective Decision Making**: Balance competing goals automatically
- **Continuous Learning**: System improves with each execution

## Configuration Files

### 1. Main Configuration Template (`config_template.yaml`)
The revolutionary intent-driven configuration structure:
- **Intent Declarations**: What you want to achieve (goals, constraints, preferences)
- **Discovery Configuration**: How to find optimal components (models, methods, datasets)
- **Universal Parameters**: Goal-oriented parameters that work with any method
- **Plugin Architecture**: Extensible system for any future components
- **Intelligent Execution**: Adaptive execution optimization and monitoring
- **AI-Driven Features**: Machine learning optimization and predictive capabilities

### 2. Method Profiles (`method_profiles_template.yaml`)
Dynamic method discovery and registration:
- **Auto-Discovery**: Automatically discover methods from `easyeditor/models/` directories
- **Plugin Interface**: Standardized interfaces for any editing method
- **Dynamic Mapping**: Universal parameters map to method-specific implementations
- **Performance Characteristics**: Auto-collected performance metrics and capabilities
- **Extensibility**: New methods automatically integrated without configuration changes

### 3. Dataset Profiles (`dataset_profiles_template.yaml`)
Intelligent dataset handling:
- **Format Inference**: Automatically detect and adapt to any dataset format
- **Schema Mapping**: Intelligent field mapping based on semantic understanding
- **Quality Assessment**: Automatic data quality evaluation and filtering
- **Compatibility Matching**: Auto-match datasets with editing methods
- **Universal Preprocessing**: Works with any dataset structure and format

### 4. Scripts Registry (`scripts_registry_template.yaml`)
Dynamic script execution system:
- **Auto-Discovery**: Automatically find execution scripts in multiple directories
- **Pattern Matching**: Flexible patterns for script organization and naming
- **Parameter Mapping**: Automatic translation between config and script parameters
- **Validation**: Runtime validation of script compatibility and requirements
- **Fallback Strategies**: Automatic selection of alternative scripts on failure

### 5. Intent-Driven Example (`example_intent_driven_experiment.yaml`)
Complete example demonstrating the power of intent-driven configuration:
- **Intent Declaration**: Specify goals and constraints
- **Auto-Discovery**: Let system find optimal components
- **Universal Parameters**: Goal-oriented parameter specification
- **Intelligent Optimization**: Automatic parameter tuning and method selection

## Example Configurations

### Intent-Driven Experiment (`example_intent_driven_experiment.yaml`)
```yaml
# Model intent - WHAT model capabilities you need
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

# Editing intent - WHAT editing you want to achieve
editing:
  intent:
    goal: "knowledge_editing"
    strategy: "precise_localization"
    constraints:
      locality_preservation: "high"
      generalization: "medium"

  method_selection:
    preferred_method: "auto"  # Let system discover optimal method
    selection_criteria:
      accuracy_weight: 0.7
      speed_weight: 0.2
      memory_weight: 0.1

# Dataset intent - WHAT data you need
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

### Legacy Compatibility Example (`example_rome_experiment.yaml`)
For users who prefer traditional configuration:
```yaml
experiment:
  name: "rome_zsre_llama2_experiment"
  description: "ROME knowledge editing on LLaMA-2 using ZsRE dataset"

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

### Batch Processing with Intent (`example_batch_experiment.yaml`)
```yaml
# Intent-driven batch processing
editing_tasks:
  - task_name: "auto_knowledge_editing"
    intent:
      goal: "knowledge_editing"
      strategy: "precise_localization"
    auto_discover: true  # System finds optimal method and dataset

  - task_name: "efficient_behavior_modification"
    intent:
      goal: "behavior_modification"
      constraints:
        computational_cost: "low"
        execution_time: "fast"
    auto_discover: true
```

## Benefits

### For Researchers
- **Focus on Research**: Concentrate on experimental design, not execution details
- **Rapid Prototyping**: Quickly test different methods and configurations
- **Reproducible Science**: Exact reproduction of any experiment
- **Collaboration**: Easy sharing and replication of experiments

### For System Administrators
- **Resource Management**: Efficient allocation of computational resources
- **Monitoring**: Centralized monitoring and logging
- **Scalability**: Easy scaling from single experiments to large batches
- **Maintenance**: Simplified maintenance and updates

### For Framework Developers
- **Extensibility**: Easy addition of new methods and features
- **Modularity**: Clean separation of concerns
- **Testing**: Standardized testing framework
- **Documentation**: Self-documenting configuration system

## Roadmap

### Phase 1: Foundation
- [x] Configuration template design
- [x] Method and dataset profiling
- [x] Script registry system
- [ ] Core API implementation

### Phase 2: Core Features
- [ ] Basic execution engine
- [ ] Parameter mapping and validation
- [ ] Progress monitoring
- [ ] Error handling and recovery

### Phase 3: Advanced Features
- [ ] Batch processing
- [ ] Sequential editing workflows
- [ ] Resource management
- [ ] Distributed execution

### Phase 4: Ecosystem
- [ ] Web interface
- [ ] Visualization tools
- [ ] Integration with ML platforms
- [ ] Community plugins

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/zjunlp/EasyEdit.git
   cd EasyEdit
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Your Experiment**
   - Copy `config/config_template.yaml` to your experiment directory
   - Modify the configuration according to your needs
   - Use method and dataset profiles as reference

4. **Execute Your Experiment**
   ```bash
   # When Core API is implemented
   python easyedit/core_api.py --config your_config.yaml
   ```

## Contributing

We welcome contributions to the declarative framework:

1. **New Methods**: Add your editing method to the method profiles
2. **New Datasets**: Register your dataset in the dataset profiles
3. **New Scripts**: Map your execution scripts in the scripts registry
4. **Improvements**: Suggest improvements to the configuration system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The EasyEdit team for the foundational work
- Contributors to the various editing methods
- The research community for feedback and suggestions