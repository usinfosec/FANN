# Agent 2: Neural Network Specialist Implementation Plan

## 🧠 Agent Profile
- **Type**: Neural Network Specialist
- **Cognitive Pattern**: Divergent Thinking  
- **Specialization**: ruv-FANN integration, neural network operations, training algorithms
- **Focus**: Exposing complete neural network capabilities through WASM

## 🎯 Mission
Integrate the complete ruv-FANN neural network library into WASM, exposing all 18 activation functions, 5 training algorithms, cascade correlation, and advanced neural network features through high-performance WebAssembly interfaces.

## 📋 Responsibilities

### 1. Complete ruv-FANN WASM Integration

#### Neural Network Core Interface
```rust
// neural_wasm.rs - Main neural network WASM interface

use wasm_bindgen::prelude::*;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingData};
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    inner: Network<f32>,
    training_data: Option<TrainingData<f32>>,
    metrics: NetworkMetrics,
}

#[derive(Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub training_error: f32,
    pub validation_error: f32,
    pub epochs_trained: u32,
    pub total_connections: usize,
    pub memory_usage: usize,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmNeuralNetwork, JsValue> {
        let config: NetworkConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        
        let mut builder = NetworkBuilder::<f32>::new()
            .input_layer(config.input_size);
            
        // Add hidden layers with specified activation functions
        for layer in config.hidden_layers {
            builder = builder.hidden_layer_with_activation(
                layer.size,
                parse_activation_function(&layer.activation)?,
                layer.steepness.unwrap_or(1.0)
            );
        }
        
        let network = builder
            .output_layer_with_activation(
                config.output_size,
                parse_activation_function(&config.output_activation)?,
                1.0
            )
            .connection_rate(config.connection_rate.unwrap_or(1.0))
            .random_seed(config.random_seed)
            .build();
            
        Ok(WasmNeuralNetwork {
            inner: network,
            training_data: None,
            metrics: NetworkMetrics {
                training_error: 0.0,
                validation_error: 0.0,
                epochs_trained: 0,
                total_connections: 0,
                memory_usage: 0,
            }
        })
    }
    
    #[wasm_bindgen]
    pub fn run(&mut self, inputs: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.run(inputs)
            .map_err(|e| JsValue::from_str(&format!("Network run error: {}", e)))
    }
    
    #[wasm_bindgen]
    pub fn set_training_data(&mut self, data: JsValue) -> Result<(), JsValue> {
        let training_data: TrainingDataConfig = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
            
        self.training_data = Some(TrainingData {
            inputs: training_data.inputs,
            outputs: training_data.outputs,
        });
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f32> {
        self.inner.get_weights()
    }
    
    #[wasm_bindgen]
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), JsValue> {
        self.inner.set_weights(weights)
            .map_err(|e| JsValue::from_str(&format!("Set weights error: {}", e)))
    }
    
    #[wasm_bindgen]
    pub fn get_network_info(&self) -> JsValue {
        let info = serde_json::json!({
            "num_layers": self.inner.num_layers(),
            "num_inputs": self.inner.num_inputs(),
            "num_outputs": self.inner.num_outputs(),
            "total_neurons": self.inner.total_neurons(),
            "total_connections": self.inner.total_connections(),
            "metrics": self.metrics
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_size: usize,
    pub hidden_layers: Vec<LayerConfig>,
    pub output_size: usize,
    pub output_activation: String,
    pub connection_rate: Option<f32>,
    pub random_seed: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerConfig {
    pub size: usize,
    pub activation: String,
    pub steepness: Option<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingDataConfig {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}
```

### 2. Complete Activation Function Support

#### All 18 FANN Activation Functions
```rust
// activation_wasm.rs - Complete activation function support

use wasm_bindgen::prelude::*;
use ruv_fann::ActivationFunction;

#[wasm_bindgen]
pub struct ActivationFunctionManager;

#[wasm_bindgen]
impl ActivationFunctionManager {
    #[wasm_bindgen]
    pub fn get_all_functions() -> JsValue {
        let functions = vec![
            ("linear", "Linear function: f(x) = x"),
            ("sigmoid", "Sigmoid: f(x) = 1/(1+e^(-2sx))"),
            ("sigmoid_symmetric", "Symmetric sigmoid: f(x) = tanh(sx)"),
            ("gaussian", "Gaussian: f(x) = e^(-x²s²)"),
            ("gaussian_symmetric", "Symmetric Gaussian"),
            ("gaussian_stepwise", "Stepwise Gaussian"),
            ("elliot", "Elliot function (fast sigmoid approximation)"),
            ("elliot_symmetric", "Symmetric Elliot function"),
            ("relu", "Rectified Linear Unit: f(x) = max(0, x)"),
            ("relu_leaky", "Leaky ReLU: f(x) = x > 0 ? x : 0.01x"),
            ("cos", "Cosine function"),
            ("cos_symmetric", "Symmetric cosine"),
            ("sin", "Sine function"),
            ("sin_symmetric", "Symmetric sine"),
            ("threshold", "Threshold function"),
            ("threshold_symmetric", "Symmetric threshold"),
            ("linear2", "Alternative linear function"),
            ("sinus", "Sinus function"),
        ];
        
        serde_wasm_bindgen::to_value(&functions).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn test_activation_function(name: &str, input: f32, steepness: f32) -> Result<f32, JsValue> {
        let activation = parse_activation_function(name)?;
        
        // Create temporary single neuron for testing
        use ruv_fann::neuron::Neuron;
        let mut neuron = Neuron::new(1, activation, steepness);
        
        Ok(neuron.activate(&[input]))
    }
    
    #[wasm_bindgen]
    pub fn compare_functions(input: f32) -> JsValue {
        let mut results = std::collections::HashMap::new();
        
        let functions = [
            "linear", "sigmoid", "sigmoid_symmetric", "gaussian",
            "elliot", "relu", "relu_leaky", "cos", "sin", "threshold"
        ];
        
        for &func_name in &functions {
            if let Ok(result) = Self::test_activation_function(func_name, input, 1.0) {
                results.insert(func_name.to_string(), result);
            }
        }
        
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}

pub fn parse_activation_function(name: &str) -> Result<ActivationFunction, JsValue> {
    match name.to_lowercase().as_str() {
        "linear" => Ok(ActivationFunction::Linear),
        "sigmoid" => Ok(ActivationFunction::Sigmoid),
        "sigmoid_symmetric" => Ok(ActivationFunction::SigmoidSymmetric),
        "gaussian" => Ok(ActivationFunction::Gaussian),
        "gaussian_symmetric" => Ok(ActivationFunction::GaussianSymmetric),
        "gaussian_stepwise" => Ok(ActivationFunction::GaussianStepwise),
        "elliot" => Ok(ActivationFunction::Elliot),
        "elliot_symmetric" => Ok(ActivationFunction::ElliotSymmetric),
        "relu" => Ok(ActivationFunction::ReLU),
        "relu_leaky" => Ok(ActivationFunction::ReLULeaky),
        "cos" => Ok(ActivationFunction::Cos),
        "cos_symmetric" => Ok(ActivationFunction::CosSymmetric),
        "sin" => Ok(ActivationFunction::Sin),
        "sin_symmetric" => Ok(ActivationFunction::SinSymmetric),
        "threshold" => Ok(ActivationFunction::Threshold),
        "threshold_symmetric" => Ok(ActivationFunction::ThresholdSymmetric),
        "linear2" => Ok(ActivationFunction::Linear2),
        "sinus" => Ok(ActivationFunction::Sinus),
        _ => Err(JsValue::from_str(&format!("Unknown activation function: {}", name))),
    }
}
```

### 3. Complete Training Algorithm Implementation

#### All 5 Training Algorithms with WASM Interface
```rust
// training_wasm.rs - Complete training algorithm support

use wasm_bindgen::prelude::*;
use ruv_fann::training::{IncrementalBackprop, BatchBackprop, Rprop, Quickprop, Sarprop};
use ruv_fann::{TrainingAlgorithm, TrainingData};

#[wasm_bindgen]
pub struct WasmTrainer {
    algorithm: TrainingAlgorithmWasm,
    training_history: Vec<TrainingEpochResult>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: String,
    pub learning_rate: Option<f32>,
    pub momentum: Option<f32>,
    pub max_epochs: u32,
    pub target_error: f32,
    pub validation_split: Option<f32>,
    pub early_stopping: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingEpochResult {
    pub epoch: u32,
    pub training_error: f32,
    pub validation_error: Option<f32>,
    pub time_ms: f64,
}

enum TrainingAlgorithmWasm {
    IncrementalBackprop(IncrementalBackprop),
    BatchBackprop(BatchBackprop),
    Rprop(Rprop),
    Quickprop(Quickprop),
    Sarprop(Sarprop),
}

#[wasm_bindgen]
impl WasmTrainer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmTrainer, JsValue> {
        let config: TrainingConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid training config: {}", e)))?;
        
        let algorithm = match config.algorithm.to_lowercase().as_str() {
            "incremental_backprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::IncrementalBackprop(IncrementalBackprop::new(lr))
            },
            "batch_backprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::BatchBackprop(BatchBackprop::new(lr))
            },
            "rprop" => {
                TrainingAlgorithmWasm::Rprop(Rprop::new())
            },
            "quickprop" => {
                let lr = config.learning_rate.unwrap_or(0.7);
                TrainingAlgorithmWasm::Quickprop(Quickprop::new(lr))
            },
            "sarprop" => {
                TrainingAlgorithmWasm::Sarprop(Sarprop::new())
            },
            _ => return Err(JsValue::from_str(&format!("Unknown training algorithm: {}", config.algorithm))),
        };
        
        Ok(WasmTrainer {
            algorithm,
            training_history: Vec::new(),
        })
    }
    
    #[wasm_bindgen]
    pub fn train_epoch(&mut self, network: &mut WasmNeuralNetwork, training_data: JsValue) -> Result<f32, JsValue> {
        let data: TrainingDataConfig = serde_wasm_bindgen::from_value(training_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
        
        let training_data = TrainingData {
            inputs: data.inputs,
            outputs: data.outputs,
        };
        
        let start_time = js_sys::Date::now();
        
        let error = match &mut self.algorithm {
            TrainingAlgorithmWasm::IncrementalBackprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::BatchBackprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Rprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Quickprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
            TrainingAlgorithmWasm::Sarprop(trainer) => {
                trainer.train_epoch(&mut network.inner, &training_data)
                    .map_err(|e| JsValue::from_str(&format!("Training error: {}", e)))?
            },
        };
        
        let end_time = js_sys::Date::now();
        
        // Record training history
        self.training_history.push(TrainingEpochResult {
            epoch: self.training_history.len() as u32 + 1,
            training_error: error,
            validation_error: None, // TODO: Add validation support
            time_ms: end_time - start_time,
        });
        
        // Update network metrics
        network.metrics.training_error = error;
        network.metrics.epochs_trained += 1;
        
        Ok(error)
    }
    
    #[wasm_bindgen]
    pub fn train_until_target(&mut self, 
                            network: &mut WasmNeuralNetwork, 
                            training_data: JsValue,
                            target_error: f32,
                            max_epochs: u32) -> Result<JsValue, JsValue> {
        
        let mut epochs = 0;
        let mut final_error = f32::MAX;
        
        while epochs < max_epochs && final_error > target_error {
            final_error = self.train_epoch(network, training_data.clone())?;
            epochs += 1;
            
            // Allow other tasks to run
            if epochs % 10 == 0 {
                // Yield control briefly
                let _ = js_sys::Promise::resolve(&JsValue::NULL);
            }
        }
        
        let result = serde_json::json!({
            "converged": final_error <= target_error,
            "final_error": final_error,
            "epochs": epochs,
            "target_error": target_error
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_training_history(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.training_history).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_algorithm_info(&self) -> JsValue {
        let info = match &self.algorithm {
            TrainingAlgorithmWasm::IncrementalBackprop(_) => {
                serde_json::json!({
                    "name": "Incremental Backpropagation",
                    "type": "gradient_descent",
                    "description": "Online learning with immediate weight updates"
                })
            },
            TrainingAlgorithmWasm::BatchBackprop(_) => {
                serde_json::json!({
                    "name": "Batch Backpropagation", 
                    "type": "gradient_descent",
                    "description": "Batch learning with accumulated gradients"
                })
            },
            TrainingAlgorithmWasm::Rprop(_) => {
                serde_json::json!({
                    "name": "RPROP",
                    "type": "adaptive",
                    "description": "Resilient backpropagation with adaptive step sizes"
                })
            },
            TrainingAlgorithmWasm::Quickprop(_) => {
                serde_json::json!({
                    "name": "Quickprop",
                    "type": "second_order", 
                    "description": "Quasi-Newton method with quadratic approximation"
                })
            },
            TrainingAlgorithmWasm::Sarprop(_) => {
                serde_json::json!({
                    "name": "SARPROP",
                    "type": "adaptive",
                    "description": "Super-accelerated resilient backpropagation"
                })
            },
        };
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}
```

### 4. Cascade Correlation WASM Interface

#### Dynamic Network Growth
```rust
// cascade_wasm.rs - Cascade correlation implementation

use wasm_bindgen::prelude::*;
use ruv_fann::cascade::{CascadeTrainer, CascadeConfig};

#[wasm_bindgen]
pub struct WasmCascadeTrainer {
    inner: Option<CascadeTrainer<f32>>,
    config: CascadeConfig,
}

#[wasm_bindgen]
impl WasmCascadeTrainer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue, network: &WasmNeuralNetwork, training_data: JsValue) -> Result<WasmCascadeTrainer, JsValue> {
        let cascade_config: CascadeConfigWasm = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid cascade config: {}", e)))?;
        
        let data: TrainingDataConfig = serde_wasm_bindgen::from_value(training_data)
            .map_err(|e| JsValue::from_str(&format!("Invalid training data: {}", e)))?;
        
        let training_data = TrainingData {
            inputs: data.inputs,
            outputs: data.outputs,
        };
        
        let config = CascadeConfig {
            max_hidden_neurons: cascade_config.max_hidden_neurons,
            num_candidates: cascade_config.num_candidates,
            output_max_epochs: cascade_config.output_max_epochs,
            candidate_max_epochs: cascade_config.candidate_max_epochs,
            output_learning_rate: cascade_config.output_learning_rate,
            candidate_learning_rate: cascade_config.candidate_learning_rate,
            output_target_error: cascade_config.output_target_error,
            candidate_target_correlation: cascade_config.candidate_target_correlation,
            min_correlation_improvement: cascade_config.min_correlation_improvement,
            candidate_weight_range: (cascade_config.candidate_weight_min, cascade_config.candidate_weight_max),
            candidate_activations: cascade_config.candidate_activations.iter()
                .map(|name| parse_activation_function(name))
                .collect::<Result<Vec<_>, _>>()?,
            verbose: cascade_config.verbose,
            ..Default::default()
        };
        
        let trainer = CascadeTrainer::new(config.clone(), network.inner.clone(), training_data)
            .map_err(|e| JsValue::from_str(&format!("Cascade trainer creation error: {}", e)))?;
        
        Ok(WasmCascadeTrainer {
            inner: Some(trainer),
            config,
        })
    }
    
    #[wasm_bindgen]
    pub fn train(&mut self) -> Result<JsValue, JsValue> {
        let trainer = self.inner.take()
            .ok_or_else(|| JsValue::from_str("Trainer already consumed"))?;
        
        let result = trainer.train()
            .map_err(|e| JsValue::from_str(&format!("Cascade training error: {}", e)))?;
        
        let result_info = serde_json::json!({
            "converged": result.converged,
            "final_error": result.final_error,
            "hidden_neurons_added": result.hidden_neurons_added,
            "epochs": result.epochs,
            "training_time_ms": result.training_time.as_millis(),
            "network_structure": {
                "total_neurons": result.final_network.total_neurons(),
                "total_connections": result.final_network.total_connections(),
                "layers": result.final_network.num_layers()
            }
        });
        
        Ok(serde_wasm_bindgen::to_value(&result_info).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_config(&self) -> JsValue {
        let config_info = serde_json::json!({
            "max_hidden_neurons": self.config.max_hidden_neurons,
            "num_candidates": self.config.num_candidates,
            "output_max_epochs": self.config.output_max_epochs,
            "candidate_max_epochs": self.config.candidate_max_epochs,
            "output_learning_rate": self.config.output_learning_rate,
            "candidate_learning_rate": self.config.candidate_learning_rate,
            "output_target_error": self.config.output_target_error,
            "candidate_target_correlation": self.config.candidate_target_correlation
        });
        
        serde_wasm_bindgen::to_value(&config_info).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct CascadeConfigWasm {
    pub max_hidden_neurons: usize,
    pub num_candidates: usize,
    pub output_max_epochs: usize,
    pub candidate_max_epochs: usize,
    pub output_learning_rate: f32,
    pub candidate_learning_rate: f32,
    pub output_target_error: f32,
    pub candidate_target_correlation: f32,
    pub min_correlation_improvement: f32,
    pub candidate_weight_min: f32,
    pub candidate_weight_max: f32,
    pub candidate_activations: Vec<String>,
    pub verbose: bool,
}
```

## 🔧 Implementation Tasks

### Week 1: Foundation
- [ ] **Day 1-2**: Implement core WasmNeuralNetwork interface
- [ ] **Day 3**: Add all 18 activation functions
- [ ] **Day 4-5**: Create basic training interface
- [ ] **Day 6-7**: Implement network serialization/deserialization

### Week 2: Training Algorithms
- [ ] **Day 1**: Implement Incremental & Batch Backpropagation
- [ ] **Day 2**: Add RPROP training algorithm
- [ ] **Day 3**: Implement Quickprop algorithm
- [ ] **Day 4**: Add SARPROP algorithm
- [ ] **Day 5-7**: Create training monitoring and visualization

### Week 3: Advanced Features
- [ ] **Day 1-3**: Implement Cascade Correlation WASM interface
- [ ] **Day 4**: Add network analysis and visualization tools
- [ ] **Day 5**: Create performance benchmarking
- [ ] **Day 6-7**: Optimize WASM for neural operations

### Week 4: Integration & Polish
- [ ] **Day 1-2**: Integration testing with Agent 1's architecture
- [ ] **Day 3**: Create comprehensive examples and tutorials
- [ ] **Day 4**: Performance optimization and memory tuning
- [ ] **Day 5-7**: Documentation and API reference

## 📊 Success Metrics

### Performance Targets
- **Training Speed**: 10x faster than JavaScript implementations
- **Memory Usage**: < 1MB per network for typical sizes
- **Activation Functions**: All 18 functions with < 1μs execution time
- **WASM Bundle Size**: < 500KB for neural module

### Functionality Targets
- **API Coverage**: 100% of ruv-FANN capabilities exposed
- **Training Algorithms**: All 5 algorithms fully functional
- **Cascade Correlation**: Dynamic network growth working
- **Serialization**: Save/load networks with full fidelity

## 🔗 Dependencies & Coordination

### Dependencies on Agent 1
- WASM build pipeline and optimization framework
- Memory management utilities
- SIMD optimization interfaces
- TypeScript definition generation

### Coordination with Other Agents
- **Agent 3**: Neural networks for forecasting model backends
- **Agent 4**: Neural networks for agent cognitive processing
- **Agent 5**: JavaScript interfaces for NPX integration

### Deliverables to Other Agents
- Complete neural network WASM module
- Training algorithm interfaces
- Performance optimization examples
- Neural network utilities for cognitive processing

This comprehensive neural network implementation provides the foundation for advanced AI capabilities across the entire ruv-swarm ecosystem.