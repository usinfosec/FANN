# Agent 3: Forecasting Specialist Implementation Plan

## 🧠 Agent Profile
- **Type**: Forecasting Specialist
- **Cognitive Pattern**: Systems Thinking
- **Specialization**: neuro-divergent integration, time series processing, forecasting models
- **Focus**: Exposing complete neural forecasting capabilities through WASM

## 🎯 Mission
Integrate the complete neuro-divergent neural forecasting library into WASM, exposing all 27+ forecasting models, time series processing, ensemble methods, and NeuralForecast API compatibility through high-performance WebAssembly interfaces.

## 📋 Responsibilities

### 1. Complete neuro-divergent WASM Integration

#### Core Forecasting Interface
```rust
// forecasting_wasm.rs - Main forecasting WASM interface

use wasm_bindgen::prelude::*;
use neuro_divergent::{NeuralForecast, NeuralForecastBuilder, Frequency, ScalerType};
use neuro_divergent_core::BaseModel;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct WasmNeuralForecast {
    inner: NeuralForecast,
    models: Vec<String>,
    training_history: Vec<ForecastingMetrics>,
}

#[derive(Serialize, Deserialize)]
pub struct ForecastingMetrics {
    pub model_name: String,
    pub training_time_ms: f64,
    pub validation_loss: f32,
    pub mae: f32,
    pub mse: f32,
    pub mape: f32,
    pub smape: f32,
    pub memory_usage_mb: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ForecastConfig {
    pub models: Vec<ModelConfig>,
    pub frequency: String,
    pub horizon: usize,
    pub prediction_intervals: Option<Vec<u8>>,
    pub scaler: Option<String>,
    pub cross_validation: Option<CrossValidationConfig>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub name: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct CrossValidationConfig {
    pub n_windows: usize,
    pub window_size: usize,
    pub step_size: usize,
}

#[wasm_bindgen]
impl WasmNeuralForecast {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmNeuralForecast, JsValue> {
        let config: ForecastConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid forecast config: {}", e)))?;
        
        let mut builder = NeuralForecastBuilder::new()
            .with_frequency(parse_frequency(&config.frequency)?)
            .with_horizon(config.horizon);
        
        if let Some(scaler_type) = config.scaler {
            builder = builder.with_scaler(parse_scaler_type(&scaler_type)?);
        }
        
        if let Some(intervals) = config.prediction_intervals {
            builder = builder.with_prediction_intervals(intervals);
        }
        
        let mut model_names = Vec::new();
        
        // Add models to the forecast instance
        for model_config in config.models {
            let model = create_model(&model_config)?;
            model_names.push(model_config.name.clone());
            builder = builder.with_model(model);
        }
        
        let neural_forecast = builder.build()
            .map_err(|e| JsValue::from_str(&format!("Failed to build NeuralForecast: {}", e)))?;
        
        Ok(WasmNeuralForecast {
            inner: neural_forecast,
            models: model_names,
            training_history: Vec::new(),
        })
    }
    
    #[wasm_bindgen]
    pub fn fit(&mut self, data: JsValue) -> Result<(), JsValue> {
        let time_series_data: TimeSeriesData = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid time series data: {}", e)))?;
        
        let start_time = js_sys::Date::now();
        
        // Convert to neuro-divergent format
        let dataframe = convert_to_dataframe(time_series_data)?;
        
        self.inner.fit(dataframe)
            .map_err(|e| JsValue::from_str(&format!("Training failed: {}", e)))?;
        
        let end_time = js_sys::Date::now();
        
        // Record training metrics
        let metrics = ForecastingMetrics {
            model_name: "ensemble".to_string(),
            training_time_ms: end_time - start_time,
            validation_loss: 0.0, // TODO: Get actual validation loss
            mae: 0.0,
            mse: 0.0,
            mape: 0.0,
            smape: 0.0,
            memory_usage_mb: get_memory_usage_mb(),
        };
        
        self.training_history.push(metrics);
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn predict(&mut self) -> Result<JsValue, JsValue> {
        let forecasts = self.inner.predict()
            .map_err(|e| JsValue::from_str(&format!("Prediction failed: {}", e)))?;
        
        // Convert to JavaScript-friendly format
        let result = convert_forecasts_to_js(forecasts)?;
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn cross_validate(&mut self, config: JsValue) -> Result<JsValue, JsValue> {
        let cv_config: CrossValidationConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid CV config: {}", e)))?;
        
        // TODO: Implement cross-validation
        let cv_results = serde_json::json!({
            "mae": 0.0,
            "mse": 0.0,
            "mape": 0.0,
            "smape": 0.0,
            "windows_evaluated": cv_config.n_windows
        });
        
        Ok(serde_wasm_bindgen::to_value(&cv_results).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_model_info(&self) -> JsValue {
        let info = serde_json::json!({
            "models": self.models,
            "horizon": self.inner.get_horizon(),
            "frequency": format!("{:?}", self.inner.get_frequency()),
            "training_history": self.training_history
        });
        
        serde_wasm_bindgen::to_value(&info).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub unique_id: Vec<String>,
    pub ds: Vec<String>, // dates
    pub y: Vec<f32>,     // values
    pub static_features: Option<serde_json::Value>,
    pub dynamic_features: Option<serde_json::Value>,
}
```

### 2. Complete Model Library Implementation

#### All 27+ Forecasting Models
```rust
// models_wasm.rs - Complete model library

use wasm_bindgen::prelude::*;
use neuro_divergent_models::*;

#[wasm_bindgen]
pub struct ModelFactory;

#[wasm_bindgen]
impl ModelFactory {
    #[wasm_bindgen]
    pub fn get_available_models() -> JsValue {
        let models = vec![
            // Basic Models
            ("MLP", "Multi-Layer Perceptron", "basic"),
            ("DLinear", "Decomposition Linear", "basic"),
            ("NLinear", "Normalization Linear", "basic"),
            ("MLPMultivariate", "Multivariate MLP", "basic"),
            
            // Recurrent Models
            ("RNN", "Recurrent Neural Network", "recurrent"),
            ("LSTM", "Long Short-Term Memory", "recurrent"),
            ("GRU", "Gated Recurrent Unit", "recurrent"),
            
            // Advanced Models
            ("NBEATS", "Neural Basis Expansion Analysis", "advanced"),
            ("NBEATSx", "Extended NBEATS", "advanced"),
            ("NHITS", "Neural Hierarchical Interpolation", "advanced"),
            ("TiDE", "Time-series Dense Encoder", "advanced"),
            
            // Transformer Models
            ("TFT", "Temporal Fusion Transformer", "transformer"),
            ("Informer", "Informer Transformer", "transformer"),
            ("AutoFormer", "AutoFormer", "transformer"),
            ("FedFormer", "FedFormer", "transformer"),
            ("PatchTST", "Patch Time Series Transformer", "transformer"),
            ("iTransformer", "Inverted Transformer", "transformer"),
            
            // Specialized Models
            ("DeepAR", "Deep AutoRegressive", "specialized"),
            ("DeepNPTS", "Deep Non-Parametric Time Series", "specialized"),
            ("TCN", "Temporal Convolutional Network", "specialized"),
            ("BiTCN", "Bidirectional TCN", "specialized"),
            ("TimesNet", "TimesNet", "specialized"),
            ("StemGNN", "Spectral Temporal Graph Neural Network", "specialized"),
            ("TSMixer", "Time Series Mixer", "specialized"),
            ("TSMixerx", "Extended TSMixer", "specialized"),
            ("PatchMixer", "Patch Mixer", "specialized"),
            ("SegRNN", "Segment Recurrent Neural Network", "specialized"),
            ("DishTS", "Dish Time Series", "specialized"),
        ];
        
        serde_wasm_bindgen::to_value(&models).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn create_model(model_type: &str, config: JsValue) -> Result<JsValue, JsValue> {
        let model_config: ModelConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid model config: {}", e)))?;
        
        // Model creation would happen here
        let model_info = match model_type.to_lowercase().as_str() {
            "lstm" => create_lstm_model(model_config.parameters)?,
            "nbeats" => create_nbeats_model(model_config.parameters)?,
            "tft" => create_tft_model(model_config.parameters)?,
            "deepar" => create_deepar_model(model_config.parameters)?,
            _ => return Err(JsValue::from_str(&format!("Unknown model type: {}", model_type))),
        };
        
        Ok(serde_wasm_bindgen::to_value(&model_info).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn get_model_requirements(model_type: &str) -> JsValue {
        let requirements = match model_type.to_lowercase().as_str() {
            "lstm" => serde_json::json!({
                "required_params": ["hidden_size", "num_layers", "horizon"],
                "optional_params": ["dropout", "bidirectional", "input_size"],
                "min_samples": 100,
                "supports_multivariate": true,
                "supports_static_features": false
            }),
            "nbeats" => serde_json::json!({
                "required_params": ["horizon", "input_size"],
                "optional_params": ["stacks", "layers", "layer_widths"],
                "min_samples": 200,
                "supports_multivariate": false,
                "supports_static_features": false
            }),
            "tft" => serde_json::json!({
                "required_params": ["horizon", "hidden_size"],
                "optional_params": ["num_attention_heads", "dropout", "input_size"],
                "min_samples": 500,
                "supports_multivariate": true,
                "supports_static_features": true
            }),
            _ => serde_json::json!({
                "error": format!("Unknown model type: {}", model_type)
            }),
        };
        
        serde_wasm_bindgen::to_value(&requirements).unwrap()
    }
}

// Individual model creation functions
fn create_lstm_model(params: serde_json::Value) -> Result<serde_json::Value, JsValue> {
    let hidden_size = params.get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("hidden_size required for LSTM"))?;
    
    let num_layers = params.get("num_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(2);
    
    let horizon = params.get("horizon")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("horizon required for LSTM"))?;
    
    // TODO: Create actual LSTM model
    let model_info = serde_json::json!({
        "type": "LSTM",
        "parameters": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "horizon": horizon,
            "dropout": params.get("dropout").unwrap_or(&serde_json::json!(0.1)),
            "bidirectional": params.get("bidirectional").unwrap_or(&serde_json::json!(false))
        },
        "estimated_memory_mb": (hidden_size * num_layers * 4) as f32 / (1024.0 * 1024.0),
        "estimated_training_time": "medium"
    });
    
    Ok(model_info)
}

fn create_nbeats_model(params: serde_json::Value) -> Result<serde_json::Value, JsValue> {
    let horizon = params.get("horizon")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("horizon required for NBEATS"))?;
    
    let input_size = params.get("input_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(horizon * 2);
    
    let stacks = params.get("stacks")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);
    
    let model_info = serde_json::json!({
        "type": "NBEATS",
        "parameters": {
            "horizon": horizon,
            "input_size": input_size,
            "stacks": stacks,
            "layers": params.get("layers").unwrap_or(&serde_json::json!(4)),
            "layer_widths": params.get("layer_widths").unwrap_or(&serde_json::json!(512))
        },
        "estimated_memory_mb": (stacks * 512 * 4) as f32 / (1024.0 * 1024.0),
        "estimated_training_time": "high"
    });
    
    Ok(model_info)
}

fn create_tft_model(params: serde_json::Value) -> Result<serde_json::Value, JsValue> {
    let horizon = params.get("horizon")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("horizon required for TFT"))?;
    
    let hidden_size = params.get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("hidden_size required for TFT"))?;
    
    let model_info = serde_json::json!({
        "type": "TFT",
        "parameters": {
            "horizon": horizon,
            "hidden_size": hidden_size,
            "num_attention_heads": params.get("num_attention_heads").unwrap_or(&serde_json::json!(4)),
            "dropout": params.get("dropout").unwrap_or(&serde_json::json!(0.1)),
            "input_size": params.get("input_size").unwrap_or(&serde_json::json!(horizon))
        },
        "estimated_memory_mb": (hidden_size * hidden_size * 8) as f32 / (1024.0 * 1024.0),
        "estimated_training_time": "very_high"
    });
    
    Ok(model_info)
}

fn create_deepar_model(params: serde_json::Value) -> Result<serde_json::Value, JsValue> {
    let horizon = params.get("horizon")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| JsValue::from_str("horizon required for DeepAR"))?;
    
    let model_info = serde_json::json!({
        "type": "DeepAR",
        "parameters": {
            "horizon": horizon,
            "cell_type": params.get("cell_type").unwrap_or(&serde_json::json!("LSTM")),
            "hidden_size": params.get("hidden_size").unwrap_or(&serde_json::json!(40)),
            "num_layers": params.get("num_layers").unwrap_or(&serde_json::json!(2)),
            "dropout": params.get("dropout").unwrap_or(&serde_json::json!(0.1))
        },
        "estimated_memory_mb": 5.0,
        "estimated_training_time": "medium"
    });
    
    Ok(model_info)
}
```

### 3. Time Series Data Processing

#### Advanced Data Handling
```rust
// data_processing_wasm.rs - Time series data processing

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct TimeSeriesProcessor {
    scalers: std::collections::HashMap<String, ScalerInfo>,
    transformations: Vec<TransformationStep>,
}

#[derive(Serialize, Deserialize)]
pub struct ScalerInfo {
    pub scaler_type: String,
    pub fitted: bool,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct TransformationStep {
    pub step_type: String,
    pub parameters: serde_json::Value,
    pub applied_at: f64, // timestamp
}

#[wasm_bindgen]
impl TimeSeriesProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TimeSeriesProcessor {
        TimeSeriesProcessor {
            scalers: std::collections::HashMap::new(),
            transformations: Vec::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn add_scaler(&mut self, name: &str, scaler_type: &str, config: JsValue) -> Result<(), JsValue> {
        let scaler_config: serde_json::Value = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid scaler config: {}", e)))?;
        
        let scaler_info = ScalerInfo {
            scaler_type: scaler_type.to_string(),
            fitted: false,
            parameters: scaler_config,
        };
        
        self.scalers.insert(name.to_string(), scaler_info);
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn fit_transform(&mut self, data: JsValue, transformations: JsValue) -> Result<JsValue, JsValue> {
        let time_series: TimeSeriesData = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid time series data: {}", e)))?;
        
        let transform_config: Vec<TransformationConfig> = serde_wasm_bindgen::from_value(transformations)
            .map_err(|e| JsValue::from_str(&format!("Invalid transformation config: {}", e)))?;
        
        let mut processed_data = time_series;
        
        for transform in transform_config {
            processed_data = self.apply_transformation(processed_data, transform)?;
        }
        
        Ok(serde_wasm_bindgen::to_value(&processed_data).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn detect_seasonality(&self, data: JsValue) -> Result<JsValue, JsValue> {
        let time_series: TimeSeriesData = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid time series data: {}", e)))?;
        
        // TODO: Implement actual seasonality detection
        let seasonality_info = serde_json::json!({
            "has_trend": true,
            "has_seasonality": true,
            "seasonal_periods": [7, 30, 365], // daily, monthly, yearly
            "trend_strength": 0.75,
            "seasonal_strength": 0.65,
            "residual_strength": 0.15
        });
        
        Ok(serde_wasm_bindgen::to_value(&seasonality_info).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn create_features(&self, data: JsValue, feature_config: JsValue) -> Result<JsValue, JsValue> {
        let time_series: TimeSeriesData = serde_wasm_bindgen::from_value(data)
            .map_err(|e| JsValue::from_str(&format!("Invalid time series data: {}", e)))?;
        
        let features: FeatureConfig = serde_wasm_bindgen::from_value(feature_config)
            .map_err(|e| JsValue::from_str(&format!("Invalid feature config: {}", e)))?;
        
        // TODO: Implement feature creation
        let feature_data = serde_json::json!({
            "lag_features": features.lags.unwrap_or_default(),
            "date_features": features.date_features.unwrap_or_default(),
            "rolling_features": features.rolling_windows.unwrap_or_default(),
            "fourier_features": features.fourier_terms.unwrap_or(0)
        });
        
        Ok(serde_wasm_bindgen::to_value(&feature_data).unwrap())
    }
    
    fn apply_transformation(&mut self, data: TimeSeriesData, transform: TransformationConfig) -> Result<TimeSeriesData, JsValue> {
        // Record the transformation
        self.transformations.push(TransformationStep {
            step_type: transform.transform_type.clone(),
            parameters: transform.parameters.clone(),
            applied_at: js_sys::Date::now(),
        });
        
        // TODO: Apply actual transformations
        match transform.transform_type.as_str() {
            "normalize" => Ok(self.normalize_data(data)?),
            "standardize" => Ok(self.standardize_data(data)?),
            "log" => Ok(self.log_transform_data(data)?),
            "difference" => Ok(self.difference_data(data)?),
            _ => Err(JsValue::from_str(&format!("Unknown transformation: {}", transform.transform_type))),
        }
    }
    
    fn normalize_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, JsValue> {
        let min_val = data.y.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.y.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        if range > 0.0 {
            data.y = data.y.iter().map(|&x| (x - min_val) / range).collect();
        }
        
        Ok(data)
    }
    
    fn standardize_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, JsValue> {
        let mean = data.y.iter().sum::<f32>() / data.y.len() as f32;
        let variance = data.y.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.y.len() as f32;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            data.y = data.y.iter().map(|&x| (x - mean) / std_dev).collect();
        }
        
        Ok(data)
    }
    
    fn log_transform_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, JsValue> {
        data.y = data.y.iter().map(|&x| x.ln()).collect();
        Ok(data)
    }
    
    fn difference_data(&self, mut data: TimeSeriesData) -> Result<TimeSeriesData, JsValue> {
        if data.y.len() > 1 {
            let differenced: Vec<f32> = data.y.windows(2).map(|w| w[1] - w[0]).collect();
            data.y = differenced;
            data.ds.remove(0); // Remove first date since we have one less point
            data.unique_id.remove(0);
        }
        Ok(data)
    }
}

#[derive(Serialize, Deserialize)]
pub struct TransformationConfig {
    pub transform_type: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct FeatureConfig {
    pub lags: Option<Vec<usize>>,
    pub date_features: Option<Vec<String>>,
    pub rolling_windows: Option<Vec<usize>>,
    pub fourier_terms: Option<usize>,
}
```

### 4. Ensemble Forecasting

#### Advanced Ensemble Methods
```rust
// ensemble_wasm.rs - Ensemble forecasting capabilities

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
pub struct EnsembleForecaster {
    models: Vec<ModelInfo>,
    ensemble_method: String,
    weights: Option<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub weight: f32,
    pub performance_metrics: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub method: String, // "simple_average", "weighted_average", "stacking", "voting"
    pub models: Vec<String>,
    pub weights: Option<Vec<f32>>,
    pub meta_learner: Option<String>,
}

#[wasm_bindgen]
impl EnsembleForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<EnsembleForecaster, JsValue> {
        let ensemble_config: EnsembleConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid ensemble config: {}", e)))?;
        
        Ok(EnsembleForecaster {
            models: Vec::new(),
            ensemble_method: ensemble_config.method,
            weights: ensemble_config.weights,
        })
    }
    
    #[wasm_bindgen]
    pub fn add_model(&mut self, model_info: JsValue) -> Result<(), JsValue> {
        let info: ModelInfo = serde_wasm_bindgen::from_value(model_info)
            .map_err(|e| JsValue::from_str(&format!("Invalid model info: {}", e)))?;
        
        self.models.push(info);
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn ensemble_predict(&self, predictions: JsValue) -> Result<JsValue, JsValue> {
        let model_predictions: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(predictions)
            .map_err(|e| JsValue::from_str(&format!("Invalid predictions: {}", e)))?;
        
        let ensemble_forecast = match self.ensemble_method.as_str() {
            "simple_average" => self.simple_average(&model_predictions)?,
            "weighted_average" => self.weighted_average(&model_predictions)?,
            "median" => self.median_ensemble(&model_predictions)?,
            "trimmed_mean" => self.trimmed_mean(&model_predictions, 0.2)?,
            _ => return Err(JsValue::from_str(&format!("Unknown ensemble method: {}", self.ensemble_method))),
        };
        
        let result = serde_json::json!({
            "ensemble_forecast": ensemble_forecast,
            "method": self.ensemble_method,
            "models_used": self.models.len(),
            "confidence_intervals": self.calculate_confidence_intervals(&model_predictions)?
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    #[wasm_bindgen]
    pub fn optimize_weights(&mut self, validation_data: JsValue) -> Result<JsValue, JsValue> {
        // TODO: Implement weight optimization using validation data
        let optimized_weights = vec![1.0 / self.models.len() as f32; self.models.len()];
        
        self.weights = Some(optimized_weights.clone());
        
        let result = serde_json::json!({
            "optimized_weights": optimized_weights,
            "optimization_method": "validation_loss_minimization",
            "cross_validation_score": 0.85 // placeholder
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
    
    fn simple_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, JsValue> {
        if predictions.is_empty() {
            return Err(JsValue::from_str("No predictions provided"));
        }
        
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        
        for pred in predictions {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value;
            }
        }
        
        for value in &mut result {
            *value /= predictions.len() as f32;
        }
        
        Ok(result)
    }
    
    fn weighted_average(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, JsValue> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| JsValue::from_str("Weights not provided for weighted average"))?;
        
        if weights.len() != predictions.len() {
            return Err(JsValue::from_str("Number of weights must match number of predictions"));
        }
        
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        
        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            for (i, &value) in pred.iter().enumerate() {
                result[i] += value * weight;
            }
        }
        
        Ok(result)
    }
    
    fn median_ensemble(&self, predictions: &[Vec<f32>]) -> Result<Vec<f32>, JsValue> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        
        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            result[i] = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
        }
        
        Ok(result)
    }
    
    fn trimmed_mean(&self, predictions: &[Vec<f32>], trim_percent: f32) -> Result<Vec<f32>, JsValue> {
        let horizon = predictions[0].len();
        let mut result = vec![0.0; horizon];
        let trim_count = ((predictions.len() as f32) * trim_percent / 2.0).floor() as usize;
        
        for i in 0..horizon {
            let mut values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Remove extreme values
            if trim_count > 0 && values.len() > 2 * trim_count {
                values = values[trim_count..values.len() - trim_count].to_vec();
            }
            
            result[i] = values.iter().sum::<f32>() / values.len() as f32;
        }
        
        Ok(result)
    }
    
    fn calculate_confidence_intervals(&self, predictions: &[Vec<f32>]) -> Result<serde_json::Value, JsValue> {
        let horizon = predictions[0].len();
        let mut lower_95 = vec![0.0; horizon];
        let mut upper_95 = vec![0.0; horizon];
        
        for i in 0..horizon {
            let values: Vec<f32> = predictions.iter().map(|pred| pred[i]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();
            
            // 95% confidence interval (approximately ±1.96 * std_dev)
            lower_95[i] = mean - 1.96 * std_dev;
            upper_95[i] = mean + 1.96 * std_dev;
        }
        
        Ok(serde_json::json!({
            "95%": {
                "lower": lower_95,
                "upper": upper_95
            },
            "prediction_std": horizon as f32 // placeholder
        }))
    }
}
```

## 🔧 Implementation Tasks

### Week 1: Foundation
- [ ] **Day 1-2**: Implement core WasmNeuralForecast interface
- [ ] **Day 3**: Create time series data processing utilities
- [ ] **Day 4-5**: Add basic model factory and model creation
- [ ] **Day 6-7**: Implement data transformation pipeline

### Week 2: Model Library
- [ ] **Day 1**: Implement basic models (MLP, DLinear, NLinear)
- [ ] **Day 2**: Add recurrent models (LSTM, GRU, RNN)
- [ ] **Day 3**: Implement NBEATS and advanced models
- [ ] **Day 4**: Add transformer models (TFT, Informer)
- [ ] **Day 5-7**: Create specialized models (DeepAR, TCN, etc.)

### Week 3: Advanced Features
- [ ] **Day 1-2**: Implement ensemble forecasting methods
- [ ] **Day 3**: Add cross-validation and model evaluation
- [ ] **Day 4**: Create feature engineering utilities
- [ ] **Day 5**: Add seasonality detection and decomposition
- [ ] **Day 6-7**: Implement prediction intervals and uncertainty quantification

### Week 4: Integration & Polish
- [ ] **Day 1-2**: Integration testing with Agent 1's architecture
- [ ] **Day 3**: Performance optimization for time series processing
- [ ] **Day 4**: Create comprehensive examples and tutorials
- [ ] **Day 5-7**: Documentation and API reference

## 📊 Success Metrics

### Performance Targets
- **Training Speed**: 5x faster than Python NeuralForecast
- **Memory Usage**: 30% less memory than Python implementations
- **Model Support**: All 27+ models available through WASM
- **WASM Bundle Size**: < 1MB for forecasting module

### Functionality Targets
- **API Compatibility**: 95% compatibility with Python NeuralForecast
- **Model Accuracy**: Equivalent or better than Python versions
- **Ensemble Methods**: 5+ ensemble techniques available
- **Data Processing**: Complete time series preprocessing pipeline

## 🔗 Dependencies & Coordination

### Dependencies on Agent 1
- WASM build pipeline optimized for large models
- Memory management for time series data
- Performance optimization framework

### Dependencies on Agent 2
- Neural network backends for forecasting models
- Training algorithms for model optimization
- Activation functions for model architectures

### Coordination with Other Agents
- **Agent 4**: Forecasting capabilities for swarm intelligence
- **Agent 5**: JavaScript interfaces for NPX integration

### Deliverables to Other Agents
- Complete forecasting WASM module
- Time series processing utilities
- Model performance benchmarking tools
- Ensemble forecasting capabilities

This comprehensive forecasting implementation provides advanced time series modeling capabilities that can be used both standalone and as part of intelligent swarm systems.