use wasm_bindgen::prelude::*;

mod agent;
mod swarm;
mod utils;

pub use agent::JsAgent;
pub use swarm::RuvSwarm;
pub use utils::{set_panic_hook, RuntimeFeatures};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn init() {
    set_panic_hook();
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Softmax,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    MSE,
    MAE,
    CrossEntropy,
    BinaryCrossEntropy,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Optimizer,
    Coordinator,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum SwarmStrategy {
    Research,
    Development,
    Analysis,
    Testing,
    Optimization,
    Maintenance,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum CoordinationMode {
    Centralized,
    Distributed,
    Hierarchical,
    Mesh,
    Hybrid,
}

// Feature detection at compile time
#[cfg(target_feature = "simd128")]
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    true
}

#[cfg(not(target_feature = "simd128"))]
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    false
}

// Version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}