use std::env;

fn main() {
    // Enable SIMD target features for WebAssembly builds
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32" {
        println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+simd128");
        
        // Tell cargo to rerun if environment changes
        println!("cargo:rerun-if-env-changed=RUSTFLAGS");
        println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
        
        // Feature detection
        if env::var("CARGO_FEATURE_SIMD").is_ok() {
            println!("cargo:rustc-cfg=feature=\"simd\"");
            println!("cargo:warning=SIMD feature enabled for WebAssembly build");
        }
        
        // Check for target features
        if env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default().contains("simd128") {
            println!("cargo:warning=SIMD128 target feature detected");
        }
        
        // Output build information
        println!("cargo:warning=Building ruv-swarm-wasm with SIMD optimizations");
        
        // Enable wasm-bindgen features for SIMD
        println!("cargo:rustc-env=WASM_BINDGEN_FEATURES=simd");
    }
    
    // For non-WASM targets, check for native SIMD support
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match target_arch.as_str() {
        "x86_64" | "x86" => {
            // Enable x86 SIMD features if available
            if env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default().contains("sse") {
                println!("cargo:rustc-cfg=has_sse");
            }
            if env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default().contains("avx") {
                println!("cargo:rustc-cfg=has_avx");
            }
        }
        "aarch64" => {
            // Enable ARM NEON if available
            if env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default().contains("neon") {
                println!("cargo:rustc-cfg=has_neon");
            }
        }
        _ => {}
    }
    
    // Version information
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    println!("cargo:rustc-env=RUV_SWARM_WASM_VERSION={}", version);
}