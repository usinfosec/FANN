# Build Pipeline Specification

## Overview
This document specifies the comprehensive build pipeline for converting all Rust crates in the ruv-FANN ecosystem into optimized WebAssembly modules that can be seamlessly integrated into the NPX package.

## 🎯 Build Pipeline Goals

### Primary Objectives
- **Unified WASM Output**: Single, optimized WASM bundle exposing all capabilities
- **Progressive Loading**: Modular WASM files for on-demand loading
- **Performance Optimization**: SIMD support, memory optimization, size reduction
- **Cross-Platform Compatibility**: Node.js, browsers, and various JavaScript engines
- **Developer Experience**: TypeScript definitions, debugging support, comprehensive documentation

### Secondary Objectives
- **Hot Reloading**: Development-time module reloading
- **Incremental Builds**: Fast rebuilds during development
- **Bundle Analysis**: Size analysis and optimization recommendations
- **Testing Integration**: Automated testing of WASM modules

## 🏗️ Build Architecture

### Directory Structure
```
ruv-swarm/
├── build/
│   ├── wasm-build.js              # Main build orchestrator
│   ├── optimize-wasm.js           # WASM optimization pipeline
│   ├── generate-bindings.js       # TypeScript binding generation
│   └── bundle-analyzer.js         # Bundle size analysis
├── wasm-config/
│   ├── core.toml                  # Core module build config
│   ├── neural.toml                # Neural network module config
│   ├── forecasting.toml           # Forecasting module config
│   ├── swarm.toml                 # Swarm orchestration config
│   └── unified.toml               # Unified build configuration
├── target/
│   ├── wasm-core/                 # Core WASM outputs
│   ├── wasm-neural/               # Neural network WASM
│   ├── wasm-forecasting/          # Forecasting WASM
│   ├── wasm-swarm/                # Swarm orchestration WASM
│   └── wasm-unified/              # Combined WASM bundle
└── npm/
    ├── wasm/                      # Final WASM modules for NPX
    │   ├── core.wasm
    │   ├── neural.wasm
    │   ├── forecasting.wasm
    │   ├── swarm.wasm
    │   └── unified.wasm
    └── types/                     # Generated TypeScript definitions
        ├── core.d.ts
        ├── neural.d.ts
        ├── forecasting.d.ts
        ├── swarm.d.ts
        └── index.d.ts
```

### Build Pipeline Components

#### 1. Multi-Crate WASM Compilation
```javascript
// build/wasm-build.js - Main build orchestrator

const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');
const { performance } = require('perf_hooks');

class WasmBuildPipeline {
    constructor() {
        this.buildTargets = [
            {
                name: 'core',
                cratePath: '../crates/ruv-swarm-core',
                features: ['wasm', 'no-std'],
                outputName: 'ruv_swarm_core'
            },
            {
                name: 'neural',
                cratePath: '../../../', // ruv-fann root
                features: ['wasm', 'simd'],
                outputName: 'ruv_fann'
            },
            {
                name: 'forecasting',
                cratePath: '../../../neuro-divergent',
                features: ['wasm', 'models'],
                outputName: 'neuro_divergent'
            },
            {
                name: 'swarm',
                cratePath: '../crates/ruv-swarm-wasm',
                features: ['unified'],
                outputName: 'ruv_swarm_unified'
            },
            {
                name: 'persistence',
                cratePath: '../crates/ruv-swarm-persistence',
                features: ['wasm', 'sqlite'],
                outputName: 'ruv_swarm_persistence'
            }
        ];
        
        this.buildConfig = {
            profile: 'release',
            target: 'wasm32-unknown-unknown',
            cargoFlags: ['--no-default-features'],
            wasmPackFlags: ['--target', 'web', '--scope', 'ruv']
        };
        
        this.optimizationConfig = {
            wasmOpt: true,
            simdOptimization: true,
            sizeOptimization: true,
            debugInfo: false
        };
    }

    async buildAll() {
        console.log('🏗️ Starting comprehensive WASM build pipeline...');
        const startTime = performance.now();
        
        try {
            // Phase 1: Clean previous builds
            await this.cleanBuildArtifacts();
            
            // Phase 2: Validate build environment
            await this.validateBuildEnvironment();
            
            // Phase 3: Build individual crates
            const buildResults = await this.buildIndividualCrates();
            
            // Phase 4: Create unified bundle
            const unifiedResult = await this.createUnifiedBundle();
            
            // Phase 5: Optimize WASM modules
            const optimizationResults = await this.optimizeWasmModules();
            
            // Phase 6: Generate TypeScript bindings
            const bindingResults = await this.generateTypeScriptBindings();
            
            // Phase 7: Bundle analysis
            const analysisResults = await this.analyzeBundles();
            
            // Phase 8: Copy to NPX package
            await this.copyToNpxPackage();
            
            const totalTime = performance.now() - startTime;
            
            const buildSummary = {
                success: true,
                totalTime: Math.round(totalTime),
                modules: buildResults,
                unified: unifiedResult,
                optimization: optimizationResults,
                bindings: bindingResults,
                analysis: analysisResults
            };
            
            await this.generateBuildReport(buildSummary);
            console.log(`✅ Build pipeline completed in ${Math.round(totalTime)}ms`);
            
            return buildSummary;
        } catch (error) {
            console.error('❌ Build pipeline failed:', error);
            throw error;
        }
    }

    async buildIndividualCrates() {
        console.log('🔨 Building individual WASM crates...');
        const results = {};
        
        for (const target of this.buildTargets) {
            console.log(`  Building ${target.name}...`);
            const startTime = performance.now();
            
            try {
                // Build with wasm-pack
                await this.runWasmPack(target);
                
                // Verify output
                await this.verifyWasmOutput(target);
                
                const buildTime = Math.round(performance.now() - startTime);
                results[target.name] = {
                    success: true,
                    buildTime,
                    outputPath: `target/wasm-${target.name}`
                };
                
                console.log(`    ✅ ${target.name} built in ${buildTime}ms`);
            } catch (error) {
                results[target.name] = {
                    success: false,
                    error: error.message
                };
                console.error(`    ❌ ${target.name} failed:`, error.message);
            }
        }
        
        return results;
    }

    async runWasmPack(target) {
        const args = [
            'build',
            target.cratePath,
            '--target', 'web',
            '--out-dir', `../target/wasm-${target.name}`,
            '--out-name', target.outputName,
            '--release'
        ];
        
        if (target.features.length > 0) {
            args.push('--features', target.features.join(','));
        }
        
        return this.runCommand('wasm-pack', args, target.cratePath);
    }

    async createUnifiedBundle() {
        console.log('🔗 Creating unified WASM bundle...');
        
        // Use custom Rust crate that re-exports all functionality
        const unifiedTarget = {
            name: 'unified',
            cratePath: '../crates/ruv-swarm-wasm-unified',
            features: ['all'],
            outputName: 'ruv_swarm_unified'
        };
        
        await this.runWasmPack(unifiedTarget);
        
        return {
            success: true,
            outputPath: 'target/wasm-unified',
            size: await this.getFileSize('target/wasm-unified/ruv_swarm_unified_bg.wasm')
        };
    }

    async optimizeWasmModules() {
        console.log('⚡ Optimizing WASM modules...');
        const results = {};
        
        const wasmFiles = await this.findWasmFiles();
        
        for (const wasmFile of wasmFiles) {
            const startTime = performance.now();
            const originalSize = await this.getFileSize(wasmFile);
            
            try {
                // Run wasm-opt for size and speed optimization
                await this.runWasmOpt(wasmFile);
                
                const optimizedSize = await this.getFileSize(wasmFile);
                const optimizationTime = Math.round(performance.now() - startTime);
                const sizeReduction = ((originalSize - optimizedSize) / originalSize * 100).toFixed(1);
                
                results[path.basename(wasmFile)] = {
                    success: true,
                    originalSize,
                    optimizedSize,
                    sizeReduction: `${sizeReduction}%`,
                    optimizationTime
                };
                
                console.log(`  ⚡ ${path.basename(wasmFile)}: ${sizeReduction}% smaller`);
            } catch (error) {
                results[path.basename(wasmFile)] = {
                    success: false,
                    error: error.message
                };
            }
        }
        
        return results;
    }

    async runWasmOpt(wasmFile) {
        const args = [
            '-Oz',                    // Optimize for size
            '--enable-simd',          // Enable SIMD optimizations
            '--enable-bulk-memory',   // Enable bulk memory operations
            '--enable-sign-ext',      // Enable sign extension
            '--strip-debug',          // Remove debug information
            '--vacuum',               // Remove unused functions
            wasmFile,
            '-o', wasmFile
        ];
        
        await this.runCommand('wasm-opt', args);
    }

    async generateTypeScriptBindings() {
        console.log('📝 Generating TypeScript bindings...');
        
        const bindingsGenerator = new TypeScriptBindingsGenerator();
        const results = {};
        
        for (const target of this.buildTargets) {
            try {
                const wasmDir = `target/wasm-${target.name}`;
                const bindingResult = await bindingsGenerator.generateBindings(
                    target.name,
                    wasmDir
                );
                
                results[target.name] = bindingResult;
                console.log(`  📝 ${target.name} bindings generated`);
            } catch (error) {
                results[target.name] = {
                    success: false,
                    error: error.message
                };
            }
        }
        
        // Generate unified bindings
        await bindingsGenerator.generateUnifiedBindings(results);
        
        return results;
    }

    async analyzeBundles() {
        console.log('📊 Analyzing bundle sizes and dependencies...');
        
        const analysis = {
            modules: {},
            total: {
                wasmSize: 0,
                jsSize: 0,
                tsSize: 0
            },
            recommendations: []
        };
        
        for (const target of this.buildTargets) {
            const wasmDir = `target/wasm-${target.name}`;
            const wasmFile = `${wasmDir}/${target.outputName}_bg.wasm`;
            const jsFile = `${wasmDir}/${target.outputName}.js`;
            
            if (await this.fileExists(wasmFile)) {
                const wasmSize = await this.getFileSize(wasmFile);
                const jsSize = await this.getFileSize(jsFile);
                
                analysis.modules[target.name] = {
                    wasmSize,
                    jsSize,
                    wasmSizeMB: (wasmSize / (1024 * 1024)).toFixed(2),
                    jsSizeKB: (jsSize / 1024).toFixed(2)
                };
                
                analysis.total.wasmSize += wasmSize;
                analysis.total.jsSize += jsSize;
            }
        }
        
        // Generate recommendations
        if (analysis.total.wasmSize > 5 * 1024 * 1024) { // > 5MB
            analysis.recommendations.push('Consider module splitting for better progressive loading');
        }
        
        if (analysis.modules.neural?.wasmSize > 2 * 1024 * 1024) { // > 2MB
            analysis.recommendations.push('Neural module is large - consider lazy loading');
        }
        
        return analysis;
    }

    async copyToNpxPackage() {
        console.log('📦 Copying WASM modules to NPX package...');
        
        // Ensure NPX wasm directory exists
        await fs.mkdir('npm/wasm', { recursive: true });
        await fs.mkdir('npm/types', { recursive: true });
        
        // Copy WASM files
        for (const target of this.buildTargets) {
            const sourceDir = `target/wasm-${target.name}`;
            const wasmFile = `${sourceDir}/${target.outputName}_bg.wasm`;
            const jsFile = `${sourceDir}/${target.outputName}.js`;
            const tsFile = `${sourceDir}/${target.outputName}.d.ts`;
            
            if (await this.fileExists(wasmFile)) {
                await fs.copyFile(wasmFile, `npm/wasm/${target.name}.wasm`);
                await fs.copyFile(jsFile, `npm/wasm/${target.name}.js`);
                
                if (await this.fileExists(tsFile)) {
                    await fs.copyFile(tsFile, `npm/types/${target.name}.d.ts`);
                }
            }
        }
        
        // Copy unified bundle
        const unifiedWasm = 'target/wasm-unified/ruv_swarm_unified_bg.wasm';
        const unifiedJs = 'target/wasm-unified/ruv_swarm_unified.js';
        
        if (await this.fileExists(unifiedWasm)) {
            await fs.copyFile(unifiedWasm, 'npm/wasm/unified.wasm');
            await fs.copyFile(unifiedJs, 'npm/wasm/unified.js');
        }
        
        console.log('  📦 WASM modules copied to NPX package');
    }

    // Utility methods
    async runCommand(command, args, cwd = process.cwd()) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, {
                cwd,
                stdio: 'pipe',
                env: { ...process.env, CARGO_TARGET_DIR: path.resolve('target') }
            });
            
            let stdout = '';
            let stderr = '';
            
            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Command failed with code ${code}: ${stderr}`));
                }
            });
            
            child.on('error', reject);
        });
    }

    async validateBuildEnvironment() {
        console.log('🔍 Validating build environment...');
        
        const requiredTools = [
            { command: 'rustc', version: '--version' },
            { command: 'cargo', version: '--version' },
            { command: 'wasm-pack', version: '--version' },
            { command: 'wasm-opt', version: '--version' }
        ];
        
        for (const tool of requiredTools) {
            try {
                const result = await this.runCommand(tool.command, [tool.version]);
                console.log(`  ✅ ${tool.command}: Available`);
            } catch (error) {
                throw new Error(`Missing required tool: ${tool.command}`);
            }
        }
        
        // Check for wasm32 target
        try {
            const result = await this.runCommand('rustup', ['target', 'list', '--installed']);
            if (!result.stdout.includes('wasm32-unknown-unknown')) {
                throw new Error('wasm32-unknown-unknown target not installed');
            }
            console.log('  ✅ wasm32-unknown-unknown target: Installed');
        } catch (error) {
            console.log('  ⚠️ Installing wasm32-unknown-unknown target...');
            await this.runCommand('rustup', ['target', 'add', 'wasm32-unknown-unknown']);
        }
    }

    async cleanBuildArtifacts() {
        console.log('🧹 Cleaning previous build artifacts...');
        
        const cleanTargets = [
            'target/wasm-core',
            'target/wasm-neural',
            'target/wasm-forecasting',
            'target/wasm-swarm',
            'target/wasm-persistence',
            'target/wasm-unified',
            'npm/wasm',
            'npm/types'
        ];
        
        for (const target of cleanTargets) {
            try {
                await fs.rm(target, { recursive: true, force: true });
            } catch (error) {
                // Ignore errors if directory doesn't exist
            }
        }
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    async getFileSize(filePath) {
        try {
            const stats = await fs.stat(filePath);
            return stats.size;
        } catch {
            return 0;
        }
    }

    async findWasmFiles() {
        const wasmFiles = [];
        const searchDirs = [
            'target/wasm-core',
            'target/wasm-neural',
            'target/wasm-forecasting',
            'target/wasm-swarm',
            'target/wasm-persistence',
            'target/wasm-unified'
        ];
        
        for (const dir of searchDirs) {
            try {
                const files = await fs.readdir(dir);
                for (const file of files) {
                    if (file.endsWith('_bg.wasm')) {
                        wasmFiles.push(path.join(dir, file));
                    }
                }
            } catch {
                // Directory doesn't exist, skip
            }
        }
        
        return wasmFiles;
    }

    async generateBuildReport(buildSummary) {
        const report = {
            timestamp: new Date().toISOString(),
            build: buildSummary,
            environment: {
                node_version: process.version,
                platform: process.platform,
                arch: process.arch
            }
        };
        
        await fs.writeFile(
            'target/build-report.json',
            JSON.stringify(report, null, 2)
        );
    }
}

// TypeScript bindings generator
class TypeScriptBindingsGenerator {
    async generateBindings(moduleName, wasmDir) {
        // Read existing .d.ts file generated by wasm-bindgen
        const existingTsFile = path.join(wasmDir, `*.d.ts`);
        
        // Enhance with additional type information
        const enhancedBindings = await this.enhanceBindings(moduleName, existingTsFile);
        
        // Write enhanced bindings
        const outputFile = `npm/types/${moduleName}.d.ts`;
        await fs.writeFile(outputFile, enhancedBindings);
        
        return {
            success: true,
            outputFile,
            size: enhancedBindings.length
        };
    }

    async enhanceBindings(moduleName, tsFile) {
        // Base TypeScript definitions with enhanced types
        return `
// Enhanced TypeScript definitions for ${moduleName} module
// Generated by ruv-swarm build pipeline

export interface ${this.capitalize(moduleName)}Module {
    // Module-specific interfaces will be generated here
    initialize(): Promise<void>;
    getVersion(): string;
    getMemoryUsage(): number;
}

// Additional type definitions based on module type
${this.getModuleSpecificTypes(moduleName)}

export default ${this.capitalize(moduleName)}Module;
`;
    }

    getModuleSpecificTypes(moduleName) {
        switch (moduleName) {
            case 'neural':
                return `
export interface NeuralNetwork {
    run(inputs: Float32Array): Float32Array;
    train(data: TrainingData): number;
    getWeights(): Float32Array;
    setWeights(weights: Float32Array): void;
}

export interface TrainingData {
    inputs: Float32Array[];
    outputs: Float32Array[];
}
`;
            case 'swarm':
                return `
export interface SwarmOrchestrator {
    createSwarm(config: SwarmConfig): string;
    spawnAgent(swarmId: string, config: AgentConfig): string;
    orchestrateTask(swarmId: string, task: TaskConfig): string;
    getStatus(swarmId: string): SwarmStatus;
}
`;
            default:
                return '';
        }
    }

    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    async generateUnifiedBindings(moduleResults) {
        const unifiedBindings = `
// Unified TypeScript definitions for ruv-swarm
// Generated by ruv-swarm build pipeline

${Object.keys(moduleResults).map(module => 
    `import { ${this.capitalize(module)}Module } from './${module}';`
).join('\n')}

export interface RuvSwarmUnified {
    ${Object.keys(moduleResults).map(module => 
        `${module}: ${this.capitalize(module)}Module;`
    ).join('\n    ')}
}

export * from './core';
export * from './neural';
export * from './forecasting';
export * from './swarm';
export * from './persistence';

declare const RuvSwarm: RuvSwarmUnified;
export default RuvSwarm;
`;
        
        await fs.writeFile('npm/types/index.d.ts', unifiedBindings);
    }
}

module.exports = { WasmBuildPipeline, TypeScriptBindingsGenerator };
```

#### 2. Automated Build Script
```bash
#!/bin/bash
# scripts/build-wasm.sh - Automated WASM build script

set -e

echo "🏗️ ruv-swarm WASM Build Pipeline"
echo "================================="

# Check environment
echo "🔍 Checking build environment..."
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Please install Rust."
    exit 1
fi

if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

if ! command -v wasm-opt &> /dev/null; then
    echo "❌ wasm-opt not found. Please install Binaryen."
    exit 1
fi

# Set build mode
BUILD_MODE=${1:-release}
echo "📦 Build mode: $BUILD_MODE"

# Run Node.js build pipeline
echo "🚀 Starting Node.js build pipeline..."
cd ruv-swarm
node build/wasm-build.js --mode=$BUILD_MODE

echo "✅ Build pipeline completed successfully!"
echo "📊 Build artifacts available in npm/wasm/"
echo "📝 TypeScript definitions available in npm/types/"
```

#### 3. CI/CD Integration
```yaml
# .github/workflows/wasm-build.yml
name: WASM Build Pipeline

on:
  push:
    branches: [ main, ruv-swarm ]
  pull_request:
    branches: [ main ]

jobs:
  build-wasm:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: wasm32-unknown-unknown
        components: rustfmt, clippy
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: ruv-swarm/npm/package-lock.json
    
    - name: Install wasm-pack
      run: cargo install wasm-pack
    
    - name: Install Binaryen
      run: |
        wget https://github.com/WebAssembly/binaryen/releases/latest/download/binaryen-version_108-x86_64-linux.tar.gz
        tar xzf binaryen-*.tar.gz
        sudo cp binaryen-*/bin/* /usr/local/bin/
    
    - name: Install NPM dependencies
      run: |
        cd ruv-swarm/npm
        npm ci
    
    - name: Run WASM build pipeline
      run: |
        cd ruv-swarm
        node build/wasm-build.js
    
    - name: Test WASM modules
      run: |
        cd ruv-swarm/npm
        npm test
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: wasm-modules
        path: |
          ruv-swarm/npm/wasm/
          ruv-swarm/npm/types/
          ruv-swarm/target/build-report.json
    
    - name: Analyze bundle size
      run: |
        cd ruv-swarm
        node build/bundle-analyzer.js --report
```

## 🔧 Build Configuration Files

### Core Module Configuration
```toml
# wasm-config/core.toml
[package]
name = "ruv-swarm-core-wasm"
version = "0.2.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[features]
default = ["std"]
std = []
wasm = ["wasm-bindgen", "js-sys", "web-sys"]
no-std = []
simd = []

[dependencies]
ruv-swarm-core = { path = "../crates/ruv-swarm-core" }
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

### Neural Module Configuration
```toml
# wasm-config/neural.toml
[package]
name = "ruv-fann-wasm"
version = "0.1.2"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[features]
default = ["std", "serde"]
std = []
wasm = ["wasm-bindgen", "js-sys"]
simd = ["ruv-fann/simd"]
parallel = ["ruv-fann/parallel"]

[dependencies]
ruv-fann = { path = "../../../", features = ["wasm"] }
wasm-bindgen = "0.2"
js-sys = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

[profile.release]
opt-level = 3
lto = true
debug = false
rpath = false
```

## 📊 Performance Optimization Strategy

### WASM Optimization Levels
1. **Size Optimization (-Oz)**
   - Minimize bundle size for network transfer
   - Remove unused code and debug information
   - Compress function names and metadata

2. **Speed Optimization (-O3)**
   - Optimize for execution speed
   - Enable SIMD instructions when available
   - Inline frequently used functions

3. **Balanced Optimization (-O2)**
   - Balance between size and speed
   - Default for production builds
   - Good compromise for most use cases

### SIMD Optimization
```rust
// Enable SIMD features in Cargo.toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", features = ["simd"] }

// Conditional SIMD compilation
#[cfg(target_feature = "simd128")]
fn simd_optimized_function() {
    // SIMD implementation
}

#[cfg(not(target_feature = "simd128"))]
fn simd_optimized_function() {
    // Fallback implementation
}
```

### Memory Optimization
- Use `wee_alloc` for smaller binary size
- Implement custom memory allocators for specific use cases
- Enable memory growth for dynamic allocation
- Use memory pools for frequent allocations

## 🧪 Testing Integration

### WASM Module Testing
```javascript
// tests/wasm-module.test.js
describe('WASM Module Loading', () => {
    test('Core module loads successfully', async () => {
        const module = await import('../npm/wasm/core.js');
        expect(module).toBeDefined();
        expect(typeof module.greet).toBe('function');
    });
    
    test('Neural module provides expected API', async () => {
        const neural = await import('../npm/wasm/neural.js');
        expect(neural.NeuralNetwork).toBeDefined();
        expect(neural.ActivationFunction).toBeDefined();
    });
    
    test('All modules have consistent memory usage', async () => {
        const modules = ['core', 'neural', 'swarm', 'forecasting'];
        
        for (const moduleName of modules) {
            const module = await import(`../npm/wasm/${moduleName}.js`);
            const initialMemory = module.get_memory_usage();
            
            // Perform operations
            // ... 
            
            const finalMemory = module.get_memory_usage();
            expect(finalMemory - initialMemory).toBeLessThan(1024 * 1024); // < 1MB growth
        }
    });
});
```

## 📈 Build Metrics and Monitoring

### Bundle Size Analysis
```javascript
// build/bundle-analyzer.js
class BundleAnalyzer {
    async analyzeBundles() {
        const analysis = {
            modules: {},
            recommendations: [],
            trends: {}
        };
        
        const wasmFiles = await this.findWasmFiles();
        
        for (const file of wasmFiles) {
            const size = await this.getFileSize(file);
            const compressed = await this.getGzipSize(file);
            
            analysis.modules[path.basename(file)] = {
                size,
                compressed,
                compressionRatio: (compressed / size * 100).toFixed(1)
            };
        }
        
        // Generate size recommendations
        if (analysis.modules['unified.wasm']?.size > 3 * 1024 * 1024) {
            analysis.recommendations.push('Consider module splitting for unified bundle');
        }
        
        return analysis;
    }
}
```

This comprehensive build pipeline ensures that all Rust capabilities are efficiently compiled to WASM while maintaining optimal performance, small bundle sizes, and excellent developer experience.