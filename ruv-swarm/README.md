# ruv-swarm 🐝

[![Crates.io](https://img.shields.io/crates/v/ruv-swarm.svg)](https://crates.io/crates/ruv-swarm)
[![Documentation](https://docs.rs/ruv-swarm/badge.svg)](https://docs.rs/ruv-swarm)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/ruv-swarm.svg)](#license)
[![CI](https://github.com/ruvnet/ruv-FANN/workflows/CI/badge.svg)](https://github.com/ruvnet/ruv-FANN/actions)

High-performance neural network swarm orchestration framework with WebAssembly acceleration and Model Context Protocol (MCP) integration for Claude Code.

## 🚀 Features

### Core Architecture
- **Multi-Agent Orchestration**: Distributed swarm coordination with 4 topology types (mesh, hierarchical, ring, star)
- **WebAssembly Performance**: High-speed WASM modules with SIMD optimization support
- **Neural Network Integration**: 18 activation functions, 5 training algorithms, cascade correlation
- **Cognitive Diversity**: 5 cognitive patterns for enhanced problem-solving
- **Real-time Persistence**: SQLite-backed state management with ACID compliance

### Advanced Capabilities
- **🧠 Neural Networks**: Built-in neural network management with per-agent training
- **📈 Forecasting**: 27+ time series forecasting models with ensemble methods
- **⚡ SIMD Acceleration**: WebAssembly SIMD optimizations for 2-4x performance gains
- **🔧 MCP Integration**: Complete Model Context Protocol support for Claude Code
- **💾 Persistence Layer**: Automatic state persistence across sessions
- **📊 Performance Monitoring**: Real-time metrics and benchmarking tools

### SDK & Development Tools
- **Rust SDK**: Complete Rust crate for native integration
- **JavaScript SDK**: NPM package with TypeScript definitions
- **WebAssembly SDK**: Browser-ready WASM modules
- **CLI Tools**: Command-line interface for swarm management
- **MCP Server**: Integrated MCP server for Claude Code workflows

## 📦 Installation

### Rust Crate
```toml
[dependencies]
ruv-swarm = "0.1.0"
```

### NPM Package
```bash
npm install ruv-swarm
# or
npx ruv-swarm --help
```

### From Source
```bash
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm
cargo build --release
```

## 🏃 Quick Start

### Rust API
```rust
use ruv_swarm::{Swarm, Agent, TopologyType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize swarm with mesh topology
    let mut swarm = Swarm::new(TopologyType::Mesh, 5).await?;
    
    // Spawn agents with different types
    let researcher = Agent::new("researcher").await?;
    let coder = Agent::new("coder").await?;
    
    swarm.add_agent(researcher).await?;
    swarm.add_agent(coder).await?;
    
    // Orchestrate tasks across the swarm
    let task = swarm.orchestrate("Analyze performance metrics").await?;
    let results = task.await_completion().await?;
    
    println!("Task completed: {:?}", results);
    Ok(())
}
```

### JavaScript/TypeScript SDK
```javascript
import { RuvSwarm } from 'ruv-swarm';

// Initialize with WASM acceleration
const swarm = await RuvSwarm.initialize({
    topology: 'mesh',
    maxAgents: 5,
    enableWASM: true,
    enableSIMD: true
});

// Create swarm and spawn agents
await swarm.createSwarm('research-swarm');
const agent1 = await swarm.spawnAgent('researcher', 'data-analyst');
const agent2 = await swarm.spawnAgent('coder', 'ml-engineer');

// Orchestrate distributed tasks
const task = await swarm.orchestrateTask({
    description: 'Build ML pipeline',
    agents: [agent1.id, agent2.id],
    strategy: 'collaborative'
});

console.log('Task result:', await task.getResults());
```

### CLI Interface
```bash
# Initialize a mesh swarm with 5 agents
ruv-swarm init mesh 5

# Spawn specialized agents
ruv-swarm spawn researcher data-scientist
ruv-swarm spawn coder ml-engineer

# Orchestrate tasks
ruv-swarm orchestrate "Optimize neural network architecture"

# Monitor real-time performance
ruv-swarm monitor --duration 30s

# Neural network operations
ruv-swarm neural status
ruv-swarm neural train agent-123 --iterations 100

# Benchmarking and features
ruv-swarm benchmark --type wasm --iterations 10
ruv-swarm features --category all
```

### MCP Integration with Claude Code
```bash
# Add ruv-swarm MCP server to Claude Code
claude mcp add ruv-swarm node ./ruv-swarm/npm/bin/ruv-swarm-enhanced.js mcp start

# Available MCP tools in Claude Code:
# - swarm_init: Initialize swarm topology
# - agent_spawn: Create specialized agents  
# - task_orchestrate: Distribute tasks across swarm
# - benchmark_run: Performance testing
# - neural_train: Train agent neural networks
# - features_detect: Runtime feature detection
# - memory_usage: System monitoring
```

## 🧠 Neural Network Integration

### Cognitive Diversity Patterns
- **Convergent**: Linear, focused problem-solving (optimization, debugging)
- **Divergent**: Creative, exploratory thinking (research, design)
- **Lateral**: Unconventional, cross-domain solutions (integration, architecture)
- **Systems**: Holistic, interconnected analysis (orchestration, coordination)
- **Critical**: Analytical, evaluative assessment (testing, security)

### Neural Network Features
- **18 Activation Functions**: ReLU, Sigmoid, Tanh, Swish, GELU, and more
- **5 Training Algorithms**: Backpropagation, RProp, Quickprop, Adam, SGD
- **Cascade Correlation**: Dynamic topology optimization
- **SIMD Acceleration**: Vectorized operations for 2-4x speedup

## ⚡ WebAssembly & Performance

### WASM Modules
- **Core Module** (512KB): Essential swarm operations
- **Neural Module** (1MB): Neural network computations  
- **Forecasting Module** (1.5MB): Time series analysis
- **SIMD Module**: Optimized mathematical operations

### Performance Benchmarks
- **Agent Spawning**: 0.01ms average (exceptional)
- **Task Orchestration**: 4-7ms (excellent)
- **Neural Operations**: 593+ ops/sec
- **Forecasting**: 6454+ predictions/sec
- **WASM Loading**: 100% success rate, <1ms

## 🔧 MCP (Model Context Protocol) Tools

Complete integration with Claude Code via 13 MCP tools:

### Swarm Management
- `swarm_init` - Initialize topology (mesh/hierarchical/ring/star)
- `swarm_status` - Real-time swarm state and metrics
- `swarm_monitor` - Live activity monitoring

### Agent Operations
- `agent_spawn` - Create specialized agents (researcher/coder/analyst/optimizer/coordinator)
- `agent_list` - Inventory active agents
- `agent_metrics` - Performance analytics

### Task Orchestration  
- `task_orchestrate` - Distribute tasks across swarm
- `task_status` - Monitor task progress
- `task_results` - Retrieve completed results

### System & Performance
- `benchmark_run` - WASM/swarm/agent benchmarks
- `features_detect` - Runtime capability detection
- `memory_usage` - Resource monitoring
- `neural_status` - Neural network diagnostics

## 🏗️ Architecture

### Modular Design
```
ruv-swarm/
├── crates/
│   ├── ruv-swarm-core/      # Core swarm logic
│   ├── ruv-swarm-agents/    # Agent implementations
│   ├── ruv-swarm-ml/        # Neural networks & forecasting
│   ├── ruv-swarm-wasm/      # WebAssembly modules
│   ├── ruv-swarm-mcp/       # MCP server integration
│   ├── ruv-swarm-transport/ # Communication protocols
│   ├── ruv-swarm-persistence/ # SQLite persistence
│   └── ruv-swarm-cli/       # Command-line interface
├── npm/                     # JavaScript/TypeScript SDK
├── docs/                    # Documentation
├── examples/                # Usage examples
└── benches/                 # Performance benchmarks
```

### Technology Stack
- **Backend**: Rust with async/await (tokio)
- **WebAssembly**: wasm-bindgen with SIMD support
- **Frontend**: TypeScript with WebAssembly integration
- **Persistence**: SQLite with migrations
- **Communication**: WebSocket, shared memory, in-process
- **MCP Integration**: JSON-RPC 2.0 protocol

## 📊 Use Cases

### Development Workflows
- **Multi-agent code review**: Distribute review tasks across specialized agents
- **Parallel testing**: Coordinate test execution across agent swarm
- **Architecture analysis**: System-level analysis with cognitive diversity
- **Performance optimization**: Distributed benchmarking and tuning

### Research & Analysis
- **Data pipeline orchestration**: Coordinate data processing workflows
- **ML experiment management**: Parallel hyperparameter optimization
- **Time series forecasting**: Ensemble prediction models
- **Cognitive task distribution**: Leverage different thinking patterns

### Integration Scenarios
- **Claude Code workflows**: MCP-based swarm orchestration
- **CI/CD pipelines**: Distributed build and test coordination
- **Microservice orchestration**: Service mesh coordination
- **Real-time monitoring**: Distributed system health checks

## 🧪 Development

### Building from Source
```bash
# Clone repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm

# Build Rust components
cargo build --release

# Build WebAssembly modules  
cd crates/ruv-swarm-wasm
wasm-pack build --target web --out-dir ../../npm/wasm

# Build NPM package
cd ../../npm
npm run build
npm test
```

### Testing
```bash
# Run Rust tests
cargo test --all-features

# Run WebAssembly tests  
cd crates/ruv-swarm-wasm
cargo test --target wasm32-unknown-unknown

# Run JavaScript tests
cd npm
npm test

# Integration tests
cargo test --test integration

# Benchmarks
cargo bench
```

### MCP Development
```bash
# Test MCP server
cd npm
node bin/ruv-swarm-enhanced.js mcp start --protocol=stdio

# Test with Claude Code
claude mcp add ruv-swarm node ./npm/bin/ruv-swarm-enhanced.js mcp start
```

## 📚 Documentation

- **[API Reference](./docs/API_REFERENCE.md)** - Complete API documentation
- **[MCP Usage Guide](./docs/MCP_USAGE.md)** - Claude Code integration
- **[Neural Integration](./docs/NEURAL_INTEGRATION.md)** - Neural network features
- **[Benchmarks](./docs/BENCHMARKS.md)** - Performance metrics
- **[Examples](./examples/)** - Code examples and demos

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup
1. Install Rust (1.75+) and Node.js (18+)
2. Install wasm-pack: `cargo install wasm-pack`
3. Clone and build: `git clone && cd ruv-swarm && cargo build`
4. Run tests: `cargo test && npm test`

## 📄 License

This project is licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🔗 Links

- **Crates.io**: https://crates.io/crates/ruv-swarm
- **NPM Package**: https://www.npmjs.com/package/ruv-swarm  
- **Documentation**: https://docs.rs/ruv-swarm
- **Repository**: https://github.com/ruvnet/ruv-FANN
- **Issues**: https://github.com/ruvnet/ruv-FANN/issues

---

Built with ❤️ by the rUv team. Part of the [ruv-FANN](https://github.com/ruvnet/ruv-FANN) neural network framework.