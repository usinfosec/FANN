# ruv-swarm 🧠🐝

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-supported-blue.svg)](https://webassembly.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![NPM](https://img.shields.io/badge/npm-ruv--swarm-red.svg)](https://www.npmjs.com/package/ruv-swarm)
[![MCP](https://img.shields.io/badge/MCP-supported-green.svg)](https://github.com/anthropics/model-context-protocol)

A high-performance, cognitive diversity-enabled distributed agent orchestration framework with native Model Context Protocol (MCP) support. Built on ruv-FANN neural networks for seamless integration with Claude Code and other MCP-enabled tools.

## ✨ Features

- 🤖 **12 MCP Tools**: Complete Model Context Protocol implementation for Claude Code
- 🧠 **Neural Networks**: WebAssembly-powered neural processing for each agent
- 💾 **SQLite Persistence**: Full state persistence with agent memory and task history
- 🚀 **High Performance**: WASM optimization with SIMD support detection
- 🔄 **Multiple Topologies**: Mesh, star, hierarchical, and ring networks
- 📊 **Real-time Monitoring**: Built-in swarm monitoring and metrics
- 🎯 **Cognitive Diversity**: 5 specialized agent types with unique capabilities
- 🌐 **Universal Deployment**: NPX, native Rust, WebAssembly, and MCP server

## 🚀 Quick Start

### NPX Installation (Recommended)
```bash
# Run directly with npx
npx ruv-swarm --help

# Initialize a swarm with MCP
npx ruv-swarm mcp start --protocol=stdio

# Create a swarm and spawn agents
npx ruv-swarm swarm_init --topology mesh --maxAgents 10
npx ruv-swarm agent_spawn researcher
npx ruv-swarm agent_spawn coder
npx ruv-swarm agent_spawn analyst

# Monitor swarm activity
npx ruv-swarm swarm_monitor --duration 10 --interval 1
```

### MCP Integration with Claude Code

#### 1. Configure Claude Code
Add to your `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm", "mcp", "start", "--protocol=stdio"],
      "capabilities": {
        "tools": true
      }
    }
  }
}
```

#### 2. Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `swarm_init` | Initialize a new swarm | `topology`, `maxAgents`, `strategy` |
| `swarm_status` | Get swarm status and statistics | `verbose` |
| `swarm_monitor` | Real-time swarm monitoring | `duration`, `interval` |
| `agent_spawn` | Create a new agent | `type`, `name`, `capabilities` |
| `agent_list` | List all agents | `filter` |
| `agent_metrics` | Get agent performance metrics | `agentId`, `metric` |
| `task_orchestrate` | Distribute task across agents | `task`, `priority`, `strategy` |
| `task_status` | Check task progress | `taskId`, `detailed` |
| `task_results` | Retrieve task results | `taskId`, `format` |
| `benchmark_run` | Run performance benchmarks | `type`, `iterations` |
| `features_detect` | Detect runtime capabilities | `category` |
| `memory_usage` | Get memory usage statistics | `detail` |

## 🏗️ Architecture

### Core Components

```
ruv-swarm/
├── src/               # JavaScript API and core logic
│   ├── index.js       # Main RuvSwarm class with WASM integration
│   ├── neural-agent.js # Neural network-enhanced agents
│   └── persistence.js  # SQLite database layer
├── wasm/              # WebAssembly modules
├── bin/               # CLI and MCP server
└── data/              # SQLite database storage
```

### Agent Types & Cognitive Patterns

| Agent Type | Specialization | Cognitive Pattern | Use Cases |
|------------|----------------|-------------------|-----------|
| **Researcher** | Data gathering, analysis | Divergent thinking | Information discovery, pattern recognition |
| **Coder** | Implementation, debugging | Convergent thinking | Code generation, optimization |
| **Analyst** | Statistical analysis, visualization | Analytical thinking | Data processing, insights |
| **Optimizer** | Performance tuning, efficiency | Systems thinking | Resource optimization, bottleneck removal |
| **Coordinator** | Task distribution, workflow | Strategic thinking | Team coordination, planning |

### Neural Network Architecture

Each agent has a dedicated neural network:
```javascript
{
  input_size: 10,      // Sensory inputs
  hidden_layers: [64, 32],  // Processing layers
  output_size: 5,      // Action outputs
  activation: 'relu',
  optimizer: 'adam'
}
```

## 📊 Performance & Benchmarks

### Current Performance Metrics
- **Agent Spawning**: ~15ms per agent with full neural network
- **Task Orchestration**: ~50ms for 5-agent distribution
- **Memory Usage**: 
  - Base: ~10MB
  - Per agent: ~5MB (including neural network)
  - SQLite DB: ~100KB per 1000 operations
- **MCP Response Time**: <10ms for all tools
- **Concurrent Operations**: 100+ agents supported

### Optimization Features
- ✅ WASM with SIMD detection (when available)
- ✅ SQLite with indexed queries
- ✅ Efficient JSON-RPC communication
- ✅ Connection pooling for database
- ✅ Lazy loading of neural networks

## 🔧 Advanced Usage

### JavaScript/TypeScript API
```javascript
import { RuvSwarm } from 'ruv-swarm';

// Initialize with options
const ruvSwarm = await RuvSwarm.initialize({
  wasmPath: './wasm',
  useSIMD: true,
  debug: false
});

// Create a swarm
const swarm = await ruvSwarm.createSwarm({
  name: 'my-swarm',
  strategy: 'balanced',
  mode: 'distributed',
  maxAgents: 10
});

// Spawn specialized agents
const researcher = await swarm.spawn({
  name: 'researcher-1',
  type: 'researcher',
  capabilities: ['data_analysis', 'pattern_recognition']
});

// Execute tasks
const result = await swarm.orchestrate({
  id: 'task-1',
  description: 'Analyze codebase performance',
  priority: 'high',
  dependencies: []
});
```

### Persistence & State Management
```javascript
// All state is automatically persisted to SQLite
const persistence = new SwarmPersistence('./data/ruv-swarm.db');

// Query historical data
const agents = persistence.getSwarmAgents(swarmId);
const tasks = persistence.getSwarmTasks(swarmId, 'completed');
const events = persistence.getSwarmEvents(swarmId, 100);

// Agent memory storage
persistence.storeAgentMemory(agentId, 'learned_patterns', patterns);
const memory = persistence.getAgentMemory(agentId, 'learned_patterns');
```

### Real-time Monitoring
```bash
# Monitor swarm activity
npx ruv-swarm swarm_monitor --duration 60 --interval 5

# Get detailed metrics
npx ruv-swarm agent_metrics --metric all

# Check memory usage
npx ruv-swarm memory_usage --detail by-agent
```

## 🧪 Testing

```bash
# Run all tests
npm test

# Test MCP tools
npm run test:mcp

# Test persistence
npm run test:persistence

# Test neural integration
npm run test:neural

# Comprehensive MCP test
node test/test-all-mcp-tools.js
```

## 📦 Installation Options

### Global Installation
```bash
npm install -g ruv-swarm
ruv-swarm --help
```

### Local Development
```bash
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/npm
npm install
npm link
```

### Docker
```dockerfile
FROM node:20-alpine
RUN npm install -g ruv-swarm
ENTRYPOINT ["ruv-swarm"]
```

## 🔮 Roadmap

- [x] Complete MCP implementation (12 tools)
- [x] SQLite persistence layer
- [x] Neural network integration per agent
- [x] Real-time monitoring
- [x] Performance benchmarking
- [ ] GPU acceleration via WebGPU
- [ ] Distributed swarm networking
- [ ] Advanced consensus algorithms
- [ ] Visual swarm dashboard
- [ ] Python bindings

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Clone the repository
2. Install dependencies: `npm install`
3. Build WASM (optional): `npm run build:wasm`
4. Run tests: `npm test`
5. Start MCP server: `npm run mcp:server`

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## ⭐ Acknowledgments

Built on the [ruv-FANN](https://github.com/ruvnet/ruv-FANN) neural network foundation with native Model Context Protocol support for seamless Claude Code integration.

---

**Made with ❤️ by the RUV team** | [GitHub](https://github.com/ruvnet/ruv-FANN) | [NPM](https://www.npmjs.com/package/ruv-swarm)