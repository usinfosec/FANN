# ruv-swarm 🧠🐝

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-supported-blue.svg)](https://webassembly.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![NPM](https://img.shields.io/badge/npm-ruv--swarm-red.svg)](https://www.npmjs.com/package/ruv-swarm)
[![MCP](https://img.shields.io/badge/MCP-supported-green.svg)](https://github.com/anthropics/model-context-protocol)

## 🌟 What is ruv-swarm?

**ruv-swarm** transforms how you work with AI agents by providing a production-ready distributed orchestration system that thinks like a diverse team of experts. Instead of managing individual AI interactions, you coordinate intelligent swarms that can tackle complex, multi-faceted problems through cognitive diversity and specialized agent types.

### 🎯 **The Problem We Solve**

Modern AI workflows often involve repetitive prompting, context switching between tools, and manual coordination of complex tasks. You need a researcher to gather data, a coder to implement solutions, an analyst to process results, and a coordinator to manage the workflow. Traditional approaches require you to manually orchestrate these roles across multiple AI sessions.

### 💡 **Our Solution**

ruv-swarm provides **intelligent agent orchestration** where:

- **🧠 Neural-Enhanced Agents**: Each agent has its own neural network for specialized cognitive processing
- **🤝 Cognitive Diversity**: 5 distinct agent types (Researcher, Coder, Analyst, Optimizer, Coordinator) with unique thinking patterns
- **🔄 Autonomous Coordination**: Swarms self-organize using mesh, star, hierarchical, or ring topologies
- **💾 Persistent Memory**: SQLite-based persistence ensures agents learn and remember across sessions
- **⚡ Production Performance**: WebAssembly optimization delivers enterprise-grade speed and efficiency

### 🚀 **Why Choose ruv-swarm?**

| Traditional Approach | ruv-swarm Advantage |
|---------------------|-------------------|
| Manual task switching between AI tools | **Automated orchestration** across specialized agents |
| Context loss between sessions | **Persistent memory** and learning |
| Single cognitive approach | **Cognitive diversity** with 5 specialized thinking patterns |
| Limited scalability | **100+ concurrent agents** with efficient resource management |
| Complex integration setup | **Zero-config NPX deployment** + Claude Code MCP integration |

### 🎬 **See It In Action**

```bash
# Deploy instantly with zero installation
npx ruv-swarm mcp start

# Create a research and development swarm
npx ruv-swarm init mesh 10
npx ruv-swarm spawn researcher data-researcher
npx ruv-swarm spawn coder backend-developer 
npx ruv-swarm spawn analyst performance-analyst

# Orchestrate complex multi-agent tasks
npx ruv-swarm orchestrate "Build secure authentication API with performance monitoring"

# Monitor real-time swarm activity
npx ruv-swarm monitor
```

**Result**: Your swarm autonomously coordinates research, implementation, and analysis—delivering a complete solution with documentation, tests, and performance metrics.

### 🏢 **Production Use Cases**

- **🔬 Research & Analysis**: Multi-agent research teams for comprehensive market analysis
- **💻 Software Development**: Coordinated development teams for feature implementation  
- **📊 Data Processing**: Parallel analysis pipelines with specialized processing agents
- **🛡️ Security Auditing**: Comprehensive security reviews with specialized scanning agents
- **📈 Performance Optimization**: System analysis and optimization recommendation engines

## ✨ Core Features

### 🤖 **Complete MCP Integration**
- **12 MCP Tools**: Full Model Context Protocol implementation for Claude Code
- **JSON-RPC 2.0**: Standards-compliant communication with stdio/HTTP protocols
- **Zero Configuration**: Works out-of-the-box with `npx ruv-swarm mcp start`

### 🧠 **Neural-Enhanced Agents**
- **Cognitive Diversity**: 5 specialized agent types with unique thinking patterns
- **Neural Processing**: JavaScript-based neural networks for intelligent task routing
- **Adaptive Learning**: Agents learn from task execution and improve performance
- **Cognitive Patterns**: Convergent, Divergent, Systems, Critical, and Lateral thinking

### 💾 **Production Persistence**
- **SQLite Database**: Full state persistence with indexed queries
- **Agent Memory**: Persistent learning and experience storage
- **Task History**: Complete audit trail of orchestration events
- **Cross-Session State**: Agents remember and build on previous interactions

### 🚀 **High Performance**
- **WASM Framework**: Core swarm coordination powered by WebAssembly
- **SIMD Detection**: Automatic optimization when hardware supports it
- **Concurrent Operations**: 100+ agents with efficient resource management
- **Sub-50ms Orchestration**: Fast task distribution across agent networks

### 🔄 **Flexible Topologies**
- **Mesh Networks**: Fully connected agent communication
- **Star Topology**: Centralized coordination with hub agent
- **Hierarchical**: Multi-level agent organization
- **Ring Networks**: Circular agent communication patterns

### 📊 **Real-time Monitoring**
- **Live Metrics**: Agent performance, memory usage, task completion rates
- **Swarm Health**: Network topology status and agent availability
- **Performance Benchmarks**: Built-in benchmarking and optimization analysis
- **Event Logging**: Comprehensive logging with SQLite persistence

### 🌐 **Universal Deployment**
- **NPX Ready**: Zero-install deployment with `npx ruv-swarm`
- **MCP Server**: Native Claude Code integration via Model Context Protocol
- **CLI Interface**: Full command-line interface for all operations
- **JavaScript API**: Programmatic access for custom integrations

## 🚀 Quick Start

### NPX Installation (Recommended)
```bash
# Run directly with npx
npx ruv-swarm --help

# Initialize a swarm 
npx ruv-swarm init mesh 10

# Spawn specialized agents
npx ruv-swarm spawn researcher data-researcher
npx ruv-swarm spawn coder backend-developer
npx ruv-swarm spawn analyst performance-analyst

# Orchestrate complex tasks
npx ruv-swarm orchestrate "Build secure authentication API with monitoring"

# Monitor swarm activity
npx ruv-swarm monitor

# Check swarm status
npx ruv-swarm status
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

| Category | Tool | Description | Parameters |
|----------|------|-------------|------------|
| **Swarm** | `swarm_init` | Initialize swarm topology | `topology`, `maxAgents`, `strategy` |
| **Swarm** | `swarm_status` | Get swarm status | `verbose` |
| **Swarm** | `swarm_monitor` | Monitor swarm activity | `duration`, `interval` |
| **Agent** | `agent_spawn` | Spawn new agents | `type`, `name`, `capabilities` |
| **Agent** | `agent_list` | List active agents | `filter` |
| **Agent** | `agent_metrics` | Get agent performance metrics | `agentId`, `metric` |
| **Task** | `task_orchestrate` | Orchestrate distributed tasks | `task`, `priority`, `strategy` |
| **Task** | `task_status` | Check task progress | `taskId`, `detailed` |
| **Task** | `task_results` | Get task results | `taskId`, `format` |
| **Analytics** | `benchmark_run` | Run performance benchmarks | `type`, `iterations` |
| **Analytics** | `features_detect` | Detect runtime features | `category` |
| **Analytics** | `memory_usage` | Get memory usage statistics | `detail` |

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
npx ruv-swarm monitor

# Get swarm status
npx ruv-swarm status

# Run performance benchmarks
npx ruv-swarm benchmark

# Check neural network status
npx ruv-swarm neural status

# Show cognitive patterns
npx ruv-swarm neural patterns
```

## 🧪 Testing

```bash
# Run built-in functionality tests
npx ruv-swarm test

# Test all MCP tools
npx ruv-swarm mcp tools

# Run comprehensive test suite (if cloned)
cd ruv-swarm/npm
npm test

# Test specific components
node test/test-all-mcp-tools.js         # All MCP tools
node test/neural-integration.test.js    # Neural networks
node test/persistence.test.js           # SQLite persistence
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
1. Clone the repository: `git clone https://github.com/ruvnet/ruv-FANN.git`
2. Navigate to swarm: `cd ruv-FANN/ruv-swarm/npm`
3. Install dependencies: `npm install`
4. Link for local development: `npm link`
5. Run tests: `npm test`
6. Start MCP server: `npx ruv-swarm mcp start`

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## ⭐ Acknowledgments

Built on the [ruv-FANN](https://github.com/ruvnet/ruv-FANN) neural network foundation with native Model Context Protocol support for seamless Claude Code integration.

---

**Made with ❤️ by the RUV team** | [GitHub](https://github.com/ruvnet/ruv-FANN) | [NPM](https://www.npmjs.com/package/ruv-swarm)