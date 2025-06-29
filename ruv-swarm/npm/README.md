# @ruv/swarm

High-performance neural network swarm orchestration in WebAssembly.

## Features

- **WebAssembly Performance**: Near-native execution speed in browsers and Node.js
- **SIMD Optimization**: Automatic detection and use of SIMD instructions when available
- **Multi-Agent Orchestration**: Spawn and coordinate multiple AI agents for complex tasks
- **Type-Safe API**: Complete TypeScript definitions included
- **Cross-Platform**: Works in browsers, Node.js, and Deno
- **Zero Dependencies**: Standalone WASM module with no runtime dependencies

## Installation

```bash
npm install @ruv/swarm
```

Or use directly via npx:

```bash
npx @ruv/swarm --help
```

## Quick Start

### Node.js / JavaScript

```javascript
const { RuvSwarm } = require('@ruv/swarm');

async function main() {
  // Initialize WASM module
  const ruvSwarm = await RuvSwarm.initialize({
    useSIMD: true, // Enable SIMD if available
    debug: false
  });

  // Create a swarm
  const swarm = await ruvSwarm.createSwarm({
    name: 'my-swarm',
    strategy: 'development',
    mode: 'distributed',
    maxAgents: 10
  });

  // Spawn an agent
  const agent = await swarm.spawn({
    name: 'researcher-1',
    type: 'researcher',
    capabilities: ['web_search', 'data_analysis']
  });

  // Execute a task
  const result = await agent.execute({
    id: 'task-1',
    description: 'Research latest AI trends',
    parameters: { depth: 'comprehensive' }
  });

  console.log('Task result:', result);
}

main().catch(console.error);
```

### TypeScript

```typescript
import { RuvSwarm, SwarmConfig, AgentConfig } from '@ruv/swarm';

const config: SwarmConfig = {
  name: 'ai-swarm',
  strategy: 'research',
  mode: 'hierarchical',
  maxAgents: 5
};

const swarm = await RuvSwarm.initialize().then(rs => rs.createSwarm(config));
```

### Browser

```html
<script type="module">
  import { RuvSwarm } from 'https://unpkg.com/@ruv/swarm/dist/ruv-swarm.browser.js';
  
  const ruvSwarm = await RuvSwarm.initialize();
  const swarm = await ruvSwarm.createSwarm({
    name: 'browser-swarm',
    strategy: 'analysis',
    mode: 'centralized'
  });
  
  // Use the swarm...
</script>
```

## CLI Usage

The package includes a powerful CLI tool:

```bash
# Spawn an agent
npx @ruv/swarm spawn researcher my-researcher

# Orchestrate a task
npx @ruv/swarm orchestrate "Analyze this dataset and generate insights"

# Check swarm status
npx @ruv/swarm status

# Run benchmarks
npx @ruv/swarm benchmark

# Check runtime features
npx @ruv/swarm features
```

## Agent Types

- **Researcher**: Information gathering and research tasks
- **Coder**: Code generation and implementation
- **Analyst**: Data analysis and pattern recognition
- **Optimizer**: Performance optimization and tuning
- **Coordinator**: Task distribution and workflow management

## API Reference

### RuvSwarm

```typescript
class RuvSwarm {
  static initialize(options?: InitOptions): Promise<RuvSwarm>
  static detectSIMDSupport(): boolean
  static getRuntimeFeatures(): RuntimeFeatures
  static getVersion(): string
  static getMemoryUsage(): number
  
  createSwarm(config: SwarmConfig): Promise<Swarm>
}
```

### Swarm

```typescript
class Swarm {
  spawn(config: AgentConfig): Promise<Agent>
  orchestrate(task: TaskConfig): Promise<OrchestrationResult>
  getAgents(): string[]
  getStatus(): SwarmStatus
}
```

### Agent

```typescript
class Agent {
  execute(task: TaskRequest): Promise<TaskResponse>
  getMetrics(): AgentMetrics
  getCapabilities(): string[]
  reset(): void
}
```

## Performance

The WASM module provides near-native performance with additional optimizations:

- **SIMD Support**: Automatically uses SIMD instructions when available
- **Memory Efficiency**: Optimized memory usage with configurable limits
- **Parallel Execution**: Support for concurrent task execution
- **Smart Caching**: Efficient caching of WASM modules and results

## Development

To build from source:

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ruv-swarm/npm

# Install dependencies
npm install

# Build WASM modules
npm run build:all
```

## Requirements

- Node.js >= 14.0.0 (for Node.js usage)
- Modern browser with WebAssembly support (for browser usage)
- Rust toolchain (for building from source)
- wasm-pack (for building WASM modules)

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/ruvnet/ruv-FANN/blob/main/CONTRIBUTING.md) for details.

## Support

- Documentation: [https://github.com/ruvnet/ruv-FANN](https://github.com/ruvnet/ruv-FANN)
- Issues: [https://github.com/ruvnet/ruv-FANN/issues](https://github.com/ruvnet/ruv-FANN/issues)
- Discussions: [https://github.com/ruvnet/ruv-FANN/discussions](https://github.com/ruvnet/ruv-FANN/discussions)