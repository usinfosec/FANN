# ruv-swarm-wasm

WebAssembly bindings for ruv-swarm neural network orchestration with cognitive diversity and distributed intelligence.

## Features

### 🧠 Advanced Swarm Orchestration
- **Multiple Topologies**: Mesh, Star, Hierarchical, and Ring configurations
- **Dynamic Agent Management**: Spawn and manage agents with cognitive patterns
- **Task Distribution**: Intelligent task assignment based on capabilities
- **Real-time Monitoring**: Sub-second swarm performance metrics

### 🎭 Cognitive Diversity Engine
- **6 Cognitive Patterns**: Convergent, Divergent, Systems, Critical, Lateral, Abstract thinking
- **Diversity Analysis**: Shannon diversity index for optimal team composition
- **Pattern Interactions**: Synergistic, complementary, and conflicting relationships
- **Adaptive Recommendations**: AI-driven suggestions for team optimization

### 🔗 Neural Network Coordination
- **Distributed Training**: Data-parallel, model-parallel, federated, and swarm optimization
- **Knowledge Synchronization**: <100ms swarm-wide sync with multiple strategies
- **Collective Intelligence**: Emergent behavior patterns and self-organization
- **Neural Architectures**: Pattern-specific network templates for each cognitive style

### ⚡ Performance Targets
- Agent spawning: <20ms with full neural network setup
- Knowledge sync: <100ms swarm-wide
- Task orchestration: <100ms for complex multi-agent tasks
- Memory usage: <5MB per agent neural network
- WASM bundle size: <800KB

## Usage

### JavaScript/TypeScript

```javascript
import init, { 
    WasmSwarmOrchestrator, 
    CognitiveDiversityEngine,
    NeuralSwarmCoordinator 
} from 'ruv-swarm-wasm';

await init();

// Create orchestrator
const orchestrator = new WasmSwarmOrchestrator();

// Create swarm with cognitive diversity
const swarmConfig = {
    name: "Research Swarm",
    topology_type: "mesh",
    max_agents: 10,
    enable_cognitive_diversity: true
};

const swarm = orchestrator.create_swarm(swarmConfig);

// Spawn diverse agents
const researcher = orchestrator.spawn_agent(swarm.swarm_id, {
    agent_type: "researcher",
    name: "Darwin"
});

// Orchestrate complex tasks
const task = orchestrator.orchestrate_task(swarm.swarm_id, {
    description: "Analyze neural network architectures",
    required_capabilities: ["data_analysis", "pattern_recognition"],
    priority: "high"
});
```

### Rust (via wasm-bindgen)

```rust
use ruv_swarm_wasm::{WasmSwarmOrchestrator, CognitiveDiversityEngine};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn create_intelligent_swarm() -> Result<JsValue, JsValue> {
    let mut orchestrator = WasmSwarmOrchestrator::new();
    
    let config = serde_json::json!({
        "name": "AI Research Swarm",
        "topology_type": "hierarchical",
        "max_agents": 20,
        "enable_cognitive_diversity": true
    });
    
    orchestrator.create_swarm(serde_wasm_bindgen::to_value(&config)?)
}
```

## Cognitive Patterns

### Convergent Thinking (Analytical)
- **Focus**: Optimization and finding single best solutions
- **Neural Architecture**: Deep feedforward networks with attention
- **Best For**: Debugging, optimization, quality assurance

### Divergent Thinking (Creative)
- **Focus**: Generating multiple alternative solutions
- **Neural Architecture**: Wide parallel paths with fusion mechanisms
- **Best For**: Research, ideation, exploration

### Systems Thinking (Holistic)
- **Focus**: Understanding relationships and emergent properties
- **Neural Architecture**: Modular networks with bidirectional connections
- **Best For**: Architecture design, coordination, planning

### Critical Thinking (Evaluative)
- **Focus**: Analysis, validation, and error detection
- **Neural Architecture**: Validation layers with confidence scoring
- **Best For**: Testing, code review, analysis

### Lateral Thinking (Innovative)
- **Focus**: Unconventional problem-solving approaches
- **Neural Architecture**: Cross-domain connections with random projections
- **Best For**: Innovation, breakthrough solutions

### Abstract Thinking (Conceptual)
- **Focus**: High-level conceptual understanding and pattern abstraction
- **Neural Architecture**: Hierarchical representation networks with concept layers
- **Best For**: Theoretical analysis, conceptual design, strategic planning

## Swarm Topologies

### Mesh Topology
- Fully connected network
- Every agent can communicate with every other agent
- Highest redundancy and fault tolerance
- Best for: Small teams requiring high collaboration

### Star Topology
- Central hub agent coordinates all others
- Efficient for centralized decision-making
- Lower communication overhead
- Best for: Hierarchical organizations

### Hierarchical Topology
- Tree-like structure with multiple levels
- Natural delegation and specialization
- Scalable to large swarms
- Best for: Complex projects with clear structure

### Ring Topology
- Agents connected in a circular chain
- Efficient for sequential processing
- Low connection overhead
- Best for: Pipeline-style workflows

## Neural Coordination

### Distributed Training Modes

1. **Data Parallel**: Same model, different data partitions
2. **Model Parallel**: Split model across agents
3. **Federated**: Privacy-preserving distributed learning
4. **Swarm Optimization**: Evolutionary approach with collective intelligence

### Knowledge Synchronization Types

- **Weights**: Direct neural network weight sharing
- **Gradients**: Gradient exchange for collaborative learning
- **Features**: Share learned feature representations
- **Knowledge**: High-level knowledge transfer
- **All**: Complete synchronization

## Performance Optimization

### Memory Management
- Efficient WASM memory allocation
- Shared memory for inter-agent communication
- Automatic garbage collection for completed tasks

### Parallelization
- Web Workers for true parallel execution
- SIMD operations for neural computations
- Async/await for non-blocking operations

## Examples

See the `examples/` directory for complete demonstrations:
- `wasm_swarm_demo.js`: Full-featured swarm orchestration demo
- `cognitive_diversity_example.js`: Cognitive pattern analysis
- `neural_coordination_example.js`: Distributed neural training

## Building

```bash
# Install dependencies
npm install

# Build WASM module
wasm-pack build --target web --out-dir pkg

# Run tests
wasm-pack test --headless --chrome
```

## License

MIT OR Apache-2.0