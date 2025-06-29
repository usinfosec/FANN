# RUV-Swarm MCP Server

Model Context Protocol (MCP) server implementation for RUV-Swarm, enabling Claude and other MCP-compatible clients to interact with the swarm orchestration system.

## Features

- WebSocket-based MCP server
- Full swarm orchestration capabilities
- Tool discovery and execution
- Event streaming and monitoring
- Session management
- Performance optimization tools

## Available Tools

### Agent Management
- `ruv-swarm.spawn` - Spawn new agents
- `ruv-swarm.agent.list` - List active agents

### Task Orchestration
- `ruv-swarm.orchestrate` - Orchestrate swarm tasks
- `ruv-swarm.task.create` - Create new tasks
- `ruv-swarm.workflow.execute` - Execute workflow files

### Monitoring & Analysis
- `ruv-swarm.query` - Query swarm state
- `ruv-swarm.monitor` - Subscribe to events
- `ruv-swarm.optimize` - Performance optimization

### Memory Management
- `ruv-swarm.memory.store` - Store persistent data
- `ruv-swarm.memory.get` - Retrieve stored data

## Running the Server

```bash
# From the ruv-swarm directory
cargo run -p ruv-swarm-mcp

# The server will start on http://127.0.0.1:3000
# WebSocket endpoint: ws://127.0.0.1:3000/mcp
```

## Claude-Flow Integration

The MCP server integrates with claude-flow commands:

```bash
# Start MCP server
./claude-flow mcp start --port 3000

# Check server status
./claude-flow mcp status

# List available tools
./claude-flow mcp tools
```

## Example Usage

### Connect with WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:3000/mcp');

// Initialize connection
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  method: 'initialize',
  params: {},
  id: 1
}));

// List available tools
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  method: 'tools/list',
  params: {},
  id: 2
}));

// Spawn an agent
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  method: 'tools/call',
  params: {
    name: 'ruv-swarm.spawn',
    arguments: {
      agent_type: 'researcher',
      name: 'Research Agent 1'
    }
  },
  id: 3
}));
```

### Using with Claude

Claude can directly interact with the MCP server when configured:

```json
{
  "mcpServers": {
    "ruv-swarm": {
      "command": "ruv-swarm-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

## API Endpoints

- `GET /` - Server information
- `GET /mcp` - WebSocket endpoint
- `GET /tools` - List available tools
- `GET /health` - Health check

## Configuration

Create a `mcp-config.json`:

```json
{
  "bind_addr": "127.0.0.1:3000",
  "max_connections": 100,
  "request_timeout_secs": 300,
  "debug": true
}
```

## Development

```bash
# Run tests
cargo test -p ruv-swarm-mcp

# Run with debug logging
RUST_LOG=debug cargo run -p ruv-swarm-mcp

# Build release version
cargo build -p ruv-swarm-mcp --release
```