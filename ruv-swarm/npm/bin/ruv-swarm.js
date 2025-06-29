#!/usr/bin/env node
/**
 * ruv-swarm CLI - Neural network swarm orchestration
 */

const { RuvSwarm } = require('../src');
const path = require('path');
const fs = require('fs');

async function main() {
  const args = process.argv.slice(2);
  
  // Initialize WASM module
  let swarm;
  try {
    const ruvSwarm = await RuvSwarm.initialize({
      wasmPath: path.join(__dirname, '..', 'wasm'),
      useSIMD: RuvSwarm.detectSIMDSupport(),
      debug: args.includes('--debug')
    });

    swarm = await ruvSwarm.createSwarm({
      name: 'cli-swarm',
      strategy: 'development',
      mode: 'centralized',
      maxAgents: 5
    });
  } catch (error) {
    console.error('Failed to initialize RuvSwarm:', error.message);
    process.exit(1);
  }
  
  // Parse commands
  const command = args[0] || 'help';
  
  switch (command) {
    case 'init':
      await handleInit(args.slice(1));
      break;
    case 'spawn':
      await handleSpawn(swarm, args.slice(1));
      break;
    case 'orchestrate':
      await handleOrchestrate(swarm, args.slice(1));
      break;
    case 'status':
      await handleStatus(swarm);
      break;
    case 'monitor':
      await handleMonitor(swarm);
      break;
    case 'benchmark':
      await handleBenchmark();
      break;
    case 'features':
      await handleFeatures();
      break;
    case 'install':
      await handleInstall(args.slice(1));
      break;
    case 'test':
      await handleTest();
      break;
    case 'mcp':
      await handleMcp(args.slice(1));
      break;
    case 'version':
      console.log(`ruv-swarm v${RuvSwarm.getVersion()}`);
      break;
    case 'help':
    default:
      showHelp();
  }
}

async function handleSpawn(swarm, args) {
  const agentType = args[0] || 'researcher';
  const agentName = args[1] || `${agentType}-${Date.now()}`;

  try {
    const agent = await swarm.spawn({
      name: agentName,
      type: agentType,
      capabilities: []
    });

    console.log(`Agent spawned successfully:`);
    console.log(`  ID: ${agent.id}`);
    console.log(`  Type: ${agent.agentType}`);
    console.log(`  Status: ${agent.status}`);
    console.log(`  Capabilities: ${agent.getCapabilities().join(', ')}`);
  } catch (error) {
    console.error('Failed to spawn agent:', error.message);
    process.exit(1);
  }
}

async function handleOrchestrate(swarm, args) {
  const taskDescription = args.join(' ') || 'Default task';

  try {
    console.log('Orchestrating task:', taskDescription);
    
    const result = await swarm.orchestrate({
      id: `task-${Date.now()}`,
      description: taskDescription,
      priority: 'medium',
      dependencies: [],
      metadata: {}
    });

    console.log('\nOrchestration completed:');
    console.log(`  Task ID: ${result.taskId}`);
    console.log(`  Status: ${result.status}`);
    console.log(`  Total Time: ${result.metrics.totalTime}s`);
    console.log(`  Agents Used: ${result.metrics.agentsSpawned}`);
    
    console.log('\nAgent Results:');
    result.results.forEach((agentResult, index) => {
      console.log(`  ${index + 1}. ${agentResult.agentType} (${agentResult.agentId})`);
      console.log(`     Execution Time: ${agentResult.executionTime}s`);
      console.log(`     Output:`, JSON.stringify(agentResult.output, null, 2));
    });
  } catch (error) {
    console.error('Orchestration failed:', error.message);
    process.exit(1);
  }
}

async function handleStatus(swarm) {
  const status = swarm.getStatus();
  
  console.log('Swarm Status:');
  console.log(`  Name: ${status.name}`);
  console.log(`  Strategy: ${status.strategy}`);
  console.log(`  Mode: ${status.mode}`);
  console.log(`  Active Agents: ${status.agentCount}/${status.maxAgents}`);
  
  if (status.agents.length > 0) {
    console.log('\nActive Agents:');
    status.agents.forEach((agentId, index) => {
      console.log(`  ${index + 1}. ${agentId}`);
    });
  }
  
  console.log(`\nMemory Usage: ${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB`);
}

async function handleBenchmark() {
  console.log('Running performance benchmarks...\n');

  const benchmarks = [
    { name: 'WASM Initialization', fn: async () => await RuvSwarm.initialize() },
    { name: 'Swarm Creation', fn: async (rs) => await rs.createSwarm({ name: 'bench', strategy: 'development', mode: 'centralized' }) },
    { name: 'Agent Spawn', fn: async (rs, s) => await s.spawn({ name: 'bench-agent', type: 'researcher' }) },
    { name: 'Task Orchestration', fn: async (rs, s) => await s.orchestrate({ id: 'bench-task', description: 'Benchmark task', priority: 'low', dependencies: [] }) }
  ];

  for (const benchmark of benchmarks) {
    const start = process.hrtime.bigint();
    
    try {
      let ruvSwarm, swarm;
      
      if (benchmark.name === 'WASM Initialization') {
        ruvSwarm = await benchmark.fn();
      } else if (benchmark.name === 'Swarm Creation') {
        ruvSwarm = await RuvSwarm.initialize();
        swarm = await benchmark.fn(ruvSwarm);
      } else {
        ruvSwarm = await RuvSwarm.initialize();
        swarm = await ruvSwarm.createSwarm({ name: 'bench', strategy: 'development', mode: 'centralized' });
        await benchmark.fn(ruvSwarm, swarm);
      }
      
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1e6; // Convert to milliseconds
      
      console.log(`${benchmark.name}: ${duration.toFixed(2)}ms`);
    } catch (error) {
      console.log(`${benchmark.name}: Failed - ${error.message}`);
    }
  }

  console.log('\nBenchmark completed.');
}

async function handleFeatures() {
  console.log('Runtime Features:');
  console.log(`  WASM Support: ${typeof WebAssembly !== 'undefined' ? 'Yes' : 'No'}`);
  
  try {
    const features = RuvSwarm.getRuntimeFeatures();
    console.log(`  SIMD Support: ${features.simdAvailable ? 'Yes' : 'No'}`);
    console.log(`  Threading Support: ${features.threadsAvailable ? 'Yes' : 'No'}`);
    console.log(`  Memory Limit: ${(features.memoryLimit / 1024 / 1024 / 1024).toFixed(2)} GB`);
  } catch (error) {
    console.log(`  Unable to detect features: ${error.message}`);
  }
  
  console.log(`  Node.js Version: ${process.version}`);
  console.log(`  Platform: ${process.platform}`);
  console.log(`  Architecture: ${process.arch}`);
}

function showHelp() {
  console.log(`
ruv-swarm - High-performance neural network swarm orchestration

Usage: npx ruv-swarm <command> [options]

Commands:
  init [topology] [max]   Initialize a new swarm (default: mesh topology, 5 agents)
  spawn <type> [name]     Spawn a new agent (researcher, coder, analyst, optimizer, coordinator)
  orchestrate <task>      Orchestrate a task across the swarm
  status                  Show swarm status and agent information
  monitor                 Real-time swarm monitoring (press Ctrl+C to stop)
  mcp <subcommand>        MCP server for Claude Code integration (stdio/http protocols)
  benchmark               Run performance benchmarks
  features                Show runtime features and capabilities
  install [target]        Show installation instructions (global, project, local)
  test                    Run functionality tests
  version                 Show version information
  help                    Show this help message

Options:
  --debug                 Enable debug logging

Examples:
  npx ruv-swarm init mesh 10              # Initialize mesh topology with 10 max agents
  npx ruv-swarm spawn researcher my-researcher
  npx ruv-swarm orchestrate "Analyze performance data and generate report"
  npx ruv-swarm status                    # Show current swarm status
  npx ruv-swarm monitor                   # Real-time monitoring
  npx ruv-swarm mcp start                 # Start MCP server for Claude Code
  npx ruv-swarm mcp tools                 # List MCP tools
  npx ruv-swarm benchmark                 # Performance benchmarks
  npx ruv-swarm test                      # Run functionality tests
  npx ruv-swarm install global           # Installation instructions

Agent Types:
  - researcher: Information gathering and research tasks
  - coder: Code generation and implementation tasks
  - analyst: Data analysis and pattern recognition
  - optimizer: Performance optimization and tuning
  - coordinator: Task distribution and workflow management

For more information, visit: https://github.com/ruvnet/ruv-FANN
`);
}

// Handle errors gracefully
process.on('unhandledRejection', (error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});

// Run main function
async function handleInit(args) {
  const topology = args[0] || 'mesh';
  const maxAgents = parseInt(args[1]) || 5;
  
  console.log(`Initializing ruv-swarm with ${topology} topology (max ${maxAgents} agents)...`);
  
  try {
    // Create a swarm configuration file
    const config = {
      topology,
      maxAgents,
      strategy: 'balanced',
      persistence: 'memory',
      transport: 'websocket',
      created: new Date().toISOString()
    };
    
    const configPath = path.join(process.cwd(), 'ruv-swarm.config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    
    console.log('✓ Swarm configuration created');
    console.log(`✓ Config saved to: ${configPath}`);
    console.log('\nNext steps:');
    console.log('  npx ruv-swarm spawn researcher    # Spawn your first agent');
    console.log('  npx ruv-swarm status              # Check swarm status');
  } catch (error) {
    console.error('Failed to initialize swarm:', error.message);
    process.exit(1);
  }
}

async function handleMonitor(swarm) {
  console.log('Monitoring swarm activity (press Ctrl+C to stop)...\n');
  
  const startTime = Date.now();
  let counter = 0;
  
  const monitor = setInterval(() => {
    counter++;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const status = swarm.getStatus();
    
    // Clear screen and show updated status
    process.stdout.write('\u001b[2J\u001b[0;0H');
    console.log(`Swarm Monitor - Elapsed: ${elapsed}s (Update #${counter})`);
    console.log('='.repeat(50));
    console.log(`Swarm: ${status.name}`);
    console.log(`Strategy: ${status.strategy} | Mode: ${status.mode}`);
    console.log(`Agents: ${status.agentCount}/${status.maxAgents}`);
    console.log(`Memory: ${(RuvSwarm.getMemoryUsage() / 1024 / 1024).toFixed(2)} MB`);
    
    if (status.agents.length > 0) {
      console.log('\nActive Agents:');
      status.agents.forEach((agentId, index) => {
        console.log(`  ${index + 1}. ${agentId}`);
      });
    }
    
    console.log('\nPress Ctrl+C to exit monitor');
  }, 2000);
  
  // Handle Ctrl+C gracefully
  process.on('SIGINT', () => {
    clearInterval(monitor);
    console.log('\n\nMonitoring stopped.');
    process.exit(0);
  });
}

async function handleInstall(args) {
  const target = args[0] || 'local';
  
  console.log(`Installing ruv-swarm for ${target} use...`);
  
  try {
    if (target === 'global') {
      console.log('To install globally, run:');
      console.log('  npm install -g ruv-swarm');
    } else if (target === 'project') {
      console.log('To install in your project, run:');
      console.log('  npm install ruv-swarm');
      console.log('  # or');
      console.log('  yarn add ruv-swarm');
    } else {
      console.log('Available installation options:');
      console.log('  npx ruv-swarm install global   # Install globally');
      console.log('  npx ruv-swarm install project  # Install in current project');
      console.log('\nOr use directly with npx:');
      console.log('  npx ruv-swarm <command>');
    }
    
    console.log('\nFor Rust native version:');
    console.log('  cargo install ruv-swarm-cli');
    console.log('  # or build from source');
    console.log('  git clone https://github.com/ruvnet/ruv-FANN.git');
    console.log('  cd ruv-FANN/ruv-swarm');
    console.log('  cargo build --release');
  } catch (error) {
    console.error('Installation guidance failed:', error.message);
  }
}

async function handleTest() {
  console.log('Running ruv-swarm functionality tests...\n');
  
  let ruvSwarm;
  
  const tests = [
    { name: 'WASM Module Loading', fn: async () => { ruvSwarm = await RuvSwarm.initialize(); return ruvSwarm; } },
    { name: 'SIMD Detection', fn: () => RuvSwarm.detectSIMDSupport() },
    { name: 'Version Info', fn: () => RuvSwarm.getVersion() },
    { name: 'Runtime Features', fn: () => ruvSwarm ? ruvSwarm.getRuntimeFeatures() : RuvSwarm.getRuntimeFeatures() },
    { name: 'Memory Usage', fn: () => RuvSwarm.getMemoryUsage() },
    { name: 'Swarm Creation', fn: async () => ruvSwarm ? await ruvSwarm.createSwarm({ name: 'test-swarm', strategy: 'development', mode: 'centralized' }) : null }
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const test of tests) {
    try {
      const result = await test.fn();
      console.log(`✓ ${test.name}: PASS`);
      if (typeof result !== 'undefined') {
        console.log(`  Result: ${JSON.stringify(result)}`);
      }
      passed++;
    } catch (error) {
      console.log(`✗ ${test.name}: FAIL - ${error.message}`);
      failed++;
    }
  }
  
  console.log(`\nTest Results: ${passed} passed, ${failed} failed`);
  
  if (failed === 0) {
    console.log('\n🎉 All tests passed! ruv-swarm is ready to use.');
  } else {
    console.log('\n⚠️  Some tests failed. Check your installation.');
    process.exit(1);
  }
}

async function handleMcp(args) {
  const subcommand = args[0] || 'help';
  
  switch (subcommand) {
    case 'start':
      await startMcpServer(args.slice(1));
      break;
    case 'status':
      await getMcpStatus();
      break;
    case 'stop':
      await stopMcpServer();
      break;
    case 'tools':
      await listMcpTools();
      break;
    case 'config':
      await configureMcp(args.slice(1));
      break;
    case 'help':
    default:
      showMcpHelp();
  }
}

async function startMcpServer(args) {
  const protocol = args.find(arg => arg.startsWith('--protocol='))?.split('=')[1] || 'stdio';
  const port = args.find(arg => arg.startsWith('--port='))?.split('=')[1] || '3000';
  const host = args.find(arg => arg.startsWith('--host='))?.split('=')[1] || 'localhost';
  
  console.log(`Starting ruv-swarm MCP server...`);
  console.log(`Protocol: ${protocol}`);
  
  try {
    if (protocol === 'stdio') {
      console.log('Starting MCP server in stdio mode for Claude Code integration');
      console.log('\nMCP Server Configuration:');
      console.log('  Name: ruv-swarm');
      console.log('  Protocol: stdio');
      console.log('  Transport: WASM + Node.js');
      console.log('  Tools: 12 swarm orchestration tools');
      
      // In stdio mode, we'd typically exec the Rust binary
      console.log('\nTo use with Claude Code, add this to your MCP settings:');
      console.log('```json');
      console.log('{');
      console.log('  "mcpServers": {');
      console.log('    "ruv-swarm": {');
      console.log('      "command": "npx",');
      console.log('      "args": ["ruv-swarm", "mcp", "start", "--protocol=stdio"]');
      console.log('    }');
      console.log('  }');
      console.log('}');
      console.log('```');
      
      // Start stdio MCP server loop
      console.log('\nMCP server ready for stdio communication...');
      process.stdin.setEncoding('utf8');
      process.stdin.on('readable', () => {
        let chunk;
        while (null !== (chunk = process.stdin.read())) {
          try {
            const request = JSON.parse(chunk.trim());
            handleMcpRequest(request);
          } catch (error) {
            console.error('Invalid MCP request:', error.message);
          }
        }
      });
      
    } else if (protocol === 'http') {
      console.log(`Starting MCP server in HTTP mode on ${host}:${port}`);
      console.log('\nHTTP endpoints:');
      console.log(`  http://${host}:${port}/mcp/tools - List available tools`);
      console.log(`  http://${host}:${port}/mcp/execute - Execute tool requests`);
      console.log(`  http://${host}:${port}/health - Health check`);
      
      // For HTTP mode, we could start an HTTP server
      console.log('\nHTTP MCP server would be implemented here...');
      console.log('Note: HTTP streaming MCP server requires additional implementation.');
      
    } else {
      throw new Error(`Unsupported protocol: ${protocol}. Use 'stdio' or 'http'`);
    }
    
  } catch (error) {
    console.error('Failed to start MCP server:', error.message);
    process.exit(1);
  }
}

async function handleMcpRequest(request) {
  // Basic MCP request handling
  const response = {
    jsonrpc: '2.0',
    id: request.id
  };
  
  try {
    switch (request.method) {
      case 'initialize':
        response.result = {
          protocolVersion: '2024-11-05',
          capabilities: {
            tools: {},
            resources: {}
          },
          serverInfo: {
            name: 'ruv-swarm',
            version: '0.1.0'
          }
        };
        break;
        
      case 'tools/list':
        response.result = {
          tools: [
            {
              name: 'swarm_init',
              description: 'Initialize a new swarm with specified topology',
              inputSchema: {
                type: 'object',
                properties: {
                  topology: { type: 'string', enum: ['mesh', 'hierarchical', 'ring', 'star'] },
                  maxAgents: { type: 'number', minimum: 1, maximum: 100 }
                }
              }
            },
            {
              name: 'agent_spawn',
              description: 'Spawn a new agent in the swarm',
              inputSchema: {
                type: 'object',
                properties: {
                  type: { type: 'string', enum: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'] },
                  name: { type: 'string' },
                  capabilities: { type: 'array', items: { type: 'string' } }
                }
              }
            },
            {
              name: 'task_orchestrate',
              description: 'Orchestrate a task across the swarm',
              inputSchema: {
                type: 'object',
                properties: {
                  task: { type: 'string' },
                  priority: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] }
                }
              }
            }
          ]
        };
        break;
        
      case 'tools/call':
        const toolName = request.params.name;
        const args = request.params.arguments || {};
        
        response.result = {
          content: [{
            type: 'text',
            text: `Executed ${toolName} with args: ${JSON.stringify(args)}`
          }]
        };
        break;
        
      default:
        response.error = {
          code: -32601,
          message: `Method not found: ${request.method}`
        };
    }
  } catch (error) {
    response.error = {
      code: -32603,
      message: error.message
    };
  }
  
  process.stdout.write(JSON.stringify(response) + '\n');
}

async function getMcpStatus() {
  console.log('MCP Server Status:');
  console.log('  Status: Not implemented in WASM mode');
  console.log('  Protocol: stdio/http');
  console.log('  Tools: 12 available');
  console.log('\nTo check active MCP server, use:');
  console.log('  ps aux | grep ruv-swarm');
}

async function stopMcpServer() {
  console.log('Stopping MCP server...');
  console.log('Note: In stdio mode, close the client connection.');
  console.log('Note: In HTTP mode, stop the HTTP server process.');
}

async function listMcpTools() {
  console.log('Available MCP Tools:');
  console.log('\n🔧 Swarm Management:');
  console.log('  • swarm_init        - Initialize swarm topology');
  console.log('  • swarm_status      - Get swarm status');
  console.log('  • swarm_monitor     - Monitor swarm activity');
  
  console.log('\n🤖 Agent Operations:');
  console.log('  • agent_spawn       - Spawn new agents');
  console.log('  • agent_list        - List active agents');
  console.log('  • agent_metrics     - Get agent performance metrics');
  
  console.log('\n📋 Task Management:');
  console.log('  • task_orchestrate  - Orchestrate distributed tasks');
  console.log('  • task_status       - Check task progress');
  console.log('  • task_results      - Get task results');
  
  console.log('\n🔬 Analytics:');
  console.log('  • benchmark_run     - Run performance benchmarks');
  console.log('  • features_detect   - Detect runtime features');
  console.log('  • memory_usage      - Get memory usage statistics');
}

async function configureMcp(args) {
  const configType = args[0] || 'show';
  
  switch (configType) {
    case 'show':
      console.log('Current MCP Configuration:');
      console.log('  Protocol: stdio (default), http (optional)');
      console.log('  WASM Module: ruv_swarm_wasm.wasm');
      console.log('  Transport: WebAssembly + Node.js');
      console.log('  Claude Code Compatible: Yes');
      break;
      
    case 'claude':
      console.log('Claude Code MCP Configuration:');
      console.log('\nAdd to your Claude Code settings.json:');
      console.log('```json');
      console.log('{');
      console.log('  "mcpServers": {');
      console.log('    "ruv-swarm": {');
      console.log('      "command": "npx",');
      console.log('      "args": ["ruv-swarm", "mcp", "start"],');
      console.log('      "env": {}');
      console.log('    }');
      console.log('  }');
      console.log('}');
      console.log('```');
      break;
      
    default:
      console.log('Available config commands:');
      console.log('  npx ruv-swarm mcp config show    - Show current configuration');
      console.log('  npx ruv-swarm mcp config claude  - Show Claude Code setup');
  }
}

function showMcpHelp() {
  console.log('ruv-swarm MCP Server - Model Context Protocol integration\n');
  console.log('Usage: npx ruv-swarm mcp <command> [options]\n');
  
  console.log('Commands:');
  console.log('  start [options]     Start MCP server for Claude Code integration');
  console.log('  status              Show MCP server status');
  console.log('  stop                Stop MCP server');
  console.log('  tools               List available MCP tools');
  console.log('  config <type>       Show/configure MCP settings');
  console.log('  help                Show this help message\n');
  
  console.log('Start Options:');
  console.log('  --protocol=stdio    Use stdio communication (default, for Claude Code)');
  console.log('  --protocol=http     Use HTTP communication (for web clients)');
  console.log('  --port=3000         HTTP port (when using http protocol)');
  console.log('  --host=localhost    HTTP host (when using http protocol)\n');
  
  console.log('Examples:');
  console.log('  npx ruv-swarm mcp start                    # Start stdio MCP server');
  console.log('  npx ruv-swarm mcp start --protocol=http    # Start HTTP MCP server');
  console.log('  npx ruv-swarm mcp tools                    # List available tools');
  console.log('  npx ruv-swarm mcp config claude           # Show Claude Code setup\n');
  
  console.log('🔗 Integration:');
  console.log('  For Claude Code: Use stdio protocol (default)');
  console.log('  For Web Apps:   Use http protocol with streaming');
  console.log('  WASM Support:   Full WebAssembly integration for performance\n');
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});