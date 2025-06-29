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
    case 'spawn':
      await handleSpawn(swarm, args.slice(1));
      break;
    case 'orchestrate':
      await handleOrchestrate(swarm, args.slice(1));
      break;
    case 'status':
      await handleStatus(swarm);
      break;
    case 'benchmark':
      await handleBenchmark();
      break;
    case 'features':
      await handleFeatures();
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

Usage: npx @ruv/swarm <command> [options]

Commands:
  spawn <type> [name]     Spawn a new agent (researcher, coder, analyst, optimizer, coordinator)
  orchestrate <task>      Orchestrate a task across the swarm
  status                  Show swarm status
  benchmark               Run performance benchmarks
  features                Show runtime features
  version                 Show version information
  help                    Show this help message

Options:
  --debug                 Enable debug logging

Examples:
  npx @ruv/swarm spawn researcher my-researcher
  npx @ruv/swarm orchestrate "Analyze performance data and generate report"
  npx @ruv/swarm status
  npx @ruv/swarm benchmark

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
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});