#!/usr/bin/env node

/**
 * Test script for ruv-swarm MCP tools
 * Demonstrates the functionality of the first 6 MCP tools with SQLite integration
 */

const { spawn } = require('child_process');
const path = require('path');

// MCP tool test cases
const testCases = [
  {
    name: 'swarm_init',
    tool: 'swarm_init',
    args: {
      topology: 'mesh',
      maxAgents: 10,
      strategy: 'balanced'
    },
    description: 'Initialize a mesh swarm with 10 max agents'
  },
  {
    name: 'swarm_status',
    tool: 'swarm_status',
    args: {
      verbose: true
    },
    description: 'Get detailed swarm status with agent info'
  },
  {
    name: 'agent_spawn_researcher',
    tool: 'agent_spawn',
    args: {
      type: 'researcher',
      name: 'research-bot-1',
      capabilities: ['web-scraping', 'data-mining']
    },
    description: 'Spawn a researcher agent with custom capabilities'
  },
  {
    name: 'agent_spawn_coder',
    tool: 'agent_spawn',
    args: {
      type: 'coder',
      name: 'code-bot-1',
      capabilities: ['typescript', 'rust']
    },
    description: 'Spawn a coder agent'
  },
  {
    name: 'agent_list',
    tool: 'agent_list',
    args: {
      filter: 'all'
    },
    description: 'List all agents with enriched data'
  },
  {
    name: 'agent_metrics',
    tool: 'agent_metrics',
    args: {
      metric: 'all'
    },
    description: 'Get comprehensive metrics for all agents'
  },
  {
    name: 'swarm_monitor',
    tool: 'swarm_monitor',
    args: {
      duration: 5,
      interval: 1
    },
    description: 'Monitor swarm for 5 seconds with 1-second intervals'
  }
];

// Execute MCP tool via stdio protocol
async function executeMcpTool(toolName, args) {
  return new Promise((resolve, reject) => {
    const mcpProcess = spawn('node', [
      path.join(__dirname, 'npm', 'bin', 'ruv-swarm.js'),
      'mcp',
      'start'
    ], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let responseBuffer = '';
    let errorBuffer = '';

    mcpProcess.stdout.on('data', (data) => {
      responseBuffer += data.toString();
      
      // Try to parse complete JSON responses
      const lines = responseBuffer.split('\n');
      for (const line of lines) {
        if (line.trim() && line.includes('{')) {
          try {
            const response = JSON.parse(line);
            if (response.result) {
              resolve(response.result);
              mcpProcess.kill();
            }
          } catch (e) {
            // Not a complete JSON yet
          }
        }
      }
    });

    mcpProcess.stderr.on('data', (data) => {
      errorBuffer += data.toString();
    });

    mcpProcess.on('error', reject);

    // Send initialize request
    const initRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {}
      }
    };
    mcpProcess.stdin.write(JSON.stringify(initRequest) + '\n');

    // Send tool call request after a short delay
    setTimeout(() => {
      const toolRequest = {
        jsonrpc: '2.0',
        id: 2,
        method: 'tools/call',
        params: {
          name: toolName,
          arguments: args
        }
      };
      mcpProcess.stdin.write(JSON.stringify(toolRequest) + '\n');
    }, 100);

    // Timeout after 10 seconds
    setTimeout(() => {
      reject(new Error('MCP tool execution timeout'));
      mcpProcess.kill();
    }, 10000);
  });
}

// Run all test cases
async function runTests() {
  console.log('🚀 Testing ruv-swarm MCP tools with SQLite integration\n');

  for (const testCase of testCases) {
    console.log(`\n📋 Test: ${testCase.name}`);
    console.log(`   Description: ${testCase.description}`);
    console.log(`   Tool: ${testCase.tool}`);
    console.log(`   Args:`, JSON.stringify(testCase.args, null, 2));

    try {
      const result = await executeMcpTool(testCase.tool, testCase.args);
      console.log(`   ✅ Success!`);
      console.log(`   Result:`, JSON.stringify(result, null, 2).split('\n').map(l => '   ' + l).join('\n'));
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
    }

    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log('\n\n✨ All tests completed!');
  console.log('\n📊 Database file created at: .ruv-swarm.db');
  console.log('   You can inspect it with: sqlite3 .ruv-swarm.db');
}

// Direct execution test (without MCP protocol)
async function directTest() {
  console.log('\n🔧 Direct execution test (bypassing MCP protocol)\n');

  const { executeSwarmTool } = require('./npm/bin/ruv-swarm.js');
  
  // Test swarm_init directly
  console.log('Testing swarm_init directly...');
  const initResult = await executeSwarmTool('swarm_init', {
    topology: 'hierarchical',
    maxAgents: 5,
    strategy: 'specialized'
  });
  console.log('Result:', JSON.stringify(JSON.parse(initResult.content[0].text), null, 2));

  // Test swarm_status directly
  console.log('\nTesting swarm_status directly...');
  const statusResult = await executeSwarmTool('swarm_status', { verbose: true });
  console.log('Result:', JSON.stringify(JSON.parse(statusResult.content[0].text), null, 2));
}

// Main execution
if (require.main === module) {
  if (process.argv.includes('--direct')) {
    directTest().catch(console.error);
  } else {
    runTests().catch(console.error);
  }
}

module.exports = { executeMcpTool, testCases };