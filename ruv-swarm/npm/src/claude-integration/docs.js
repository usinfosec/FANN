/**
 * Documentation generation module for Claude Code integration
 * Generates claude.md and .claude/commands/ documentation
 */

import { promises as fs } from 'fs';
import path from 'path';

class ClaudeDocsGenerator {
    constructor(options = {}) {
        this.workingDir = options.workingDir || process.cwd();
    }

    /**
     * Generate main claude.md configuration file
     */
    async generateClaudeMd() {
        const content = `# Claude Code Configuration for ruv-swarm

## 🎯 IMPORTANT: Separation of Responsibilities

### Claude Code Handles:
- ✅ **ALL file operations** (Read, Write, Edit, MultiEdit)
- ✅ **ALL code generation** and development tasks
- ✅ **ALL bash commands** and system operations
- ✅ **ALL actual implementation** work
- ✅ **Project navigation** and code analysis

### ruv-swarm MCP Tools Handle:
- 🧠 **Coordination only** - Orchestrating Claude Code's actions
- 💾 **Memory management** - Persistent state across sessions
- 🤖 **Neural features** - Cognitive patterns and learning
- 📊 **Performance tracking** - Monitoring and metrics
- 🐝 **Swarm orchestration** - Multi-agent coordination

### ⚠️ Key Principle:
**MCP tools DO NOT create content or write code.** They coordinate and enhance Claude Code's native capabilities. Think of them as an orchestration layer that helps Claude Code work more efficiently.

## 🚀 Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)
\`\`\`bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
\`\`\`

### 2. Use MCP Tools for Coordination in Claude Code
Once configured, ruv-swarm MCP tools enhance Claude Code's coordination:

**Initialize a swarm:**
- Use the \`mcp__ruv-swarm__swarm_init\` tool to set up coordination topology
- Choose: mesh, hierarchical, ring, or star
- This creates a coordination framework for Claude Code's work

**Spawn agents:**
- Use \`mcp__ruv-swarm__agent_spawn\` tool to create specialized coordinators
- Agent types represent different thinking patterns, not actual coders
- They help Claude Code approach problems from different angles

**Orchestrate tasks:**
- Use \`mcp__ruv-swarm__task_orchestrate\` tool to coordinate complex workflows
- This breaks down tasks for Claude Code to execute systematically
- The agents don't write code - they coordinate Claude Code's actions

## Available MCP Tools for Coordination

### Coordination Tools:
- \`mcp__ruv-swarm__swarm_init\` - Set up coordination topology for Claude Code
- \`mcp__ruv-swarm__agent_spawn\` - Create cognitive patterns to guide Claude Code
- \`mcp__ruv-swarm__task_orchestrate\` - Break down and coordinate complex tasks

### Monitoring Tools:
- \`mcp__ruv-swarm__swarm_status\` - Monitor coordination effectiveness
- \`mcp__ruv-swarm__agent_list\` - View active cognitive patterns
- \`mcp__ruv-swarm__agent_metrics\` - Track coordination performance
- \`mcp__ruv-swarm__task_status\` - Check workflow progress
- \`mcp__ruv-swarm__task_results\` - Review coordination outcomes

### Memory & Neural Tools:
- \`mcp__ruv-swarm__memory_usage\` - Persistent memory across sessions
- \`mcp__ruv-swarm__neural_status\` - Neural pattern effectiveness
- \`mcp__ruv-swarm__neural_train\` - Improve coordination patterns
- \`mcp__ruv-swarm__neural_patterns\` - Analyze thinking approaches

### System Tools:
- \`mcp__ruv-swarm__benchmark_run\` - Measure coordination efficiency
- \`mcp__ruv-swarm__features_detect\` - Available capabilities
- \`mcp__ruv-swarm__swarm_monitor\` - Real-time coordination tracking

## Workflow Examples (Coordination-Focused)

### Research Coordination Example
**Context:** Claude Code needs to research a complex topic systematically

**Step 1:** Set up research coordination
- Tool: \`mcp__ruv-swarm__swarm_init\`
- Parameters: \`{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}\`
- Result: Creates a mesh topology for comprehensive exploration

**Step 2:** Define research perspectives
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "researcher", "name": "Literature Review"}\`
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "analyst", "name": "Data Analysis"}\`
- Result: Different cognitive patterns for Claude Code to use

**Step 3:** Coordinate research execution
- Tool: \`mcp__ruv-swarm__task_orchestrate\`
- Parameters: \`{"task": "Research neural architecture search papers", "strategy": "adaptive"}\`
- Result: Claude Code systematically searches, reads, and analyzes papers

**What Actually Happens:**
1. The swarm sets up a coordination framework
2. Agents represent different analytical approaches
3. Claude Code uses its native Read, WebSearch, and Task tools
4. The swarm coordinates how Claude Code approaches the research
5. Results are synthesized by Claude Code, not the agents

### Development Coordination Example
**Context:** Claude Code needs to build a complex system with multiple components

**Step 1:** Set up development coordination
- Tool: \`mcp__ruv-swarm__swarm_init\`
- Parameters: \`{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}\`
- Result: Hierarchical structure for organized development

**Step 2:** Define development perspectives
- Tool: \`mcp__ruv-swarm__agent_spawn\`
- Parameters: \`{"type": "architect", "name": "System Design"}\`
- Result: Architectural thinking pattern for Claude Code

**Step 3:** Coordinate implementation
- Tool: \`mcp__ruv-swarm__task_orchestrate\`
- Parameters: \`{"task": "Implement user authentication with JWT", "strategy": "parallel"}\`
- Result: Claude Code implements features using its native tools

**What Actually Happens:**
1. The swarm creates a development coordination plan
2. Agents guide Claude Code's approach to the problem
3. Claude Code uses Write, Edit, Bash tools for implementation
4. The swarm ensures systematic coverage of all aspects
5. All code is written by Claude Code, not the agents

## Best Practices for Coordination

### ✅ DO:
- Use MCP tools to coordinate Claude Code's approach to complex tasks
- Let the swarm break down problems into manageable pieces
- Use memory tools to maintain context across sessions
- Monitor coordination effectiveness with status tools
- Train neural patterns for better coordination over time

### ❌ DON'T:
- Expect agents to write code (Claude Code does all implementation)
- Use MCP tools for file operations (use Claude Code's native tools)
- Try to make agents execute bash commands (Claude Code handles this)
- Confuse coordination with execution (MCP coordinates, Claude executes)

## Memory and Persistence

The swarm provides persistent memory that helps Claude Code:
- Remember project context across sessions
- Track decisions and rationale
- Maintain consistency in large projects
- Learn from previous coordination patterns

## Performance Benefits

When using ruv-swarm coordination with Claude Code:
- **84.8% SWE-Bench solve rate** - Better problem-solving through coordination
- **32.3% token reduction** - Efficient task breakdown reduces redundancy
- **2.8-4.4x speed improvement** - Parallel coordination strategies
- **27+ neural models** - Diverse cognitive approaches

## Integration Tips

1. **Start Simple**: Begin with basic swarm init and single agent
2. **Scale Gradually**: Add more agents as task complexity increases
3. **Use Memory**: Store important decisions and context
4. **Monitor Progress**: Regular status checks ensure effective coordination
5. **Train Patterns**: Let neural agents learn from successful coordinations

## Support

- Documentation: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm
- Issues: https://github.com/ruvnet/ruv-FANN/issues
- Examples: https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm/examples

---

Remember: **ruv-swarm coordinates, Claude Code creates!** Start with \`mcp__ruv-swarm__swarm_init\` to enhance your development workflow.
`;

        const filePath = path.join(this.workingDir, 'claude.md');
        await fs.writeFile(filePath, content);
        return { file: 'claude.md', success: true };
    }

    /**
     * Generate command documentation files in organized subdirectories
     */
    async generateCommandDocs() {
        const commandsDir = path.join(this.workingDir, '.claude', 'commands');
        await fs.mkdir(commandsDir, { recursive: true });

        // Create subdirectories
        const subdirs = ['coordination', 'monitoring', 'memory', 'workflows'];
        for (const subdir of subdirs) {
            await fs.mkdir(path.join(commandsDir, subdir), { recursive: true });
        }

        const commands = {
            // Coordination commands
            'coordination/init.md': {
                title: 'Initialize Coordination Framework',
                tool: 'mcp__ruv-swarm__swarm_init',
                params: '{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}',
                description: 'Set up a coordination topology to guide Claude Code\'s approach to complex tasks',
                details: `This tool creates a coordination framework that helps Claude Code:
- Break down complex problems systematically
- Approach tasks from multiple perspectives
- Maintain consistency across large projects
- Work more efficiently through structured coordination

Remember: This does NOT create actual coding agents. It creates a coordination pattern for Claude Code to follow.`
            },
            'coordination/spawn.md': {
                title: 'Create Cognitive Patterns',
                tool: 'mcp__ruv-swarm__agent_spawn',
                params: '{"type": "researcher", "name": "Literature Analysis", "capabilities": ["deep-analysis"]}',
                description: 'Define cognitive patterns that represent different approaches Claude Code can take',
                details: `Agent types represent thinking patterns, not actual coders:
- **researcher**: Systematic exploration approach
- **coder**: Implementation-focused thinking
- **analyst**: Data-driven decision making
- **architect**: Big-picture system design
- **reviewer**: Quality and consistency checking

These patterns guide how Claude Code approaches different aspects of your task.`
            },
            'coordination/orchestrate.md': {
                title: 'Coordinate Task Execution',
                tool: 'mcp__ruv-swarm__task_orchestrate',
                params: '{"task": "Implement authentication system", "strategy": "parallel", "priority": "high"}',
                description: 'Break down and coordinate complex tasks for systematic execution by Claude Code',
                details: `Orchestration strategies:
- **parallel**: Claude Code works on independent components simultaneously
- **sequential**: Step-by-step execution for dependent tasks
- **adaptive**: Dynamically adjusts based on task complexity

The orchestrator creates a plan that Claude Code follows using its native tools.`
            },

            // Monitoring commands
            'monitoring/status.md': {
                title: 'Check Coordination Status',
                tool: 'mcp__ruv-swarm__swarm_status',
                params: '{"verbose": true}',
                description: 'Monitor the effectiveness of current coordination patterns',
                details: `Shows:
- Active coordination topologies
- Current cognitive patterns in use
- Task breakdown and progress
- Resource utilization for coordination
- Overall system health`
            },
            'monitoring/agents.md': {
                title: 'List Active Patterns',
                tool: 'mcp__ruv-swarm__agent_list',
                params: '{"filter": "active"}',
                description: 'View all active cognitive patterns and their current focus areas',
                details: `Filters:
- **all**: Show all defined patterns
- **active**: Currently engaged patterns
- **idle**: Available but unused patterns
- **busy**: Patterns actively coordinating tasks`
            },

            // Memory commands
            'memory/usage.md': {
                title: 'Memory Management',
                tool: 'mcp__ruv-swarm__memory_usage',
                params: '{"detail": "detailed"}',
                description: 'Track persistent memory usage across Claude Code sessions',
                details: `Memory helps Claude Code:
- Maintain context between sessions
- Remember project decisions
- Track implementation patterns
- Store coordination strategies that worked well`
            },
            'memory/neural.md': {
                title: 'Neural Pattern Training',
                tool: 'mcp__ruv-swarm__neural_train',
                params: '{"iterations": 10}',
                description: 'Improve coordination patterns through neural network training',
                details: `Training improves:
- Task breakdown effectiveness
- Coordination pattern selection
- Resource allocation strategies
- Overall coordination efficiency`
            },

            // Workflow examples
            'workflows/research.md': {
                title: 'Research Workflow Coordination',
                content: `# Research Workflow Coordination

## Purpose
Coordinate Claude Code's research activities for comprehensive, systematic exploration.

## Step-by-Step Coordination

### 1. Initialize Research Framework
\`\`\`
Tool: mcp__ruv-swarm__swarm_init
Parameters: {"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}
\`\`\`
Creates a mesh topology for comprehensive exploration from multiple angles.

### 2. Define Research Perspectives
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "researcher", "name": "Literature Review"}
\`\`\`
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn  
Parameters: {"type": "analyst", "name": "Data Analysis"}
\`\`\`
Sets up different analytical approaches for Claude Code to use.

### 3. Execute Coordinated Research
\`\`\`
Tool: mcp__ruv-swarm__task_orchestrate
Parameters: {"task": "Research modern web frameworks performance", "strategy": "adaptive"}
\`\`\`

## What Claude Code Actually Does
1. Uses **WebSearch** tool for finding resources
2. Uses **Read** tool for analyzing documentation
3. Uses **Task** tool for parallel exploration
4. Synthesizes findings using coordination patterns
5. Stores insights in memory for future reference

Remember: The swarm coordinates HOW Claude Code researches, not WHAT it finds.`
            },
            'workflows/development.md': {
                title: 'Development Workflow Coordination',
                content: `# Development Workflow Coordination

## Purpose
Structure Claude Code's approach to complex development tasks for maximum efficiency.

## Step-by-Step Coordination

### 1. Initialize Development Framework
\`\`\`
Tool: mcp__ruv-swarm__swarm_init
Parameters: {"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}
\`\`\`
Creates hierarchical structure for organized, top-down development.

### 2. Define Development Perspectives
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "architect", "name": "System Design"}
\`\`\`
\`\`\`
Tool: mcp__ruv-swarm__agent_spawn
Parameters: {"type": "coder", "name": "Implementation Focus"}
\`\`\`
Sets up architectural and implementation thinking patterns.

### 3. Coordinate Implementation
\`\`\`
Tool: mcp__ruv-swarm__task_orchestrate
Parameters: {"task": "Build REST API with authentication", "strategy": "parallel", "priority": "high"}
\`\`\`

## What Claude Code Actually Does
1. Uses **Write** tool to create new files
2. Uses **Edit/MultiEdit** tools for code modifications
3. Uses **Bash** tool for testing and building
4. Uses **TodoWrite** tool for task tracking
5. Follows coordination patterns for systematic implementation

Remember: All code is written by Claude Code using its native tools!`
            }
        };

        const createdFiles = [];
        
        // Generate command files
        for (const [filepath, config] of Object.entries(commands)) {
            let content;
            
            if (config.content) {
                // Use provided content for workflow files
                content = config.content;
            } else {
                // Generate content for tool documentation
                content = `# ${config.title}

## 🎯 Key Principle
**This tool coordinates Claude Code's actions. It does NOT write code or create content.**

## MCP Tool Usage in Claude Code

**Tool:** \`${config.tool}\`

## Parameters
\`\`\`json
${config.params}
\`\`\`

## Description
${config.description}

## Details
${config.details}

## Example Usage

**In Claude Code:**
1. Use the tool: \`${config.tool}\`
2. With parameters: \`${config.params}\`
3. Claude Code then executes the coordinated plan using its native tools

## Important Reminders
- ✅ This tool provides coordination and structure
- ✅ Claude Code performs all actual implementation
- ❌ The tool does NOT write code
- ❌ The tool does NOT access files directly
- ❌ The tool does NOT execute commands

## See Also
- Main documentation: /claude.md
- Other commands in this category
- Workflow examples in /workflows/
`;
            }

            const filePath = path.join(commandsDir, filepath);
            await fs.writeFile(filePath, content);
            createdFiles.push(filepath);
        }

        return { files: createdFiles, success: true };
    }

    /**
     * Generate all documentation files
     */
    async generateAll() {
        console.log('📚 Generating Claude Code documentation...');
        
        try {
            const results = {
                claudeMd: await this.generateClaudeMd(),
                commands: await this.generateCommandDocs(),
                success: true
            };

            console.log('✅ Documentation generated successfully');
            console.log('   - claude.md');
            console.log('   - .claude/commands/ directory with ' + results.commands.files.length + ' files');
            
            return results;
        } catch (error) {
            console.error('❌ Failed to generate documentation:', error.message);
            throw error;
        }
    }
}

export { ClaudeDocsGenerator };