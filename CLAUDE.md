# Claude Code Configuration for ruv-swarm
#ruv 
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
```bash
# Add ruv-swarm MCP server to Claude Code using stdio
claude mcp add ruv-swarm npx ruv-swarm mcp start
```

### 2. Use MCP Tools for Coordination in Claude Code
Once configured, ruv-swarm MCP tools enhance Claude Code's coordination:

**Initialize a swarm:**
- Use the `mcp__ruv-swarm__swarm_init` tool to set up coordination topology
- Choose: mesh, hierarchical, ring, or star
- This creates a coordination framework for Claude Code's work

**Spawn agents:**
- Use `mcp__ruv-swarm__agent_spawn` tool to create specialized coordinators
- Agent types represent different thinking patterns, not actual coders
- They help Claude Code approach problems from different angles

**Orchestrate tasks:**
- Use `mcp__ruv-swarm__task_orchestrate` tool to coordinate complex workflows
- This breaks down tasks for Claude Code to execute systematically
- The agents don't write code - they coordinate Claude Code's actions

## Available MCP Tools for Coordination

### Coordination Tools:
- `mcp__ruv-swarm__swarm_init` - Set up coordination topology for Claude Code
- `mcp__ruv-swarm__agent_spawn` - Create cognitive patterns to guide Claude Code
- `mcp__ruv-swarm__task_orchestrate` - Break down and coordinate complex tasks

### Monitoring Tools:
- `mcp__ruv-swarm__swarm_status` - Monitor coordination effectiveness
- `mcp__ruv-swarm__agent_list` - View active cognitive patterns
- `mcp__ruv-swarm__agent_metrics` - Track coordination performance
- `mcp__ruv-swarm__task_status` - Check workflow progress
- `mcp__ruv-swarm__task_results` - Review coordination outcomes

### Memory & Neural Tools:
- `mcp__ruv-swarm__memory_usage` - Persistent memory across sessions
- `mcp__ruv-swarm__neural_status` - Neural pattern effectiveness
- `mcp__ruv-swarm__neural_train` - Improve coordination patterns
- `mcp__ruv-swarm__neural_patterns` - Analyze thinking approaches

### System Tools:
- `mcp__ruv-swarm__benchmark_run` - Measure coordination efficiency
- `mcp__ruv-swarm__features_detect` - Available capabilities
- `mcp__ruv-swarm__swarm_monitor` - Real-time coordination tracking

## Workflow Examples (Coordination-Focused)

### Research Coordination Example
**Context:** Claude Code needs to research a complex topic systematically

**Step 1:** Set up research coordination
- Tool: `mcp__ruv-swarm__swarm_init`
- Parameters: `{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}`
- Result: Creates a mesh topology for comprehensive exploration

**Step 2:** Define research perspectives
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "researcher", "name": "Literature Review"}`
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "analyst", "name": "Data Analysis"}`
- Result: Different cognitive patterns for Claude Code to use

**Step 3:** Coordinate research execution
- Tool: `mcp__ruv-swarm__task_orchestrate`
- Parameters: `{"task": "Research neural architecture search papers", "strategy": "adaptive"}`
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
- Tool: `mcp__ruv-swarm__swarm_init`
- Parameters: `{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}`
- Result: Hierarchical structure for organized development

**Step 2:** Define development perspectives
- Tool: `mcp__ruv-swarm__agent_spawn`
- Parameters: `{"type": "architect", "name": "System Design"}`
- Result: Architectural thinking pattern for Claude Code

**Step 3:** Coordinate implementation
- Tool: `mcp__ruv-swarm__task_orchestrate`
- Parameters: `{"task": "Implement user authentication with JWT", "strategy": "parallel"}`
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

Remember: **ruv-swarm coordinates, Claude Code creates!** Start with `mcp__ruv-swarm__swarm_init` to enhance your development workflow.
