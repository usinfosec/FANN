/**
 * DAA Service Layer - Manages JS-WASM Communication
 * Provides comprehensive agent lifecycle management, cross-agent state persistence,
 * and multi-agent workflow coordination with < 1ms cross-boundary call latency
 */

import { WasmModuleLoader } from './wasm-loader.js';
import { performance } from 'perf_hooks';
import EventEmitter from 'events';

// Performance monitoring utilities
class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.thresholds = {
      crossBoundaryCall: 1.0, // 1ms threshold
      agentSpawn: 10.0,
      stateSync: 5.0,
      workflowStep: 20.0
    };
  }

  startTimer(operation) {
    const id = `${operation}-${Date.now()}-${Math.random()}`;
    this.metrics.set(id, {
      operation,
      start: performance.now(),
      id
    });
    return id;
  }

  endTimer(id) {
    const metric = this.metrics.get(id);
    if (!metric) return null;

    const duration = performance.now() - metric.start;
    this.metrics.delete(id);

    const threshold = this.thresholds[metric.operation];
    if (threshold && duration > threshold) {
      console.warn(`⚠️ Performance warning: ${metric.operation} took ${duration.toFixed(2)}ms (threshold: ${threshold}ms)`);
    }

    return {
      operation: metric.operation,
      duration,
      withinThreshold: !threshold || duration <= threshold
    };
  }

  getAverageLatency(operation) {
    const relevantMetrics = Array.from(this.metrics.values())
      .filter(m => m.operation === operation);
    
    if (relevantMetrics.length === 0) return 0;
    
    const totalDuration = relevantMetrics.reduce((sum, m) => {
      const duration = performance.now() - m.start;
      return sum + duration;
    }, 0);

    return totalDuration / relevantMetrics.length;
  }
}

// Agent state management with persistence
class AgentStateManager {
  constructor() {
    this.states = new Map();
    this.stateHistory = new Map();
    this.persistenceEnabled = true;
    this.maxHistorySize = 100;
  }

  saveState(agentId, state) {
    const timestamp = Date.now();
    const stateEntry = {
      ...state,
      timestamp,
      version: (this.states.get(agentId)?.version || 0) + 1
    };

    this.states.set(agentId, stateEntry);

    // Maintain history
    if (!this.stateHistory.has(agentId)) {
      this.stateHistory.set(agentId, []);
    }
    
    const history = this.stateHistory.get(agentId);
    history.push(stateEntry);
    
    // Trim history if needed
    if (history.length > this.maxHistorySize) {
      history.shift();
    }

    // Persist to storage if enabled
    if (this.persistenceEnabled) {
      this.persistToStorage(agentId, stateEntry);
    }

    return stateEntry;
  }

  getState(agentId) {
    return this.states.get(agentId);
  }

  getStateHistory(agentId, limit = 10) {
    const history = this.stateHistory.get(agentId) || [];
    return history.slice(-limit);
  }

  async persistToStorage(agentId, state) {
    // In a real implementation, this would persist to IndexedDB or file system
    // For now, we'll use a simple in-memory simulation
    if (typeof localStorage !== 'undefined') {
      try {
        const key = `daa-agent-state-${agentId}`;
        localStorage.setItem(key, JSON.stringify(state));
      } catch (e) {
        console.warn('Failed to persist agent state:', e);
      }
    }
  }

  async loadFromStorage(agentId) {
    if (typeof localStorage !== 'undefined') {
      try {
        const key = `daa-agent-state-${agentId}`;
        const stored = localStorage.getItem(key);
        if (stored) {
          return JSON.parse(stored);
        }
      } catch (e) {
        console.warn('Failed to load agent state:', e);
      }
    }
    return null;
  }

  clearState(agentId) {
    this.states.delete(agentId);
    this.stateHistory.delete(agentId);
    
    if (typeof localStorage !== 'undefined') {
      localStorage.removeItem(`daa-agent-state-${agentId}`);
    }
  }
}

// Workflow coordination manager
class WorkflowCoordinator {
  constructor() {
    this.workflows = new Map();
    this.activeSteps = new Map();
    this.completedSteps = new Map();
    this.dependencies = new Map();
  }

  createWorkflow(workflowId, steps, dependencies = {}) {
    const workflow = {
      id: workflowId,
      steps: new Map(steps.map(s => [s.id, s])),
      dependencies,
      status: 'pending',
      createdAt: Date.now(),
      completedSteps: new Set(),
      activeSteps: new Set(),
      pendingSteps: new Set(steps.map(s => s.id))
    };

    this.workflows.set(workflowId, workflow);
    return workflow;
  }

  async executeStep(workflowId, stepId, agents) {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) throw new Error(`Workflow ${workflowId} not found`);

    const step = workflow.steps.get(stepId);
    if (!step) throw new Error(`Step ${stepId} not found in workflow ${workflowId}`);

    // Check dependencies
    const deps = workflow.dependencies[stepId] || [];
    for (const dep of deps) {
      if (!workflow.completedSteps.has(dep)) {
        throw new Error(`Dependency ${dep} not completed for step ${stepId}`);
      }
    }

    // Mark as active
    workflow.pendingSteps.delete(stepId);
    workflow.activeSteps.add(stepId);
    workflow.status = 'running';

    try {
      // Execute step with assigned agents
      const result = await this.runStepWithAgents(step, agents);
      
      // Mark as completed
      workflow.activeSteps.delete(stepId);
      workflow.completedSteps.add(stepId);
      
      // Check if workflow is complete
      if (workflow.pendingSteps.size === 0 && workflow.activeSteps.size === 0) {
        workflow.status = 'completed';
      }

      return result;
    } catch (error) {
      workflow.status = 'failed';
      throw error;
    }
  }

  async runStepWithAgents(step, agents) {
    const results = [];
    
    // Parallel execution for independent agent tasks
    const promises = agents.map(async (agent) => {
      if (step.agentFilter && !step.agentFilter(agent)) {
        return null;
      }

      const task = step.task || step.action;
      if (typeof task === 'function') {
        return await task(agent);
      } else {
        // Direct WASM call
        return await agent[task.method](...(task.args || []));
      }
    });

    const agentResults = await Promise.all(promises);
    return agentResults.filter(r => r !== null);
  }

  getWorkflowStatus(workflowId) {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) return null;

    return {
      id: workflow.id,
      status: workflow.status,
      progress: {
        total: workflow.steps.size,
        completed: workflow.completedSteps.size,
        active: workflow.activeSteps.size,
        pending: workflow.pendingSteps.size
      },
      completedSteps: Array.from(workflow.completedSteps),
      activeSteps: Array.from(workflow.activeSteps),
      pendingSteps: Array.from(workflow.pendingSteps)
    };
  }
}

// Main DAA Service Layer
export class DAAService extends EventEmitter {
  constructor() {
    super();
    this.wasmLoader = new WasmModuleLoader();
    this.agents = new Map();
    this.agentStates = new AgentStateManager();
    this.workflows = new WorkflowCoordinator();
    this.performance = new PerformanceMonitor();
    this.initialized = false;
    this.wasmModule = null;
    this.coordinatorModule = null;
    this.resourceManagerModule = null;
  }

  async initialize() {
    if (this.initialized) return;

    const timerId = this.performance.startTimer('initialization');

    try {
      // Initialize WASM loader with progressive strategy
      await this.wasmLoader.initialize('progressive');
      
      // Load core module
      const coreModule = await this.wasmLoader.loadModule('core');
      this.wasmModule = coreModule.exports;

      // Initialize WASM utilities
      if (this.wasmModule.WasmUtils) {
        this.wasmModule.WasmUtils.init();
      }

      // Create coordinator and resource manager
      if (this.wasmModule.WasmCoordinator) {
        this.coordinatorModule = new this.wasmModule.WasmCoordinator();
      }

      if (this.wasmModule.WasmResourceManager) {
        this.resourceManagerModule = new this.wasmModule.WasmResourceManager(1024); // 1GB limit
      }

      this.initialized = true;
      this.emit('initialized');

      const timing = this.performance.endTimer(timerId);
      console.log(`✅ DAA Service initialized in ${timing.duration.toFixed(2)}ms`);

    } catch (error) {
      console.error('Failed to initialize DAA Service:', error);
      throw error;
    }
  }

  // Agent Lifecycle Management
  async createAgent(id, capabilities = []) {
    if (!this.initialized) await this.initialize();

    const timerId = this.performance.startTimer('agentSpawn');

    try {
      // Create WASM agent
      const wasmAgent = new this.wasmModule.WasmAutonomousAgent(id);
      
      // Add capabilities
      for (const capability of capabilities) {
        wasmAgent.add_capability(capability);
      }

      // Create agent wrapper with enhanced functionality
      const agent = {
        id,
        wasmAgent,
        capabilities: new Set(capabilities),
        status: 'active',
        createdAt: Date.now(),
        lastActivity: Date.now(),
        metrics: {
          decisionsMade: 0,
          tasksCompleted: 0,
          errors: 0,
          averageResponseTime: 0
        }
      };

      // Store agent
      this.agents.set(id, agent);

      // Add to coordinator
      if (this.coordinatorModule) {
        this.coordinatorModule.add_agent(wasmAgent);
      }

      // Load persisted state if available
      const persistedState = await this.agentStates.loadFromStorage(id);
      if (persistedState) {
        agent.state = persistedState;
        console.log(`📂 Restored persisted state for agent ${id}`);
      }

      // Save initial state
      this.agentStates.saveState(id, {
        status: agent.status,
        capabilities: Array.from(agent.capabilities),
        metrics: agent.metrics
      });

      this.emit('agentCreated', { agentId: id, capabilities });

      const timing = this.performance.endTimer(timerId);
      console.log(`🤖 Created agent ${id} in ${timing.duration.toFixed(2)}ms`);

      return agent;

    } catch (error) {
      console.error(`Failed to create agent ${id}:`, error);
      throw error;
    }
  }

  async destroyAgent(id) {
    const agent = this.agents.get(id);
    if (!agent) return false;

    try {
      // Remove from coordinator
      if (this.coordinatorModule) {
        this.coordinatorModule.remove_agent(id);
      }

      // Clear state
      this.agentStates.clearState(id);

      // Remove from active agents
      this.agents.delete(id);

      this.emit('agentDestroyed', { agentId: id });
      console.log(`🗑️ Destroyed agent ${id}`);

      return true;

    } catch (error) {
      console.error(`Failed to destroy agent ${id}:`, error);
      return false;
    }
  }

  // Cross-boundary communication with < 1ms latency
  async makeDecision(agentId, context) {
    const agent = this.agents.get(agentId);
    if (!agent) throw new Error(`Agent ${agentId} not found`);

    const timerId = this.performance.startTimer('crossBoundaryCall');

    try {
      // Prepare context for WASM
      const contextJson = JSON.stringify(context);
      
      // Make decision through WASM
      const decisionPromise = agent.wasmAgent.make_decision(contextJson);
      const decision = await decisionPromise;

      // Update metrics
      agent.lastActivity = Date.now();
      agent.metrics.decisionsMade++;

      // Update state
      this.agentStates.saveState(agentId, {
        lastDecision: decision,
        lastContext: context,
        timestamp: Date.now()
      });

      const timing = this.performance.endTimer(timerId);
      
      // Update average response time
      const prevAvg = agent.metrics.averageResponseTime;
      agent.metrics.averageResponseTime = 
        (prevAvg * (agent.metrics.decisionsMade - 1) + timing.duration) / agent.metrics.decisionsMade;

      this.emit('decisionMade', { 
        agentId, 
        decision, 
        latency: timing.duration,
        withinThreshold: timing.withinThreshold 
      });

      return decision;

    } catch (error) {
      agent.metrics.errors++;
      console.error(`Decision making failed for agent ${agentId}:`, error);
      throw error;
    }
  }

  // Multi-agent workflow coordination
  async createWorkflow(workflowId, steps, dependencies) {
    const workflow = this.workflows.createWorkflow(workflowId, steps, dependencies);
    
    this.emit('workflowCreated', { 
      workflowId, 
      steps: steps.map(s => s.id),
      dependencies 
    });

    return workflow;
  }

  async executeWorkflowStep(workflowId, stepId, agentIds) {
    const timerId = this.performance.startTimer('workflowStep');

    try {
      // Get agents for execution
      const agents = agentIds.map(id => {
        const agent = this.agents.get(id);
        if (!agent) throw new Error(`Agent ${id} not found`);
        return agent.wasmAgent;
      });

      // Execute step
      const result = await this.workflows.executeStep(workflowId, stepId, agents);

      const timing = this.performance.endTimer(timerId);
      
      this.emit('workflowStepCompleted', {
        workflowId,
        stepId,
        agentIds,
        duration: timing.duration,
        result
      });

      return result;

    } catch (error) {
      console.error(`Workflow step execution failed:`, error);
      throw error;
    }
  }

  // State synchronization across agents
  async synchronizeStates(agentIds) {
    const timerId = this.performance.startTimer('stateSync');

    try {
      // Collect all agent states
      const states = new Map();
      for (const id of agentIds) {
        const state = this.agentStates.getState(id);
        if (state) {
          states.set(id, state);
        }
      }

      // Coordinate through WASM
      if (this.coordinatorModule) {
        await this.coordinatorModule.coordinate();
      }

      const timing = this.performance.endTimer(timerId);

      this.emit('statesSynchronized', {
        agentIds,
        duration: timing.duration
      });

      return states;

    } catch (error) {
      console.error('State synchronization failed:', error);
      throw error;
    }
  }

  // Resource optimization
  async optimizeResources() {
    if (!this.resourceManagerModule) {
      console.warn('Resource manager not available');
      return null;
    }

    try {
      const result = await this.resourceManagerModule.optimize();
      
      this.emit('resourcesOptimized', { result });
      
      return result;

    } catch (error) {
      console.error('Resource optimization failed:', error);
      throw error;
    }
  }

  // Performance monitoring
  getPerformanceMetrics() {
    const metrics = {
      agents: {},
      workflows: {},
      system: {
        totalAgents: this.agents.size,
        activeWorkflows: this.workflows.workflows.size,
        averageLatencies: {
          crossBoundaryCall: this.performance.getAverageLatency('crossBoundaryCall'),
          agentSpawn: this.performance.getAverageLatency('agentSpawn'),
          stateSync: this.performance.getAverageLatency('stateSync'),
          workflowStep: this.performance.getAverageLatency('workflowStep')
        }
      }
    };

    // Collect per-agent metrics
    for (const [id, agent] of this.agents) {
      metrics.agents[id] = {
        ...agent.metrics,
        uptime: Date.now() - agent.createdAt,
        status: agent.status
      };
    }

    // Collect workflow metrics
    for (const [id, workflow] of this.workflows.workflows) {
      metrics.workflows[id] = this.workflows.getWorkflowStatus(id);
    }

    return metrics;
  }

  // Batch operations for efficiency
  async batchCreateAgents(configs) {
    const results = [];
    
    for (const config of configs) {
      try {
        const agent = await this.createAgent(config.id, config.capabilities || []);
        results.push({ success: true, agent });
      } catch (error) {
        results.push({ success: false, error: error.message, config });
      }
    }

    return results;
  }

  async batchMakeDecisions(decisions) {
    const promises = decisions.map(async ({ agentId, context }) => {
      try {
        const decision = await this.makeDecision(agentId, context);
        return { success: true, agentId, decision };
      } catch (error) {
        return { success: false, agentId, error: error.message };
      }
    });

    return await Promise.all(promises);
  }

  // Cleanup and resource management
  async cleanup() {
    try {
      // Destroy all agents
      for (const id of this.agents.keys()) {
        await this.destroyAgent(id);
      }

      // Clear caches
      this.wasmLoader.clearCache();

      // Optimize memory
      const optimization = this.wasmLoader.optimizeMemory();
      
      console.log('🧹 DAA Service cleanup completed', optimization);
      
      this.emit('cleanup', optimization);

    } catch (error) {
      console.error('Cleanup failed:', error);
    }
  }

  // Get service status
  getStatus() {
    return {
      initialized: this.initialized,
      agents: {
        count: this.agents.size,
        ids: Array.from(this.agents.keys()),
        states: this.agentStates.states.size
      },
      workflows: {
        count: this.workflows.workflows.size,
        active: Array.from(this.workflows.workflows.values())
          .filter(w => w.status === 'running').length
      },
      wasm: {
        modules: this.wasmLoader.getModuleStatus(),
        memoryUsage: this.wasmLoader.getTotalMemoryUsage()
      },
      performance: this.getPerformanceMetrics()
    };
  }
}

// Export singleton instance
export const daaService = new DAAService();

// Default export
export default DAAService;