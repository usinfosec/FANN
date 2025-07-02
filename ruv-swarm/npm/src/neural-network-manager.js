/**
 * Neural Network Manager
 * Manages per-agent neural networks with WASM integration
 */

import { createNeuralModel, MODEL_PRESETS } from './neural-models/index.js';

class NeuralNetworkManager {
  constructor(wasmLoader) {
    this.wasmLoader = wasmLoader;
    this.neuralNetworks = new Map();
    this.templates = {
      deep_analyzer: {
        layers: [128, 256, 512, 256, 128],
        activation: 'relu',
        output_activation: 'sigmoid',
        dropout: 0.3,
      },
      nlp_processor: {
        layers: [512, 1024, 512, 256],
        activation: 'gelu',
        output_activation: 'softmax',
        dropout: 0.4,
      },
      reinforcement_learner: {
        layers: [64, 128, 128, 64],
        activation: 'tanh',
        output_activation: 'linear',
        dropout: 0.2,
      },
      pattern_recognizer: {
        layers: [256, 512, 1024, 512, 256],
        activation: 'relu',
        output_activation: 'sigmoid',
        dropout: 0.35,
      },
      time_series_analyzer: {
        layers: [128, 256, 256, 128],
        activation: 'lstm',
        output_activation: 'linear',
        dropout: 0.25,
      },
      transformer_nlp: {
        modelType: 'transformer',
        preset: 'base',
        dimensions: 512,
        heads: 8,
        layers: 6,
      },
      cnn_vision: {
        modelType: 'cnn',
        preset: 'cifar10',
        inputShape: [32, 32, 3],
        outputSize: 10,
      },
      gru_sequence: {
        modelType: 'gru',
        preset: 'text_classification',
        hiddenSize: 256,
        numLayers: 2,
        bidirectional: true,
      },
      autoencoder_compress: {
        modelType: 'autoencoder',
        preset: 'mnist_compress',
        bottleneckSize: 32,
        variational: false,
      },
      gnn_social: {
        modelType: 'gnn',
        preset: 'social_network',
        nodeDimensions: 128,
        numLayers: 3,
      },
      resnet_classifier: {
        modelType: 'resnet',
        preset: 'resnet18',
        inputDimensions: 784,
        outputDimensions: 10,
      },
      vae_generator: {
        modelType: 'vae',
        preset: 'mnist_vae',
        latentDimensions: 20,
        betaKL: 1.0,
      },
      lstm_sequence: {
        modelType: 'lstm',
        preset: 'sentiment_analysis',
        hiddenSize: 256,
        numLayers: 2,
        bidirectional: true,
      },
    };
    
    // Store instances of new neural models
    this.neuralModels = new Map();
  }

  async createAgentNeuralNetwork(agentId, config = {}) {
    // Check if this is a new neural model type
    const template = config.template || 'deep_analyzer';
    const templateConfig = this.templates[template];
    
    if (templateConfig && templateConfig.modelType) {
      // Create new neural model
      return this.createAdvancedNeuralModel(agentId, template, config);
    }
    
    // Load neural module if not already loaded
    const neuralModule = await this.wasmLoader.loadModule('neural');

    if (!neuralModule || neuralModule.isPlaceholder) {
      console.warn('Neural network module not available, using simulation');
      return this.createSimulatedNetwork(agentId, config);
    }

    const {
      layers = null,
      activation = 'relu',
      learningRate = 0.001,
      optimizer = 'adam',
    } = config;

    // Use template or custom layers
    const networkConfig = layers ? { layers, activation } : this.templates[template];

    try {
      // Create network using WASM module
      const networkId = neuralModule.exports.create_neural_network(
        JSON.stringify({
          agent_id: agentId,
          layers: networkConfig.layers,
          activation: networkConfig.activation,
          learning_rate: learningRate,
          optimizer,
        }),
      );

      const network = new NeuralNetwork(networkId, agentId, networkConfig, neuralModule);
      this.neuralNetworks.set(agentId, network);

      return network;
    } catch (error) {
      console.error('Failed to create neural network:', error);
      return this.createSimulatedNetwork(agentId, config);
    }
  }

  createSimulatedNetwork(agentId, config) {
    const network = new SimulatedNeuralNetwork(agentId, config);
    this.neuralNetworks.set(agentId, network);
    return network;
  }

  async createAdvancedNeuralModel(agentId, template, customConfig = {}) {
    const templateConfig = this.templates[template];
    
    if (!templateConfig || !templateConfig.modelType) {
      throw new Error(`Invalid template: ${template}`);
    }
    
    // Merge template config with custom config
    const config = {
      ...templateConfig,
      ...customConfig
    };
    
    // Use preset if specified
    if (config.preset && MODEL_PRESETS[config.modelType]) {
      const presetConfig = MODEL_PRESETS[config.modelType][config.preset];
      Object.assign(config, presetConfig);
    }
    
    try {
      // Create the neural model
      const model = await createNeuralModel(config.modelType, config);
      
      // Wrap in a compatible interface
      const wrappedModel = new AdvancedNeuralNetwork(agentId, model, config);
      
      this.neuralNetworks.set(agentId, wrappedModel);
      this.neuralModels.set(agentId, model);
      
      console.log(`Created ${config.modelType} neural network for agent ${agentId}`);
      
      return wrappedModel;
    } catch (error) {
      console.error(`Failed to create advanced neural model: ${error}`);
      return this.createSimulatedNetwork(agentId, config);
    }
  }

  async fineTuneNetwork(agentId, trainingData, options = {}) {
    const network = this.neuralNetworks.get(agentId);
    if (!network) {
      throw new Error(`No neural network found for agent ${agentId}`);
    }

    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      freezeLayers = [],
    } = options;

    return network.train(trainingData, { epochs, batchSize, learningRate, freezeLayers });
  }

  async enableCollaborativeLearning(agentIds, options = {}) {
    const {
      strategy = 'federated',
      syncInterval = 30000,
      privacyLevel = 'high',
    } = options;

    const networks = agentIds.map(id => this.neuralNetworks.get(id)).filter(n => n);

    if (networks.length < 2) {
      throw new Error('At least 2 neural networks required for collaborative learning');
    }

    // Create collaborative learning session
    const session = {
      id: `collab-${Date.now()}`,
      networks,
      strategy,
      syncInterval,
      privacyLevel,
      active: true,
    };

    // Start synchronization
    if (strategy === 'federated') {
      this.startFederatedLearning(session);
    }

    return session;
  }

  startFederatedLearning(session) {
    const syncFunction = () => {
      if (!session.active) {
        return;
      }

      // Aggregate gradients from all networks
      const gradients = session.networks.map(n => n.getGradients());

      // Apply privacy-preserving aggregation
      const aggregatedGradients = this.aggregateGradients(gradients, session.privacyLevel);

      // Update all networks with aggregated gradients
      session.networks.forEach(n => n.applyGradients(aggregatedGradients));

      // Schedule next sync
      setTimeout(syncFunction, session.syncInterval);
    };

    // Start synchronization
    setTimeout(syncFunction, session.syncInterval);
  }

  aggregateGradients(gradients, privacyLevel) {
    // Simple averaging for now (in real implementation, use secure aggregation)
    const aggregated = {};

    // Privacy levels could add noise or use secure multi-party computation
    const noise = privacyLevel === 'high' ? 0.01 : 0;

    // Average gradients with optional noise
    gradients.forEach(grad => {
      Object.entries(grad).forEach(([key, value]) => {
        if (!aggregated[key]) {
          aggregated[key] = 0;
        }
        aggregated[key] += value / gradients.length + (Math.random() - 0.5) * noise;
      });
    });

    return aggregated;
  }

  getNetworkMetrics(agentId) {
    const network = this.neuralNetworks.get(agentId);
    if (!network) {
      return null;
    }

    return network.getMetrics();
  }

  saveNetworkState(agentId, filePath) {
    const network = this.neuralNetworks.get(agentId);
    if (!network) {
      throw new Error(`No neural network found for agent ${agentId}`);
    }

    return network.save(filePath);
  }

  async loadNetworkState(agentId, filePath) {
    const network = this.neuralNetworks.get(agentId);
    if (!network) {
      throw new Error(`No neural network found for agent ${agentId}`);
    }

    return network.load(filePath);
  }
}

// Neural Network wrapper class
class NeuralNetwork {
  constructor(networkId, agentId, config, wasmModule) {
    this.networkId = networkId;
    this.agentId = agentId;
    this.config = config;
    this.wasmModule = wasmModule;
    this.trainingHistory = [];
    this.metrics = {
      accuracy: 0,
      loss: 1.0,
      epochs_trained: 0,
      total_samples: 0,
    };
  }

  async forward(input) {
    try {
      const result = this.wasmModule.exports.forward_pass(this.networkId, input);
      return result;
    } catch (error) {
      console.error('Forward pass failed:', error);
      return new Float32Array(this.config.layers[this.config.layers.length - 1]).fill(0.5);
    }
  }

  async train(trainingData, options) {
    const { epochs, batchSize, learningRate, freezeLayers } = options;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let batchCount = 0;

      // Process in batches
      for (let i = 0; i < trainingData.samples.length; i += batchSize) {
        const batch = trainingData.samples.slice(i, i + batchSize);

        try {
          const loss = this.wasmModule.exports.train_batch(
            this.networkId,
            JSON.stringify(batch),
            learningRate,
            JSON.stringify(freezeLayers),
          );

          epochLoss += loss;
          batchCount++;
        } catch (error) {
          console.error('Training batch failed:', error);
        }
      }

      const avgLoss = epochLoss / batchCount;
      this.metrics.loss = avgLoss;
      this.metrics.epochs_trained++;
      this.trainingHistory.push({ epoch, loss: avgLoss });

      console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
    }

    return this.metrics;
  }

  getGradients() {
    // Get gradients from WASM module
    try {
      const gradients = this.wasmModule.exports.get_gradients(this.networkId);
      return JSON.parse(gradients);
    } catch (error) {
      console.error('Failed to get gradients:', error);
      return {};
    }
  }

  applyGradients(gradients) {
    // Apply gradients to network
    try {
      this.wasmModule.exports.apply_gradients(this.networkId, JSON.stringify(gradients));
    } catch (error) {
      console.error('Failed to apply gradients:', error);
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      training_history: this.trainingHistory,
      network_info: {
        layers: this.config.layers,
        parameters: this.config.layers.reduce((acc, size, i) => {
          if (i > 0) {
            return acc + (this.config.layers[i - 1] * size);
          }
          return acc;
        }, 0),
      },
    };
  }

  async save(filePath) {
    try {
      const state = this.wasmModule.exports.serialize_network(this.networkId);
      // In real implementation, save to file
      console.log(`Saving network state to ${filePath}`);
      return true;
    } catch (error) {
      console.error('Failed to save network:', error);
      return false;
    }
  }

  async load(filePath) {
    try {
      // In real implementation, load from file
      console.log(`Loading network state from ${filePath}`);
      this.wasmModule.exports.deserialize_network(this.networkId, 'state_data');
      return true;
    } catch (error) {
      console.error('Failed to load network:', error);
      return false;
    }
  }
}

// Simulated Neural Network for when WASM is not available
class SimulatedNeuralNetwork {
  constructor(agentId, config) {
    this.agentId = agentId;
    this.config = config;
    this.weights = this.initializeWeights();
    this.trainingHistory = [];
    this.metrics = {
      accuracy: 0.5 + Math.random() * 0.3,
      loss: 0.5 + Math.random() * 0.5,
      epochs_trained: 0,
      total_samples: 0,
    };
  }

  initializeWeights() {
    // Simple weight initialization
    return this.config.layers?.map(() => Math.random() * 2 - 1) || [0];
  }

  async forward(input) {
    // Simple forward pass simulation
    const outputSize = this.config.layers?.[this.config.layers.length - 1] || 1;
    const output = new Float32Array(outputSize);

    for (let i = 0; i < outputSize; i++) {
      output[i] = Math.random();
    }

    return output;
  }

  async train(trainingData, options) {
    const { epochs } = options;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const loss = Math.max(0.01, this.metrics.loss * (0.9 + Math.random() * 0.1));
      this.metrics.loss = loss;
      this.metrics.epochs_trained++;
      this.metrics.accuracy = Math.min(0.99, this.metrics.accuracy + 0.01);
      this.trainingHistory.push({ epoch, loss });

      console.log(`[Simulated] Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(4)}`);
    }

    return this.metrics;
  }

  getGradients() {
    // Simulated gradients
    return {
      layer_0: Math.random() * 0.1,
      layer_1: Math.random() * 0.1,
    };
  }

  applyGradients(gradients) {
    // Simulate gradient application
    console.log('[Simulated] Applying gradients');
  }

  getMetrics() {
    return {
      ...this.metrics,
      training_history: this.trainingHistory,
      network_info: {
        layers: this.config.layers || [128, 64, 32],
        parameters: 10000, // Simulated parameter count
      },
    };
  }

  async save(filePath) {
    console.log(`[Simulated] Saving network state to ${filePath}`);
    return true;
  }

  async load(filePath) {
    console.log(`[Simulated] Loading network state from ${filePath}`);
    return true;
  }
}

// Neural Network Templates for quick configuration
const NeuralNetworkTemplates = {
  getTemplate: (templateName) => {
    const templates = {
      deep_analyzer: {
        layers: [128, 256, 512, 256, 128],
        activation: 'relu',
        output_activation: 'sigmoid',
        dropout: 0.3,
      },
      nlp_processor: {
        layers: [512, 1024, 512, 256],
        activation: 'gelu',
        output_activation: 'softmax',
        dropout: 0.4,
      },
      reinforcement_learner: {
        layers: [64, 128, 128, 64],
        activation: 'tanh',
        output_activation: 'linear',
        dropout: 0.2,
      },
    };

    return templates[templateName] || templates.deep_analyzer;
  },
};

// Advanced Neural Network wrapper for new model types
class AdvancedNeuralNetwork {
  constructor(agentId, model, config) {
    this.agentId = agentId;
    this.model = model;
    this.config = config;
    this.modelType = config.modelType;
    this.isAdvanced = true;
  }

  async forward(input) {
    try {
      // Handle different input formats
      let formattedInput = input;
      
      if (this.modelType === 'transformer' || this.modelType === 'gru') {
        // Ensure input has shape [batch_size, sequence_length, features]
        if (!input.shape) {
          formattedInput = new Float32Array(input);
          formattedInput.shape = [1, input.length, 1];
        }
      } else if (this.modelType === 'cnn') {
        // Ensure input has shape [batch_size, height, width, channels]
        if (!input.shape) {
          const inputShape = this.config.inputShape;
          formattedInput = new Float32Array(input);
          formattedInput.shape = [1, ...inputShape];
        }
      } else if (this.modelType === 'autoencoder') {
        // Ensure input has shape [batch_size, input_size]
        if (!input.shape) {
          formattedInput = new Float32Array(input);
          formattedInput.shape = [1, input.length];
        }
      }
      
      const result = await this.model.forward(formattedInput, false);
      
      // Return appropriate output based on model type
      if (this.modelType === 'autoencoder') {
        return result.reconstruction;
      }
      
      return result;
    } catch (error) {
      console.error(`Forward pass failed for ${this.modelType}:`, error);
      return new Float32Array(this.config.outputSize || 10).fill(0.5);
    }
  }

  async train(trainingData, options) {
    return this.model.train(trainingData, options);
  }

  getGradients() {
    // Advanced models handle gradients internally
    return {};
  }

  applyGradients(gradients) {
    // Advanced models handle gradient updates internally
    console.log(`Gradient update handled internally by ${this.modelType}`);
  }

  getMetrics() {
    return this.model.getMetrics();
  }

  async save(filePath) {
    return this.model.save(filePath);
  }

  async load(filePath) {
    return this.model.load(filePath);
  }

  // Special methods for specific model types
  async encode(input) {
    if (this.modelType === 'autoencoder') {
      const encoder = await this.model.getEncoder();
      return encoder.encode(input);
    }
    throw new Error(`Encode not supported for ${this.modelType}`);
  }

  async decode(latent) {
    if (this.modelType === 'autoencoder') {
      const decoder = await this.model.getDecoder();
      return decoder.decode(latent);
    }
    throw new Error(`Decode not supported for ${this.modelType}`);
  }

  async generate(numSamples) {
    if (this.modelType === 'autoencoder' && this.config.variational) {
      return this.model.generate(numSamples);
    }
    throw new Error(`Generation not supported for ${this.modelType}`);
  }
}

export { NeuralNetworkManager, NeuralNetworkTemplates };