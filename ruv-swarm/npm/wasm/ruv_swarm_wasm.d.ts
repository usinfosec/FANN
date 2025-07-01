/* tslint:disable */
/* eslint-disable */
export function js_array_to_vec_f32(array: any): Float32Array;
export function vec_f32_to_js_array(vec: Float32Array): Float32Array;
export function get_wasm_memory_usage(): bigint;
export function console_log(message: string): void;
export function console_error(message: string): void;
export function console_warn(message: string): void;
export function format_js_error(error: any): string;
export function init(): void;
export function create_neural_network(layers: Uint32Array, activation: ActivationFunction): WasmNeuralNetwork;
export function create_swarm_orchestrator(topology: string): WasmSwarmOrchestrator;
export function create_forecasting_model(model_type: string): WasmForecastingModel;
export function get_version(): string;
export function get_features(): string;
export enum ActivationFunction {
  Linear = 0,
  Sigmoid = 1,
  SymmetricSigmoid = 2,
  Tanh = 3,
  ReLU = 4,
  LeakyReLU = 5,
  Swish = 6,
  Gaussian = 7,
  Elliot = 8,
  SymmetricElliot = 9,
  Sine = 10,
  Cosine = 11,
  SinSymmetric = 12,
  CosSymmetric = 13,
  ThresholdSymmetric = 14,
  Threshold = 15,
  StepSymmetric = 16,
  Step = 17,
}
export class PerformanceTimer {
  free(): void;
  constructor(name: string);
  elapsed(): number;
  log(): void;
}
export class RuntimeFeatures {
  free(): void;
  constructor();
  get_features_json(): string;
  readonly simd_available: boolean;
  readonly threads_available: boolean;
  readonly memory_limit: bigint;
}
export class WasmForecastingModel {
  free(): void;
  constructor(model_type: string);
  predict(input: Float64Array): Float64Array;
  get_model_type(): string;
}
export class WasmNeuralNetwork {
  free(): void;
  constructor(layers: Uint32Array, activation: ActivationFunction);
  randomize_weights(min: number, max: number): void;
  set_weights(weights: Float64Array): void;
  get_weights(): Float64Array;
  run(inputs: Float64Array): Float64Array;
}
export class WasmSwarmOrchestrator {
  free(): void;
  constructor(topology: string);
  add_agent(agent_id: string): void;
  get_agent_count(): number;
  get_topology(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_runtimefeatures_free: (a: number, b: number) => void;
  readonly runtimefeatures_new: () => number;
  readonly runtimefeatures_simd_available: (a: number) => number;
  readonly runtimefeatures_threads_available: (a: number) => number;
  readonly runtimefeatures_memory_limit: (a: number) => bigint;
  readonly runtimefeatures_get_features_json: (a: number, b: number) => void;
  readonly js_array_to_vec_f32: (a: number, b: number) => void;
  readonly vec_f32_to_js_array: (a: number, b: number) => number;
  readonly __wbg_performancetimer_free: (a: number, b: number) => void;
  readonly performancetimer_new: (a: number, b: number) => number;
  readonly performancetimer_elapsed: (a: number) => number;
  readonly performancetimer_log: (a: number) => void;
  readonly get_wasm_memory_usage: () => bigint;
  readonly console_log: (a: number, b: number) => void;
  readonly console_error: (a: number, b: number) => void;
  readonly console_warn: (a: number, b: number) => void;
  readonly format_js_error: (a: number, b: number) => void;
  readonly init: () => void;
  readonly __wbg_wasmneuralnetwork_free: (a: number, b: number) => void;
  readonly wasmneuralnetwork_new: (a: number, b: number, c: number) => number;
  readonly wasmneuralnetwork_randomize_weights: (a: number, b: number, c: number) => void;
  readonly wasmneuralnetwork_set_weights: (a: number, b: number, c: number) => void;
  readonly wasmneuralnetwork_get_weights: (a: number, b: number) => void;
  readonly wasmneuralnetwork_run: (a: number, b: number, c: number, d: number) => void;
  readonly __wbg_wasmswarmorchestrator_free: (a: number, b: number) => void;
  readonly wasmswarmorchestrator_add_agent: (a: number, b: number, c: number) => void;
  readonly wasmswarmorchestrator_get_agent_count: (a: number) => number;
  readonly wasmswarmorchestrator_get_topology: (a: number, b: number) => void;
  readonly __wbg_wasmforecastingmodel_free: (a: number, b: number) => void;
  readonly wasmforecastingmodel_predict: (a: number, b: number, c: number, d: number) => void;
  readonly wasmforecastingmodel_get_model_type: (a: number, b: number) => void;
  readonly create_neural_network: (a: number, b: number, c: number) => number;
  readonly create_swarm_orchestrator: (a: number, b: number) => number;
  readonly create_forecasting_model: (a: number, b: number) => number;
  readonly get_version: (a: number) => void;
  readonly get_features: (a: number) => void;
  readonly wasmswarmorchestrator_new: (a: number, b: number) => number;
  readonly wasmforecastingmodel_new: (a: number, b: number) => number;
  readonly __wbindgen_export_0: (a: number) => void;
  readonly __wbindgen_export_1: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export_2: (a: number, b: number) => number;
  readonly __wbindgen_export_3: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
