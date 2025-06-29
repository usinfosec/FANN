# ruv-swarm-transport

Transport layer for RUV-FANN swarm communication with multiple backend implementations.

## Features

- **Multiple Transport Backends**:
  - WebSocket with TLS support and auto-reconnection
  - Shared memory with lock-free ring buffers
  - In-process channels for local communication
  
- **Message Protocol**:
  - Request/Response pattern with correlation IDs
  - Event notifications
  - Broadcast messaging
  - Heartbeat monitoring
  - Protocol version negotiation
  
- **Performance Optimizations**:
  - Zero-copy message passing in shared memory
  - Message compression (gzip)
  - Efficient binary serialization (MessagePack)
  - Lock-free data structures
  
- **WASM Support**:
  - SharedArrayBuffer integration
  - Web Workers compatibility
  - Fallback for environments without shared memory

## Usage

### In-Process Transport

```rust
use ruv_swarm_transport::{Transport, TransportConfig, in_process::InProcessTransport};

// Create a pair of connected transports
let config = TransportConfig::default();
let (mut transport1, mut transport2) = InProcessTransport::create_pair(
    "agent1".to_string(),
    "agent2".to_string(),
    config,
).await?;

// Send a message
let msg = Message::event("agent1".to_string(), "test", serde_json::json!({"data": "hello"}));
transport1.send("agent2", msg).await?;

// Receive the message
let (from, received) = transport2.receive().await?;
```

### WebSocket Transport

```rust
use ruv_swarm_transport::{websocket::{WebSocketTransport, WsMode}};

// Client mode
let client = WebSocketTransport::new(
    WsMode::Client { url: "wss://example.com:8080".to_string() },
    TransportConfig::default()
).await?;

// Server mode
let server = WebSocketTransport::new(
    WsMode::Server { bind_addr: "0.0.0.0:8080".to_string() },
    TransportConfig::default()
).await?;
```

### Shared Memory Transport

```rust
use ruv_swarm_transport::{shared_memory::{SharedMemoryTransport, SharedMemoryInfo}};

let info = SharedMemoryInfo {
    name: "swarm_shmem".to_string(),
    size: 10 * 1024 * 1024, // 10MB
    ring_buffer_size: 1024 * 1024, // 1MB per buffer
};

let transport = SharedMemoryTransport::new(info, TransportConfig::default()).await?;
```

## Configuration

```rust
let config = TransportConfig {
    max_message_size: 10 * 1024 * 1024, // 10MB
    connection_timeout_ms: 5000,
    retry_attempts: 3,
    enable_compression: true,
    compression_threshold: 1024, // 1KB
};
```

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Benchmark results on typical hardware:
- In-process send: ~500ns per message
- Shared memory throughput: ~1M messages/sec
- WebSocket with compression: ~50μs per message
- Binary serialization: 10x faster than JSON

## Safety

- All transports are thread-safe and can be shared across tasks
- Shared memory uses lock-free algorithms for concurrent access
- WebSocket handles reconnection automatically with exponential backoff
- Message size limits prevent memory exhaustion

## License

MIT OR Apache-2.0