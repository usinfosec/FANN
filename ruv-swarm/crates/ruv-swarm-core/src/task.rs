//! Task definitions and task distribution logic

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec};

use core::fmt;

/// Unique identifier for a task
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    /// Create a new task ID
    pub fn new(id: impl Into<String>) -> Self {
        TaskId(id.into())
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is waiting to be assigned
    Pending,
    /// Task has been assigned to an agent
    Assigned,
    /// Task is currently being processed
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

/// Task definition
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub task_type: String,
    pub priority: TaskPriority,
    pub payload: TaskPayload,
    pub required_capabilities: Vec<String>,
    pub timeout_ms: Option<u64>,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl Task {
    /// Create a new task
    pub fn new(id: impl Into<String>, task_type: impl Into<String>) -> Self {
        Task {
            id: TaskId::new(id),
            task_type: task_type.into(),
            priority: TaskPriority::Normal,
            payload: TaskPayload::Empty,
            required_capabilities: Vec::new(),
            timeout_ms: None,
            retry_count: 0,
            max_retries: 3,
        }
    }
    
    /// Set task priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set task payload
    pub fn with_payload(mut self, payload: TaskPayload) -> Self {
        self.payload = payload;
        self
    }
    
    /// Add required capability
    pub fn require_capability(mut self, capability: impl Into<String>) -> Self {
        self.required_capabilities.push(capability.into());
        self
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// Check if task can be retried
    #[inline]
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
    
    /// Increment retry count
    #[inline]
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Task payload variants
#[derive(Debug, Clone)]
pub enum TaskPayload {
    /// Empty payload
    Empty,
    /// Text data
    Text(String),
    /// Binary data
    Binary(Vec<u8>),
    /// JSON data (as string)
    Json(String),
    /// Custom data
    Custom(Box<dyn CustomPayload>),
}

/// Trait for custom task payloads
pub trait CustomPayload: Send + Sync + fmt::Debug {
    /// Clone the payload
    fn clone_box(&self) -> Box<dyn CustomPayload>;
}

impl Clone for Box<dyn CustomPayload> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Task result
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub status: TaskStatus,
    pub output: Option<TaskOutput>,
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(output: impl Into<TaskOutput>) -> Self {
        TaskResult {
            task_id: TaskId::new(""),
            status: TaskStatus::Completed,
            output: Some(output.into()),
            error: None,
            execution_time_ms: 0,
        }
    }
    
    /// Create a failed task result
    pub fn failure(error: impl Into<String>) -> Self {
        TaskResult {
            task_id: TaskId::new(""),
            status: TaskStatus::Failed,
            output: None,
            error: Some(error.into()),
            execution_time_ms: 0,
        }
    }
    
    /// Set task ID
    pub fn with_task_id(mut self, task_id: TaskId) -> Self {
        self.task_id = task_id;
        self
    }
    
    /// Set execution time
    pub fn with_execution_time(mut self, time_ms: u64) -> Self {
        self.execution_time_ms = time_ms;
        self
    }
}

/// Task output variants
#[derive(Debug, Clone)]
pub enum TaskOutput {
    /// Text output
    Text(String),
    /// Binary output
    Binary(Vec<u8>),
    /// JSON output (as string)
    Json(String),
    /// No output
    None,
}

impl From<String> for TaskOutput {
    fn from(s: String) -> Self {
        TaskOutput::Text(s)
    }
}

impl From<&str> for TaskOutput {
    fn from(s: &str) -> Self {
        TaskOutput::Text(s.to_string())
    }
}

impl From<Vec<u8>> for TaskOutput {
    fn from(bytes: Vec<u8>) -> Self {
        TaskOutput::Binary(bytes)
    }
}

/// Task distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded agent first
    LeastLoaded,
    /// Random distribution
    Random,
    /// Priority-based distribution
    Priority,
    /// Capability-based distribution
    CapabilityBased,
}

impl Default for DistributionStrategy {
    fn default() -> Self {
        DistributionStrategy::LeastLoaded
    }
}