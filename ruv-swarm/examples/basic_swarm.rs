//! Basic Swarm Example
//! 
//! This example demonstrates how to create a simple swarm, spawn agents,
//! and orchestrate a basic task.

use ruv_swarm::{
    agent::{AgentId, AgentType},
    swarm::{Swarm, SwarmConfig},
    topology::Topology,
    Result,
};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Basic Swarm Example ===\n");

    // Create swarm configuration
    let config = SwarmConfig {
        max_agents: 10,
        topology: Topology::FullyConnected,
        heartbeat_interval: Duration::from_secs(5),
        message_timeout: Duration::from_secs(30),
        ..Default::default()
    };

    // Initialize swarm
    println!("Initializing swarm...");
    let mut swarm = Swarm::new(config)?;

    // Spawn worker agents
    println!("Spawning agents...");
    let agent1 = swarm.spawn_agent(AgentType::Worker).await?;
    println!("  Agent 1 spawned: {:?}", agent1);

    let agent2 = swarm.spawn_agent(AgentType::Worker).await?;
    println!("  Agent 2 spawned: {:?}", agent2);

    let agent3 = swarm.spawn_agent(AgentType::Worker).await?;
    println!("  Agent 3 spawned: {:?}", agent3);

    // Wait for agents to be ready
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Orchestrate a simple task
    println!("\nOrchestrating task: 'Process data batch'");
    let task = swarm.create_task("Process data batch", vec![
        "Load dataset",
        "Preprocess entries",
        "Apply transformations",
        "Aggregate results",
    ])?;

    // Execute task with timeout
    let result = timeout(
        Duration::from_secs(10),
        swarm.orchestrate(task)
    ).await;

    match result {
        Ok(Ok(task_result)) => {
            println!("\nTask completed successfully!");
            println!("Result: {:?}", task_result);
        }
        Ok(Err(e)) => {
            eprintln!("\nTask failed: {}", e);
        }
        Err(_) => {
            eprintln!("\nTask timed out!");
        }
    }

    // Get swarm statistics
    let stats = swarm.get_statistics();
    println!("\nSwarm Statistics:");
    println!("  Active agents: {}", stats.active_agents);
    println!("  Tasks completed: {}", stats.tasks_completed);
    println!("  Messages sent: {}", stats.messages_sent);
    println!("  Average response time: {:?}", stats.avg_response_time);

    // Graceful shutdown
    println!("\nShutting down swarm...");
    swarm.shutdown().await?;

    println!("Example completed!");
    Ok(())
}