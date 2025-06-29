use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ruv_swarm_core::{
    task::{Task, Priority},
    agent::CognitivePattern,
};

#[cfg(feature = "std")]
use ruv_swarm_core::{
    swarm::{Swarm, SwarmConfig, OrchestrationStrategy, Topology},
};

fn task_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_creation");
    
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    let _task = Task::new(
                        format!("task_{}", i),
                        Priority::Medium,
                    );
                }
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "std")]
fn swarm_agent_registration(c: &mut Criterion) {
    use tokio::runtime::Runtime;
    let rt = Runtime::new().unwrap();
    
    c.bench_function("agent_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = SwarmConfig::default();
                let swarm = Swarm::new(config);
                
                for i in 0..100 {
                    let _ = swarm.register_agent(
                        format!("agent_{}", i),
                        vec!["compute".to_string()],
                        CognitivePattern::Convergent,
                    ).await;
                }
            });
        });
    });
}

#[cfg(feature = "std")]
fn task_distribution_strategies(c: &mut Criterion) {
    use tokio::runtime::Runtime;
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("task_distribution");
    
    let strategies = [
        ("round_robin", OrchestrationStrategy::RoundRobin),
        ("least_loaded", OrchestrationStrategy::LeastLoaded),
        ("capability_based", OrchestrationStrategy::CapabilityBased),
        ("priority_based", OrchestrationStrategy::PriorityBased),
    ];
    
    for (name, strategy) in strategies.iter() {
        group.bench_function(name, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut config = SwarmConfig::default();
                    config.strategy = *strategy;
                    let swarm = Swarm::new(config);
                    
                    // Register agents
                    for i in 0..10 {
                        let _ = swarm.register_agent(
                            format!("agent_{}", i),
                            vec!["compute".to_string()],
                            CognitivePattern::Convergent,
                        ).await;
                    }
                    
                    // Submit tasks
                    for i in 0..100 {
                        let task = Task::new(
                            format!("payload_{}", i),
                            Priority::Medium,
                        ).with_capabilities(vec!["compute".to_string()]);
                        
                        let _ = swarm.submit_task(task).await;
                    }
                });
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "std")]
criterion_group!(
    benches,
    task_creation_benchmark,
    swarm_agent_registration,
    task_distribution_strategies
);

#[cfg(not(feature = "std"))]
criterion_group!(
    benches,
    task_creation_benchmark
);

criterion_main!(benches);