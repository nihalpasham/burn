use burn_ir::{OperationIr, TensorId};
use crate::stream::store::{ExecutionPlan, ExecutionPlanSummary};
use std::collections::{HashMap, HashSet};

/// Generate an ASCII representation of a sequence of operations (pre-optimized graph).
pub fn operations_to_ascii_graph(operations: &[OperationIr]) -> String {
    let mut graph = String::new();
    graph.push_str("Pre-optimized Operation Graph:\n");
    graph.push_str("============================\n\n");

    if operations.is_empty() {
        graph.push_str("No operations found.\n");
        return graph;
    }

    // Track tensor dependencies
    let mut tensor_producers: HashMap<TensorId, usize> = HashMap::new();
    let mut tensor_consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();

    // First pass: identify producers and consumers
    for (i, op) in operations.iter().enumerate() {
        let nodes = op.nodes();
        
        for node in &nodes {
            match node.status {
                burn_ir::TensorStatus::NotInit => {
                    // This operation produces this tensor
                    tensor_producers.insert(node.id, i);
                }
                burn_ir::TensorStatus::ReadOnly | burn_ir::TensorStatus::ReadWrite => {
                    // This operation consumes this tensor
                    tensor_consumers.entry(node.id).or_insert_with(Vec::new).push(i);
                }
            }
        }
    }

    // Generate the graph
    for (i, op) in operations.iter().enumerate() {
        graph.push_str(&format!("Op[{}]: {}\n", i, operation_to_string(op)));
        
        let nodes = op.nodes();
        let inputs: Vec<_> = nodes.iter()
            .filter(|n| matches!(n.status, burn_ir::TensorStatus::ReadOnly | burn_ir::TensorStatus::ReadWrite))
            .collect();
        let outputs: Vec<_> = nodes.iter()
            .filter(|n| matches!(n.status, burn_ir::TensorStatus::NotInit))
            .collect();

        if !inputs.is_empty() {
            graph.push_str("  Inputs:  ");
            for input in &inputs {
                if let Some(&producer_id) = tensor_producers.get(&input.id) {
                    graph.push_str(&format!("{}(from Op[{}]) ", input.id, producer_id));
                } else {
                    graph.push_str(&format!("{}(external) ", input.id));
                }
            }
            graph.push('\n');
        }

        if !outputs.is_empty() {
            graph.push_str("  Outputs: ");
            for output in &outputs {
                graph.push_str(&format!("{} ", output.id));
            }
            graph.push('\n');
        }

        graph.push('\n');
    }

    // Add dependency visualization
    graph.push_str("Dependency Flow:\n");
    graph.push_str("================\n");
    for (i, _) in operations.iter().enumerate() {
        let dependencies: Vec<usize> = tensor_consumers.values()
            .flatten()
            .filter(|&&consumer| consumer == i)
            .map(|_| {
                // Find which operations this one depends on
                operations[i].nodes().iter()
                    .filter_map(|node| {
                        if matches!(node.status, burn_ir::TensorStatus::ReadOnly | burn_ir::TensorStatus::ReadWrite) {
                            tensor_producers.get(&node.id).copied()
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        if !dependencies.is_empty() {
            let mut unique_deps: Vec<_> = dependencies.into_iter().collect::<HashSet<_>>().into_iter().collect();
            unique_deps.sort();
            graph.push_str(&format!("Op[{}] depends on: {:?}\n", i, unique_deps));
        }
    }

    graph
}

#[allow(dead_code)]
/// Generate an ASCII representation of execution plans (post-optimized graph).
pub(crate) fn execution_plans_to_ascii_graph<O: std::fmt::Debug>(plans: &[ExecutionPlan<O>]) -> String {
    let mut graph = String::new();
    graph.push_str("Post-optimized Execution Plans:\n");
    graph.push_str("===============================\n\n");

    if plans.is_empty() {
        graph.push_str("No execution plans found.\n");
        return graph;
    }

    for (i, plan) in plans.iter().enumerate() {
        graph.push_str(&format!("Plan[{}]:\n", i));
        graph.push_str(&format!("  Operations: {} ops\n", plan.operations.len()));
        graph.push_str(&format!("  Triggers: {} triggers\n", plan.triggers.len()));
        
        // Show the operations in this plan
        graph.push_str("  Operation sequence:\n");
        for (j, op) in plan.operations.iter().enumerate() {
            graph.push_str(&format!("    [{}] {}\n", j, operation_to_string(op)));
        }
        
        graph.push_str(&format!("  Strategy: {:?}\n", plan.optimization.strategy));
        graph.push('\n');
    }

    graph
}

/// Generate a summary comparison between pre and post-optimized graphs.
pub fn generate_optimization_summary(
    pre_ops: &[OperationIr],
    summaries: &[ExecutionPlanSummary],
) -> String {
    let mut summary = String::new();
    summary.push_str("Optimization Summary:\n");
    summary.push_str("====================\n\n");

    summary.push_str(&format!("Pre-optimization:  {} operations\n", pre_ops.len()));
    summary.push_str(&format!("Post-optimization: {} execution plans\n", summaries.len()));
    
    let total_optimized_ops: usize = summaries.iter().map(|s| s.operation_count).sum();
    summary.push_str(&format!("Total operations in plans: {}\n", total_optimized_ops));
    
    if pre_ops.len() > 0 {
        let reduction_ratio = (pre_ops.len() as f32 - total_optimized_ops as f32) / pre_ops.len() as f32 * 100.0;
        summary.push_str(&format!("Operation reduction: {:.1}%\n", reduction_ratio));
    }

    summary.push('\n');

    // Show operation type distribution
    let mut op_types: HashMap<String, usize> = HashMap::new();
    for op in pre_ops {
        let op_type = operation_type_name(op);
        *op_types.entry(op_type).or_insert(0) += 1;
    }

    summary.push_str("Operation type distribution:\n");
    for (op_type, count) in op_types {
        summary.push_str(&format!("  {}: {}\n", op_type, count));
    }

    summary
}

/// Convert an OperationIr to a human-readable string.
fn operation_to_string(op: &OperationIr) -> String {
    match op {
        OperationIr::BaseFloat(base_op) => format!("BaseFloat({:?})", base_op),
        OperationIr::BaseInt(base_op) => format!("BaseInt({:?})", base_op),
        OperationIr::BaseBool(base_op) => format!("BaseBool({:?})", base_op),
        OperationIr::NumericFloat(dtype, numeric_op) => {
            format!("NumericFloat({:?}, {:?})", dtype, numeric_op)
        }
        OperationIr::NumericInt(dtype, numeric_op) => {
            format!("NumericInt({:?}, {:?})", dtype, numeric_op)
        }
        OperationIr::Bool(bool_op) => format!("Bool({:?})", bool_op),
        OperationIr::Int(int_op) => format!("Int({:?})", int_op),
        OperationIr::Float(dtype, float_op) => format!("Float({:?}, {:?})", dtype, float_op),
        OperationIr::Module(module_op) => format!("Module({:?})", module_op),
        OperationIr::Init(init_op) => format!("Init({:?})", init_op),
        OperationIr::Custom(custom_op) => format!("Custom({})", custom_op.id),
        OperationIr::Drop(tensor) => format!("Drop({})", tensor.id),
    }
}

/// Get the operation type name for categorization.
pub fn operation_type_name(op: &OperationIr) -> String {
    match op {
        OperationIr::BaseFloat(_) => "BaseFloat".to_string(),
        OperationIr::BaseInt(_) => "BaseInt".to_string(),
        OperationIr::BaseBool(_) => "BaseBool".to_string(),
        OperationIr::NumericFloat(_, _) => "NumericFloat".to_string(),
        OperationIr::NumericInt(_, _) => "NumericInt".to_string(),
        OperationIr::Bool(_) => "Bool".to_string(),
        OperationIr::Int(_) => "Int".to_string(),
        OperationIr::Float(_, _) => "Float".to_string(),
        OperationIr::Module(_) => "Module".to_string(),
        OperationIr::Init(_) => "Init".to_string(),
        OperationIr::Custom(_) => "Custom".to_string(),
        OperationIr::Drop(_) => "Drop".to_string(),
    }
}

/// Generate a DOT graph format for visualization tools like Graphviz.
pub fn operations_to_dot_graph(operations: &[OperationIr]) -> String {
    let mut dot = String::new();
    dot.push_str("digraph OperationGraph {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box];\n\n");

    // Add nodes
    for (i, op) in operations.iter().enumerate() {
        let label = operation_to_string(op).replace('"', "'");
        dot.push_str(&format!("  op{} [label=\"Op[{}]\\n{}\"];\n", i, i, label));
    }

    // Add edges based on tensor dependencies
    let mut tensor_producers: HashMap<TensorId, usize> = HashMap::new();
    
    // Find producers
    for (i, op) in operations.iter().enumerate() {
        for node in op.nodes() {
            if matches!(node.status, burn_ir::TensorStatus::NotInit) {
                tensor_producers.insert(node.id, i);
            }
        }
    }

    // Add edges
    for (i, op) in operations.iter().enumerate() {
        for node in op.nodes() {
            if matches!(node.status, burn_ir::TensorStatus::ReadOnly | burn_ir::TensorStatus::ReadWrite) {
                if let Some(&producer) = tensor_producers.get(&node.id) {
                    dot.push_str(&format!("  op{} -> op{} [label=\"{}\"];\n", producer, i, node.id));
                }
            }
        }
    }

    dot.push_str("}\n");
    dot
}
