// This file shows how to add debugging capabilities to the fusion system
// to access both pre and post-optimized compute graphs

use burn::backend::Wgpu;
use burn::tensor::Tensor;

// Type alias for the backend to use (Wgpu already includes fusion when fusion feature is enabled)
type Backend = Wgpu;

fn main() {
    env_logger::init();

    let device = Default::default();

    println!("=== FUSION DEBUG PATCH EXAMPLE ===");
    println!("This shows what you need to implement to access compute graphs");
    
    // Create a tensor with some data.
    let tensor1 = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

    println!("\n1. Operations that will be fused:");
    let tensor2 = tensor1 * 2.0;
    let tensor3 = tensor2 + 1.0;
    let tensor4 = tensor3.tanh();

    println!("\n2. Forcing execution:");
    let result = tensor4.to_data();
    println!("   Result: {:?}", result);
    
    println!("\n=== IMPLEMENTATION GUIDE ===");
    print_implementation_guide();
}

fn print_implementation_guide() {
    println!("To access compute graphs, add these debugging hooks:");
    
    println!("\n1. ADD TO MultiStream:");
    println!("```rust");
    println!("impl<R: FusionRuntime> MultiStream<R> {{");
    println!("    pub fn debug_operation_queue(&self, stream_id: StreamId) -> Option<&Vec<OperationIr>> {{");
    println!("        self.streams.get(&stream_id).map(|s| &s.queue.global)");
    println!("    }}");
    println!("    ");
    println!("    pub fn debug_execution_plans(&self) -> &ExecutionPlanStore<R::Optimization> {{");
    println!("        &self.optimizations");
    println!("    }}");
    println!("}}");
    println!("```");
    
    println!("\n2. ADD TO ExecutionPlanStore:");
    println!("```rust");
    println!("impl<O> ExecutionPlanStore<O> {{");
    println!("    pub fn debug_plans(&self) -> &Vec<ExecutionPlan<O>> {{");
    println!("        &self.plans");
    println!("    }}");
    println!("    ");
    println!("    pub fn to_json(&self) -> Result<String, serde_json::Error>");
    println!("    where O: serde::Serialize {{");
    println!("        serde_json::to_string_pretty(&self.plans)");
    println!("    }}");
    println!("}}");
    println!("```");
    
    println!("\n3. ADD TO FusionServer:");
    println!("```rust");
    println!("impl<R: FusionRuntime> FusionServer<R> {{");
    println!("    pub fn debug_pre_optimized(&self, stream_id: StreamId) -> Option<&Vec<OperationIr>> {{");
    println!("        self.streams.debug_operation_queue(stream_id)");
    println!("    }}");
    println!("    ");
    println!("    pub fn debug_post_optimized(&self) -> &ExecutionPlanStore<R::Optimization> {{");
    println!("        self.streams.debug_execution_plans()");
    println!("    }}");
    println!("}}");
    println!("```");
    
    println!("\n4. USAGE EXAMPLE:");
    println!("```rust");
    println!("// Access the client and server");
    println!("let client = get_client::<Backend>(&device);");
    println!("let server = client.server.lock();");
    println!("");
    println!("// Get pre-optimized graph");
    println!("let stream_id = StreamId::current();");
    println!("if let Some(operations) = server.debug_pre_optimized(stream_id) {{");
    println!("    println!(\"Pre-optimized operations:\");");
    println!("    for (i, op) in operations.iter().enumerate() {{");
    println!("        println!(\"  [{{}}] {{:?}}\", i, op);");
    println!("    }}");
    println!("}}");
    println!("");
    println!("// Get post-optimized graph");
    println!("let execution_plans = server.debug_post_optimized();");
    println!("for (i, plan) in execution_plans.debug_plans().iter().enumerate() {{");
    println!("    println!(\"Execution Plan {{}}:\", i);");
    println!("    println!(\"  Operations: {{}} ops\", plan.operations.len());");
    println!("    println!(\"  Strategy: {{:?}}\", plan.optimization.strategy);");
    println!("}}");
    println!("```");
    
    println!("\n5. ASCII GRAPH GENERATION:");
    println!("```rust");
    println!("fn operations_to_ascii_graph(operations: &[OperationIr]) -> String {{");
    println!("    let mut graph = String::new();");
    println!("    graph.push_str(\"digraph G {{\\n\");");
    println!("    ");
    println!("    for (i, op) in operations.iter().enumerate() {{");
    println!("        let op_name = format!(\"op_{{}}\", i);");
    println!("        let op_type = match op {{");
    println!("            OperationIr::NumericFloat(_, numeric_op) => format!(\"{{:?}}\", numeric_op),");
    println!("            OperationIr::NumericInt(_, numeric_op) => format!(\"{{:?}}\", numeric_op),");
    println!("            _ => \"Other\".to_string(),");
    println!("        }};");
    println!("        graph.push_str(&format!(\"  {{}} [label=\\\"{{}}\\\"]; \\n\", op_name, op_type));");
    println!("        ");
    println!("        // Add edges based on tensor dependencies");
    println!("        for node in op.nodes() {{");
    println!("            // Connect to previous operations that produced this tensor");
    println!("        }}");
    println!("    }}");
    println!("    ");
    println!("    graph.push_str(\"}}\");");
    println!("    graph");
    println!("}}");
    println!("```");
    
    println!("\n=== NEXT STEPS ===");
    println!("1. Apply these patches to burn-fusion");
    println!("2. Rebuild and test with your example");
    println!("3. Export graphs to JSON/DOT format");
    println!("4. Create ASCII visualization tools");
    println!("5. Compare pre vs post-optimized graphs");
}
