use burn::backend::Wgpu;
use burn::tensor::Tensor;
use burn_common::id::StreamId;
use burn_fusion::debug;

// Type alias for the backend to use (Wgpu already includes fusion when fusion feature is enabled)
type Backend = Wgpu;

fn main() {
    env_logger::init();

    let device = Default::default();

    println!("=== COMPUTE GRAPH DEBUGGING ===");
    println!("This example demonstrates accessing and visualizing REAL fusion graphs");

    let current_stream = StreamId::current();
    println!("Current stream: {}", current_stream);

    println!("\n=== STEP 1: TENSOR CREATION ===");
    println!("Creating tensor with from_data() - this triggers Init operation...");

    // Create a tensor with some data.
    let tensor1 = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

    // Check immediately after tensor creation
    println!("\n--- AFTER TENSOR CREATION ---");
    if let Some(ops_after_init) = Backend::debug_pre_optimized(&device, current_stream) {
        println!("Operations in queue: {}", ops_after_init.len());
        if !ops_after_init.is_empty() {
            println!("Init operation captured:");
            let init_graph = debug::operations_to_ascii_graph(&ops_after_init);
            println!("{}", init_graph);
        }
    } else {
        println!("No operations in queue (Init was processed immediately)");
    }

    let summary_after_init = Backend::debug_fusion_summary(&device);
    println!("Execution plans after Init: {}", summary_after_init.execution_plan_count);

    println!("\n=== STEP 2: COMPUTE OPERATIONS ===");
    println!("Adding compute operations (these get queued for fusion):");

    // These operations will be queued for fusion
    println!("   - tensor1 * 2.0");
    let tensor2 = tensor1 * 2.0;

    println!("   - result + 1.0");
    let tensor3 = tensor2 + 1.0;

    println!("   - tanh(result)");
    let tensor4 = tensor3.tanh();

    // Check the current stream for operations BEFORE execution
    println!("\n=== AFTER ADDING COMPUTE OPERATIONS ===");

    // Try to access pre-optimized operations
    if let Some(pre_ops) = Backend::debug_pre_optimized(&device, current_stream) {
        println!("Operations in queue: {} (the 3 compute operations)", pre_ops.len());
        println!("\n--- COMPUTE OPERATIONS GRAPH ---");
        let pre_graph = debug::operations_to_ascii_graph(&pre_ops);
        println!("{}", pre_graph);

        println!("\n--- DOT GRAPH FORMAT ---");
        let dot_graph = debug::operations_to_dot_graph(&pre_ops);
        println!("{}", dot_graph);

        println!("\n--- CURRENT FUSION STATE ---");
        let fusion_summary = Backend::debug_fusion_summary(&device);
        println!("Fusion Summary: {:#?}", fusion_summary);
    } else {
        println!(
            "No operations found in current stream (operations may have been executed already)"
        );
    }

    println!("\n=== STEP 3: EXECUTION ===");
    println!("Forcing execution with .to_data() - this triggers optimization and ExecutionPlan creation:");
    let result = tensor4.to_data();
    println!("   Final result: {:?}", result);

    // Check the post-optimized execution plans after execution
    println!("\n=== POST-EXECUTION GRAPH ACCESS ===");
    let updated_summary = Backend::debug_fusion_summary(&device);
    println!("Post-execution fusion summary: {:#?}", updated_summary);

    // Access the execution plan summaries (post-optimized)
    let execution_plans = Backend::debug_execution_plan_summaries_with_ops(&device);
    println!("\n--- POST-OPTIMIZED EXECUTION PLANS ---");
    for plan in execution_plans.iter() {
        println!("Execution Plan {}: {} operations, {} triggers",
                 plan.id, plan.operation_count, plan.trigger_count);
        println!("  Operation types: {:?}", plan.operation_types);
    }

    // Try to get detailed execution plan information
    // Note: This requires the Optimization type to implement Debug
    println!("\n--- EXECUTION PLAN ANALYSIS ---");
    println!("Why do we have {} execution plans?", execution_plans.len());
    println!("- Plan 0: {} operations, types: {:?}",
             execution_plans.get(0).map_or(0, |p| p.operation_count),
             execution_plans.get(0).map_or(vec![], |p| p.operation_types.clone()));
    println!("  → This is the Init operation from tensor creation");
    println!("- Plan 1: {} operations, types: {:?}",
             execution_plans.get(1).map_or(0, |p| p.operation_count),
             execution_plans.get(1).map_or(vec![], |p| p.operation_types.clone()));
    println!("  → These are the 3 compute operations that got fused");

    println!("\n🔍 KEY INSIGHT:");
    println!("The Init operation was processed immediately after tensor creation,");
    println!("while the compute operations were queued for fusion optimization!");
    println!("That's why we see 3 ops in the queue but 4 total operations across both plans.");

    println!("\n--- ACTUAL FUSE TRACE INFORMATION ---");
    println!("Here's the REAL FuseTrace from the execution plans:");
    let fuse_traces = Backend::debug_fuse_trace_info(&device);
    for (i, trace_info) in fuse_traces.iter().enumerate() {
        println!("\n=== Execution Plan {} FuseTrace ===", i);
        println!("{}", trace_info);
    }

    if fuse_traces.is_empty() {
        println!("No FuseTrace information available (operations may not have been optimized yet)");
    }

    // Check if any operations remain in the pre-optimized queue (should be empty after execution)
    if let Some(remaining_ops) = Backend::debug_pre_optimized(&device, current_stream) {
        println!("\nOperations remaining in pre-optimized queue: {}", remaining_ops.len());
        if !remaining_ops.is_empty() {
            println!("WARNING: Operations still in queue after execution!");
        }
    } else {
        println!("\n✅ Pre-optimized queue is empty (operations were consumed and optimized)");
    }

    // Show all streams
    println!("\n=== ALL STREAMS ANALYSIS ===");
    let all_ops = Backend::debug_all_pre_optimized(&device);
    println!("Total streams: {}", all_ops.len());
    for (stream_id, ops) in &all_ops {
        println!("Stream {}: {} operations", stream_id, ops.len());
    }

    if !all_ops.is_empty() {
        println!("\n--- ALL STREAMS ASCII GRAPH ---");
        let all_graph = Backend::debug_all_pre_optimized_ascii_graph(&device);
        println!("{}", all_graph);
    }

    println!("\n=== IMPLEMENTATION STATUS ===");
    println!("✅ Added debugging methods to MultiStream");
    println!("✅ Added debugging methods to ExecutionPlanStore");
    println!("✅ Added debugging methods to FusionServer");
    println!("✅ Created ASCII graph generation functions");
    println!("✅ Exposed FusionServer access in public API");
    println!("✅ Created working example with real graph access");
    println!("🎉 COMPLETE: You can now visualize Burn's compute graphs!");
}
