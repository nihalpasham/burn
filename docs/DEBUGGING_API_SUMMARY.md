# Burn Fusion Debugging API - Summary

## What's New

Burn now includes comprehensive debugging capabilities for visualizing and analyzing fusion compute graphs. This allows developers to:

- **See exactly what operations are being fused**
- **Understand tensor dependencies and data flow**  
- **Generate visual graphs for debugging and optimization**
- **Monitor fusion performance and behavior**

## Key Features

### 🔍 **Operation Inspection**
```rust
// Access raw operations before they're optimized
let operations = Backend::debug_pre_optimized(&device, stream_id);
```

### 📊 **ASCII Visualization**
```rust
// Generate human-readable operation graphs
let graph = debug::operations_to_ascii_graph(&operations);
println!("{}", graph);
```

### 🎨 **GraphViz Export**
```rust
// Create DOT files for professional visualization tools
let dot = debug::operations_to_dot_graph(&operations);
std::fs::write("graph.dot", dot)?;
```

### 📈 **Fusion Analytics**
```rust
// Get detailed fusion statistics
let summary = Backend::debug_fusion_summary(&device);
println!("Streams: {}, Operations: {}, Plans: {}", 
    summary.stream_count, 
    summary.total_operations, 
    summary.execution_plan_count
);
```

## Quick Example

```rust
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use burn_fusion::{debug, Fusion};
use burn_common::id::StreamId;

type Backend = Wgpu;

fn main() {
    let device = Default::default();
    
    // Create some operations
    let x = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let y = x * 2.0;
    let z = y + 1.0;
    let result = z.tanh();
    
    // Visualize the compute graph BEFORE execution
    let stream = StreamId::current();
    if let Some(ops) = Backend::debug_pre_optimized(&device, stream) {
        println!("{}", debug::operations_to_ascii_graph(&ops));
    }
    
    // Execute and get results
    let data = result.to_data();
}
```

## Output Example

```
Pre-optimized Operation Graph:
============================

Op[0]: NumericFloat(F32, MulScalar(...))
  Inputs:  TensorId(0)(external) 
  Outputs: TensorId(1) 

Op[1]: NumericFloat(F32, AddScalar(...))
  Inputs:  TensorId(1)(from Op[0]) 
  Outputs: TensorId(2) 

Op[2]: Float(F32, Tanh(...))
  Inputs:  TensorId(2)(from Op[1]) 
  Outputs: TensorId(3) 

Dependency Flow:
================
Op[1] depends on: [0]
Op[2] depends on: [1]
```

## Use Cases

### 🐛 **Debugging**
- Understand why operations aren't fusing as expected
- Identify bottlenecks in compute graphs
- Verify tensor shapes and data flow

### 🎓 **Learning**
- Visualize how Burn optimizes your code
- Understand fusion behavior
- Learn about tensor operation dependencies

### 🔧 **Optimization**
- Analyze fusion effectiveness
- Identify opportunities for manual optimization
- Monitor performance characteristics

### 📚 **Documentation**
- Generate diagrams for papers and presentations
- Create visual documentation of model architectures
- Share compute graph designs with team members

## Getting Started

1. **Enable fusion** in your backend (most backends support this automatically)
2. **Import the debugging module**: `use burn_fusion::{debug, Fusion};`
3. **Access operations** before execution: `Backend::debug_pre_optimized(&device, stream_id)`
4. **Generate visualizations**: `debug::operations_to_ascii_graph(&operations)`

## Documentation

- **Full Guide**: See `docs/FUSION_DEBUGGING.md` for comprehensive documentation
- **API Reference**: All debugging methods are documented with examples
- **Working Example**: `examples/burn-test/examples/graph-debug.rs`

## Compatibility

- ✅ **All fusion-enabled backends** (Wgpu, CubeCL, etc.)
- ✅ **All tensor types** (Float, Int, Bool)
- ✅ **All operation types** (Numeric, Module, Custom)
- ✅ **Cross-platform** (Windows, macOS, Linux)

---

**🎉 Start visualizing your Burn compute graphs today!**
