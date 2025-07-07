use burn::backend::Wgpu;
use burn::tensor::Tensor;

// Type alias for the backend to use.
type Backend = Wgpu;

fn main() {
    // Initialize logging to see Burn's internal operations
    env_logger::init();

    let device = Default::default();

    println!("=== FUSION DEMONSTRATION ===");
    println!("Creating operations without forcing execution...");

    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);

    // These operations get queued for fusion
    println!("Queuing: tensor_1 * 2.0");
    let temp = tensor_1.clone() * 2.0;

    println!("Queuing: + 1.0");
    let y = temp + 1.0;

    println!("Queuing: tanh()");
    let z = y.tanh();

    println!("Final result (this triggers fusion): {}", z);


    // println!("\n=== COMPARISON: FORCED EXECUTION ===");
    // println!("Now doing the same with forced execution at each step...");

    // let tensor_2 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);

    // println!("Step 1: tensor_2 * 2.0");
    // let temp2 = tensor_2.clone() * 2.0;
    // println!("Result: {}", temp2); // Forces execution

    // println!("Step 2: + 1.0");
    // let y2 = temp2 + 1.0;
    // println!("Result: {}", y2); // Forces execution

    // println!("Step 3: tanh()");
    // let z2 = y2.tanh();
    // println!("Result: {}", z2); // Forces execution

    // Create a sample TensorIr (this is what would be in the operation)
    // let sample_tensor_ir = TensorIr {
    //     id: burn_ir::TensorId::new(1),
    //     shape: vec![2, 2],
    //     dtype: DType::F32,
    //     status: TensorStatus::ReadOnly,
    // };

    // let sample_output_ir = TensorIr {
    //     id: burn_ir::TensorId::new(2),
    //     shape: vec![2, 2],
    //     dtype: DType::F32,
    //     status: TensorStatus::NotInit,
    // };

    // // Create a ScalarOpIr (this is what gets passed to MulScalar)
    // let scalar_op = ScalarOpIr {
    //     lhs: sample_tensor_ir.clone(),
    //     rhs: 2.0f32,
    //     out: sample_output_ir.clone(),
    // };

    // // Create the OperationIr that would be registered for tensor * 2.0
    // let mul_scalar_op = OperationIr::NumericFloat(
    //     DType::F32,
    //     NumericOperationIr::MulScalar(scalar_op)
    // );

    // println!("MulScalar OperationIr: {:#?}", mul_scalar_op);
}
