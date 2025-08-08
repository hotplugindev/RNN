//! Direct Weight Initialization Test
//!
//! This example tests weight initialization directly to identify why weights
//! are not being properly initialized in the neural network.

use nnl::layers::DenseLayer;
use nnl::prelude::*;

fn main() -> Result<()> {
    env_logger::init();

    println!("Direct Weight Initialization Test");
    println!("=================================");

    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

    // Test 1: Direct tensor creation and initialization
    println!("\n1. Testing direct tensor initialization...");
    let mut test_tensor = Tensor::zeros_on_device(&[3, 3], device.clone())?;
    println!("Initial tensor data: {:?}", test_tensor.to_vec()?);

    // Test Xavier initialization directly
    let init = WeightInit::Xavier;
    init.initialize(&mut test_tensor, 3, 3)?;
    println!("After Xavier init: {:?}", test_tensor.to_vec()?);

    // Test 2: Dense layer creation
    println!("\n2. Testing DenseLayer creation...");
    let dense_layer = DenseLayer::new_on_device(
        2,
        3,
        Activation::ReLU,
        true,
        WeightInit::Xavier,
        device.clone(),
    )?;

    println!("Dense layer created");
    let params = dense_layer.parameters();
    println!("Number of parameters: {}", params.len());

    for (i, param) in params.iter().enumerate() {
        let data = param.to_vec()?;
        println!(
            "Parameter {}: shape {:?}, first 5 values: {:?}",
            i,
            param.shape(),
            &data[..data.len().min(5)]
        );
    }

    // Test 3: Manual weight modification
    println!("\n3. Testing manual weight modification...");
    let mut manual_tensor = Tensor::zeros_on_device(&[2, 2], device.clone())?;
    println!("Before manual fill: {:?}", manual_tensor.to_vec()?);

    manual_tensor.fill(0.5)?;
    println!("After fill(0.5): {:?}", manual_tensor.to_vec()?);

    manual_tensor.copy_from_slice(&[1.0, 2.0, 3.0, 4.0])?;
    println!("After copy_from_slice: {:?}", manual_tensor.to_vec()?);

    // Test 4: Different weight initializations
    println!("\n4. Testing different weight initializations...");
    let mut test_weights = vec![];
    let inits = vec![
        ("Zeros", WeightInit::Zeros),
        ("Ones", WeightInit::Ones),
        ("Xavier", WeightInit::Xavier),
        ("He", WeightInit::He),
        ("Constant(0.5)", WeightInit::Constant(0.5)),
        ("Normal(0.1)", WeightInit::Normal(0.1)),
        ("Uniform(0.5)", WeightInit::Uniform(0.5)),
    ];

    for (name, init) in inits {
        let mut tensor = Tensor::zeros_on_device(&[2, 3], device.clone())?;
        init.initialize(&mut tensor, 2, 3)?;
        let data = tensor.to_vec()?;
        println!("{}: {:?}", name, data);
        test_weights.push((name, data));
    }

    // Test 5: Verify non-zero initializations worked
    println!("\n5. Verification...");
    for (name, data) in test_weights {
        let all_zeros = data.iter().all(|&x| x == 0.0);
        let all_same = data.iter().all(|&x| x == data[0]);

        match name {
            "Zeros" => {
                if all_zeros {
                    println!("✅ {}: Correctly all zeros", name);
                } else {
                    println!("❌ {}: Should be all zeros but isn't", name);
                }
            }
            "Ones" | "Constant(0.5)" => {
                if all_same && !all_zeros {
                    println!("✅ {}: Correctly all same value", name);
                } else {
                    println!("❌ {}: Should be all same value but isn't", name);
                }
            }
            _ => {
                if all_zeros {
                    println!("❌ {}: ERROR - All zeros (should be random)", name);
                } else if all_same {
                    println!("⚠️  {}: WARNING - All same value (suspicious)", name);
                } else {
                    println!("✅ {}: Correctly randomized", name);
                }
            }
        }
    }

    // Test 6: Network builder weight initialization
    println!("\n6. Testing NetworkBuilder weight initialization...");
    let network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 3,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError)
        .optimizer(OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        })
        .device(device.clone())
        .build()?;

    println!("Network built successfully");

    // Check if network weights are properly initialized
    let mut all_weights_zero = true;
    for (layer_idx, layer) in network.layers().iter().enumerate() {
        let params = layer.parameters();
        for (param_idx, param) in params.iter().enumerate() {
            let data = param.to_vec()?;
            let is_zero = data.iter().all(|&x| x == 0.0);

            if is_zero {
                println!(
                    "❌ Network Layer {} Param {}: All zeros",
                    layer_idx, param_idx
                );
            } else {
                println!(
                    "✅ Network Layer {} Param {}: Non-zero values {:?}",
                    layer_idx,
                    param_idx,
                    &data[..data.len().min(3)]
                );
                all_weights_zero = false;
            }
        }
    }

    if all_weights_zero {
        println!("\n🚨 CRITICAL: All network weights are zero - initialization failed!");
    } else {
        println!("\n✅ SUCCESS: Network weights are properly initialized");
    }

    Ok(())
}
