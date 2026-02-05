mod backend;
mod layer;
mod mnist;
mod network;
mod trainer;

use crate::{backend::GPUBackend, layer::Layer, mnist::MnistManager, trainer::Trainer};

const OUTPUT_LABELS: usize = 10;
const LEARNING_RATE: f32 = 0.01f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let l1 = Layer::new(28 * 28);
    let l2 = l1.matmul(256);
    let l2 = l2.sigmoid();
    let output = l2.matmul(10);
    let output = output.sigmoid();
    let mut net = output.compile();

    println!("{:?}", net);

    let ctx = cudarc::driver::CudaContext::new(0)?;
    net.init_gpu(ctx.clone())?;

    let backend = GPUBackend::new(ctx.clone())?;
    net.randomise_weights(&backend)?;
    let mut trainer = Trainer::new(5000, net);
    let data = MnistManager::new();
    let mut test_data = backend
        .stream
        .clone_htod(&data.tst_img[0..(5000 * 28 * 28)])?;
    let test_label = backend.stream.clone_htod(&data.tst_lbl[0..5000])?;
    let loss = trainer.calculate_loss(&backend, &mut test_data, &test_label)?;
    println!("Net loss: {}", loss);

    let mut train_data = backend.stream.clone_htod(&data.trn_img[0..(5000*28*28)])?;
    let train_label = backend.stream.clone_htod(&data.trn_lbl[0..5000])?;

    trainer.backprop(&backend, &mut train_data, &train_label)?;

    let loss = trainer.calculate_loss(&backend, &mut test_data, &test_label)?;
    println!("Net loss: {}", loss);

    Ok(())
}
