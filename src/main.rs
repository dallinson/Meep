mod backend;
mod layer;
mod mnist;
mod network;
mod trainer;

use crate::{backend::GPUBackend, layer::Layer, mnist::MnistManager, trainer::Trainer};

const OUTPUT_LABELS: usize = 10;
const BATCH_SIZE: usize = 5_000;
const EPOCH_SIZE: usize = 60_000;
const EPOCH_COUNT: usize = 5_000;
const LEARNING_RATE: f32 = -0.001f32 * (BATCH_SIZE as f32 / EPOCH_SIZE as f32);

fn calculate_loss(
    backend: &GPUBackend,
    trainer: &Trainer,
    data: &MnistManager,
) -> Result<(f32, usize), Box<dyn std::error::Error>> {
    let test_size = data.tst_lbl.len();
    assert_eq!(test_size % BATCH_SIZE, 0);
    let mut sum = 0.0f32;
    let mut match_count = 0;
    for batch in 0..(test_size / BATCH_SIZE) {
        let offset = batch * BATCH_SIZE;
        let (trn_img, trn_lbl) = data.get_test_data(offset..(offset + BATCH_SIZE));

        let data = trainer.calculate_loss(backend, trn_img, trn_lbl)?;
        sum += data.0;
        match_count += data.1;
    }
    Ok((sum, match_count))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let ctx = cudarc::driver::CudaContext::new(0)?;
    let backend = GPUBackend::new(ctx.clone())?;

    let l1 = Layer::new(28 * 28);
    let l2 = l1.matmul(256);
    let l2 = l2.sigmoid();
    let output = l2.matmul(10);
    let output = output.sigmoid();
    let mut net = output.compile(&backend)?;

    println!("{:?}", net);

    net.randomise_weights(&backend)?;
    let mut trainer = Trainer::new(BATCH_SIZE, net);
    let data = MnistManager::new(&backend)?;
    let loss = calculate_loss(&backend, &trainer, &data)?;
    println!("Net loss: {}, matches: {}", loss.0, loss.1);

    println!("Training...");
    for epoch in 0..EPOCH_COUNT {
        for i in 0..(EPOCH_SIZE / BATCH_SIZE) {
            let offset = i * BATCH_SIZE;
            let (trn_img, trn_lbl) = data.get_train_data(offset..(offset + BATCH_SIZE));

            trainer.backprop(&backend, trn_img, trn_lbl)?;
        }
        if epoch % 100 == 0 {
            let loss = calculate_loss(&backend, &trainer, &data)?;
            println!("Epoch {} loss: {}, matches: {}", epoch, loss.0, loss.1);
        }
    }
    let loss = calculate_loss(&backend, &trainer, &data)?;
    println!("Net loss: {}, matches: {}", loss.0, loss.1);

    Ok(())
}
