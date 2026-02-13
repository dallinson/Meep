use cudarc::driver::{CudaView, CudaViewMut};

use crate::{
    LEARNING_RATE, OUTPUT_LABELS,
    backend::{CublasShape, GPUBackend},
    layer::ActivationFunction,
    network::Network, optimiser::{self, Optimiser},
};

pub(crate) struct Trainer {
    batch_size: usize,
    network: Network,
    optimiser: Box<dyn Optimiser>
}

impl Trainer {
    pub(crate) fn new(batch_size: usize, network: Network, optimiser: Box<dyn Optimiser>) -> Self {
        Self {
            batch_size,
            network,
            optimiser
        }
    }

    fn feedforward(
        &self,
        backend: &GPUBackend,
        data: CudaView<f32>,
        mut z: CudaViewMut<f32>,
        mut output: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(data.len() % self.network.layer_sizes[0], 0); // Make sure we send the right amount of data!
        assert_eq!(data.len() / self.network.layer_sizes[0], self.batch_size);
        assert_eq!(z.len(), output.len());
        assert_eq!(
            z.len(),
            self.network.layer_sizes[1..].iter().sum::<usize>() * self.batch_size
        );
        let (weights, biases): (Vec<_>, Vec<_>) =
            self.network.get_weights_biases().into_iter().unzip();
        let mut z_out_delta_ranges = Vec::new();
        let mut curr_idx = 0;
        for size in self.network.layer_sizes[1..].iter() {
            z_out_delta_ranges.push(curr_idx..(curr_idx + (size * self.batch_size)));
            curr_idx += size * self.batch_size;
        }
        let z_out_delta_ranges = z_out_delta_ranges;

        let mut matmul_idx = 0;
        for layer in self.network.layers[1..].iter() {
            let current_activated_data = if matmul_idx == 0 {
                data.slice(..)
            } else {
                output.slice(z_out_delta_ranges[matmul_idx - 1].clone())
            };
            match layer {
                crate::layer::LayerType::NODE(_) => {
                    // Matmul!
                    let mut output_matrix = z.slice_mut(z_out_delta_ranges[matmul_idx].clone());
                    backend.splat(biases[matmul_idx].slice(..), output_matrix.slice_mut(..))?;
                    backend.matmul_unstrided(
                        false,
                        false,
                        CublasShape::new(
                            self.network.layer_sizes[matmul_idx + 1],
                            self.network.layer_sizes[matmul_idx],
                        ),
                        CublasShape::new(self.network.layer_sizes[matmul_idx], self.batch_size),
                        CublasShape::new(self.network.layer_sizes[matmul_idx + 1], self.batch_size),
                        1.0f32,
                        1.0f32,
                        weights[matmul_idx].slice(..),
                        current_activated_data,
                        output_matrix.slice_mut(..),
                    )?;
                    matmul_idx += 1;
                }
                crate::layer::LayerType::ACTIVATION(activation_function) => {
                    match activation_function {
                        crate::layer::ActivationFunction::SIGMOID => backend.unary(
                            "sigmoid",
                            z.slice(z_out_delta_ranges[matmul_idx.checked_sub(1).unwrap()].clone()),
                            output.slice_mut(
                                z_out_delta_ranges[matmul_idx.checked_sub(1).unwrap()].clone(),
                            ),
                        )?,
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn calculate_loss(
        &self,
        backend: &GPUBackend,
        data: CudaView<f32>,
        labels: CudaView<usize>,
    ) -> Result<(f32, usize), Box<dyn std::error::Error>> {
        assert_eq!(labels.len() * self.network.layer_sizes[0], data.len());
        let mut output = backend
            .stream
            .alloc_zeros(self.network.layer_sizes[1..].iter().sum::<usize>() * self.batch_size)?;
        let mut z = backend
            .stream
            .alloc_zeros(self.network.layer_sizes[1..].iter().sum::<usize>() * self.batch_size)?;
        self.feedforward(backend, data, z.as_view_mut(), output.as_view_mut())?; // Feedforwards first
        let last_idx = self.network.layer_sizes[1..(self.network.layer_sizes.len() - 1)]
            .iter()
            .sum::<usize>()
            * self.batch_size;
        backend.synchronize()?;
        let result = backend.calculate_loss(output.slice(last_idx..), labels.slice(..))?;
        let match_count =
            backend.calculate_matching(output.slice(last_idx..), labels, OUTPUT_LABELS)?;
        Ok((result, match_count))
    }

    pub(crate) fn compute_gradient(
        &mut self,
        backend: &GPUBackend,
        data: CudaView<f32>,
        labels: CudaView<usize>,
        mut gradient: CudaViewMut<f32>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let out_size = self.network.layer_sizes[1..].iter().sum::<usize>() * self.batch_size;
        let mut output = backend.stream.alloc_zeros(out_size)?;
        let mut z = backend.stream.alloc_zeros(out_size)?;
        let mut delta = backend.stream.alloc_zeros(out_size)?;
        let (weights, _): (Vec<_>, Vec<_>) = self.network.get_weights_biases().into_iter().unzip();

        let mut z_out_delta_ranges = Vec::new();
        let mut curr_idx = 0;
        for size in self.network.layer_sizes[1..].iter() {
            let offset = size * self.batch_size;
            z_out_delta_ranges.push(curr_idx..(curr_idx + offset));
            curr_idx += offset;
        }

        let mut curr_idx = 0;
        let mut weight_changes = Vec::new();
        let mut bias_changes = Vec::new();
        for layer_sizes in self.network.layer_sizes.windows(2) {
            weight_changes.push(curr_idx..(curr_idx + (layer_sizes[0] * layer_sizes[1])));
            curr_idx += layer_sizes[0] * layer_sizes[1];
            bias_changes.push(curr_idx..(curr_idx + layer_sizes[1]));
            curr_idx += layer_sizes[1];
        }

        let weight_changes = weight_changes;
        let bias_changes = bias_changes;
        let z_out_delta_len = z_out_delta_ranges.len();

        self.feedforward(
            backend,
            data.slice(..),
            z.as_view_mut(),
            output.as_view_mut(),
        )?;
        let activations = self
            .network
            .layers
            .iter()
            .filter_map(|x| match x {
                crate::layer::LayerType::NODE(_) => None,
                crate::layer::LayerType::ACTIVATION(activation_function) => {
                    Some(activation_function)
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(z_out_delta_len, activations.len());
        for (z_range, function) in z_out_delta_ranges.iter().zip(activations.iter()) {
            backend.unary_inplace(
                match function {
                    ActivationFunction::SIGMOID => "sigmoid_prime",
                },
                z.slice_mut(z_range.clone()),
            )?;
        }
        let filled_activations = backend.fill_estimated(labels)?;
        backend.binary(
            "squared_diff_prime",
            output.slice(z_out_delta_ranges.last().unwrap().clone()),
            filled_activations.as_view(),
            delta.slice_mut(z_out_delta_ranges.last().unwrap().clone()),
        )?;
        for (rev_idx, layer_size) in self.network.layer_sizes[1..].iter().rev().enumerate() {
            if rev_idx != 0 {
                // We need to fill curr_delta (the last curr_delta in the chain was filled by the earlier squared_diff_prime call)
                let first_range = z_out_delta_ranges[z_out_delta_len - rev_idx - 1].clone();
                let second_range = z_out_delta_ranges[z_out_delta_len - rev_idx].clone();
                assert_eq!(first_range.end, second_range.start); // the end of a range is exclusive
                let mut split_delta = delta.split_at_mut(second_range.start);
                backend.matmul_unstrided(
                    true,
                    false,
                    CublasShape::new(
                        self.network.layer_sizes[self.network.layer_sizes.len() - rev_idx],
                        *layer_size,
                    ),
                    CublasShape::new(
                        self.network.layer_sizes[self.network.layer_sizes.len() - rev_idx],
                        self.batch_size,
                    ),
                    CublasShape::new(*layer_size, self.batch_size), // rows in b =
                    1.0f32,
                    0.0f32,
                    weights[weights.len() - rev_idx].slice(..),
                    split_delta
                        .1
                        .slice(0..(second_range.end - second_range.start)), // Since we're going backwards, split_delta.1 is b and 0 is c
                    split_delta.0.slice_mut(
                        first_range.end - (first_range.end - first_range.start)..first_range.end,
                    ),
                )?;
            }
            let mut curr_delta =
                delta.slice_mut(z_out_delta_ranges[z_out_delta_len - 1 - rev_idx].clone());
            let curr_z = z.slice(z_out_delta_ranges[z_out_delta_len - 1 - rev_idx].clone());
            backend.binary_inplace("multiply", curr_z, curr_delta.slice_mut(..))?;
            // This computes the delta of this layer
            // We next need to compute the derivative of the loss wrt the weights (the delta is the biases!)
            backend.matmul_unstrided(
                false,
                true,
                CublasShape::new(
                    self.network.layer_sizes[z_out_delta_len - rev_idx],
                    self.batch_size,
                ),
                CublasShape::new(
                    self.network.layer_sizes[z_out_delta_len - rev_idx - 1],
                    self.batch_size,
                ),
                CublasShape::new(
                    self.network.layer_sizes[z_out_delta_len - rev_idx],
                    self.network.layer_sizes[z_out_delta_len - rev_idx - 1],
                ),
                1.0f32,
                1.0f32,
                curr_delta.slice(..),
                if z_out_delta_len < 2 + rev_idx {
                    // z_out_data_len is at least 2 if a hidden layer exists (hl + ol)
                    data.slice(..)
                } else {
                    assert!(z_out_delta_len >= 2 + rev_idx);
                    // if we have e.g. 3 layers (input, hl, ol) then output has 2 entries
                    // so len - 2 is legal
                    output.slice(z_out_delta_ranges[z_out_delta_len - 2 - rev_idx].clone())
                },
                gradient.slice_mut(weight_changes[weight_changes.len() - 1 - rev_idx].clone()),
            )?;
        }

        // We have now computed the weight and bias changes for each layer!

        // Now we need to apply it:
        for ((delta_range, layer_size), bias_range) in z_out_delta_ranges
            .iter()
            .zip(self.network.layer_sizes[1..].iter())
            .zip(bias_changes.iter())
        {
            backend.reduce_strided(
                delta.slice_mut(delta_range.clone()),
                self.batch_size,
                *layer_size,
                *layer_size,
            )?;

            backend.stream.memcpy_dtod(
                &delta.slice(0..*layer_size),
                &mut gradient.slice_mut(bias_range.clone()),
            )?;
            // This moves the biases to the main gradient
        }

        Ok(())
    }

    pub(crate) fn optimise(&mut self, backend: &GPUBackend, data: CudaView<f32>, labels: CudaView<usize>) -> Result<(), Box<dyn std::error::Error>> {
        let mut gradient = backend.stream.alloc_zeros(self.network.weights.len())?;
        self.compute_gradient(backend, data, labels, gradient.as_view_mut());
        self.optimiser.optimise(backend, gradient.as_view_mut(), self.network.weights.as_view_mut())?;
        Ok(())
    }
}
