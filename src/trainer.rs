use cudarc::driver::{CudaSlice, CudaView};

use crate::{
    LEARNING_RATE, OUTPUT_LABELS, backend::{CublasShape, GPUBackend}, layer::ActivationFunction, network::Network
};

pub(crate) struct Trainer {
    batch_size: usize,
    network: Network,
}

impl Trainer {
    pub(crate) fn new(batch_size: usize, network: Network) -> Self {
        Self {
            batch_size,
            network,
        }
    }

    fn feedforward(
        &self,
        backend: &GPUBackend,
        data: CudaView<f32>,
        zs: &mut Vec<CudaSlice<f32>>,
        outputs: &mut Vec<CudaSlice<f32>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(data.len() % self.network.layer_sizes[0], 0); // Make sure we send the right amount of data!
        assert_eq!(outputs.len(), self.network.layer_sizes.len() - 1);
        assert_eq!(data.len() / self.network.layer_sizes[0], self.batch_size);
        assert_eq!(zs.len(), outputs.len());
        for (size, slice) in self.network.layer_sizes[1..].iter().zip(outputs.iter()) {
            assert_eq!(size * self.batch_size, slice.len());
        }

        let mut matmul_idx = 0;
        for layer in self.network.layers[1..].iter() {
            let current_activated_data = if matmul_idx == 0 {
                data.slice(..)
            } else {
                outputs[matmul_idx - 1].as_view()
            };
            match layer {
                crate::layer::LayerType::NODE(_) => {
                    // Matmul!
                    let output_matrix = &mut zs[matmul_idx];
                    backend.splat(
                        self.network.biases[matmul_idx].as_view(),
                        output_matrix.as_view_mut(),
                    )?;
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
                        self.network.weights[matmul_idx].as_view(),
                        current_activated_data,
                        output_matrix.as_view_mut(),
                    )?;
                    matmul_idx += 1;
                }
                crate::layer::LayerType::ACTIVATION(activation_function) => {
                    match activation_function {
                        crate::layer::ActivationFunction::SIGMOID => backend.unary(
                            "sigmoid",
                            zs[matmul_idx.checked_sub(1).unwrap()].as_view(),
                            outputs[matmul_idx.checked_sub(1).unwrap()].as_view_mut(),
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
        let mut outputs = Vec::new();
        let mut zs = Vec::new();
        for layer_size in self.network.layer_sizes[1..].iter() {
            outputs.push(backend.stream.alloc_zeros(layer_size * self.batch_size)?);
            zs.push(backend.stream.alloc_zeros(layer_size * self.batch_size)?);
        }

        self.feedforward(backend, data, &mut zs, &mut outputs)?; // Feedforwards first
        backend.synchronize()?;
        let result = backend.calculate_loss(outputs.last().unwrap().as_view(), labels.slice(..))?;
        let match_count = backend.calculate_matching(outputs.last().unwrap().as_view(), labels, OUTPUT_LABELS)?;
        Ok((result, match_count))
    }

    pub(crate) fn backprop(
        &mut self,
        backend: &GPUBackend,
        data: CudaView<f32>,
        labels: CudaView<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut outputs = Vec::new();
        let mut zs = Vec::new();
        let mut deltas = Vec::new();
        let mut weight_changes = Vec::new();
        for layer_size in self.network.layer_sizes[1..].iter() {
            outputs.push(backend.stream.alloc_zeros(layer_size * self.batch_size)?);
            zs.push(backend.stream.alloc_zeros(layer_size * self.batch_size)?);
            deltas.push(backend.stream.alloc_zeros(layer_size * self.batch_size)?);
        }
        let z_out_delta_len = outputs.len();
        for weights in self.network.weights.iter() {
            weight_changes.push(backend.stream.alloc_zeros::<f32>(weights.len())?);
        }

        self.feedforward(backend, data.slice(..), &mut zs, &mut outputs)?;
        let outputs = outputs;
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
        assert_eq!(zs.len(), activations.len());
        for (z, function) in zs.iter_mut().zip(activations.iter()) {
            backend.unary_inplace(
                match function {
                    ActivationFunction::SIGMOID => "sigmoid_prime",
                },
                z.as_view_mut(),
            )?;
        }
        let filled_activations = backend.fill_estimated(labels)?;
        backend.binary(
            "squared_diff_prime",
            outputs.last().unwrap().as_view(),
            filled_activations.as_view(),
            deltas.last_mut().unwrap().as_view_mut(),
        )?;
        for (rev_idx, layer_size) in self.network.layer_sizes[1..].iter().rev().enumerate() {
            if rev_idx != 0 {
                // We need to fill curr_delta (the last curr_delta in the chain was filled by the earlier squared_diff_prime call)
                let split = deltas.split_at_mut(z_out_delta_len - rev_idx);
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
                    self.network.weights[self.network.weights.len() - rev_idx].as_view(),
                    split.1.first().unwrap().as_view(),
                    split.0.last_mut().unwrap().as_view_mut(),
                )?;
            }
            let curr_delta = &mut deltas[z_out_delta_len - 1 - rev_idx];
            let curr_z = &zs[zs.len() - 1 - rev_idx];
            backend.binary_inplace("multiply", curr_z.as_view(), curr_delta.as_view_mut())?;
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
                curr_delta.as_view(),
                if z_out_delta_len < 2 + rev_idx {
                    // z_out_data_len is at least 2 if a hidden layer exists (hl + ol)
                    data.slice(..)
                } else {
                    assert!(z_out_delta_len >= 2 + rev_idx);
                    // if we have e.g. 3 layers (input, hl, ol) then output has 2 entries
                    // so len - 2 is legal
                    outputs[z_out_delta_len - 2 - rev_idx].as_view()
                },
                weight_changes[self.network.weights.len() - 1 - rev_idx].as_view_mut(),
            )?;
        }

        // We have now computed the weight and bias changes for each layer!

        for (change, weights) in weight_changes
            .iter_mut()
            .zip(self.network.weights.iter_mut())
        {
            backend.mult_by_val(change.as_view_mut(), LEARNING_RATE)?;
            backend.binary_inplace("add", change.as_view(), weights.as_view_mut())?; // Adjust changes by learning rate and add to weights
        }
        for ((delta, biases), layer_size) in deltas
            .iter_mut()
            .zip(self.network.biases.iter_mut())
            .zip(self.network.layer_sizes[1..].iter())
        {
            // Compute the biases for this delta
            backend.reduce_strided(
                delta.as_view_mut(),
                self.batch_size,
                *layer_size,
                *layer_size,
            )?;
            backend.mult_by_val(delta.as_view_mut().slice_mut(0..*layer_size), LEARNING_RATE)?;
            backend.binary_inplace(
                "add",
                delta.as_view().slice(0..*layer_size),
                biases.as_view_mut(),
            )?;
        }

        backend.synchronize()?;

        Ok(())
    }
}
