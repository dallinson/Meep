use std::fmt::Debug;

use cudarc::driver::{CudaSlice, CudaView};

use crate::{
    backend::GPUBackend,
    layer::{Layer, LayerType},
};

pub(crate) struct Network {
    pub(crate) layers: Vec<LayerType>,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) weights: CudaSlice<f32>,
}

impl Network {
    pub(crate) fn from_layers(
        layer: &Layer,
        backend: &GPUBackend,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut layers = Vec::new();
        layers.push(layer);
        let mut layer = layer;
        while let Some(prev_layer) = layer.previous_layer {
            layers.push(prev_layer);
            layer = prev_layer;
        }
        layers.reverse();
        let layers = layers;
        let layer_sizes = layers
            .iter()
            .filter_map(|x| match x.layer_type {
                LayerType::NODE(size) => Some(size),
                _ => None,
            })
            .collect::<Vec<_>>();

        let sum = layer_sizes
            .windows(2)
            .map(|elems| elems[0] * elems[1])
            .sum::<usize>()
            + layer_sizes.iter().sum::<usize>();

        Ok(Self {
            layers: layers.iter().map(|x| x.layer_type).collect(),
            layer_sizes,
            weights: backend.stream.alloc_zeros::<f32>(sum)?,
        })
    }

    pub(crate) fn randomise_weights(
        &mut self,
        backend: &GPUBackend,
    ) -> Result<(), Box<dyn std::error::Error>> {
        backend.fill_with_uniform(self.weights.as_view_mut())?;
        Ok(())
    }

    pub(crate) fn get_weights_biases(&self) -> Vec<(CudaView<'_, f32>, CudaView<'_, f32>)> {
        let mut to_return = Vec::new();
        let mut curr_idx = 0;
        for layer_sizes in self.layer_sizes.windows(2) {
            let prev_layer_size = layer_sizes[0];
            let curr_layer_size = layer_sizes[1];
            let weights = self
                .weights
                .slice(curr_idx..(curr_idx + (prev_layer_size * curr_layer_size)));
            curr_idx += prev_layer_size * curr_layer_size;
            let biases = self.weights.slice(curr_idx..(curr_idx + curr_layer_size));
            curr_idx += curr_layer_size;
            to_return.push((weights, biases));
        }
        to_return
    }
}

impl Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sizes = self
            .layer_sizes
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        //f.debug_struct("Network").field("layer_sizes", &self.layer_sizes).finish()
        f.write_str(&sizes.join(" -> "))
    }
}
