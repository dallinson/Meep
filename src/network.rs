use std::{fmt::Debug, sync::Arc};

use cudarc::driver::{CudaContext, CudaSlice};

use crate::{
    backend::GPUBackend,
    layer::{Layer, LayerType},
};

pub(crate) struct Network {
    pub(crate) layers: Vec<LayerType>,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) weights: Vec<CudaSlice<f32>>,
    pub(crate) biases: Vec<CudaSlice<f32>>,
}

impl Network {
    pub(crate) fn from_layers(layer: &Layer) -> Self {
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
            .collect();

        Self {
            layers: layers.iter().map(|x| x.layer_type).collect(),
            layer_sizes,
            weights: Vec::new(),
            biases: Vec::new(),
        }
    }

    pub(crate) fn init_gpu(
        &mut self,
        ctx: Arc<CudaContext>,
    ) -> Result<&mut Self, Box<dyn std::error::Error>> {
        self.weights = Vec::new();
        self.biases = Vec::new();

        for i in 1..self.layer_sizes.len() {
            self.weights.push(
                ctx.default_stream()
                    .alloc_zeros::<f32>(self.layer_sizes[i - 1] * self.layer_sizes[i])?,
            );
            self.biases.push(
                ctx.default_stream()
                    .alloc_zeros::<f32>(self.layer_sizes[i])?,
            );
        }

        Ok(self)
    }

    pub(crate) fn randomise_weights(
        &mut self,
        backend: &GPUBackend,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.layer_sizes.len() - 1 {
            backend.fill_with_uniform(self.weights[i].as_view_mut())?;
            backend.fill_with_uniform(self.biases[i].as_view_mut())?;
        }
        Ok(())
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
