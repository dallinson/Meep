use crate::{backend::GPUBackend, network::Network};

#[derive(Copy, Clone)]
pub(crate) enum ActivationFunction {
    SIGMOID,
}

#[derive(Copy, Clone)]
pub(crate) enum LayerType {
    NODE(usize),
    ACTIVATION(ActivationFunction),
}

pub(crate) struct Layer<'a> {
    pub(crate) layer_type: LayerType,
    pub(crate) previous_layer: Option<&'a Layer<'a>>,
}

impl<'a> Layer<'a> {
    pub(crate) fn from_this(&'a self, layer_type: LayerType) -> Self {
        Self {
            layer_type,
            previous_layer: Some(self),
        }
    }

    pub(crate) fn sigmoid(&'a self) -> Self {
        self.from_this(LayerType::ACTIVATION(ActivationFunction::SIGMOID))
    }

    pub(crate) fn matmul(&'a self, size: usize) -> Self {
        self.from_this(LayerType::NODE(size))
    }

    pub(crate) fn new(size: usize) -> Self {
        Self {
            layer_type: LayerType::NODE(size),
            previous_layer: None,
        }
    }

    pub(crate) fn compile(
        &'a self,
        backend: &GPUBackend,
    ) -> Result<Network, Box<dyn std::error::Error>> {
        Network::from_layers(self, backend)
    }
}
