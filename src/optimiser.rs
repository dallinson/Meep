use cudarc::driver::{CudaSlice, CudaViewMut};

use crate::{backend::GPUBackend, network::Network};

pub(crate) trait Optimiser {
    fn optimise(
        &mut self,
        backend: &GPUBackend,
        gradient: CudaViewMut<f32>,
        weights: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>>;
    // We make the gradient mutable so optimisers can do work in-place on them
}

pub(crate) struct MultiplierOptimiser {
    scalar: f32,
}

impl Optimiser for MultiplierOptimiser {
    fn optimise(
        &mut self,
        backend: &GPUBackend,
        mut gradient: CudaViewMut<f32>,
        weights: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        backend.mult_by_val(gradient.slice_mut(..), self.scalar)?;
        backend.binary_inplace("add", gradient.as_view(), weights)?;
        Ok(())
    }
}

impl MultiplierOptimiser {
    pub(crate) fn new(scalar: f32) -> Self {
        Self { scalar }
    }
}

pub(crate) struct AdamOptimiser {
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    t: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl Optimiser for AdamOptimiser {
    fn optimise(
        &mut self,
        backend: &GPUBackend,
        gradient: CudaViewMut<f32>,
        weights: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        backend.adam(
            gradient.as_view(),
            weights,
            self.m.as_view_mut(),
            self.v.as_view_mut(),
            self.t,
            self.lr,
            self.beta1,
            self.beta2,
            self.epsilon,
        )
    }
}

impl AdamOptimiser {
    pub(crate) fn new(
        backend: &GPUBackend,
        net: &Network,
        t: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gradient_size = net.weights.len();
        let m = backend.stream.alloc_zeros(gradient_size)?;
        let v = backend.stream.alloc_zeros(gradient_size)?;

        Ok(Self {
            m,
            v,
            t,
            lr,
            beta1,
            beta2,
            epsilon,
        })
    }
}
