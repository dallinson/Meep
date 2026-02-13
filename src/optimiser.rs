use cudarc::driver::CudaViewMut;

use crate::backend::GPUBackend;

pub(crate) trait Optimiser {
    fn optimise(&mut self, backend: &GPUBackend, gradient: CudaViewMut<f32>, weights: CudaViewMut<f32>) -> Result<(), Box<dyn std::error::Error>>;
    // We make the gradient mutable so optimisers can do work in-place on them
}

pub(crate) struct MultiplierOptimiser {
    scalar: f32,
}

impl Optimiser for MultiplierOptimiser {
    fn optimise(&mut self, backend: &GPUBackend, mut gradient: CudaViewMut<f32>, weights: CudaViewMut<f32>) -> Result<(), Box<dyn std::error::Error>> {
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