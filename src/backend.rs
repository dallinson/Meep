use crate::OUTPUT_LABELS;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaModule, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig},
    curand::CudaRng,
    driver::{CudaContext, CudaSlice, CudaStream},
};
use std::sync::Arc;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CublasShape {
    rows: usize,
    cols: usize,
}

impl CublasShape {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
    pub(crate) fn elems(&self) -> usize {
        self.rows * self.cols
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }
}

pub(crate) struct GPUBackend {
    ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    blas: CudaBlas,
    rng: CudaRng,
    kernels: Arc<CudaModule>,
}

impl GPUBackend {
    pub(crate) fn new(ctx: Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let rng = CudaRng::new(0, stream.clone())?;
        let src = include_str!("kernels.cu");
        let ptx = cudarc::nvrtc::compile_ptx(src)?;
        let kernels = ctx.load_module(ptx)?;

        ctx.set_blocking_synchronize()?;

        Ok(Self {
            ctx,
            stream,
            blas,
            rng,
            kernels,
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn fill_with_uniform(
        &self,
        mut dat: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.rng.fill_with_uniform(&mut dat)?;
        {
            let func = self.kernels.load_function("map_uniform")?;
            let mut builder = self.stream.launch_builder(&func);
            let len = dat.len();
            builder.arg(&mut dat);
            builder.arg(&len);
            unsafe { builder.launch(LaunchConfig::for_num_elems(len as u32))? };
        }
        Ok(())
    }

    pub(crate) fn matmul_unstrided(
        &self,
        transa: bool,
        transb: bool,
        a_dims: CublasShape,
        b_dims: CublasShape,
        c_dims: CublasShape,
        alpha: f32,
        beta: f32,
        a: CudaView<f32>,
        b: CudaView<f32>,
        mut c: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let m = if transa { a_dims.cols() } else { a_dims.rows() };
        let n = if transb { b_dims.rows() } else { b_dims.cols() };
        let k = if transa { a_dims.rows() } else { a_dims.cols() };
        assert_eq!(CublasShape::new(m, n), c_dims);
        assert_eq!(a_dims.elems(), a.len());
        assert_eq!(b_dims.elems(), b.len());
        assert_eq!(c_dims.elems(), c.len());
        let cfg = GemmConfig {
            transa: if transa {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            transb: if transb {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            },
            m: m as i32,
            n: n as i32,
            k: k as i32,
            alpha: alpha,
            lda: a_dims.rows() as i32,
            ldb: b_dims.rows() as i32,
            beta: beta,
            ldc: c_dims.rows() as i32,
        };

        unsafe {
            self.blas.gemm(cfg, &a, &b, &mut c)?;
        };
        Ok(())
    }

    pub(crate) fn unary(
        &self,
        func_name: &str,
        src: CudaView<f32>,
        mut dest: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(src.len(), dest.len());
        let func = self.kernels.load_function(func_name)?;
        let mut builder = self.stream.launch_builder(&func);
        let to_adj_len = src.len();
        builder.arg(&src);
        builder.arg(&mut dest);
        builder.arg(&to_adj_len);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(to_adj_len.div_ceil(4) as u32))?;
        };
        Ok(())
    }

    pub(crate) fn unary_inplace(
        &self,
        func_name: &str,
        mut dat: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let func = self
            .kernels
            .load_function(format!("{}_inplace", func_name).as_str())?;
        let mut builder = self.stream.launch_builder(&func);
        let dat_len = dat.len();
        builder.arg(&mut dat);
        builder.arg(&dat_len);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(dat_len.div_ceil(4) as u32))?;
        };
        Ok(())
    }

    pub(crate) fn binary(
        &self,
        func_name: &str,
        a: CudaView<f32>,
        b: CudaView<f32>,
        mut dest: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), dest.len());
        let len = a.len();
        let func = self.kernels.load_function(func_name)?;
        let mut builder = self.stream.launch_builder(&func);
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&mut dest);
        builder.arg(&len);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(len.div_ceil(4) as u32))?;
        };
        Ok(())
    }

    pub(crate) fn binary_inplace(
        &self,
        func_name: &str,
        src: CudaView<f32>,
        dest: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unary(format!("{}_inplace", func_name).as_str(), src, dest)
    }

    pub(crate) fn splat(
        &self,
        src: CudaView<f32>,
        mut dest: CudaViewMut<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(dest.len() % src.len(), 0);
        let func = self.kernels.load_function("splat")?;
        let mut builder = self.stream.launch_builder(&func);
        let src_len = src.len();
        let dest_len = dest.len();
        let dest_ratio = dest_len / src_len;
        builder.arg(&src);
        builder.arg(&mut dest);
        builder.arg(&src_len);
        builder.arg(&dest_ratio);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(((dest_len / 4) + 1) as u32))?;
        };
        Ok(())
    }

    pub(crate) fn fill_estimated(
        &self,
        labels: CudaView<usize>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut labelled_outputs = self
            .stream
            .alloc_zeros::<f32>(labels.len() * OUTPUT_LABELS)?;
        {
            let func = self.kernels.load_function("fill_estimated")?;
            let mut builder = self.stream.launch_builder(&func);
            let label_len = labels.len();
            builder.arg(&labels);
            builder.arg(&mut labelled_outputs);
            builder.arg(&OUTPUT_LABELS);
            builder.arg(&label_len);
            unsafe {
                builder.launch(LaunchConfig::for_num_elems(label_len as u32))?;
            };
        }
        Ok(labelled_outputs)
    }

    pub(crate) fn calculate_loss(
        &self,
        net_output: CudaView<f32>,
        labels: CudaView<usize>,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        assert_eq!(net_output.len(), labels.len() * OUTPUT_LABELS);

        let mut labelled_outputs = self.fill_estimated(labels.slice(..))?;

        {
            let func = self.kernels.load_function("squared_diff_inplace")?;
            let mut builder = self.stream.launch_builder(&func);
            let output_len = net_output.len();
            builder.arg(&net_output);
            builder.arg(&mut labelled_outputs);
            builder.arg(&output_len);
            unsafe {
                builder.launch(LaunchConfig::for_num_elems(output_len.div_ceil(4) as u32))?;
            };
        }

        self.reduce_strided(
            labelled_outputs.as_view_mut(),
            labels.len(),
            OUTPUT_LABELS,
            OUTPUT_LABELS,
        )?;

        {
            let func = self.kernels.load_function("reduce_vecwise")?;
            let mut builder = self.stream.launch_builder(&func);
            let label_len = labels.len();
            builder.arg(&mut labelled_outputs);
            builder.arg(&OUTPUT_LABELS);
            builder.arg(&1);
            unsafe {
                builder.launch(LaunchConfig::for_num_elems(label_len as u32))?;
            };
        }

        self.synchronize()?;

        let result = self.stream.clone_dtoh(&labelled_outputs)?;

        Ok(*result.first().unwrap())
    }

    pub(crate) fn mult_by_val(
        &self,
        mut dat: CudaViewMut<f32>,
        mult: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.kernels.load_function("mult_by_float")?;
        let mut builder = self.stream.launch_builder(&func);
        let dat_len = dat.len();
        builder.arg(&mut dat);
        builder.arg(&mult);
        builder.arg(&dat_len);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(dat_len.div_ceil(4) as u32))?;
        }
        Ok(())
    }

    pub(crate) fn reduce_strided(
        &self,
        mut dat: CudaViewMut<f32>,
        elem_count: usize,
        offset: usize,
        vector_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.kernels.load_function("reduce_strided")?;
        let mut to_reduce = elem_count;
        let mut stride = offset;
        while to_reduce > 1 {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut dat);
            builder.arg(&to_reduce);
            builder.arg(&stride);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (to_reduce.div_ceil(1024) as u32, vector_size as u32, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            }
            to_reduce = to_reduce.div_ceil(32);
            stride *= 32;
        }

        Ok(())
    }

    pub(crate) fn calculate_matching(
        &self,
        outputs: CudaView<f32>,
        labels: CudaView<usize>,
        label_count: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut matches = self.stream.alloc_zeros::<f32>(labels.len())?;
        {
            let func = self.kernels.load_function("find_matching")?;
            let mut builder = self.stream.launch_builder(&func);
            let label_len = labels.len();
            builder.arg(&outputs);
            builder.arg(&labels);
            builder.arg(&mut matches);
            builder.arg(&label_count);
            builder.arg(&label_len);

            unsafe {
                builder.launch(LaunchConfig::for_num_elems(label_len as u32))?;
            };
        }

        self.reduce_strided(matches.as_view_mut(), labels.len(), 1, 1)?;

        let slice = matches.as_view().slice(0..1);

        let res = self.stream.clone_dtoh(&slice)?;
        let res = res.first().unwrap();
        Ok(*res as usize)
    }
}
