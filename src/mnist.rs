use std::ops::Range;

use cudarc::driver::{CudaSlice, CudaView};
use mnist::{Mnist, MnistBuilder};

use crate::backend::GPUBackend;

pub(crate) struct MnistManager {
    pub(crate) trn_img: CudaSlice<f32>,
    pub(crate) trn_lbl: CudaSlice<usize>,
    pub(crate) tst_img: CudaSlice<f32>,
    pub(crate) tst_lbl: CudaSlice<usize>,
}

impl MnistManager {
    pub(crate) fn new(backend: &GPUBackend) -> Result<Self, Box<dyn std::error::Error>> {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(60_000)
            .test_set_length(10_000)
            .finalize();

        let trn_lbl = backend
            .stream
            .clone_htod(&trn_lbl.iter().map(|x| *x as usize).collect::<Vec<_>>())?;
        let tst_lbl = backend
            .stream
            .clone_htod(&tst_lbl.iter().map(|x| *x as usize).collect::<Vec<_>>())?;

        let trn_img = backend.stream.clone_htod(
            &trn_img
                .iter()
                .map(|x| *x as f32 / 256.0f32)
                .collect::<Vec<_>>(),
        )?;
        let tst_img = backend.stream.clone_htod(
            &tst_img
                .iter()
                .map(|x| *x as f32 / 256.0f32)
                .collect::<Vec<_>>(),
        )?;

        Ok(Self {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
        })
    }

    pub(crate) fn get_train_data(
        &self,
        range: Range<usize>,
    ) -> (CudaView<'_, f32>, CudaView<'_, usize>) {
        let label_range = self.trn_lbl.as_view().slice(range.clone());
        let img_range = self
            .trn_img
            .as_view()
            .slice((range.start * 28 * 28)..(range.end * 28 * 28));
        (img_range, label_range)
    }

    pub(crate) fn get_test_data(
        &self,
        range: Range<usize>,
    ) -> (CudaView<'_, f32>, CudaView<'_, usize>) {
        let label_range = self.tst_lbl.as_view().slice(range.clone());
        let img_range = self
            .tst_img
            .as_view()
            .slice((range.start * 28 * 28)..(range.end * 28 * 28));
        (img_range, label_range)
    }
}
