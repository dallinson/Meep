use mnist::{Mnist, MnistBuilder};

pub(crate) struct MnistManager {
    pub(crate) trn_img: Vec<f32>,
    pub(crate) trn_lbl: Vec<usize>,
    pub(crate) tst_img: Vec<f32>,
    pub(crate) tst_lbl: Vec<usize>,
}

impl MnistManager {
    pub(crate) fn new() -> Self {
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

        let trn_lbl = trn_lbl.iter().map(|x| *x as usize).collect();
        let tst_lbl = tst_lbl.iter().map(|x| *x as usize).collect();

        let trn_img = trn_img.iter().map(|x| *x as f32 / 256.0f32).collect();
        let tst_img = tst_img.iter().map(|x| *x as f32 / 256.0f32).collect();

        Self {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
        }
    }
}
