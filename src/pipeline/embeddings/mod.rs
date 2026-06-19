pub mod mean_pool;
pub mod raw;
pub mod token;

use ndarray::Array2;
use tract_onnx::prelude::*;
use crate::config::ExtractorMode;


pub trait Extractor {
    fn extract(
        &self,
        outputs: &TVec<TValue>,
        output_index: usize,
        flat_mask: &[i64],
        n: usize,
        seq_len: usize,
    ) -> crate::Result<Array2<f32>>;
}


impl Extractor for ExtractorMode {
    fn extract(
        &self,
        outputs: &TVec<TValue>,
        output_index: usize,
        flat_mask: &[i64],
        n: usize,
        seq_len: usize,
    ) -> crate::Result<Array2<f32>> {
        match self {
            ExtractorMode::Raw => {
                raw::RawExtractor.extract(outputs, output_index, flat_mask, n, seq_len)
            }
            ExtractorMode::Token(idx) => {
                token::TokenExtractor(*idx).extract(outputs, output_index, flat_mask, n, seq_len)
            }
            ExtractorMode::MeanPool => {
                mean_pool::MeanPoolExtractor.extract(outputs, output_index, flat_mask, n, seq_len)
            }
        }
    }
}
