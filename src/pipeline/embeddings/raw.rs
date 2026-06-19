use ndarray::Array2;
use tract_onnx::prelude::*;

use super::Extractor;

pub(crate) struct RawExtractor;

impl Extractor for RawExtractor {
    fn extract(
        &self,
        outputs: &TVec<TValue>,
        output_index: usize,
        _flat_mask: &[i64],
        _n: usize,
        _seq_len: usize,
    ) -> crate::Result<Array2<f32>> {
        Ok(outputs[output_index]
            .to_plain_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix2>()?
            .to_owned()
            .into_owned())
    }
}
