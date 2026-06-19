use ndarray::Array2;
use tract_onnx::prelude::*;

use super::Extractor;

pub(crate) struct MeanPoolExtractor;

impl Extractor for MeanPoolExtractor {
    fn extract(
        &self,
        outputs: &TVec<TValue>,
        output_index: usize,
        flat_mask: &[i64],
        n: usize,
        seq_len: usize,
    ) -> crate::Result<Array2<f32>> {
        let hidden = outputs[output_index]
            .to_plain_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix3>()?;
        let dim = hidden.shape()[2];
        let mut result = Array2::<f32>::zeros((n, dim));
        for i in 0..n {
            let mask_sum: f32 =
                flat_mask[i * seq_len..(i + 1) * seq_len].iter().map(|&x| x as f32).sum();
            for j in 0..seq_len {
                let m = flat_mask[i * seq_len + j] as f32;
                for k in 0..dim {
                    result[[i, k]] += hidden[[i, j, k]] * m;
                }
            }
            for k in 0..dim {
                result[[i, k]] /= mask_sum;
            }
        }
        Ok(result)
    }
}
