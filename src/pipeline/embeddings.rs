use ndarray::Array2;
use tract_onnx::prelude::*;

/// How to extract a per-sequence embedding from the model output tensor.
#[derive(Debug, Clone, Default)]
pub enum ExtractorMode {
    /// Output is already `[n, dim]` — use each row directly (e.g. `sentence_embedding`).
    #[default]
    Raw,
    /// Output is `[n, seq_len, dim]` — take the token at the given index (e.g. `Token(0)` for CLS).
    Token(usize),
    /// Output is `[n, seq_len, dim]` — masked mean-pool over the sequence dimension.
    MeanPool,
}

pub(super) fn extract_embeddings(
    outputs: &TVec<TValue>,
    output_index: usize,
    mode: &ExtractorMode,
    flat_mask: &[i64],
    n: usize,
    seq_len: usize,
) -> crate::Result<Array2<f32>> {
    let output = &outputs[output_index];
    match mode {
        ExtractorMode::Raw => Ok(output
            .to_plain_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix2>()?
            .to_owned()
            .into_owned()),

        ExtractorMode::Token(idx) => {
            let hidden = output
                .to_plain_array_view::<f32>()?
                .into_dimensionality::<tract_ndarray::Ix3>()?;
            Ok(hidden.slice(ndarray::s![.., *idx, ..]).to_owned())
        }

        ExtractorMode::MeanPool => {
            let hidden = output
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
}
