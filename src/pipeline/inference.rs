use tract_onnx::prelude::*;

use super::model::Model;

pub(super) fn run_inference(model: &Model, ids: Tensor, mask: Tensor) -> crate::Result<TVec<TValue>> {
    cfg_select! {
        all(feature = "metal", target_os = "macos", target_arch = "aarch64") => {
            Ok(tract_metal::with_metal_stream(|_| model.run(tvec![ids.into(), mask.into()]))?)
        }
        _ => {
            Ok(model.run(tvec![ids.into(), mask.into()])?)
        }
    }
}
