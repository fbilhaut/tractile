use std::sync::Arc;
use tract_onnx::prelude::*;

use crate::config::Configuration;

pub(super) type Model = Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>;

pub(super) fn load_model(config: &Configuration) -> crate::Result<Model> {
    let inference_model = tract_onnx::onnx().model_for_path(config.model_path())?;
    cfg_select! {
        all(feature = "metal", target_os = "macos", target_arch = "aarch64") => {
            load_metal_model(inference_model)
        }
        _ => {
            Ok(inference_model.into_optimized()?.into_runnable()?)
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn load_metal_model(inference_model: InferenceModel) -> crate::Result<Model> {
    use tract_core::transform::ModelTransform;
    use tract_gpu::fact::DeviceTypedFactExt;
    use tract_gpu::sync::{DeviceSync, DeviceSyncKind};
    use tract_metal::MetalTransform;

    let typed = inference_model.into_typed()?;
    let mut m = MetalTransform::default().transform_into(typed)?;

    let outlets = m.output_outlets()?.to_vec();
    let synced: Vec<OutletId> = outlets
        .iter()
        .map(|&o| {
            if m.outlet_fact(o)?.as_device_fact().is_some() {
                Ok(m.wire_node(
                    format!("sync_output_{}", o.node),
                    DeviceSync::new(DeviceSyncKind::ToHost),
                    &[o],
                )?[0])
            } else {
                Ok(o)
            }
        })
        .collect::<TractResult<_>>()?;
    m.select_output_outlets(&synced)?;

    Ok(m.into_optimized()?.into_runnable()?)
}
