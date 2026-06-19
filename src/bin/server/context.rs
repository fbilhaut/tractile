use std::sync::Mutex;

use tractile::config::Configuration;
use tractile::pipeline::TextEmbeddingPipeline;

use super::config::ServerConfig;

pub struct AppContext {
    pub embedding: Option<Mutex<TextEmbeddingPipeline>>,
}

impl AppContext {
    pub fn new(config: &ServerConfig) -> tractile::Result<Self> {
        Ok(Self {
            embedding: config.embedding.as_ref().map(load_pipeline).transpose()?,
        })
    }
}

fn load_pipeline(config: &Configuration) -> tractile::Result<Mutex<TextEmbeddingPipeline>> {
    log::info!("Loading pipeline: model={:?}", config.model_path());
    let pipeline = TextEmbeddingPipeline::new(config.clone())?;
    log::info!("Pipeline ready");
    Ok(Mutex::new(pipeline))
}
