use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use tractile::pipeline::TextEmbeddingPipeline;

pub struct AppContext {
    pub embedding_pipeline: RwLock<Option<TextEmbeddingPipeline>>,
    ready: AtomicBool,
}

impl AppContext {
    pub fn new() -> Self {
        Self {
            embedding_pipeline: RwLock::new(None),
            ready: AtomicBool::new(false),
        }
    }

    pub fn set_pipeline(&self, pipeline: TextEmbeddingPipeline) {
        *self.embedding_pipeline.write().unwrap() = Some(pipeline);
        self.ready.store(true, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}
