mod config;
mod input;
mod output;
mod pipeline;

pub use config::Configuration;
pub use input::TextInput;
pub use output::TextEmbeddings;
pub use pipeline::{ExtractorMode, TextEmbeddingPipeline};

pub use ndarray::{Array1, Array2, ArrayView1};

pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;
